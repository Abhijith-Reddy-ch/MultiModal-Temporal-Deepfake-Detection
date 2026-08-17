"""
Multi-Modal DeepFake Detection Model (plan.pdf architecture).

Operates on CACHED frozen features (see training/extract_features.py):
    visual   : (B, 8, 4, 384)  - DINOv2 ViT-S/14 CLS tokens, per frame per region
    audio    : (B, 8, 384)     - Whisper-tiny encoder states, mean-pooled to 8 steps
    metadata : (B, 34)         - ffprobe-derived container/codec features

Pipeline:
    Region encoders (per-region linear projection)
      -> Cross-Region Attention (face/eyes/lips/jaw attend to each other, per frame)
      -> Temporal branch: 1-layer BiLSTM over fused per-frame region vector
    Audio branch: linear projection + 1-layer BiLSTM
    Fusion Transformer (2 layers, 4 heads) over concatenated visual+audio tokens
    GMU: gate between pooled visual and pooled audio streams
    Classifier head: fused + metadata embedding -> real/fake logit
    Auxiliary head: multi-class manipulation-type logits (shared representation)
"""
import torch
import torch.nn as nn

DINO_DIM = 384
HIDDEN_DIM = 256
NUM_REGIONS = 4
FRAMES_PER_VIDEO = 8

# Unified manipulation-type taxonomy across datasets. 'real' plus every fake
# method seen in FakeAVCeleb (method column) and PolyGlotFake (sync_tech,
# lowercased so wav2lip aligns across datasets). Extend this list when
# FaceForensics++ (Deepfakes/Face2Face/FaceSwap/NeuralTextures) is added.
MANIPULATION_TYPES = [
    "real",
    "faceswap",
    "faceswap-wav2lip",
    "fsgan",
    "fsgan-wav2lip",
    "rtvc",
    "wav2lip",
    "video_retalking",
    "unknown_fake",
]
MANIPULATION_TYPE_TO_IDX = {name: i for i, name in enumerate(MANIPULATION_TYPES)}

# Dataset-source "domains" for the domain-adversarial generalization
# experiment (DFDC investigation, attempt 7). Deliberately kept separate from
# MANIPULATION_TYPES: aux_forgery_head is trained to be forgery-type-SENSITIVE
# (plan.pdf's multi-task design), while domain_head below is trained via
# gradient reversal to be dataset-source-INVARIANT. Conflating the two axes
# would have the two heads fighting the same shared representation. DFDC is
# never assigned a domain id (it's never in train/val/test), consistent with
# never touching it during training.
DOMAINS = ["FakeAVCeleb", "PolyGlotFake", "FFPP"]
DOMAIN_TO_IDX = {name: i for i, name in enumerate(DOMAINS)}


# ------------------------------------------------------------------ #
#  Gradient Reversal Layer (Ganin & Lempitsky, 2016)                   #
# ------------------------------------------------------------------ #
class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Identity in the forward pass; negates and scales the gradient in the
    backward pass. Placed between `fused` and the domain classifier head so
    that the domain head itself trains normally (learns to actually predict
    dataset source), while everything upstream (region encoders, cross-region
    attention, BiLSTMs, fusion transformer, GMU, and in Stage B the LoRA
    backbone weights) receives the reversed gradient, pushing the shared
    representation to become non-predictive of dataset source instead."""

    def __init__(self, lambda_: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, value: float):
        self.lambda_ = value

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)


# ------------------------------------------------------------------ #
#  Cross-Region Attention                                              #
# ------------------------------------------------------------------ #
class CrossRegionAttention(nn.Module):
    """Lets face/eyes/lips/jaw region tokens attend to each other, per frame.

    Input:  (B, T, R, D)  - R=4 regions
    Output: (B, T, D)     - fused per-frame region vector (mean-pooled post-attention)
    """

    def __init__(self, dim: int = HIDDEN_DIM, num_regions: int = NUM_REGIONS, num_heads: int = 4):
        super().__init__()
        self.region_embed = nn.Parameter(torch.zeros(num_regions, dim))
        nn.init.trunc_normal_(self.region_embed, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, R, D = x.shape
        x = x + self.region_embed.unsqueeze(0).unsqueeze(0)  # broadcast region identity
        x = x.view(B * T, R, D)
        attn_out, _ = self.attn(x, x, x)
        fused = self.norm(attn_out + x)  # residual
        fused = fused.mean(dim=1)  # pool over regions -> (B*T, D)
        return fused.view(B, T, D)


# ------------------------------------------------------------------ #
#  Visual branch: region projection + cross-region attention + BiLSTM #
# ------------------------------------------------------------------ #
class VisualBranch(nn.Module):
    def __init__(self, in_dim: int = DINO_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.region_proj = nn.Linear(in_dim, hidden_dim)
        self.cross_region_attn = CrossRegionAttention(dim=hidden_dim)
        # 1-layer BiLSTM - plan.pdf explicitly warns 2-layer BiLSTM overfits on small data
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )

    def forward(self, visual):
        # visual: (B, T, R, in_dim)
        B, T, R, _ = visual.shape
        x = self.region_proj(visual)          # (B, T, R, hidden_dim)
        x = self.cross_region_attn(x)          # (B, T, hidden_dim)
        x, _ = self.lstm(x)                    # (B, T, hidden_dim)  (hidden//2 * 2 directions)
        return x


# ------------------------------------------------------------------ #
#  Audio branch: linear projection + BiLSTM over cached Whisper feats  #
# ------------------------------------------------------------------ #
class AudioBranch(nn.Module):
    def __init__(self, in_dim: int = DINO_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )

    def forward(self, audio):
        # audio: (B, T, in_dim)
        x = self.proj(audio)
        x, _ = self.lstm(x)
        return x  # (B, T, hidden_dim)


# ------------------------------------------------------------------ #
#  GMU: Gated Multimodal Unit                                          #
# ------------------------------------------------------------------ #
class GMU(nn.Module):
    def __init__(self, dim: int = HIDDEN_DIM):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, visual_pooled, audio_pooled, return_gate=False):
        z = self.gate(torch.cat([visual_pooled, audio_pooled], dim=-1))
        fused = z * visual_pooled + (1 - z) * audio_pooled
        if return_gate:
            return fused, z
        return fused


# ------------------------------------------------------------------ #
#  Full model                                                          #
# ------------------------------------------------------------------ #
class DeepfakeClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_manipulation_types: int = len(MANIPULATION_TYPES),
        metadata_dim: int = 34,
    ):
        super().__init__()
        self.visual_branch = VisualBranch(hidden_dim=hidden_dim)
        self.audio_branch = AudioBranch(hidden_dim=hidden_dim)

        # Modality embeddings so the fusion transformer can tell visual vs audio tokens apart
        self.visual_modality_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.audio_modality_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.visual_modality_embed, std=0.02)
        nn.init.trunc_normal_(self.audio_modality_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.2, batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.gmu = GMU(dim=hidden_dim)

        # Domain-adversarial branch (DFDC investigation, attempt 7): see
        # GradientReversalLayer docstring. lambda_ starts at 0 (no adversarial
        # pressure) and is ramped up per-epoch by the training script via
        # domain_grl.set_lambda(); it stays at 0 (a no-op) unless a training
        # script explicitly schedules it, so this is inert by default.
        self.domain_grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, len(DOMAINS)),
        )

        self.metadata_encoder = nn.Sequential(nn.Linear(metadata_dim, 32), nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
        )

        # Auxiliary multi-task head: predicts manipulation_type from the shared
        # representation, forcing it to be artifact-sensitive, not just binary-sensitive.
        self.aux_forgery_head = nn.Linear(hidden_dim, num_manipulation_types)

    def forward(self, visual, audio, metadata=None, return_features=False):
        # visual: (B, 8, 4, 384), audio: (B, 8, 384), metadata: (B, 34)
        B = visual.shape[0]

        v = self.visual_branch(visual)          # (B, T, hidden_dim)
        a = self.audio_branch(audio)            # (B, T, hidden_dim)

        v = v + self.visual_modality_embed
        a = a + self.audio_modality_embed

        tokens = torch.cat([v, a], dim=1)       # (B, 2T, hidden_dim)
        fused_tokens = self.fusion_transformer(tokens)

        T = v.shape[1]
        visual_out, audio_out = fused_tokens[:, :T], fused_tokens[:, T:]
        visual_pooled = visual_out.mean(dim=1)  # (B, hidden_dim)
        audio_pooled = audio_out.mean(dim=1)    # (B, hidden_dim)

        fused, gmu_gate = self.gmu(visual_pooled, audio_pooled, return_gate=True)  # (B, hidden_dim), (B, hidden_dim)

        if metadata is None:
            metadata = torch.zeros((B, 34), dtype=torch.float32, device=visual.device)
        meta_emb = self.metadata_encoder(metadata)      # (B, 32)

        logits = self.classifier(torch.cat([fused, meta_emb], dim=-1)).squeeze(-1)  # (B,)
        aux_logits = self.aux_forgery_head(fused)       # (B, num_manipulation_types)
        domain_logits = self.domain_head(self.domain_grl(fused))  # (B, num_domains)

        if return_features:
            # gmu_gate: (B, hidden_dim) elementwise visual-vs-audio weight (1=all
            # visual, 0=all audio); mean over hidden_dim gives a single scalar
            # summary per sample - the real interpretable signal plan.pdf's
            # explainability phase calls for, not a recomputation approximation.
            return logits, {
                "aux_forgery_logits": aux_logits,
                "domain_logits": domain_logits,
                "fused": fused,
                "gmu_gate": gmu_gate.mean(dim=-1),  # (B,) in [0,1], 1=visual-dominant
            }
        return logits


_DOMAIN_HEAD_KEYS = {
    "domain_head.0.weight", "domain_head.0.bias",
    "domain_head.2.weight", "domain_head.2.bias",
}


def load_classifier_state_dict(model: "DeepfakeClassifier", state_dict: dict, label: str = "checkpoint"):
    """Loads a state dict into a DeepfakeClassifier with strict=False, so
    checkpoints saved before the domain-adversarial branch (domain_head) was
    added still load cleanly - those new parameters just stay at their random
    init, which is harmless: domain_grl.lambda_ defaults to 0, so the branch
    is a no-op unless a training script explicitly enables
    USE_DOMAIN_ADVERSARIAL and schedules lambda > 0. Any OTHER
    missing/unexpected key (i.e. not explained by this specific addition) is
    printed as a warning rather than silently swallowed by strict=False,
    since that flag can also mask real bugs unrelated to this change."""
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_missing = set(result.missing_keys) - _DOMAIN_HEAD_KEYS
    if unexpected_missing or result.unexpected_keys:
        print(f"[WARNING] loading {label}: state_dict mismatch beyond the known "
              f"domain-adversarial addition - missing={sorted(unexpected_missing) or 'none'}, "
              f"unexpected={result.unexpected_keys or 'none'}")
    elif result.missing_keys:
        print(f"[{label}] loaded a pre-domain-adversarial checkpoint - domain_head "
              f"initialized fresh (inert unless USE_DOMAIN_ADVERSARIAL=1).")


def load_optimizer_state_dict(optimizer, state_dict: dict, label: str = "checkpoint"):
    """Loads an optimizer state dict, falling back to the optimizer's fresh
    (just-constructed) state if the parameter groups don't match - this
    happens when resuming a checkpoint saved before the domain_head
    parameters existed (see load_classifier_state_dict above: the model now
    has more parameters than the checkpoint's optimizer state accounts for).
    A warm-started model with a freshly-initialized optimizer is a standard,
    safe fallback (Adam re-adapts its moment estimates quickly); crashing on
    a version mismatch is not."""
    try:
        optimizer.load_state_dict(state_dict)
    except ValueError as e:
        print(f"[WARNING] optimizer state in {label} incompatible with the current model "
              f"(likely a pre-domain-adversarial checkpoint) - continuing with a freshly "
              f"initialized optimizer instead: {e}")
