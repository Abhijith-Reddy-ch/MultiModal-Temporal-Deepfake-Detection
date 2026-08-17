"""
Explainability (plan.pdf Phase 7) for the DINOv2 + Whisper + cross-region
attention + GMU architecture:

  1. Per-region Grad-CAM over the last DINOv2 block's CLS-to-patch
     attention (ViT equivalent of CNN Grad-CAM: gradient-weighted
     attention-row activations, reshaped to a 16x16 grid and upsampled -
     patch tokens themselves are a dead end for gradients since only the
     CLS token is consumed downstream), one heatmap per region per frame.
  2. Attention rollout through the fusion Transformer's 2 layers, giving
     a per-frame visual-vs-audio importance breakdown (Abnar & Zuidema
     2020 method: multiply per-layer attention matrices with residual
     connections folded in).
  3. GMU gate value (already computed in training/model.py's forward) -
     the single scalar visual-vs-audio summary.

These three are composed into one `explain_video()` output per plan.pdf's
"compose these three into one output per inference" instruction.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pipeline.phase5_extract import REGIONS, FRAMES_PER_VIDEO


# ------------------------------------------------------------------ #
#  1. Per-region Grad-CAM over DINOv2 patch tokens                     #
# ------------------------------------------------------------------ #
def _get_dinov2_base(dinov2):
    """Unwraps a PeftModel (Stage B, LoRA) to reach the underlying
    DinoVisionTransformer if needed; forward_features works on either."""
    return dinov2.get_base_model() if hasattr(dinov2, "get_base_model") else dinov2


def _eager_attention_with_weights(attn_module, x):
    """Recomputes one DINOv2 Attention module's forward with an explicit
    softmax instead of the fused scaled_dot_product_attention/xformers
    kernel, so the attention matrix is a real tensor we can retain_grad()
    on. Works transparently through PEFT/LoRA-wrapped qkv/proj Linears
    too, since we still call them as submodules (LoRA's delta is applied
    inside their own forward)."""
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)
    attn_weights = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn_weights = attn_weights.softmax(dim=-1)  # (B, heads, N, N)
    out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
    out = attn_module.proj_drop(attn_module.proj(out))
    return out, attn_weights


def _eager_block_forward_with_attn(block, x):
    """Replays one DINOv2 Block's forward (norm1 -> attn -> residual ->
    norm2 -> mlp -> residual) swapping in the eager attention above.
    DropPath is identity outside training and the model always runs in
    eval() here, so it's safely omitted."""
    attn_out, attn_weights = _eager_attention_with_weights(block.attn, block.norm1(x))
    x = x + block.ls1(attn_out)
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x, attn_weights


def forward_features_with_last_attn(dinov2_base, x):
    """Runs DINOv2's forward_features but replays the LAST block with an
    explicit attention. x_norm_patchtokens is otherwise a dead end for
    Grad-CAM: it's a sibling slice of x_norm_clstoken (both come from
    indexing the same self.norm(x) output) and is never consumed
    downstream (only the CLS token feeds the classifier), so its gradient
    is always None - there's no path from the logit back into it. The
    CLS row of the last block's own attention-to-patches, by contrast,
    IS on that path (the CLS token's block output is literally a
    weighted sum over patch value vectors using these weights), so it's
    a valid, class-discriminative Grad-CAM target."""
    x = dinov2_base.prepare_tokens_with_masks(x)
    for blk in dinov2_base.blocks[:-1]:
        x = blk(x)
    x, attn_weights = _eager_block_forward_with_attn(dinov2_base.blocks[-1], x)
    attn_weights.retain_grad()
    x_norm = dinov2_base.norm(x)
    n_reg = dinov2_base.num_register_tokens
    cls_token = x_norm[:, 0]
    patch_tokens = x_norm[:, n_reg + 1:]
    return cls_token, patch_tokens, attn_weights


def attn_weights_to_cam(attn_weights, grad, num_register_tokens=0, size=224):
    """Grad-CAM formula applied to the last block's CLS-to-patch attention
    row instead of CNN feature maps: weight = gradient of the logit w.r.t.
    each attention weight, activation = the attention weight itself
    (already spatial - one value per patch), summed over heads (the ViT
    analogue of Grad-CAM's channel sum), ReLU'd, reshaped 256 -> 16x16,
    upsampled."""
    patch_start = num_register_tokens + 1
    cls_row = attn_weights[:, :, 0, patch_start:]      # (B, heads, num_patches)
    cls_row_grad = grad[:, :, 0, patch_start:]         # (B, heads, num_patches)
    cam = (cls_row_grad * cls_row).sum(dim=1)          # (B, num_patches)
    cam = F.relu(cam).view(-1, 16, 16)
    cam = F.interpolate(cam.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False)
    cam = cam.squeeze(1).detach().cpu().numpy()

    out = np.zeros_like(cam)
    for i in range(cam.shape[0]):
        cmin, cmax = cam[i].min(), cam[i].max()
        out[i] = (cam[i] - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else 0.0
    return out.astype(np.float32)


def overlay_heatmap(bgr_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """heatmap: (H, W) in [0,1]. Returns a BGR image with a jet colormap overlay."""
    heat_uint8 = np.uint8(255 * heatmap)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    if bgr_image.shape[:2] != heatmap.shape:
        bgr_image = cv2.resize(bgr_image, (heatmap.shape[1], heatmap.shape[0]))
    return cv2.addWeighted(bgr_image, 1 - alpha, heat_color, alpha, 0)


# ------------------------------------------------------------------ #
#  2. Attention rollout through the fusion Transformer                #
# ------------------------------------------------------------------ #
def _replay_transformer_layer_capturing_attn(layer, x):
    """Manually replays one nn.TransformerEncoderLayer's forward (post-norm,
    the default/what training/model.py uses) so we can request attention
    weights from the self_attn submodule directly - the encoder's built-in
    forward doesn't expose them (may use a fused kernel that never
    materializes the weight matrix)."""
    attn_out, attn_weights = layer.self_attn(
        x, x, x, need_weights=True, average_attn_weights=False
    )  # attn_weights: (B, num_heads, T, T)
    x = layer.norm1(x + layer.dropout1(attn_out))
    ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
    x = layer.norm2(x + layer.dropout2(ff_out))
    return x, attn_weights


def compute_attention_rollout(fusion_transformer, tokens):
    """
    tokens: (1, 2T, hidden_dim) - concatenated [visual(T); audio(T)] tokens,
    exactly what training/model.py feeds into self.fusion_transformer.

    Returns: (2T, 2T) rolled-out attention matrix (Abnar & Zuidema 2020):
    residual connections are folded in by averaging each layer's attention
    with the identity before multiplying across layers, since a token's
    total influence includes "attending to itself" via the residual path.
    """
    x = tokens
    seq_len = tokens.shape[1]
    rollout = torch.eye(seq_len, device=tokens.device)

    for layer in fusion_transformer.layers:
        x, attn_weights = _replay_transformer_layer_capturing_attn(layer, x)
        attn_avg_heads = attn_weights.mean(dim=1).squeeze(0)  # (T, T), avg over heads
        attn_with_residual = 0.5 * attn_avg_heads + 0.5 * torch.eye(seq_len, device=tokens.device)
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        rollout = attn_with_residual @ rollout

    return rollout.detach().cpu().numpy(), x


def visual_audio_importance_per_frame(rollout: np.ndarray, num_frames: int = FRAMES_PER_VIDEO):
    """
    rollout: (2T, 2T) matrix from compute_attention_rollout, tokens ordered
    [visual_0..visual_{T-1}, audio_0..audio_{T-1}].

    Returns per-frame dicts: how much each output position's rolled-out
    attention mass falls on visual vs audio input tokens, averaged over all
    output positions - a per-frame temporal breakdown that complements the
    single-scalar GMU gate value.
    """
    T = num_frames
    # Average attention *received* by each input token, across all outputs
    received = rollout.mean(axis=0)  # (2T,)
    visual_received = received[:T]
    audio_received = received[T:]

    total = float(visual_received.sum() + audio_received.sum())
    if total <= 1e-8:
        visual_received = np.ones(T) / (2 * T)
        audio_received = np.ones(T) / (2 * T)
        total = 1.0

    per_frame = []
    for i in range(T):
        v, a = float(visual_received[i]), float(audio_received[i])
        s = v + a
        per_frame.append({
            "frame": i,
            "visual_importance": v / total,
            "audio_importance": a / total,
            "visual_vs_audio_ratio": (v / s) if s > 1e-8 else 0.5,
        })
    return per_frame


# ------------------------------------------------------------------ #
#  3. Compose everything for one video                                #
# ------------------------------------------------------------------ #
def explain_video(loaded_model, video_path, out_dir):
    """
    loaded_model: an infer.LoadedModel (dinov2, whisper_encoder, classifier, device, ...)
    video_path: Path to the video file
    out_dir: Path to write heatmap images into

    Returns a dict combining all three explainability signals, ready to
    attach to a /predict-style API response.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms
    import whisper as whisper_module

    from pipeline.phase5_extract import FaceRegionExtractor
    from training.model import MANIPULATION_TYPES

    device = loaded_model.device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    tmp_dir = Path(tempfile.mkdtemp())
    cudnn_was_enabled = torch.backends.cudnn.enabled
    try:
        audio_path = tmp_dir / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", "-loglevel", "error", str(audio_path)],
            check=False,
        )
        if not audio_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                 "-t", "1", "-acodec", "pcm_s16le", "-loglevel", "error", str(audio_path)],
                check=False,
            )

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Could not read frames from {video_path}")
        indices = [int(i * total / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]

        extractor = FaceRegionExtractor()
        frame_bgr_crops = []  # [(frame_idx, {region: bgr_crop}), ...]
        region_tensors = []
        for frame_idx, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            crops = extractor.extract(frame)
            frame_bgr_crops.append((frame_idx, crops))
            for region in REGIONS:
                pil = Image.fromarray(cv2.cvtColor(crops[region], cv2.COLOR_BGR2RGB))
                region_tensors.append(transform(pil))
        cap.release()

        # --- One connected forward pass (DINOv2 -> ... -> classifier) so the
        # backward pass reflects each region-frame's TRUE contribution to
        # the actual fake/real decision, not just generic feature strength.
        # cuDNN's fused LSTM kernel can't run backward() in eval mode, so it's
        # disabled here (falls back to the native LSTM impl, which can) -
        # this keeps dropout correctly off (unlike switching to .train()). ---
        loaded_model.classifier.eval()
        torch.backends.cudnn.enabled = False
        base_dinov2 = _get_dinov2_base(loaded_model.dinov2)

        all_images = torch.stack(region_tensors).to(device)  # (32, 3, 224, 224)
        all_images.requires_grad_(True)

        cls_tokens, patch_tokens, last_attn_weights = forward_features_with_last_attn(base_dinov2, all_images)

        visual = cls_tokens.view(1, FRAMES_PER_VIDEO, len(REGIONS), 384)

        audio_raw = whisper_module.load_audio(str(audio_path))
        audio_raw = whisper_module.pad_or_trim(audio_raw)
        mel = whisper_module.log_mel_spectrogram(audio_raw, n_mels=loaded_model.n_mels).unsqueeze(0).to(device)
        hidden = loaded_model.whisper_encoder(mel)
        hidden_t = hidden.transpose(1, 2)
        audio_feat = F.adaptive_avg_pool1d(hidden_t, FRAMES_PER_VIDEO).transpose(1, 2)

        v = loaded_model.classifier.visual_branch(visual)
        a = loaded_model.classifier.audio_branch(audio_feat)
        v_embed = v + loaded_model.classifier.visual_modality_embed
        a_embed = a + loaded_model.classifier.audio_modality_embed
        tokens = torch.cat([v_embed, a_embed], dim=1)

        rollout, fused_tokens = compute_attention_rollout(loaded_model.classifier.fusion_transformer, tokens)
        per_frame_importance = visual_audio_importance_per_frame(rollout)

        T = FRAMES_PER_VIDEO
        visual_pooled = fused_tokens[:, :T].mean(dim=1)
        audio_pooled = fused_tokens[:, T:].mean(dim=1)
        fused, gmu_gate = loaded_model.classifier.gmu(visual_pooled, audio_pooled, return_gate=True)
        meta_zeros = torch.zeros((1, 34), dtype=torch.float32, device=device)
        meta_emb = loaded_model.classifier.metadata_encoder(meta_zeros)
        logits = loaded_model.classifier.classifier(torch.cat([fused, meta_emb], dim=-1)).squeeze(-1)
        aux_logits = loaded_model.classifier.aux_forgery_head(fused)

        # Backprop from the real fake-logit - this is the actual decision,
        # so the resulting per-patch gradients are true class-discriminative
        # Grad-CAM, not a generic activation-strength proxy.
        loaded_model.classifier.zero_grad(set_to_none=True)
        loaded_model.dinov2.zero_grad(set_to_none=True)
        logits.sum().backward()

        prob = torch.sigmoid(logits).item()
        aux_probs = torch.softmax(aux_logits, dim=-1).squeeze(0).detach().cpu().tolist()
        gate_scalar = gmu_gate.mean().item()

        # --- Build Grad-CAM heatmaps from the gradients just computed -----
        gradcam_paths = {}
        grad = last_attn_weights.grad  # (32, heads, N, N)
        if grad is not None:
            n_reg = getattr(base_dinov2, "num_register_tokens", 0)
            cams = attn_weights_to_cam(last_attn_weights.detach(), grad, num_register_tokens=n_reg)  # (32, 224, 224)
            for i, (frame_idx, crops) in enumerate(frame_bgr_crops):
                for r_idx, region in enumerate(REGIONS):
                    flat_idx = i * len(REGIONS) + r_idx
                    overlay = overlay_heatmap(crops[region], cams[flat_idx])
                    out_name = f"gradcam_frame{frame_idx}_{region}.jpg"
                    cv2.imwrite(str(out_dir / out_name), overlay)
                    gradcam_paths.setdefault(region, {})[frame_idx] = str(out_dir / out_name)

        return {
            "fake_probability": prob,
            "gmu_gate_visual_weight": gate_scalar,
            "gmu_gate_audio_weight": 1.0 - gate_scalar,
            "manipulation_type_breakdown": dict(zip(MANIPULATION_TYPES, aux_probs)),
            "attention_rollout_per_frame": per_frame_importance,
            "gradcam_image_paths": gradcam_paths,
        }
    finally:
        torch.backends.cudnn.enabled = cudnn_was_enabled
        shutil.rmtree(tmp_dir, ignore_errors=True)
