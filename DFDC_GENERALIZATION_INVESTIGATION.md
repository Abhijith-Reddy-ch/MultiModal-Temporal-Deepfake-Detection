# DFDC Cross-Dataset Generalization — Investigation Log

## The problem

The deployed model (DINOv2 ViT-S/14 + Whisper-tiny + cross-region attention +
BiLSTM + fusion transformer + GMU, per `plan.pdf`) reaches **AUC 0.99+ /
~97-99% accuracy** on in-distribution data (FakeAVCeleb + PolyGlotFake + FFPP
test splits), but only **AUC ~0.65-0.71** on DFDC — the one dataset that was
held out from training/val/threshold-picking from day one
(`outputs/manifests/held_out_crossdataset.csv`), specifically to catch this
failure mode.

Root cause (confirmed, not a bug): the training data only spans 3-4 closely
related generation-method families. The model learns per-dataset
generator/compression fingerprints instead of generic forgery artifacts, so
it doesn't transfer to DFDC's different generation methods. Two obvious
shortcut explanations were investigated and ruled out early on: (1)
`media_metadata.json` was silently broken (all-zero metadata for every
video, every experiment) — ruled out a metadata shortcut, since metadata
contributed nothing; (2) audio presence/silence doesn't correlate with
real/fake label within any dataset — ruled out an audio-silence shortcut.

## Model selection methodology (corrected 2026-08-16)

**A reviewer flagged that attempt #4 was selected as canonical using DFDC
performance** ("gives the highest DFDC accuracy ... and is our canonical
configuration") — meaning DFDC was not actually a blind held-out set for
model-selection purposes, even though it was never used for training,
hyperparameter tuning, or threshold calibration. This is a real
methodological flaw, not a misreading.

**Fix applied**: attempt #1 (Stage A, frozen backbone) is a Stage B
prerequisite, not a competing deployable configuration — `train.py`
confirms Stage B always loads and fine-tunes from `stageA_best_model.pth`.
The real candidate pool is attempts #2–#6 (five Stage B variants). Selecting
canonical purely by **validation AUC** (never DFDC) among these five:

| Rank | Attempt | Val AUC |
|---|---|---|
| 1 | #2 (LoRA only, no aug/reg) | **0.9976** |
| 2 | #5 (+SBI) | 0.9972 |
| 3 | #3 (+aug) | 0.9971 |
| 4 | #4 (+weight_decay) | 0.9962 |
| 5 | #6 (+SRM) | 0.9945 |

**Attempt #2 is now canonical**, deployed 2026-08-16 by copying
`models/stageB_best_classifier_v2_noaug.pth` /
`stageB_best_dinov2_lora_v2_noaug/` / `stageB_best_whisper_lora_v2_noaug/`
over the unsuffixed canonical filenames (attempt #4's content remains fully
preserved at `models/archive_v4_weightdecay/`, so nothing was lost).
Re-ran `training/evaluate_stage_b.py` against the new canonical checkpoint
to regenerate a fresh val-derived threshold and confirm DFDC numbers.

**Notable finding**: attempt #2's DFDC AUC (0.6912) is *not* the best of
the five — it's one of the two weakest (attempts #4 and #5 both score
higher on DFDC). This is itself informative: genuinely DFDC-blind selection
does not land on a better cross-dataset performer, reinforcing that none of
the tested interventions (augmentation, weight decay, SBI) move the needle
in a way validation performance would ever surface.

## Six attempts to close the gap

| # | Attempt | DFDC AUC | DFDC Acc | DFDC Real Acc | DFDC Fake Acc | In-dist AUC | Status |
|---|---|---|---|---|---|---|---|
| 1 | Stage A (frozen backbone) | 0.6914 | 52.00% | 77.92% | 45.82% | 0.9978 | prerequisite, not a candidate |
| 2 | **Stage B (LoRA, no aug)** | 0.6912 | 55.50% | 70.13% | 52.01% | **0.9976** | **CANONICAL (validation-selected, current)** |
| 3 | Stage B + augmentation | 0.6683 | 60.75% | 66.23% | 59.44% | 0.9971 | archived |
| 4 | Stage B + augmentation + weight_decay=1e-4 | 0.7105 | 50.25% | 84.42% | 42.11% | 0.9962 | archived (previously deployed, best DFDC AUC/Acc of the five) |
| 5 | #4 + Self-Blended Images (SBI) | 0.7127 | 42.75% | 90.91% | 31.27% | 0.9972 | archived |
| 6 | #4 + SRM frequency-domain branch | 0.6529 | 43.50% | 88.31% | 32.82% | 0.9945 | archived, code reverted |

**Honest assessment**: ~0.69 AUC (canonical, validation-selected) is not a
fix. Six attempts spanning regularization, augmentation, synthetic data
(SBI), and an added input modality (frequency branch) have now all landed
in the same ~0.65-0.71 AUC band — strong evidence the bottleneck is
training-data generator diversity (or a genuine backbone capacity ceiling),
not something fixable by further tuning the same FakeAVCeleb/PolyGlotFake/FFPP
data. This conclusion is now on firmer footing than before, since it no
longer rests on a DFDC-informed choice among the six.

**Known design limitation (not yet fixed)**: attempts #4–#6 are built
cumulatively on top of #4's weight-decay setting rather than independently
on top of #2 or #3, so the six rows are not a clean one-factor-at-a-time
ablation. A factorial or single-baseline-relative redesign would isolate
each intervention's contribution more cleanly; flagged in the paper's
Limitations section as future work, not fixed in this pass.

**Diagnostics caveat**: the error analysis and Grad-CAM spot-check below
were run against attempt #4's DFDC predictions (when #4 was still deployed),
not re-run against the now-canonical #2. Re-running them against #2 is
cheap, high-value follow-up work — not yet done.

---

## Attempt 5 detail: Self-Blended Images (SBI)

*Shiohara & Yamasaki, CVPR 2022.* Synthesizes pseudo-fakes from real videos
alone (no generator involved) by blending a warped/color-shifted copy of a
face back onto itself via a landmark convex-hull mask — teaches the model
the generic "blending-boundary artifact" common to many face-forgery
methods, rather than any one dataset's specific fingerprint.

- Built `pipeline/phase5c_sbi_augment.py`: generated SBI pseudo-fakes from
  all 2558 real train-split videos (checkpointed/resumable, verified
  visually and via pixel-diff).
- Trained Stage B fresh with `USE_SBI_AUGMENTATION=1`: 15 max epochs,
  early-stopped at epoch 11, best val AUC 0.9957 (epoch 7).
- **Result**: DFDC AUC 0.7127 — statistically flat vs. the 0.7105 baseline.
  Its val-derived threshold (0.970, unusually high) gave *worse* practical
  DFDC accuracy (42.75%, fake-recall 31.3%) than the baseline's threshold
  (50.25% accuracy, fake-recall 42.11%) despite the near-identical AUC.
- **Decision**: restored the weight-decay winner as canonical. SBI
  checkpoints preserved at `models/archive_v5_sbi/`, not deployed. Full
  detail in `SBI_AUGMENTATION_STATUS.md`.

## Diagnostics run before attempt 6

Rather than guess at a 6th lever blind, ran two cheap diagnostics first:

**1. DFDC error analysis** (`scripts/dfdc_error_analysis.py`) — broke down
the canonical model's DFDC predictions by technical video properties.
Findings:
- Resolution (1920x1080), fps (29.97), codec (h264), bitrate: **identical**
  across correctly-classified and misclassified videos, both real and fake.
  Ruled out a covariate-shift / technical-artifact explanation.
- Misclassifications are **not borderline** — the model is extremely
  confident and extremely wrong. Median probability for the 214/323 missed
  fakes was 0.0138 (confidently "real"); for correctly-classified fakes it
  was 0.9991. Same story in reverse for the 9 misclassified reals (median
  prob 0.9995, confidently "fake"). This ruled out "just needs threshold
  recalibration" — the failures are a genuine wrong decision, not a
  miscalibrated boundary case.
- Missed fakes don't cluster on a small number of source identities (214
  misses trace to 160 distinct DFDC source videos, ~proportional spread).

**2. Grad-CAM spot-check** (`scripts/gradcam_spotcheck_dfdc.py` +
`training/explainability.py`) — ran Grad-CAM on the 6 most-confidently-wrong
DFDC fakes. Consistent finding across all of them: the model's visual
attention is dominated by **spatial boundary/edge cues** (hairline-to-
background boundaries, glasses frame edges, jaw-to-background contours) and
largely ignores intra-face skin texture, where classic GAN artifacts would
be expected. GMU gate showed visual and audio contributing roughly equally
(~48-50% visual weight), so this isn't an "audio-only model" issue.

**Interpretation**: the model over-relies on boundary/seam cleanliness as
its primary visual cue (unsurprising — that's exactly what SBI and FFPP-style
training fakes teach), but DFDC's specific blending doesn't trigger that cue
the same way, so the model falls back to confidently predicting "real." This
diagnosis directly motivated attempt #6 (a frequency-domain signal,
orthogonal to spatial boundary sharpness) — but as documented below, that
didn't pan out either. **The diagnostic findings themselves remain valid and
reusable** even though this specific fix didn't work; a future attempt
should target the same root cause (over-reliance on a single cue that
doesn't transfer) with a different technique.

## Attempt 6 detail: SRM frequency-domain branch (reverted)

Added a fixed (non-trainable) SRM-style high-pass residual descriptor
(4 classic forensic kernels — horizontal/vertical edge, Laplacian, SRM
"SQUARE 5x5" — over a 4x4 spatial grid = 64-dim per region/frame) as a third
model input modality alongside visual (DINOv2) and audio (Whisper),
extending the GMU fusion gate from a 2-way sigmoid to an N-way softmax.
Threaded through the entire stack: Stage A feature caching, Stage B raw-crop
loading, both training scripts, both eval scripts, `infer.py`,
`explainability.py`, `backend/app.py`, and the Next.js frontend's
`ModalityChart`/`PerFrameChart` components.

Full re-extraction (~45k videos, ~2.5-3hrs), fresh Stage A (~15min, best val
AUC 0.9936), fresh Stage B (early-stopped epoch 11, best val AUC 0.9942 at
epoch 7) — all completed successfully with no errors.

**Result: DFDC AUC 0.6529 — worse than every prior attempt**, including
doing nothing extra. In-distribution performance was essentially unaffected
(AUC 0.9945, ~97.5% accuracy), so the regression is specific to
cross-dataset generalization: the added modality most likely gave the model
more capacity to fit training-distribution-specific artifacts through a new
channel, without capturing anything that actually transfers to DFDC.

**Full revert performed**: since the new architecture also made the old,
better-performing weight-decay checkpoint impossible to load (incompatible
state dict), and the regression was clear and consistent, the user chose a
full revert over keeping the 3-modality code behind a flag or shipping the
regression. All 12 touched files reverted to their pre-frequency-branch
state, `training/freq_features.py` deleted entirely, canonical model
restored to the weight-decay winner (verified to load cleanly and produce
correct predictions post-revert). Frequency-branch checkpoints preserved at
`models/archive_v6_freqbranch/` (Stage B) and
`models/archive_v6_freqbranch_stageA/` (Stage A) in case ever revisited.

---

## Current state (as of 2026-08-17)

- **Canonical model**: attempt #2's recipe (Stage B, LoRA, no augmentation,
  no weight decay), **retrained on the corrected, identity-leak-free
  splits** — selected on validation AUC (0.9995) per Exp 1's methodology,
  beating the corrected-split attempt #4-recipe retrain (0.9994). See
  "Second retrain and final canonical re-selection" above for the full
  four-way comparison. This supersedes both (a) the original attempt #2
  deployed after Exp 1 (which was on the *old*, leaky splits) and (b) the
  attempt #4-recipe corrected-split retrain that was briefly the most
  recent checkpoint before this one.
  `models/stageB_best_classifier.pth`, `models/stageB_best_dinov2_lora/`,
  `models/stageB_best_whisper_lora/` contain this checkpoint (training
  auto-saves "best" to these unsuffixed paths, so it was already deployed
  the moment the winning run finished — no manual copy needed). A
  clearly-labeled backup is at
  `models/archive_postfix_attempt2recipe_CANONICAL/`.
  `training/infer.py` and `backend/app.py` load these unsuffixed paths
  directly (`training/infer.py:224-227`), so the live inference/demo
  backend serves this checkpoint.
  Note `models/stageA_best_model.pth` is **not** read by the Stage B
  inference path (`infer.py:238-241` only falls back to it if Stage B
  artifacts are missing) — Stage B always constructs a fresh pretrained
  DINOv2/Whisper base and applies the LoRA adapter dirs on top. This file
  now holds the freshly-retrained (corrected-split) Stage A checkpoint,
  shared by both corrected-split Stage B recipes above.
- **Decision threshold**: re-derived fresh via `pick_threshold_on_val()` in
  `training/evaluate_stage_b.py` against the new canonical checkpoint
  (0.990, balanced accuracy 0.9931 on val, 2026-08-17) — never assume a
  previous run's threshold still applies.
- **Codebase architecture**: 2-modality (visual + audio). No
  frequency-branch code remains. `raw_crops_dataset.py`/`train_stage_b.py`
  now have `USE_AUGMENTATION`/`WEIGHT_DECAY` env-var toggles (added
  2026-08-17, default to the historical attempt-4-recipe behavior for
  backward compatibility) so either Stage B recipe can be reproduced.
  `train_stage_b.py` also has an opt-in `SEED` env var (fixes
  torch/numpy/random RNG before model construction, unset by default) used
  for the multiple-seed variance study below.
- **All experiment artifacts preserved**, nothing deleted:
  - `models/archive_v4_weightdecay/` — attempt #4's backup (old split)
  - `models/archive_v5_sbi/` — SBI attempt's checkpoints
  - `models/archive_v6_freqbranch/`, `models/archive_v6_freqbranch_stageA/`
    — frequency-branch attempt's checkpoints
  - `models/archive_prefreq_stageA/` — pre-freq-branch Stage A
  - `models/archive_pre_identityfix_stageA/` — stale v7 domain-adversarial
    Stage A checkpoints, cleared before the first corrected-split retrain
  - `models/archive_postfix_attempt4recipe/` — corrected-split attempt
    #4-recipe retrain (runner-up in the final comparison)
  - `models/archive_postfix_attempt2recipe_CANONICAL/` — corrected-split
    attempt #2-recipe retrain (the current canonical, also live at the
    unsuffixed `models/stageB_best_*` paths)
  - `outputs/predictions/dfdc_error_analysis.csv` — full per-video DFDC
    prediction breakdown (from attempt #4, not yet re-run on #2)
  - `outputs/misclassified/dfdc_gradcam_spotcheck/` — Grad-CAM heatmaps for
    the 6 most-confidently-wrong DFDC fakes (from attempt #4)
- **Diagnostic scripts kept** (reusable for any future attempt):
  `scripts/dfdc_error_analysis.py`, `scripts/gradcam_spotcheck_dfdc.py`.
- **Undocumented attempt #7 discovered (2026-08-16)**: `models/archive_v7_domainadv/`
  and `models/archive_v7_domainadv_pre/` exist (Stage A checkpoints only,
  dated 2026-07-13, one day after the attempt #6 revert), named
  "domainadv" — presumably a domain-adversarial-training experiment
  started after this doc's previous update (2026-07-12) and never
  documented or carried through to Stage B. `models/stageA_best_model.pth`
  currently holds this v7 Stage A checkpoint, **not** the attempt
  #1/pre-freq-branch Stage A — confirmed harmless to current Stage B
  inference (see above) but should be investigated/documented or cleaned
  up; not touched as part of this pass since it's outside Exp 1's scope.
  **Update**: this is more integrated than the checkpoint files alone
  suggest — `model.py`'s classifier already has a `domain_head` module,
  currently inert, gated behind a `USE_DOMAIN_ADVERSARIAL` env flag
  (confirmed via the loader's own log line:
  `"loaded a pre-domain-adversarial checkpoint - domain_head initialized
  fresh (inert unless USE_DOMAIN_ADVERSARIAL=1)"`). Domain-adversarial
  training is a plausible 8th generalization attempt (train the backbone to
  be invariant to which source dataset a sample came from, which could
  directly target the generator-fingerprint-overfitting root cause) — but
  it was apparently started and abandoned without being logged anywhere.
  Recommend the user clarify intent before any future attempt reuses or
  removes this code path.

## Verification of the corrected canonical model (2026-08-16)

Re-ran a live check against the actual swapped `models/stageB_best_classifier.pth`
+ LoRA adapter checkpoint (not just trusting the historical log) using a
fast DFDC-only script (`training/evaluate_stage_b.py`'s loading + scoring
functions, restricted to the 400-video held-out set to avoid the full
14k-video val+test+held_out sweep, which timed out after ~37 min on this
machine's raw-crop I/O). Result: **DFDC AUC 0.6912**, n=400 (77 real, 323
fake) — an exact match to the historical attempt #2 number, confirming the
checkpoint swap was performed correctly and the reported number is real,
not stale. The accuracy/threshold figures (55.50% acc, 70.13% real-acc,
52.01% fake-acc) reported in the paper and table above are the historical
values for this identical checkpoint (unchanged weights, only the file
location changed) and were not independently re-run in this session due to
the val-threshold-sweep cost; given the AUC matched to 4 decimal places,
they should reproduce if re-run, but this hasn't been done bit-for-bit.

## Exp 4: Quantitative boundary-cue ablation (2026-08-16) — contradicts the Grad-CAM hypothesis

The Grad-CAM spot-check (above, "Diagnostic Analysis") was a 6-image
qualitative spot-check on attempt #4's confidently-wrong DFDC fakes, which
found attention dominated by boundary/edge cues and motivated the framing
that the model "over-relies on blending-boundary sharpness." A reviewer
correctly flagged that six Grad-CAM images are not quantitative evidence for
this claim. This experiment turns it into one: on the canonical (attempt #2)
model, across all 400 DFDC held-out videos, mask or blur the face-contour
boundary region (a ~24px-wide ring straddling the landmark convex-hull edge
on each 224px face crop, reusing `pipeline/phase5c_sbi_augment.py`'s
landmark/hull machinery) vs. the interior (skin) region vs. an area-matched
random-region control, and measure the AUC change. Script:
`scripts/dfdc_boundary_ablation.py`; raw per-video predictions at
`outputs/predictions/dfdc_boundary_ablation.csv`, summary at
`outputs/predictions/dfdc_boundary_ablation_summary.csv`.

| Condition | AUC | 95% CI (Hanley–McNeil) | Δ from original |
|---|---|---|---|
| Original (reconstructed via the same region-derivation pipeline) | 0.6911 | [0.631, 0.751] | — |
| Boundary blurred | 0.6850 | [0.625, 0.745] | −0.0061 |
| Boundary masked (mean-fill) | 0.7071 | [0.649, 0.765] | +0.0160 |
| Interior masked (mean-fill) | **0.5994** | **[0.532, 0.667]** | **−0.0917** |
| Random-region masked (area-matched control) | 0.6860 | [0.626, 0.746] | −0.0051 |

("Original (reconstructed)" reproduces the true production pipeline's AUC
(0.6912, see verification section above) to within 0.0001, confirming the
region-regeneration path used for all five conditions is faithful.)

**This does not support the boundary-cue-reliance hypothesis.** Masking or
blurring the boundary ring changes AUC by less than a random-region control
of the same area does — i.e., there is nothing special about the boundary
specifically; removing *any* similarly-sized patch of the face has a small,
statistically indistinguishable effect. The only condition that moves AUC by
more than its own confidence interval's width is **interior masking**,
which drops AUC by 0.092 — the model's cross-dataset discriminative power on
DFDC depends far more on interior/skin-texture information than on the
face-contour boundary. Note these are independent-sample CIs on the same
400 paired videos, which is conservative (understates significance of a
paired difference) rather than liberal — no formal paired significance test
(e.g. DeLong's test, or a bootstrap over paired per-video differences) has
been run; that would sharpen this further and is reasonable follow-up work,
but the effect-size gap (interior's −0.092 vs. the ±0.016 band of the other
three conditions) is large enough that we're confident in the qualitative
ranking even without it.

**Honest reassessment of the earlier diagnostic claim**: the Grad-CAM
spot-check's finding (attention concentrated on boundary/edge regions in 6
confidently-wrong fakes) may still be an accurate description of *where the
model looks* in those specific failure cases, but this experiment shows that
*where attention concentrates* is not the same as *what the model's
DFDC-relevant discriminative signal actually depends on* — attention maps on
6 hand-picked wrong examples do not establish causal reliance, and the
causal test disagrees with the attention-based intuition. Attempt #6 (the
SRM frequency-domain branch) was partly motivated by the boundary-cue
diagnosis; this result suggests that motivation was likely wrong, though it
doesn't change attempt #6's own outcome (DFDC AUC 0.6529, still the worst of
the six) or the overall six-attempts conclusion (a ~0.65-0.71 AUC band no
single intervention closes). The paper's Diagnostic Analysis / Discussion
sections have been corrected to reflect this rather than reporting the
original speculative interpretation as settled.

## Exp 2: Identity-safe splitting audit and fix (2026-08-16)

A reviewer flagged that the paper's "identity-disjoint 80/10/10 split" claim
was too strong for FaceForensics++, which fell back to an MD5 hash of the
source file path (i.e. no real identity grouping at all) rather than actual
identity parsing. Investigating this turned up two findings much larger than
the original scope.

### Finding 1: the "FFPP" source bucket is ~72% Celeb-DF-v2, not FaceForensics++

A filename-pattern census of all 8,627 videos under `data/raw/FFPP/` (all
blanket-labeled `source=FFPP` in the manifest) found:

| Pattern | Count | Actual origin |
|---|---|---|
| `idX_idY_NNNN.mp4` | 5,639 | Celeb-DF-v2 synthesis |
| `XXX.mp4` (3-digit) | 1,000 | Genuine FF++ originals |
| `XXX_YYY.mp4` (3digit pair) | 1,000 | Genuine FF++ manipulated pairs |
| `idX_NNNN.mp4` | 589 | Celeb-DF-v2 real |
| 5-digit only | 300 | Unresolved |
| 4-digit / `_fake` suffix | 49+49 | Unresolved |
| `XX_YY__action__hash.mp4` | 1 | DFD (Google/Jigsaw) naming |

The 589/5,639 real/fake counts match Celeb-DF-v2's well-known published
composition almost exactly. **The paper's training-data description
("FakeAVCeleb, PolyGlotFake, and FaceForensics++") was materially wrong** —
it silently trains on Celeb-DF-v2 (72% of the bucket) with only ~23% genuine
FF++. This also means **Celeb-DF can no longer be used as a future
held-out external dataset** (it was suggested as an Exp 5 candidate) without
first removing it from training, since it's already contaminated into
train/val/test.

**Fix**: `pipeline/phase1_organize.py::classify_ffpp_source()` now
classifies each file by filename convention into `FFPP` / `CelebDF` / `DFD`
/ `FFPP_unresolved` (the ~398 videos of unresolved origin are left honestly
unresolved rather than guessed). `outputs/manifests/inventory.csv`
regenerated with corrected `source` values.

### Finding 2: pairwise identity leakage — and it's much worse for FakeAVCeleb than FFPP

The pre-existing split algorithm (`pipeline/phase4_split.py`) assigned each
raw identity *token* independently to train/val/test, then routed a video to
whichever split any of its identities belonged to (test > val > train
priority). This does not guarantee disjointness when a video references
**multiple** identities (e.g. a face-swap video naming both the source-face
donor and the target-video identity): if identity A's own video lands in
train but A is paired with identity B (assigned to test) in another video,
that video must go to test — so A now has representation in both splits.

Quantifying this on the **old** (pre-fix) splits, checking whether each true
identity's videos ever spanned more than one of {train, val, test}:

| Source | Total identities | Leaked across splits |
|---|---|---|
| **FakeAVCeleb** | 500 | **455 (91.0%)** |
| FaceForensics++ (true) | 1,000 | 488 (48.8%) |
| Celeb-DF-v2 | 59 | 59 (100.0%) |

**FakeAVCeleb — the dataset the paper claimed was already correctly
identity-disjoint — was actually the worst offender.** This was never
verified empirically before; the reviewer only flagged FF++ specifically.
Root cause: FakeAVCeleb's `FakeVideo-*` face-swap videos encode two
identities in the path (the target-video directory id and the
source-face-donor id in the filename, e.g.
`.../id00166/00010_id01637_...faceswap.mp4`), and the old algorithm never
grouped them.

**Fix**: replaced per-token assignment with a connected-components approach
(`pipeline/phase4_split.py::_UnionFind`) — union every pair of identities
that ever co-occur in the same video, then split at the component level, so
no identity can ever end up in two partitions. Component sizes turned out to
be extremely skewed (top 10 of 15,280 components = 54.8% of all videos, one
alone = 10.3%), so naive "shuffle components, take 80%" would badly miss an
80/10/10 video-count target; used balanced greedy bin-packing (largest
components first, each placed into whichever split is furthest below its
target share) instead. Result: **35,649 / 4,456 / 4,456 videos (80.0% /
10.0% / 10.0%), zero identities spanning more than one split** (verified
across all 16,326 identities post-fix, vs. hundreds leaking pre-fix).

**Consequence — 19,874 videos (44.6% of the trainable pool) moved to a
different split** between the old and corrected splits. Crop directories on
disk (`data/processed/crops/<split>/<label>/<video_id>`) were reconciled to
match: 13,037 moved, 6,837 were duplicate leftovers from an earlier
extraction run already sitting at the correct new location (verified
byte-identical by filename+size before deleting the redundant old copy).
Old splits backed up at `outputs/manifests/pre_identity_fix_backup/` before
regenerating.

### Post-identity-fix retrain results (2026-08-16, same day)

Retrained Stage A + Stage B from scratch on the corrected splits, using the
**current, unmodified codebase's default recipe** — LoRA fine-tuning +
augmentation + weight_decay=1e-4, i.e. attempt #4's recipe, not attempt #2's
(the codebase has no toggle for augmentation/weight-decay; both became
unconditional once attempt #4 was adopted historically, so reproducing
attempt #2 exactly would need a second retrain with code changes — not done
in this pass). Stale checkpoints from the undocumented v7 domain-adversarial
run were archived to `models/archive_pre_identityfix_stageA/` first so
training started clean rather than silently resuming from unrelated state.

- **Stage A**: early-stopped epoch 11 (best epoch 6), val AUC 0.9985.
- **Stage B**: early-stopped epoch 9 (best epoch 5), val AUC 0.9994.
- **Full evaluation** (`training/evaluate_stage_b.py`, threshold 0.930
  chosen on val, balanced accuracy 0.9929):

| Metric | Old (attempt #4, leaky split) | New (attempt #4 recipe, corrected split) |
|---|---|---|
| In-distribution AUC | 0.9962 | **0.9989** |
| In-distribution Accuracy | ~97-99% | **99.01%** (347/350 real, 4065/4106 fake) |
| DFDC AUC | 0.7105 | **0.6572** |
| DFDC Accuracy | 50.25% | **36.50%** |
| DFDC Real Accuracy | 84.42% | 90.91% |
| DFDC Fake Accuracy (recall) | 42.11% | **23.53%** |

**Two findings, one reassuring and one not.** (1) In-distribution
performance was **not** substantially inflated by the leakage bug for this
recipe — AUC actually rose slightly post-fix (0.9962→0.9989). This is
good news for the paper's core integrity: the headline near-perfect
in-distribution numbers hold up under honest, leak-free evaluation, even
though the leakage itself was real and severe (91% of FakeAVCeleb
identities). (2) **DFDC cross-dataset performance got measurably worse**,
not better: AUC (0.6572) is now below every one of the original six
attempts (previous worst was 0.6529), and fake-recall (23.53%) is worse
than any prior attempt's 31-52% range. We do not yet know whether this is
genuine (e.g. the corrected train set's different composition/balance
across FFPP/CelebDF/FakeAVCeleb shifted what the model learns) or retrain
variance (no multiple-seed comparison has been run, per reviewer
recommendation #4, never implemented) — a single retrain cannot distinguish
these. We report the number as-is rather than dismissing it as noise
without evidence.

**Resolved same day**: see "Second retrain and final canonical re-selection"
below — attempt #2's exact recipe was retrained on the corrected splits too,
and canonical was re-selected by validation AUC exactly as Exp 1's
methodology requires. This is no longer open.

### Second retrain and final canonical re-selection (2026-08-16/17)

User asked to complete the comparison: added `USE_AUGMENTATION` and
`WEIGHT_DECAY` env-var toggles to `raw_crops_dataset.py`/`train_stage_b.py`
(mirroring the existing `USE_SBI_AUGMENTATION` pattern — both had become
hardcoded/unconditional once attempt #4 was historically adopted, with no
way to reproduce attempt #2 exactly without this change). Archived the
attempt-#4-recipe retrain to `models/archive_postfix_attempt4recipe/` and
cleared stale `stageB_checkpoint_epoch*.pth` files before launching, same as
before. Reused the same freshly-retrained Stage A checkpoint (Stage A is
shared across all Stage B recipes — no need to redo it).

**Training was interrupted twice by `WinError 1455`** (`Couldn't open
shared file mapping` — Windows page-file/shared-memory commitment limit,
not physical RAM exhaustion; 8GB+ free RAM was confirmed available both
times), a known occasional flakiness on this machine already documented in
`train_stage_b.py`'s own code comment ("workers=6 crashed with WinError
1455... this machine's real ceiling is 4" — num_workers=4 was already the
established safe setting, yet it still recurred). Both crashes happened
mid-epoch; `stageB_checkpoint_inprogress.pth` (saved every 20 batches) plus
`stageB_checkpoint_epoch*.pth` meant each relaunch resumed with at most one
epoch's redone work lost. Third launch completed cleanly: **early-stopped
epoch 14, best epoch 10, val AUC 0.9995**.

**Full four-way comparison** (`training/evaluate_stage_b.py` for both new
runs; threshold 0.990 chosen on val for attempt #2-recipe, balanced
accuracy 0.9931):

| Metric | Attempt 2 (old, leaky) | Attempt 4 (old, leaky) | Attempt 4 (new, corrected) | **Attempt 2 (new, corrected)** |
|---|---|---|---|---|
| Val AUC (training) | 0.9976 | 0.9962 | 0.9994 | **0.9995** |
| In-dist Test AUC | — | — | 0.9989 | **0.9984** |
| In-dist Test Acc | — | ~97-99% | 99.01% | **98.43%** (342/350 real, 4044/4106 fake) |
| DFDC AUC | 0.6912 | 0.7105 | 0.6572 | **0.7091** |
| DFDC Accuracy | 55.50% | 50.25% | 36.50% | **42.25%** |
| DFDC Real Accuracy | 70.13% | 84.42% | 90.91% | **93.51%** (72/77) |
| DFDC Fake Recall | 52.01% | 42.11% | 23.53% | **30.03%** (97/323) |

**Re-selection result**: attempt #2's recipe wins on validation AUC under
the corrected split too (0.9995 vs 0.9994) — the same narrow margin, same
winner, as the original (leaky-split) comparison in Exp 1 (0.9976 vs
0.9962). This is a reassuring, reproducible finding: the validation-based
selection methodology gives a consistent answer regardless of which
splitting regime is used. **Attempt #2's recipe, retrained on the
corrected splits, is now the canonical, deployed configuration** —
`models/stageB_best_classifier.pth` + LoRA adapter dirs already contain
this checkpoint (training auto-saves "best" to the canonical unsuffixed
paths, so no manual copy was needed once the winning run finished). A
clearly-labeled copy is preserved at
`models/archive_postfix_attempt2recipe_CANONICAL/`.

**Bonus finding**: unlike the old-split comparison (where the
validation-selected winner, attempt #2, had *worse* DFDC performance than
the runner-up, attempt #4 — 0.6912 vs 0.7105), under the corrected split
the validation-selected winner *also* generalizes better to DFDC (0.7091 vs
0.6572). Not guaranteed by the methodology (validation AUC says nothing
about DFDC by design), but a reassuring coincidence this time. The
now-canonical model's DFDC AUC (0.7091) is essentially unchanged from the
original 0.71 finding — the identity-leak fix did not materially change
the headline cross-dataset generalization conclusion, only which specific
checkpoint is deployed and how confident we can be that in-distribution
numbers aren't leakage artifacts.

**Resolved 2026-08-17**: see "Multiple-seed variance study" below.

### Multiple-seed variance study (2026-08-17)

User asked to run the multi-seed comparison (reviewer recommendation #4).
Given the cost of a full 5-seed × 2-recipe study (each Stage B run takes
~2-4 hours), scoped to 3 seeds of just the canonical (attempt #2) recipe —
the existing deployed run counts as seed 1 (unseeded/default RNG), plus 2
freshly-seeded runs. Added an opt-in `SEED` env var to `train_stage_b.py`
(fixes `torch`/`numpy`/`random` RNG state before model construction; unset
by default, so existing runs' behavior/reproducibility notes are
unaffected). Seed 2 and seed 3 each retrained fresh from the shared Stage A
checkpoint, each early-stopping around epoch 9-11 (val AUC 0.9994-0.9995,
essentially matching seed 1's 0.9995).

**Important operational note**: `train_stage_b.py` overwrites the live
`models/stageB_best_*` deployment paths every time it saves a new "best"
checkpoint *during training* — so for the duration of seeds 2 and 3
training, the backend was transiently serving an in-progress, not-yet-final
checkpoint from whichever seed run was active. This is expected/harmless
for a same-day research variance study but worth knowing if this pattern
is reused: the deployed model isn't stable while any Stage B training job
is running. Seed 1's canonical checkpoint was fully restored (copied from
`models/archive_postfix_attempt2recipe_CANONICAL/`, verified via matching
md5sum) once the study concluded.

**Results**:

| Metric | Seed 1 | Seed 2 | Seed 3 | Mean ± std (n=3) |
|---|---|---|---|---|
| In-dist Test AUC | 0.9984 | 0.9988 | 0.9972 | **0.9981 ± 0.0008** |
| DFDC AUC | 0.7091 | 0.7295 | 0.7083 | **0.7156 ± 0.0120** |
| DFDC Accuracy | 42.25% | 44.75% | 52.00% | **46.33% ± 5.06pp** |
| Val-derived threshold | 0.990 | 0.990 | 0.730 | — |

**Interpretation**:
- **In-distribution AUC is very stable** (std 0.0008 on a 0.997-0.999
  range) — the headline in-distribution claim is robust to seed choice.
- **DFDC AUC is fairly stable too** (std 0.012). The ~0.06 gap between
  attempt #2-recipe's mean DFDC AUC (0.7156) and attempt #4-recipe's
  single-run DFDC AUC (0.6572) is roughly **5× the attempt-2 seed std** —
  suggestive the recipe difference found earlier is a genuine effect, not
  retrain noise. Caveat: this compares a 3-seed spread for one recipe
  against a single run of the other recipe, not a symmetric two-recipe
  variance study — extending the seed study to attempt #4's recipe would
  make this comparison rigorous rather than merely suggestive.
- **DFDC accuracy at the chosen threshold is much less stable than AUC**
  (42.25% to 52.00%, std 5.06 percentage points) — a genuinely new and
  somewhat concerning finding. The three seeds' validation-derived
  thresholds were 0.99, 0.99, and 0.73 — the balanced-accuracy
  threshold-selection procedure (`pick_threshold_on_val()` in
  `evaluate_stage_b.py`) is itself seed-sensitive, even when the
  underlying rank-based AUC barely moves. This means a stable AUC across
  retrains does NOT imply a stable deployed operating point — only
  within a single training run's own threshold, which is what the
  project's existing "AUC is not sufficient evidence" argument
  (Section VI-B of the paper) was about calibration \emph{within one
  run}, not stability \emph{across} runs. This is a new, additional
  nuance worth remembering for any future retraining.
- Deployed checkpoint (seed 1) was chosen before this study for
  unrelated reasons (it was simply the first/original run) and happens to
  be the median of the three on DFDC AUC — not cherry-picked after seeing
  all three results, which would have reintroduced exactly the kind of
  post-hoc selection bias the whole Exp 1 methodology exists to prevent.

**Still open**: extending this seed study to attempt #4's recipe (for a
symmetric two-recipe comparison) and to the DFDC-side six-attempt
comparison more broadly (Table in the "Six attempts" section above) remains
unstudied. Given the cost already spent (5 total Stage B runs across this
session: 2 recipe comparisons + 2 additional seeds), further seed studies
are deferred pending explicit user request.

## Open strategic question

Six attempts (five architecture/training tweaks plus one added-data attempt)
have now converged on the same ~0.65-0.71 AUC band on DFDC, with no clear
upward trend. Two honest paths forward, not yet decided:

1. **Add genuinely different training data** — Celeb-DF (v2), DFD, or
   WildDeepfake, spanning different generation-method families than the
   current FakeAVCeleb/PolyGlotFake/FFPP mix. Directly attacks the diagnosed
   root cause (generator diversity), but is the slowest path (new dataset
   download/preprocessing/crop extraction before any retraining can even
   start).
2. **Accept 0.71 AUC as the honest generalization ceiling** for this
   backbone/data combination, and report it as such rather than continuing
   to chase incremental fixes. This is itself a defensible, honest research
   finding — not every generalization gap closes with more tuning, and that
   conclusion is exactly what plan.pdf's held-out DFDC test was designed to
   surface.
