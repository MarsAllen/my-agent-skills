---
name: colab-ai-video-production
description: >-
  End-to-end production workflow, reusable CLI toolkit, and battle-tested
  pitfall manual for making AI short videos and pet/animal lip-sync singing
  memes using local 0-CCU previews paired with Google Colab GPU V2V polishing,
  sub-pixel dolly zoom, and high-bitrate CJK subtitles. Use when creating
  singing pet/character lip-sync videos, running Wan2.2-S2V or Wan2.1 V2V on
  Colab GPUs, troubleshooting Colab CLI/OOM/lip-sync flaws, or auditing AI
  short videos before release.
---

# Colab AI Video & Pet Lip-Sync Production

## Overview

This skill packages a proven two-stage pipeline for producing natural,
syllable-synced AI short videos (especially non-human pets/reptiles/animals
singing to music) with minimal Google Colab CCU burn:

1. **Local 0-CCU Driven Preview**: Crop a crisp close-up, extract a clean
   audio envelope and reference timing, stabilize a single sharp mouth bank
   (no spring head-jerk), apply a physical vertical eyelid-close wipe for
   emotional climaxes, and verify 100% syllable-level lip-sync locally at 30fps.
2. **Colab GPU Low-Denoise V2V Polish**: Polish the verified preview with
   `Wan2.1-T2V-1.3B-Diffusers` (`WanVideoToVideoPipeline`, `strength=0.28`) on
   an L4/A100 GPU (~3 min, ~6 GB VRAM) to fuse eyelids/jaw seams naturally,
   optionally add a sub-pixel `INTER_LANCZOS4` emotional dolly zoom, burn
   seamless `-crf 15` Noto Sans CJK subtitles, and verify with dynamic playback
   plus a 4fps audit sheet before stopping the VM.

For detailed troubleshooting of every pitfall (macOS OpenSSL 4 WebSocket drops,
MCP 3-minute timeouts, A100 40GB OOM, `flash_attn` SDPA patch, GPU 503
fallbacks, tongue-smudge artifacts, etc.), read
[references/pitfalls_and_tuning.md](./references/pitfalls_and_tuning.md).

## Dependencies

- **`colab` skill** (required): All Google Colab VM lifecycle actions
  (`colab_status`, `colab_new`, `colab_exec`, `colab_files`, `colab_stop`) MUST
  use the installed `colab` MCP tools and surface the side panel link
  `[Colab](sidecar://colab-status/colab_panel/)`.
- **`image-generator` subagent** (recommended): Used in Stage 2 to generate a
  pixel-aligned closed-eyes (or custom emotional expression) edit of the
  subject's close-up photo.
- **Local tools**: `ffmpeg`, `ffprobe`, and `uv` on `PATH`.

## Core Workflow

### Stage 1: Assets, Audio & Reference Analysis (0 CCU)
1. **Framing check**: Ensure the subject's face fills at least ~50–65% of the
   frame width (9:16 vertical, e.g. `480×832` or `448×768`). If the raw photo
   is wide, crop tightly around the face and front paws while preserving the
   original background.
2. **Extract & analyze audio** (use `uv run` on
   [scripts/video_toolkit.py](./scripts/video_toolkit.py)):
   - Run `analyze-audio` to produce a 30fps syllable-energy JSON and locate
     exact phrase boundaries & pauses.
   - If a reference performance video exists, run `analyze-ref` to generate a
     contact sheet and identify emotional eye-closure windows and subtitle beats.
3. **Prepare seamless SRT**: Draft subtitles aligned to the vocal boundaries
   and run `validate-srt` to guarantee valid `HH:MM:SS,mmm` timestamps with zero
   flicker gaps.

### Stage 2: Local 0-CCU Synced Preview (Iterate Until Rhythm Is 100%)
1. **Obtain a sharp mouth-opening bank**:
   - Either run a short single-clip S2V pass (e.g., `Wan2.2-S2V-14B` with
     `--infer_frames 48`, `--t5_cpu`, SDPA patch, and decoupled `AUDIO_CFG=7.5`
     — see [pitfalls_and_tuning.md](./references/pitfalls_and_tuning.md)) to
     harvest 8–10 consecutive sharp jaw-opening frames, or reuse an existing
     clean bank.
   - **Rule**: Use **one razor-sharp stabilized bank** rather than mixing in a
     softer secondary bank (blurry pink mouth edges cause V2V models to
     hallucinate protruding tongues).
2. **Generate pixel-aligned closed-eye expression**:
   - Delegate to `image-generator` with strict instructions to keep framing,
     lighting, pose, and background identical while editing only the eyelids
     tightly shut.
   - Visually inspect with `read_file` and verify eye alignment on a grid.
3. **Render 30fps local preview**:
   - Run `build-preview` to lock the upper head (zero sideways spring jerk),
     drive the jaw with a smooth 70ms RMS + gentle syllable-dip envelope, and
     apply a **top-to-bottom vertical eyelid wipe** (never a flat alpha
     crossfade) on emotional climax windows.
   - Watch the preview via `read_file` and iterate locally (0 CCU, ~20–60s per
     tweak) until the user confirms mouth speed, eye timing, and framing.

### Stage 3: Colab GPU Low-Denoise V2V Polish & Post-Processing
1. **Check `colab_status` first** and pick an eligible GPU (`L4` 24GB is fast
   and cheap for `Wan2.1-T2V-1.3B-Diffusers`, ~3 min / ~6 GB VRAM; `A100` also
   works).
   - If Colab GPU provisioning returns `503 Service Unavailable`, do NOT hang in
     a loop: inform the user, offer a fast CPU subtitle pass if desired, or
     schedule a retry later.
2. **Always run heavy Colab jobs in the background**:
   - Launch downloads and diffusion runs via `subprocess.Popen` logging to a
     file (MCP calls have a 3-minute hard timeout). Never run `colab_exec` and
     `colab_files` concurrently on the same session.
3. **Run [scripts/colab_v2v_polish.py](./scripts/colab_v2v_polish.py)** on the
   Colab GPU:
   - Recommended golden parameters: `--model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers`,
     `--strength 0.28` (`0.25–0.30` preserves razor-sharp jaw edges while fusing
     eyelids and adding natural skin/breathing cohesion), `--steps 32`,
     `--guidance 5.0`.
   - Always include `sticking out tongue, protruding tongue, pink smudge, blurry mouth`
     in the negative prompt.
   - Optionally enable `--dolly-zoom` for a smooth sub-pixel `INTER_LANCZOS4`
     climax push-in (`1.00x → 1.11x`, zero background rotation) and pass
     `--subtitles` to burn `Noto Sans CJK TC` at `-crf 15`.

### Stage 4: Mandatory Visual Audit & Safe Shutdown
1. **Never rely on 1fps thumbnails alone**: Download the rendered `.mp4` locally,
   watch the actual video with audio via `read_file`, AND run `audit-video` to
   inspect a **4fps contact sheet** across the entire duration.
2. **CCU-safe shutdown protocol**:
   - Unless the user explicitly asked for immediate auto-shutdown, **keep the
     GPU session alive while the user inspects the final video** (avoiding
     re-downloading weights if a tweak is needed), clearly stating the live
     status and `[Colab](sidecar://colab-status/colab_panel/)` link.
   - The moment the user approves (or says stop), call `colab_stop` and verify
     `colab_status` reports `assignments: 0` and `rate: 0.0`.

## Utility Scripts

All local tools live in [scripts/video_toolkit.py](./scripts/video_toolkit.py)
and use only Python stdlib + `ffmpeg`/`ffprobe` (run via `uv run`):

```bash
TOOL=~/.gemini/config/skills/colab-ai-video-production/scripts/video_toolkit.py

# 1. Analyze vocal audio into a 30fps syllable-openness JSON
uv run "$TOOL" analyze-audio \
  --audio assets/audio/chorus.wav \
  --fps 30 \
  --output work/audio_envelope.json

# 2. Extract a contact sheet from a reference performance video
uv run "$TOOL" analyze-ref \
  --video assets/audio/reference.mp4 \
  --fps 2 \
  --tile 5x4 \
  --output work/ref_contact.jpg

# 3. Validate (and optionally auto-close flicker gaps in) an SRT subtitle file
uv run "$TOOL" validate-srt \
  --srt assets/audio/lyrics.srt \
  --fix-gaps \
  --output work/srt_report.json

# 4. Build a 0-CCU 30fps stabilized preview with vertical eyelid close
uv run "$TOOL" build-preview \
  --mouth-video outputs/s2v_source.mp4 \
  --mouth-frames 56:66 \
  --eyes-closed assets/photos/eyes_closed.jpg \
  --audio assets/audio/chorus.wav \
  --climax-windows "1.92-5.34,7.42-9.18" \
  --output outputs/preview_synced.mp4

# 5. Audit any rendered video with a 4fps contact sheet + ffprobe JSON
uv run "$TOOL" audit-video \
  --video outputs/final_subtitled.mp4 \
  --fps 4 \
  --output work/audit_4fps.jpg
```

For the Colab GPU polishing stage, upload
[scripts/colab_v2v_polish.py](./scripts/colab_v2v_polish.py) and run inside the
Colab VM (background `Popen`):

```bash
python3 colab_v2v_polish.py \
  --input preview_synced.mp4 \
  --audio chorus.wav \
  --subtitles lyrics.srt \
  --model-id /content/Wan2.1-T2V-1.3B-Diffusers \
  --strength 0.28 --steps 32 --guidance 5.0 \
  --dolly-zoom \
  --output-clean outputs/final_master.mp4 \
  --output-sub outputs/final_master_subtitled.mp4
```

## Common Mistakes

1. **Trusting pure Audio-to-Video (`Wan2.2-S2V`) on reptiles/non-humans**:
   Human-trained audio cross-attention misses syllable articulation on non-human
   mouths and resets to a closed mouth at 48-frame chunk boundaries. Always
   drive lip-sync explicitly locally first, then polish with low-denoise V2V.
2. **Scrubbing a full-frame mouth ramp without stabilizing the head**: If the
   source ramp drifts sideways as the mouth opens, scrubbing it back and forth
   creates an obvious spring/boomerang head-jerk. Lock the upper head and blend
   only the aligned jaw region.
3. **Using high V2V strength (`>= 0.35`) or blurry secondary mouth frames**:
   Causes diffusion models to turn soft pink mouth rims into protruding tongues
   or smeared scales. Keep V2V `strength` around `0.25–0.30` on a single sharp
   mouth bank.
4. **Burning subtitles with default `libx264` CRF (23)**: Destroys fine fur/scale
   texture and drops bitrate below 600 kbps. Always pass `-crf 15` and install
   `fonts-noto-cjk` on Colab Linux.
