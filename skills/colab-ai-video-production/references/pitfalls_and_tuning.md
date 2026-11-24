# Pitfalls, Root Causes & Golden Tuning Reference

Battle-tested engineering lessons captured from end-to-end production of
`gecko-singing-meme` on macOS + Google Colab (A100 40GB & L4 24GB).

---

## 1. Google Colab CLI & Connection Pitfalls

### 1.1 `WebSocketConnectionClosedException: Connection is already closed.` on macOS
- **Symptom**: `colab_new`, `colab_files`, `colab_status`, and `colab_stop` all
  succeed (HTTPS via `requests` + `certifi`), while `colab_exec` and `colab_run`
  immediately crash inside `jupyter_kernel_client/wsclient.py`.
- **Root cause**: `uv`-managed Homebrew Python 3.15 links against `openssl@4`,
  whose default CA file `/usr/local/etc/openssl@4/cert.pem` may be missing
  (`ssl.get_default_verify_paths().cafile == None`). `websocket-client` fails
  the WSS TLS handshake silently and leaves `self.sock` closed.
- **Fix**:
  ```bash
  mkdir -p /usr/local/etc/openssl@4 && \
    ln -sf /usr/local/etc/ca-certificates/cert.pem /usr/local/etc/openssl@4/cert.pem
  ```
  Verify with a 10-second CPU `colab_run(code='print("ok")')`.

### 1.2 MCP 3-Minute Hard Timeout on Downloads & Diffusion
- **Symptom**: Synchronous `colab_exec` calls running >180s terminate with
  `context deadline exceeded` even though the Colab process is still running.
- **Fix**: Always launch model downloads and inference scripts in the
  background via `subprocess.Popen(..., stdout=open("/content/job.log", "w"), stderr=subprocess.STDOUT)`, write the PID to a file, and check status with
  short 2-second `colab_exec` calls + `schedule` timers.

### 1.3 Concurrent WebSocket Collisions
- **Symptom**: Running `colab_exec` and `colab_files` in parallel on the same
  session triggers `WebSocketConnectionClosedException` or dropped connections.
- **Fix**: Always call `colab_files` and `colab_exec` **sequentially** on a
  given session. If a long-running session drops a WS token after ~45 minutes,
  call `colab_sessions` once to refresh the proxy token.

### 1.4 Colab GPU `POST /tun/m/assign?variant=GPU` Returns `503 Service Unavailable`
- **Diagnosis**: Inspect `~/.config/colab-cli/colab.log`. If GPU allocation
  retries on HTTP `503` while CPU `colab_run` succeeds in ~10s, Google's GPU
  pool is temporarily overloaded/outaged (not a client/auth bug).
- **Mitigation**: Do not loop-hammer GPU allocation. Report cleanly, confirm
  `0` active sessions, use a cheap CPU session if lightweight FFmpeg/subtitle
  work is needed, and retry GPU later.

### 1.5 CCU Safety & Interactive Review Rule
- Always finish lip-sync/timing iteration **locally at 0 CCU** before booting a
  GPU VM.
- Once a GPU render completes, download outputs and **keep the VM warm until
  the user inspects and approves** (avoiding 5–42 GB re-downloads if a quick
  parameter tweak is needed), while reminding the user in every reply that the
  GPU is live.
- On approval, call `colab_stop(all=True)` and verify via `colab_status` that
  `assignments == 0` and `rate == 0.0`.

---

## 2. Wan2.2-S2V-14B & Open-Source Audio-to-Video Pitfalls

### 2.1 `flash_attn` Compile Trap on Modern Colab PyTorch / CUDA
- **Symptom**: `pip install -r requirements.txt` tries to build `flash_attn`
  from source for 20+ minutes on PyTorch 2.14 / CUDA 13.
- **Fact**: `Wan2.2/wan/modules/attention.py` has an SDPA fallback in
  `attention()`, **but** `wan/modules/s2v/model_s2v.py` calls `flash_attention()`
  directly, which hits `assert FLASH_ATTN_2_AVAILABLE`.
- **Fix**: Skip installing `flash_attn` completely (only install `ftfy`,
  `decord`/`eva-decord`, `imageio[ffmpeg]`, `dashscope`, `demucs`, `hf_transfer`)
  and patch `flash_attention()` to delegate to
  `torch.nn.functional.scaled_dot_product_attention` (handling `b==1`
  `q_lens`/`k_lens` slicing) when `FLASH_ATTN_2_AVAILABLE` is `False`.

### 2.2 A100 40GB OOM by ~640 MiB at Default `--infer_frames 80`
- **Fix**: Pass ALL four flags together on A100 40GB:
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `--infer_frames 48`
  - `--t5_cpu` (moves 11 GB UMT5-XXL to Colab's 83 GB system RAM)
  - `--offload_model True --convert_model_dtype`

### 2.3 Hardcoded Negative Prompt & Diluted Audio CFG
- `Wan2.2-S2V`'s default Chinese negative prompt literally contains
  `情绪激动剧烈` ("intense/dramatic emotion"), actively flattening singing
  expressions! Override `n_prompt` to exclude `closed mouth, silent, frozen jaw, expressionless, human lips, human teeth`.
- Coupled CFG (`guide_scale=4.5`) dilutes audio lip-sync. Patching 3-pass
  decoupled CFG (`u + text_cfg*(ct - u) + audio_cfg*(c - ct)` with `text_cfg=3.5`,
  `audio_cfg=7.5`) dramatically strengthens mouth movement.

### 2.4 Why Pure Audio-to-Video Still Misses Syllables on Non-Humans
- Audio cross-attention layers were trained overwhelmingly on human lips. On
  reptiles/geckos, pure S2V opens the mouth only once per 48-frame chunk,
  resets to a closed mouth across every 3-second chunk boundary, and refuses to
  squeeze eyes shut on emotional high notes.
- **Solution**: Use S2V only (if needed) to harvest a clean 8–10 frame
  jaw-opening sequence, then switch to the **two-stage driven preview → V2V
  polish** architecture.

---

## 3. Two-Stage Driven Preview & V2V Polishing Pitfalls

### 3.1 Framing & Letterbox Alignment
- Subjects whose head is <35% of frame width become blurry at `448–480px` width.
  Crop tightly to a 9:16 macro close-up first.
- `Wan2.2-S2V` outputs slight top/bottom black bars if aspect ratio differs
  (`y=28..803` inside `448×832`). Always detect and crop the active non-black
  area before aligning edited expression overlays or exporting final 9:16.

### 3.2 Eliminating the "Spring / Boomerang Head-Jerk"
- **Pitfall**: Scrubbing full-frame `frames 56..65` forward/backward according
  to audio loudness makes the entire head jerk left/right on every syllable if
  the source ramp included natural head sway.
- **Fix**: Track left & right eye centers across every ramp frame, cancel the
  horizontal shift `(src_cx - base_cx)`, and feather-blend **only the lower
  snout/jaw region (`y >= eye_y + 10`)** onto the single static base head.

### 3.3 Physical Vertical Eyelid Wipe vs. Flat Alpha Crossfade
- **Pitfall**: Blending an open-eye frame and a closed-eye frame with uniform
  alpha creates a ghostly semi-transparent eyelid floating over the pupil for
  0.2s.
- **Fix**: Use a **top-to-bottom vertical curtain wipe** (`lid_edge` sweeping
  from `-rad_y` to `+rad_y` with a 5px soft feather) so the scaly eyelid
  physically closes downward and opens upward.

### 3.4 Blurry Secondary Mouth Banks Cause V2V "Protruding Tongue" Hallucinations
- **Pitfall**: Blending a softer/blurrier secondary wide-mouth bank (`Bank B`)
  at climaxes caused `Wan2.1` V2V (`strength=0.36`) to interpret the soft pink
  rim as a sticking-out tongue.
- **Fix**: Stick to **one razor-sharp stabilized mouth bank**, keep V2V
  `strength` in `0.25–0.30` (sweet spot: `0.28`), and explicitly add
  `sticking out tongue, protruding tongue, pink smudge, blurry mouth` to
  `negative_prompt`.

### 3.5 Syllable Envelope Tuning (No Chatter, Clean Breaths)
- Too short an RMS window (`35ms`) causes twitchy vibrato jaw-chatter.
- **Golden recipe**: `70ms` RMS window + noise gate (`< 0.036 -> 0.0`) + gentle
  local-mean modulation (`0.65–1.18`) + asymmetric attack/release (`attack=0.46`,
  `release=0.25`) + 3-tap `[0.25, 0.50, 0.25]` smoothing pass.

---

## 4. Post-Processing, Subtitles & Verification Pitfalls

### 4.1 Smooth Emotional Dolly Zoom Without FFmpeg Expression Errors
- FFmpeg `crop` rejects time `t` inside `w`/`h`, and `zoompan` can exhibit
  integer-pixel jitter.
- **Fix**: Use OpenCV `cv2.warpAffine` with `cv2.INTER_LANCZOS4` and a smooth
  `smoothstep` scale curve (`1.00x → 1.085x → 1.045x → 1.11x`) centered on the
  subject's face (`0` rotation, `0` border flicker).

### 4.2 Local macOS FFmpeg Missing `drawtext` / `subtitles` Filters
- Homebrew FFmpeg builds sometimes omit `libass` / `freetype`. Burn CJK
  subtitles on Colab (after `apt-get install -y fonts-noto-cjk ffmpeg`) where
  `subtitles=...:force_style='FontName=Noto Sans CJK TC,...'` works reliably.
- Always pass **`-crf 15`** when re-encoding with burned subtitles; default
  `libx264` CRF 23 smears fine textures.
- Always validate SRT files for strict `HH:MM:SS,mmm` formatting (two colons,
  one comma) and zero-gap line transitions so subtitles never blink off during
  short breaths.

### 4.3 Mandatory Dynamic + 4fps Audit Rule
- Never declare a lip-sync video "done" based on a 1fps contact sheet. Always:
  1. Watch the actual `.mp4` with audio via `read_file` to judge real-time
     rhythm and audio-visual sync.
  2. Generate and inspect a **4fps contact sheet** (`audit-video`) to catch
     frame-level artifacts, subtitle gaps, or V2V seam glitches.
