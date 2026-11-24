#!/usr/bin/env python3
"""Reusable local CLI toolkit (stdlib + ffmpeg/ffprobe) for AI video & pet lip-sync production.

Subcommands:
  analyze-audio  Extract 30fps syllable-synced vocal envelope JSON from a WAV/MP3/MP4.
  analyze-ref    Create a contact sheet + ffprobe metadata JSON from a reference video.
  validate-srt   Validate SRT syntax (HH:MM:SS,mmm), detect flicker gaps, optionally auto-fix.
  build-preview  Build a 0-CCU 30fps stabilized jaw-only + vertical-eyelid synced preview.
  audit-video    Generate a 4fps audit contact sheet + ffprobe quality report JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

SRT_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        print(f"[error] Required binary not found on PATH: {name}", file=sys.stderr)
        raise SystemExit(1)


def probe_media(path: Path) -> dict:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(out)


def load_mono_wav(audio_path: Path) -> tuple[list[float], int]:
    """Decode any audio/video file to 44.1kHz mono PCM float samples in [-1, 1]."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(audio_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-acodec",
                "pcm_s16le",
                tmp.name,
            ],
            check=True,
        )
        with wave.open(tmp.name, "rb") as wf:
            sr = wf.getframerate()
            n_samp = wf.getnframes()
            raw = wf.readframes(n_samp)
            samples = struct.unpack(f"<{n_samp}h", raw)
            mono = [s / 32768.0 for s in samples]
    return mono, sr


def compute_syllable_curve(
    mono: list[float],
    sr: int,
    fps: int = 30,
    window_ms: float = 70.0,
    noise_gate: float = 0.036,
    attack: float = 0.46,
    release: float = 0.25,
) -> tuple[list[float], list[float]]:
    dur = len(mono) / sr
    n_frames = max(1, int(round(dur * fps)))
    win = max(1, int(sr * window_ms / 1000.0))

    raw_rms: list[float] = []
    for f in range(n_frames):
        center = int((f + 0.5) * sr / fps)
        a0 = max(0, center - win // 2)
        a1 = min(len(mono), center + win // 2)
        seg = mono[a0:a1]
        raw_rms.append(math.sqrt(sum(v * v for v in seg) / len(seg)) if seg else 0.0)

    peak = max(raw_rms) or 1.0
    curve: list[float] = []
    for f in range(n_frames):
        r = raw_rms[f]
        if r < noise_gate:
            val = 0.0
        else:
            base_open = ((r - noise_gate) / max(1e-6, peak - noise_gate)) ** 0.65
            lo = max(0, f - 5)
            hi = min(n_frames, f + 6)
            local_mean = sum(raw_rms[lo:hi]) / (hi - lo)
            rel = r / max(0.045, local_mean)
            mod = max(0.65, min(1.18, 0.82 + 0.24 * (rel - 0.90)))
            val = max(0.0, min(1.0, base_open * mod * 1.08))
        curve.append(val)

    smoothed: list[float] = []
    prev = 0.0
    for v in curve:
        alpha = attack if v > prev else release
        cur = prev * (1.0 - alpha) + v * alpha
        smoothed.append(0.0 if cur < 0.035 else cur)
        prev = cur

    final: list[float] = []
    for i in range(n_frames):
        v = 0.25 * smoothed[max(0, i - 1)] + 0.50 * smoothed[i] + 0.25 * smoothed[min(n_frames - 1, i + 1)]
        final.append(0.0 if v < 0.03 else round(v, 4))
    return final, [round(x, 4) for x in raw_rms]


def cmd_analyze_audio(args: argparse.Namespace) -> int:
    require_binary("ffmpeg")
    mono, sr = load_mono_wav(args.audio)
    mouth, rms = compute_syllable_curve(
        mono,
        sr,
        fps=args.fps,
        window_ms=args.window_ms,
        noise_gate=args.noise_gate,
    )
    dur = len(mono) / sr
    # Detect silent pauses (>= 0.12s consecutive zeros)
    pauses = []
    start = None
    for i, m in enumerate(mouth):
        if m == 0.0 and start is None:
            start = i
        elif m > 0.0 and start is not None:
            if (i - start) / args.fps >= 0.12:
                pauses.append({"start_s": round(start / args.fps, 3), "end_s": round(i / args.fps, 3)})
            start = None

    report = {
        "audio": str(args.audio.resolve()),
        "duration_s": round(dur, 3),
        "fps": args.fps,
        "num_frames": len(mouth),
        "pauses": pauses,
        "frames": [
            {"frame": i, "time_s": round(i / args.fps, 3), "rms": rms[i], "mouth": mouth[i]}
            for i in range(len(mouth))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Audio envelope written to: {args.output} ({len(mouth)} frames, {len(pauses)} pauses)")
    return 0


def cmd_analyze_ref(args: argparse.Namespace) -> int:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(args.video),
            "-vf",
            f"fps={args.fps},scale={args.tile_width}:-1,tile={args.tile}",
            "-frames:v",
            "1",
            str(args.output),
        ],
        check=True,
    )
    meta = probe_media(args.video)
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Reference contact sheet written to: {args.output} (metadata: {meta_path})")
    return 0


def parse_srt_ms(ts: str) -> int:
    m = SRT_TIME_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Invalid SRT timestamp '{ts}' (expected HH:MM:SS,mmm)")
    hh, mm, ss, ms = map(int, m.groups())
    return ((hh * 3600 + mm * 60 + ss) * 1000) + ms


def fmt_srt_ms(total_ms: int) -> str:
    total_ms = max(0, total_ms)
    hh = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    mm = rem // 60_000
    rem %= 60_000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def cmd_validate_srt(args: argparse.Namespace) -> int:
    text = args.srt.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\r?\n\s*\r?\n", text)
    entries = []
    errors = []
    warnings = []

    for idx, blk in enumerate(blocks, start=1):
        lines = [l.rstrip("\r") for l in blk.splitlines()]
        if len(lines) < 3 or "-->" not in lines[1]:
            errors.append(f"Block {idx}: malformed SRT block")
            continue
        t0_s, t1_s = [p.strip() for p in lines[1].split("-->", 1)]
        try:
            t0 = parse_srt_ms(t0_s)
            t1 = parse_srt_ms(t1_s)
        except ValueError as exc:
            errors.append(f"Block {idx}: {exc}")
            continue
        if t1 <= t0:
            errors.append(f"Block {idx}: end ({t1_s}) <= start ({t0_s})")
        entries.append({"index": idx, "start_ms": t0, "end_ms": t1, "text": "\n".join(lines[2:])})

    for i in range(1, len(entries)):
        gap = entries[i]["start_ms"] - entries[i - 1]["end_ms"]
        if gap < 0:
            errors.append(f"Blocks {i}->{i+1}: overlap of {-gap} ms")
        elif 0 < gap <= 300:
            warnings.append(f"Blocks {i}->{i+1}: short flicker gap of {gap} ms")
            if args.fix_gaps:
                entries[i - 1]["end_ms"] = entries[i]["start_ms"]

    fixed_path = None
    if args.fix_gaps and not errors:
        fixed_path = args.fixed_srt or args.srt.with_name(f"{args.srt.stem}_seamless.srt")
        out_blocks = [
            f"{i}\n{fmt_srt_ms(e['start_ms'])} --> {fmt_srt_ms(e['end_ms'])}\n{e['text']}"
            for i, e in enumerate(entries, start=1)
        ]
        fixed_path.write_text("\n\n".join(out_blocks) + "\n", encoding="utf-8")

    report = {
        "srt": str(args.srt.resolve()),
        "valid": len(errors) == 0,
        "num_entries": len(entries),
        "errors": errors,
        "warnings": warnings,
        "fixed_srt": str(fixed_path.resolve()) if fixed_path else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] SRT validation report written to: {args.output} (valid={report['valid']}, warnings={len(warnings)})")
    return 0 if report["valid"] else 1


def parse_climax_windows(spec: str) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    if not spec.strip():
        return windows
    for part in spec.split(","):
        a, b = part.strip().split("-")
        windows.append((float(a), float(b)))
    return windows


def smoothstep(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return u * u * (3.0 - 2.0 * u)


def detect_active_vertical_bounds(frame: bytes, w: int, h: int) -> tuple[int, int]:
    top = 0
    for y in range(h):
        if sum(frame[(y * w + x) * 3] for x in range(w // 5, (4 * w) // 5, 10)) > 200:
            top = y
            break
    bot = h - 1
    for y in range(h - 1, -1, -1):
        if sum(frame[(y * w + x) * 3] for x in range(w // 5, (4 * w) // 5, 10)) > 200:
            bot = y
            break
    return top, bot - top + 1


def track_eye(frame: bytes, ref_frame: bytes, w: int, rx0: float, ry0: float, ax0: float, ay0: float, search: int = 8) -> tuple[float, float]:
    best = (1e18, ax0, ay0)
    ri, rj = int(round(rx0)), int(round(ry0))
    ai, aj = int(round(ax0)), int(round(ay0))
    for dy in range(-search, search + 1, 2):
        for dx in range(-search, search + 1, 2):
            err = 0
            cnt = 0
            for py in range(-16, 17, 4):
                for px in range(-20, 21, 4):
                    i1 = ((aj + dy + py) * w + (ai + dx + px)) * 3
                    i2 = ((rj + py) * w + (ri + px)) * 3
                    err += abs(frame[i1] - ref_frame[i2]) + abs(frame[i1 + 1] - ref_frame[i2 + 1]) + abs(frame[i1 + 2] - ref_frame[i2 + 2])
                    cnt += 1
            avg = err / max(1, cnt)
            if avg < best[0]:
                best = (avg, float(ai + dx), float(aj + dy))
    return best[1], best[2]


def cmd_build_preview(args: argparse.Namespace) -> int:
    require_binary("ffmpeg")
    require_binary("ffprobe")

    meta = probe_media(args.mouth_video)
    v_stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
    w, h = int(v_stream["width"]), int(v_stream["height"])
    f_start, f_end = [int(x) for x in args.mouth_frames.split(":")]

    raw_vid = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(args.mouth_video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )
    fb = w * h * 3
    ramp = [raw_vid[i * fb : (i + 1) * fb] for i in range(f_start, f_end)]
    if len(ramp) < 2:
        print("[error] --mouth-frames must span at least 2 frames (e.g. 56:66)", file=sys.stderr)
        return 1

    base = ramp[0]
    top_crop, active_h = detect_active_vertical_bounds(base, w, h)

    # Parse base eye coordinates: "lx,ly,rx,ry"
    blx, bly, brx, bry = [float(x) for x in args.base_eyes.split(",")]
    ecx_l, ecy_l, ecx_r, ecy_r = [float(x) for x in (args.closed_eyes_coords or args.base_eyes).split(",")]
    base_cx, base_cy = 0.5 * (blx + brx), 0.5 * (bly + bry)

    # Stabilize jaw region onto base head to eliminate sideways spring head-jerk
    stabilized = [base]
    prev_l, prev_r = (blx, bly), (brx, bry)
    jaw_cx = base_cx
    jaw_cy = base_cy + (brx - blx) * 0.84
    jaw_rx = (brx - blx) * 0.88
    jaw_ry = (brx - blx) * 0.72
    eye_floor = int(max(bly, bry) + 10)

    for frm in ramp[1:]:
        lx, ly = track_eye(frm, base, w, blx, bly, prev_l[0], prev_l[1], search=10)
        rx, ry = track_eye(frm, base, w, brx, bry, prev_r[0], prev_r[1], search=10)
        prev_l, prev_r = (lx, ly), (rx, ry)
        dx = 0.5 * (lx + rx) - base_cx
        dy = 0.5 * (ly + ry) - base_cy
        out = bytearray(base)
        for y in range(max(top_crop, eye_floor), min(top_crop + active_h, int(jaw_cy + jaw_ry + 2))):
            sy = int(round(y + dy))
            if not (top_crop <= sy < top_crop + active_h):
                continue
            for x in range(max(0, int(jaw_cx - jaw_rx - 2)), min(w, int(jaw_cx + jaw_rx + 2))):
                dist = math.sqrt(((x - jaw_cx) / jaw_rx) ** 2 + ((y - jaw_cy) / jaw_ry) ** 2)
                if dist >= 1.0:
                    continue
                sx = int(round(x + dx))
                if not (0 <= sx < w):
                    continue
                alpha = 1.0 if dist <= 0.62 else 0.5 * (1.0 + math.cos(math.pi * (dist - 0.62) / 0.38))
                if y < eye_floor + 18:
                    alpha *= (y - eye_floor) / 18.0
                inv = 1.0 - alpha
                ti, si = (y * w + x) * 3, (sy * w + sx) * 3
                out[ti] = int(out[ti] * inv + frm[si] * alpha)
                out[ti + 1] = int(out[ti + 1] * inv + frm[si + 1] * alpha)
                out[ti + 2] = int(out[ti + 2] * inv + frm[si + 2] * alpha)
        stabilized.append(bytes(out))

    ec_aligned = None
    if args.eyes_closed:
        ec_aligned = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(args.eyes_closed),
                "-vf",
                f"scale={w}:{active_h},pad={w}:{h}:0:{top_crop}:black",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
        )

    mono, sr = load_mono_wav(args.audio)
    mouth_curve, _ = compute_syllable_curve(mono, sr, fps=args.fps)
    n_frames = len(mouth_curve)
    climaxes = parse_climax_windows(args.climax_windows)

    rad_x, rad_y = 28.0, 22.0
    crop_start = top_crop * w * 3
    crop_end = (top_crop + active_h) * w * 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{active_h}",
            "-r",
            str(args.fps),
            "-i",
            "-",
            "-i",
            str(args.audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    na = len(stabilized)

    for f in range(n_frames):
        t = f / args.fps
        u = mouth_curve[f] * (na - 1)
        a0 = min(na - 1, int(math.floor(u)))
        a1 = min(na - 1, a0 + 1)
        frac = u - a0
        if frac <= 0.002:
            frame_full = stabilized[a0]
        elif frac >= 0.998:
            frame_full = stabilized[a1]
        else:
            k2 = int(round(frac * 256))
            k1 = 256 - k2
            frame_full = bytes(((a * k1 + b * k2) >> 8) for a, b in zip(stabilized[a0], stabilized[a1]))

        # Physical top-to-bottom vertical eyelid close wipe during climax windows
        c = 0.0
        for t0, t1 in climaxes:
            if t0 <= t <= t1:
                c = max(c, min(smoothstep(t, t0, t0 + 0.20), 1.0 - smoothstep(t, t1 - 0.24, t1)))

        if ec_aligned is not None and c > 0.005:
            out = bytearray(frame_full)
            lid_edge = -rad_y + c * (2.0 * rad_y + 6.0)
            feather_y = 5.0
            for (tx_c, ty_c), (sx_c, sy_c) in (((blx, bly), (ecx_l, ecy_l)), ((brx, bry), (ecx_r, ecy_r))):
                for y in range(max(top_crop, int(ty_c - rad_y - 2)), min(top_crop + active_h, int(ty_c + rad_y + 3))):
                    dy = y - ty_c
                    if dy > lid_edge + feather_y:
                        continue
                    wipe_a = 1.0 if dy <= lid_edge else max(0.0, 1.0 - (dy - lid_edge) / feather_y)
                    sy = int(round(sy_c + dy))
                    if not (0 <= sy < h):
                        continue
                    for x in range(max(0, int(tx_c - rad_x - 2)), min(w, int(tx_c + rad_x + 3))):
                        dx = x - tx_c
                        dist = math.sqrt((dx / rad_x) ** 2 + (dy / rad_y) ** 2)
                        if dist >= 1.0:
                            continue
                        sx = int(round(sx_c + dx))
                        if not (0 <= sx < w):
                            continue
                        oval_a = 1.0 if dist <= 0.62 else 0.5 * (1.0 + math.cos(math.pi * (dist - 0.62) / 0.38))
                        alpha = oval_a * wipe_a
                        if alpha <= 0.002:
                            continue
                        inv = 1.0 - alpha
                        ti, si = (y * w + x) * 3, (sy * w + sx) * 3
                        out[ti] = int(out[ti] * inv + ec_aligned[si] * alpha)
                        out[ti + 1] = int(out[ti + 1] * inv + ec_aligned[si + 1] * alpha)
                        out[ti + 2] = int(out[ti + 2] * inv + ec_aligned[si + 2] * alpha)
            frame_full = bytes(out)

        enc.stdin.write(frame_full[crop_start:crop_end])

    enc.stdin.close()
    if enc.wait() != 0:
        print("[error] ffmpeg preview encoding failed", file=sys.stderr)
        return 1
    print(f"[ok] 0-CCU synced preview written to: {args.output} ({args.output.stat().st_size / 1e6:.2f} MB)")
    return 0


def cmd_audit_video(args: argparse.Namespace) -> int:
    require_binary("ffmpeg")
    require_binary("ffprobe")
    meta = probe_media(args.video)
    dur = float(meta["format"]["duration"])
    est_frames = max(1, int(math.ceil(dur * args.fps)))
    cols = args.cols
    rows = max(1, int(math.ceil(est_frames / cols)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(args.video),
            "-vf",
            f"fps={args.fps},scale={args.tile_width}:-1,tile={cols}x{rows}",
            "-frames:v",
            "1",
            str(args.output),
        ],
        check=True,
    )
    report_path = args.output.with_suffix(".json")
    v_streams = [s for s in meta["streams"] if s.get("codec_type") == "video"]
    a_streams = [s for s in meta["streams"] if s.get("codec_type") == "audio"]
    warnings = []
    if not v_streams or not a_streams:
        warnings.append("Missing video or audio stream!")
    if v_streams:
        br = int(v_streams[0].get("bit_rate", 0) or 0)
        if 0 < br < 900_000:
            warnings.append(f"Low video bitrate ({br/1000:.0f} kbps) — re-encode with -crf 15 for sharper textures.")
    report = {
        "video": str(args.video.resolve()),
        "contact_sheet": str(args.output.resolve()),
        "audit_fps": args.fps,
        "tile_grid": f"{cols}x{rows}",
        "warnings": warnings,
        "ffprobe": meta,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Audit sheet written to: {args.output} (report: {report_path}, warnings={len(warnings)})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local toolkit for Colab AI video & pet lip-sync production.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_aud = sub.add_parser("analyze-audio", help="Extract 30fps syllable-synced vocal envelope JSON")
    p_aud.add_argument("--audio", required=True, type=Path)
    p_aud.add_argument("--fps", type=int, default=30)
    p_aud.add_argument("--window-ms", type=float, default=70.0)
    p_aud.add_argument("--noise-gate", type=float, default=0.036)
    p_aud.add_argument("--output", required=True, type=Path)
    p_aud.set_defaults(func=cmd_analyze_audio)

    p_ref = sub.add_parser("analyze-ref", help="Extract contact sheet + metadata from reference video")
    p_ref.add_argument("--video", required=True, type=Path)
    p_ref.add_argument("--fps", type=float, default=2.0)
    p_ref.add_argument("--tile", default="5x4")
    p_ref.add_argument("--tile-width", type=int, default=240)
    p_ref.add_argument("--output", required=True, type=Path)
    p_ref.set_defaults(func=cmd_analyze_ref)

    p_srt = sub.add_parser("validate-srt", help="Validate SRT timestamps and optionally close flicker gaps")
    p_srt.add_argument("--srt", required=True, type=Path)
    p_srt.add_argument("--fix-gaps", action="store_true")
    p_srt.add_argument("--fixed-srt", default=None, type=Path)
    p_srt.add_argument("--output", required=True, type=Path)
    p_srt.set_defaults(func=cmd_validate_srt)

    p_pre = sub.add_parser("build-preview", help="Build 0-CCU stabilized jaw + vertical-eyelid synced preview")
    p_pre.add_argument("--mouth-video", required=True, type=Path)
    p_pre.add_argument("--mouth-frames", required=True, help="Inclusive:exclusive frame slice, e.g. 56:66")
    p_pre.add_argument("--base-eyes", required=True, help="lx,ly,rx,ry pixel coords on first mouth frame")
    p_pre.add_argument("--eyes-closed", default=None, type=Path, help="Pixel-aligned closed-eyes image")
    p_pre.add_argument("--closed-eyes-coords", default=None, help="Optional lx,ly,rx,ry on aligned closed-eyes image")
    p_pre.add_argument("--audio", required=True, type=Path)
    p_pre.add_argument("--climax-windows", default="", help="Comma-separated t0-t1 sec windows, e.g. 1.92-5.34,7.42-9.18")
    p_pre.add_argument("--fps", type=int, default=30)
    p_pre.add_argument("--output", required=True, type=Path)
    p_pre.set_defaults(func=cmd_build_preview)

    p_chk = sub.add_parser("audit-video", help="Generate 4fps audit contact sheet + ffprobe quality JSON")
    p_chk.add_argument("--video", required=True, type=Path)
    p_chk.add_argument("--fps", type=float, default=4.0)
    p_chk.add_argument("--cols", type=int, default=6)
    p_chk.add_argument("--tile-width", type=int, default=200)
    p_chk.add_argument("--output", required=True, type=Path)
    p_chk.set_defaults(func=cmd_audit_video)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
