#!/usr/bin/env python3
"""Colab GPU low-denoise Wan2.1 V2V polish + optional Lanczos4 dolly zoom + CRF 15 CJK subtitles."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def extract_frames_16fps(video_path: Path, width: int, height: int):
    import numpy as np
    from PIL import Image

    raw = subprocess.check_output(
        [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-vf", f"fps=16,scale={width}:{height}:flags=lanczos",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
    )
    fb = width * height * 3
    n = len(raw) // fb
    return [
        Image.fromarray(np.frombuffer(raw[i * fb : (i + 1) * fb], dtype=np.uint8).reshape((height, width, 3)))
        for i in range(n)
    ]


def smoothstep(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return u * u * (3.0 - 2.0 * u)


def apply_dolly_zoom(in_mp4: Path, out_mp4: Path, center_x_frac: float = 0.47, center_y_frac: float = 0.43) -> None:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(in_mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    dur = len(frames) / fps
    cx, cy = w * center_x_frac, h * center_y_frac

    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(out_mp4),
        ],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    for i, frm in enumerate(frames):
        t = i / fps
        # Smooth two-climax emotional push-in (1.00x -> 1.085x -> 1.045x -> 1.11x), zero rotation
        z = (
            1.00
            + 0.020 * smoothstep(t, 0.0, dur * 0.19)
            + 0.065 * smoothstep(t, dur * 0.19, dur * 0.54)
            - 0.040 * smoothstep(t, dur * 0.54, dur * 0.75)
            + 0.065 * smoothstep(t, dur * 0.75, dur * 0.94)
            - 0.015 * smoothstep(t, dur * 0.94, dur)
        )
        m = np.array([[z, 0.0, cx * (1.0 - z)], [0.0, z, cy * (1.0 - z)]], dtype=np.float32)
        warped = cv2.warpAffine(frm, m, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        enc.stdin.write(warped.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("Dolly zoom ffmpeg encode failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab GPU low-denoise Wan2.1 V2V polish + dolly zoom + subtitles")
    parser.add_argument("--input", required=True, type=Path, help="Input 0-CCU synced preview video")
    parser.add_argument("--audio", required=True, type=Path, help="Full original audio track")
    parser.add_argument("--subtitles", default=None, type=Path, help="Optional seamless SRT subtitle file")
    parser.add_argument("--output-clean", required=True, type=Path)
    parser.add_argument("--output-sub", default=None, type=Path)
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default=(
        "Close-up static smartphone video of a cute pet singing an emotional song, "
        "natural realistic skin and eyelids, smooth organic jaw movement and subtle breathing, "
        "natural lighting, crisp focus, stable tripod camera."
    ))
    parser.add_argument("--negative-prompt", default=(
        "sticking out tongue, protruding tongue, pink smudge, blurry mouth, "
        "human lips, human teeth, distorted face, morphing background, blurry, "
        "flickering scales, low quality, watermark, text, double eyes"
    ))
    parser.add_argument("--strength", type=float, default=0.28, help="SDEdit denoise strength (recommended 0.25-0.30)")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--dolly-zoom", action="store_true", help="Apply smooth Lanczos4 climax push-in")
    parser.add_argument("--font-name", default="Noto Sans CJK TC")
    args = parser.parse_args()

    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from PIL import Image

    print(f"[v2v] Loading {args.model_id} on {torch.cuda.get_device_name(0)}...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVideoToVideoPipeline.from_pretrained(args.model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    frames = extract_frames_16fps(args.input, args.width, args.height)
    n_total = len(frames)
    chunk_len = 81
    windows = [(0, n_total)] if n_total <= chunk_len else [(0, chunk_len), (max(0, n_total - chunk_len), n_total)]

    rendered = []
    for idx, (s, e) in enumerate(windows):
        sub = frames[s:e]
        valid_len = ((len(sub) - 1) // 4) * 4 + 1
        sub = sub[:valid_len]
        print(f"[v2v] Window {idx+1}/{len(windows)} frames [{s}:{s+valid_len}] strength={args.strength}", flush=True)
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
        out = pipe(
            video=sub,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=gen,
        ).frames[0]
        arrs = [
            np.array(f.convert("RGB") if isinstance(f, Image.Image) else (np.clip(f, 0, 1) * 255).astype(np.uint8))
            for f in out
        ]
        rendered.append((s, arrs))

    if len(rendered) == 1:
        merged = rendered[0][1][:n_total]
    else:
        s1, c1 = rendered[0]
        s2, c2 = rendered[1]
        ov = (s1 + len(c1)) - s2
        merged = list(c1[:s2])
        for i in range(ov):
            w_b = 0.5 * (1.0 - math.cos(math.pi * (i + 0.5) / max(1, ov)))
            merged.append((c1[s2 + i].astype(np.float32) * (1.0 - w_b) + c2[i].astype(np.float32) * w_b).clip(0, 255).astype(np.uint8))
        merged.extend(c2[ov:])
        merged = merged[:n_total]

    work = args.output_clean.parent / "_v2v_work"
    work.mkdir(parents=True, exist_ok=True)
    raw_16 = work / "polished_16fps.mp4"
    enc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", "16", "-i", "-",
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(raw_16),
        ],
        stdin=subprocess.PIPE,
    )
    assert enc.stdin is not None
    for arr in merged:
        enc.stdin.write(arr.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("Failed encoding 16fps V2V output")

    raw_30 = work / "polished_30fps.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(raw_16),
            "-vf", "minterpolate=fps=30:mi_mode=blend",
            "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", str(raw_30),
        ],
        check=True,
    )

    video_for_mux = raw_30
    if args.dolly_zoom:
        zoomed = work / "polished_30fps_dolly.mp4"
        apply_dolly_zoom(raw_30, zoomed)
        video_for_mux = zoomed

    args.output_clean.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_for_mux), "-i", str(args.audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(args.output_clean),
        ],
        check=True,
    )

    if args.subtitles and args.subtitles.exists():
        out_sub = args.output_sub or args.output_clean.with_name(f"{args.output_clean.stem}_subtitled.mp4")
        style = (
            f"FontName={args.font_name},FontSize=20,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=52"
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(args.output_clean),
                "-vf", f"subtitles={args.subtitles}:force_style='{style}'",
                "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p",
                "-c:a", "copy", str(out_sub),
            ],
            check=True,
        )
        print(f"[ok] Subtitled master written to: {out_sub}", flush=True)

    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(args.output_clean)],
        text=True,
    )
    print(f"[ok] Clean master written to: {args.output_clean} | {json.loads(probe)['format']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
