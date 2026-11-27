"""Master build script to render 1080p 16:9 MP4 video synchronized with audio.

Usage:
  python3 build_video.py --preview   # Render keyframe PNGs across all scenes for inspection
  python3 build_video.py             # Render full 1080p MP4 synced with voice-over & underscore
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from timeline import FPS, TOTAL_DURATION, get_scene_info
from audio_mix import prepare_final_audio
from scenes.opening_closing import render_opening, render_closing
from scenes.scene1_hook import render_scene1
from scenes.scene2_definition import render_scene2
from scenes.scene3_visual import render_scene3
from scenes.scene4_example import render_scene4
from scenes.scene5_apps import render_scene5
from scenes.scene6_summary import render_scene6

RENDER_DISPATCH = {
    "opening": render_opening,
    "scene1": render_scene1,
    "scene2": render_scene2,
    "scene3": render_scene3,
    "scene4": render_scene4,
    "scene5": render_scene5,
    "scene6": render_scene6,
    "closing": render_closing,
}


def render_frame(t: float):
    """Render a single 1920x1080 RGB frame at timestamp t (seconds)."""
    scene_id = get_scene_info(t)[0]
    return RENDER_DISPATCH[scene_id](t)


def render_chunk(args):
    """Render frame index range [f0, f1) to a lossless/high-quality H.264 chunk MP4."""
    chunk_idx, f0, f1, out_path = args
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", "1920x1080",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for fi in range(f0, f1):
        t = fi / float(FPS)
        img = render_frame(t)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg chunk {chunk_idx} exited with status {ret}")
    return chunk_idx, f1 - f0, out_path


def run_preview(out_dir: str):
    """Render representative keyframes across every scene for quick visual QA."""
    os.makedirs(out_dir, exist_ok=True)
    timestamps = [
        ("01_opening", 4.5),
        ("02_scene1_paths", 22.5),
        ("03_scene1_grad", 36.0),
        ("04_scene2_1d", 48.0),
        ("05_scene2_3d", 93.5),
        ("06_scene3_dial", 118.5),
        ("07_scene3_contour", 136.0),
        ("08_scene4_3d", 184.0),
        ("09_scene4_unit", 214.0),
        ("10_scene5_apps", 248.0),
        ("11_scene6_sum", 276.0),
        ("12_closing", 286.0),
    ]
    for name, ts in timestamps:
        p = os.path.join(out_dir, f"{name}.png")
        render_frame(ts).save(p)
        print(f"Saved preview {p} (t={ts:.1f}s)")


def build_full_video(output_mp4: str, workers: int = 6, include_music: bool = True):
    """Render all frames in parallel chunks, mix audio, and mux final 1080p MP4."""
    t_start = time.time()
    build_dir = os.path.join(BASE_DIR, "_build_tmp")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_mp4), exist_ok=True)

    audio_wav = os.path.join(build_dir, "final_audio.wav")
    print("Preparing normalized voice-over and subtle underscore...")
    prepare_final_audio(audio_wav, include_music=include_music)

    total_frames = int(round(TOTAL_DURATION * FPS))
    # Split into fine chunks for balanced parallel execution
    num_chunks = max(workers * 2, 12)
    chunk_size = (total_frames + num_chunks - 1) // num_chunks
    tasks = []
    for k in range(num_chunks):
        f0 = k * chunk_size
        f1 = min(total_frames, (k + 1) * chunk_size)
        if f0 < f1:
            tasks.append((k, f0, f1, os.path.join(build_dir, f"chunk_{k:03d}.mp4")))

    print(f"Rendering {total_frames} frames ({TOTAL_DURATION:.1f}s @ {FPS}fps) across {workers} workers...")
    done_frames = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, count, _ in ex.map(render_chunk, tasks):
            done_frames += count
            print(f"  Chunk {idx + 1}/{len(tasks)} complete ({done_frames}/{total_frames} frames)")

    concat_list = os.path.join(build_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for _, _, _, path in tasks:
            f.write(f"file '{path}'\n")

    print("Muxing video chunks and audio track into final 1080p MP4...")
    cmd_mux = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list,
        "-i", audio_wav,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        output_mp4,
    ]
    subprocess.run(cmd_mux, check=True)
    print(f"Finished {output_mp4} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build M3 Gradient Animation Video")
    parser.add_argument("--preview", action="store_true", help="Render keyframe PNGs for visual inspection")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel render workers")
    parser.add_argument("--no-music", action="store_true", help="Omit ambient underscore")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(BASE_DIR), "Gradient_Video.mp4"))
    args = parser.parse_args()

    if args.preview:
        run_preview(os.path.join(BASE_DIR, "_preview"))
    else:
        build_full_video(args.output, workers=args.workers, include_music=not args.no_music)
