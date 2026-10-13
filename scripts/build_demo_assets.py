# -*- coding: utf-8 -*-
"""把 Playwright 录出来的 webm 转成 README 用的 MP4 + GIF。

用法:
    python scripts/build_demo_assets.py [输入 webm] [输出目录]

默认:
    输入  frontend/demo_output/unikb_demo.webm
    输出  docs/demo/  ->  demo.mp4 (H.264) + demo.gif (预览)

为什么需要这一步:
1. **webm 没有时长元数据**。Playwright 的 recordVideo 直接产出 VP8/WebM,
   ffprobe 读出来是 `Duration: N/A`, GitHub 与部分浏览器/播放器无法正确显示
   时长和拖动进度条。重新封装一次就能写入 duration。
2. **Safari 不支持 VP8/WebM**, README 里只放 webm 的话 Safari 用户点开是黑的。
   MP4(H.264) 是通用解。
3. **GIF 用于 README 内嵌预览**。GitHub 的 `<video>` 标签在 README 里不可控,
   `<img>` 引 GIF 才能稳定自动播放/循环。

ffmpeg 来源 (按优先级):
1. 环境变量 `FFMPEG` 指定的路径;
2. 系统 PATH 里的 `ffmpeg`;
3. `imageio-ffmpeg` 自带的二进制 (Windows 上最常见的情况: 系统没装 ffmpeg,
   但装过 imageio 就会有这个 85MB 的完整构建, 带 libx264 / libvpx)。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "frontend" / "demo_output" / "unikb_demo.webm"
DEFAULT_OUTDIR = ROOT / "docs" / "demo"

# GIF 预览参数: 宽度与帧率直接决定体积, 760px/12fps 是"看得清文字又不至于
# 十几 MB"的折中; 再低代码块就糊了, 再高 GitHub 加载会卡。
GIF_WIDTH = 760
GIF_FPS = 12
GIF_COLORS = 128

# MP4: crf 23 + slow  preset 是屏幕录制类的甜点, 文字边缘不会糊成一片。
MP4_CRF = "23"
MP4_PRESET = "slow"


def find_ffmpeg() -> str:
    """按 环境变量 -> PATH -> imageio-ffmpeg 的顺序找 ffmpeg。"""
    env = os.environ.get("FFMPEG", "").strip()
    if env and Path(env).exists():
        return env

    which = shutil.which("ffmpeg")
    if which:
        return which

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise SystemExit(
        "找不到 ffmpeg。三选一:\n"
        "  1. set FFMPEG=C:\\path\\to\\ffmpeg.exe\n"
        "  2. 把 ffmpeg 加进 PATH\n"
        "  3. pip install imageio-ffmpeg  (自带完整构建, 含 libx264)"
    )


def run(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:] + "\n")
        raise SystemExit(f"命令失败 (exit {proc.returncode})")
    return proc


def probe_duration(ffmpeg: str, path: Path) -> str:
    """读时长。webm 常常读不出来 (N/A), 这里只用于日志提示。"""
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in (proc.stderr or "").splitlines():
        if "Duration:" in line:
            return line.split("Duration:")[1].split(",")[0].strip()
    return "N/A"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTDIR

    if not src.exists():
        raise SystemExit(f"输入文件不存在: {src}\n先跑 `cd frontend && node record_demo.mjs`")
    outdir.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    print(f"ffmpeg : {ffmpeg}")
    print(f"输入   : {src}  ({src.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"原时长 : {probe_duration(ffmpeg, src)}  (webm 常为 N/A, 转码后会写入)")
    print(f"输出   : {outdir}")

    mp4 = outdir / "demo.mp4"
    gif = outdir / "demo.gif"

    print("\n[1/2] MP4 (H.264 + faststart)")
    run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        # genpts: webm 缺时长元数据时重新生成时间戳, 否则输出仍有概率没有 duration
        "-fflags", "+genpts",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", MP4_PRESET,
        "-crf", MP4_CRF,
        "-pix_fmt", "yuv420p",
        # 高 Profile 兼容性最好: 老设备/内嵌浏览器也能放
        "-profile:v", "high", "-level", "4.0",
        # faststart: 把 moov 挪到文件头, 网页点开就能播, 不用等下完
        "-movflags", "+faststart",
        "-an",
        str(mp4),
    ])
    print(f"  -> {mp4.name}  {mp4.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n[2/2] GIF (预览用, 两遍法调色板)")
    # 两遍法: 先为整段生成统一调色板, 再用它量化。
    # 单遍 palettegen 会让每一帧各自取色, 播放时颜色跳变闪烁。
    run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-filter_complex",
        (
            f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos,"
            f"split[s0][s1];[s0]palettegen=max_colors={GIF_COLORS}:stats_mode=diff[p];"
            f"[s1][p]paletteuse=dither=bayer:bayer_scale=3"
        ),
        "-loop", "0",
        str(gif),
    ])
    print(f"  -> {gif.name}  {gif.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n完成:")
    print(f"  {mp4}  {mp4.stat().st_size / 1024 / 1024:.2f} MB  {probe_duration(ffmpeg, mp4)}")
    print(f"  {gif}  {gif.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
