import argparse
import os
import sys
import time
import zipfile
import tempfile
import urllib.request
import subprocess
import threading
import socket
import json
from pathlib import Path

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".flv", ".ts", ".m4v", ".wmv", ".webm", ".mpg", ".mpeg")


def log(msg: str) -> None:
    print(msg, flush=True)


def find_video(source: Path) -> Path:
    if source.is_file():
        return source
    if source.is_dir():
        for ext in VIDEO_EXTS:
            files = sorted(source.glob(f"*{ext}"))
            if files:
                return files[0]
    raise FileNotFoundError(f"No video file found at {source}")


def ensure_ffmpeg() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except Exception as e:
        raise RuntimeError(
            "imageio-ffmpeg is required. Please install requirements: pip install -r requirements.txt"
        ) from e

    ffmpeg_path = get_ffmpeg_exe()
    if not ffmpeg_path or not Path(ffmpeg_path).exists():
        raise RuntimeError("Failed to acquire FFmpeg binary")
    return ffmpeg_path


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dst, "wb") as f:
        chunk = resp.read(8192)
        while chunk:
            f.write(chunk)
            chunk = resp.read(8192)


def _github_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)


def _resolve_mediamtx_asset_url() -> str:
    # Try the 'latest' release first
    try:
        latest = _github_json("https://api.github.com/repos/bluenviron/mediamtx/releases/latest")
        assets = latest.get("assets", [])
        for a in assets:
            name = a.get("name", "").lower()
            if name.endswith(".zip") and ("windows" in name) and ("amd64" in name or "x64" in name):
                return a.get("browser_download_url")
    except Exception:
        pass

    # Fallback: iterate recent releases
    try:
        releases = _github_json("https://api.github.com/repos/bluenviron/mediamtx/releases")
        for rel in releases[:10]:
            for a in rel.get("assets", []):
                name = a.get("name", "").lower()
                if name.endswith(".zip") and ("windows" in name) and ("amd64" in name or "x64" in name):
                    return a.get("browser_download_url")
    except Exception:
        pass

    raise RuntimeError("Could not find MediaMTX Windows asset via GitHub API. Please download mediamtx for Windows manually and place mediamtx.exe under .rtsp_streamer/mediamtx.")


def ensure_mediamtx(base_dir: Path) -> Path:
    # Only Windows is needed for this task; adjust here if you want cross-platform
    target_dir = base_dir / ".rtsp_streamer" / "mediamtx"
    target_dir.mkdir(parents=True, exist_ok=True)
    exe_path = target_dir / "mediamtx.exe"

    if exe_path.exists():
        return exe_path

    url = _resolve_mediamtx_asset_url()

    log("Downloading MediaMTX (RTSP server)...")
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "mediamtx.zip"
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

    if not exe_path.exists():
        # Some archives may contain nested directories; search for the exe
        for p in target_dir.rglob("mediamtx.exe"):
            exe_path = p
            break
    if not exe_path.exists():
        raise RuntimeError("Failed to set up MediaMTX binary")

    return exe_path


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(host, port):
            return True
        time.sleep(0.2)
    return False


def stream_subprocess_output(proc: subprocess.Popen, prefix: str):
    try:
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            print(f"[{prefix}] {line.rstrip()}", flush=True)
    except Exception:
        pass


def start_mediamtx_server(exe_path: Path, port: int) -> subprocess.Popen:
    # MediaMTX listens on 8554 by default; to change port, we provide a minimal config
    # Create a temporary config that sets the RTSP port
    cfg_dir = exe_path.parent
    cfg_path = cfg_dir / "mediamtx.yml"
    cfg_content = f"""
rtspAddress: :{port}
paths:
  all:
    source: publisher
""".lstrip()
    cfg_path.write_text(cfg_content, encoding="utf-8")

    cmd = [str(exe_path), str(cfg_path)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cfg_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_subprocess_output, args=(proc, "mediamtx"), daemon=True).start()
    return proc


def start_ffmpeg_publisher(ffmpeg: str, input_video: Path, rtsp_url: str) -> subprocess.Popen:
    # Encode to H.264 and stream via RTSP over TCP. Disable audio to avoid issues with missing AAC encoder or audio track.
    cmd = [
        ffmpeg,
        "-re",
        "-stream_loop", "-1",
        "-i", str(input_video),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-an",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=stream_subprocess_output, args=(proc, "ffmpeg"), daemon=True).start()
    return proc


def main():
    parser = argparse.ArgumentParser(description="Fake RTSP streamer from a local video file using MediaMTX + FFmpeg")
    parser.add_argument("--source", type=str, default=str(Path(r"c:\AlgoOrange Task\RTSP_reader\sample_video")), help="Path to a video file or a folder containing a video file")
    parser.add_argument("--stream-name", type=str, default="test", help="RTSP stream name (path)")
    parser.add_argument("--port", type=int, default=8554, help="RTSP TCP port to listen on")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    source_path = Path(args.source)
    try:
        video_path = find_video(source_path)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)

    ffmpeg_path = ensure_ffmpeg()
    mediamtx_path = ensure_mediamtx(base_dir)

    # Workaround for hyphenated name in f-string: use attribute directly
    rtsp_url = f"rtsp://127.0.0.1:{args.port}/{args.stream_name}"

    log("Starting RTSP server (MediaMTX)...")
    server_proc = start_mediamtx_server(mediamtx_path, args.port)

    if not wait_for_port("127.0.0.1", args.port, timeout=20):
        log("ERROR: RTSP server did not start listening in time.")
        server_proc.terminate()
        server_proc.wait(timeout=5)
        sys.exit(2)

    log(f"Publishing {video_path.name} to {rtsp_url}")
    pub_proc = start_ffmpeg_publisher(ffmpeg_path, video_path, rtsp_url)

    log("Streamer is running. Press Ctrl+C to stop.")
    try:
        # Keep the main thread alive while children run
        while True:
            if server_proc.poll() is not None:
                log("MediaMTX server exited.")
                break
            if pub_proc.poll() is not None:
                log("FFmpeg publisher exited.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        log("Stopping...")
    finally:
        for p in (pub_proc, server_proc):
            try:
                if p and p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        for p in (pub_proc, server_proc):
            try:
                if p:
                    p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
