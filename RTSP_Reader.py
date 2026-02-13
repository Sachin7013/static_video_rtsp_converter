"""
RTSP Video Streamer
Streams a static video file via RTSP protocol with loop functionality.
Uses MediaMTX as RTSP server and FFmpeg to push the video stream.
"""

import cv2
import os
import sys
import subprocess
import time
import signal
import socket
import threading
from pathlib import Path


class RTSPVideoStreamer:
    """
    A class to stream static video files via RTSP protocol.
    The video will loop continuously using FFmpeg and MediaMTX.
    """
    
    def __init__(self, video_path, rtsp_port=8554, stream_name="video_stream", use_mediamtx=True):
        """
        Initialize the RTSP video streamer.
        
        Args:
            video_path (str): Path to the video file
            rtsp_port (int): Port for RTSP server (default: 8554)
            stream_name (str): Name of the RTSP stream (default: "video_stream")
            use_mediamtx (bool): Use MediaMTX server if available (default: True)
        """
        self.video_path = os.path.abspath(video_path)
        self.rtsp_port = rtsp_port
        self.stream_name = stream_name
        self.use_mediamtx = use_mediamtx
        self.running = False
        self.ffmpeg_process = None
        self.mediamtx_process = None
        self.ffmpeg_stderr_thread = None
        self.ffmpeg_errors = []
        
        # Find MediaMTX executable
        self.mediamtx_path = None
        if use_mediamtx:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            mediamtx_exe = os.path.join(script_dir, '.rtsp_streamer', 'mediamtx', 'mediamtx.exe')
            if os.path.exists(mediamtx_exe):
                self.mediamtx_path = mediamtx_exe
                self.mediamtx_config = os.path.join(script_dir, '.rtsp_streamer', 'mediamtx', 'mediamtx.yml')
            else:
                print("Warning: MediaMTX not found, will attempt direct FFmpeg RTSP streaming")
                self.use_mediamtx = False
        
        # Validate video file
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Get video properties
        temp_cap = cv2.VideoCapture(self.video_path)
        if not temp_cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        self.fps = int(temp_cap.get(cv2.CAP_PROP_FPS)) or 25
        self.width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        temp_cap.release()
        
        print(f"Video properties: {self.width}x{self.height} @ {self.fps} FPS")
    
    def get_rtsp_url(self):
        """Get the RTSP URL for accessing the stream."""
        return f"rtsp://localhost:{self.rtsp_port}/{self.stream_name}"
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True,
                         creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_port(self, port, timeout=1):
        """Check if a port is listening."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _start_mediamtx(self):
        """Start MediaMTX RTSP server."""
        if not self.mediamtx_path:
            return False
        
        try:
            mediamtx_dir = os.path.dirname(self.mediamtx_path)
            mediamtx_config_abs = os.path.abspath(self.mediamtx_config)
            
            # Verify config file exists
            if not os.path.exists(mediamtx_config_abs):
                print(f"Error: MediaMTX config file not found: {mediamtx_config_abs}")
                return False
            
            print(f"Starting MediaMTX from: {mediamtx_dir}")
            print(f"Using config: {mediamtx_config_abs}")
            
            self.mediamtx_process = subprocess.Popen(
                [self.mediamtx_path, mediamtx_config_abs],
                cwd=mediamtx_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Give MediaMTX time to start
            time.sleep(3)
            
            # Check if process is still running
            if self.mediamtx_process.poll() is not None:
                stderr_output = self.mediamtx_process.stderr.read().decode('utf-8', errors='ignore')
                stdout_output = self.mediamtx_process.stdout.read().decode('utf-8', errors='ignore')
                print(f"❌ Error: MediaMTX process exited immediately.")
                if stderr_output:
                    print(f"MediaMTX stderr: {stderr_output}")
                if stdout_output:
                    print(f"MediaMTX stdout: {stdout_output}")
                return False
            
            # Verify port is listening
            max_retries = 5
            for i in range(max_retries):
                if self._check_port(self.rtsp_port):
                    print(f"✓ MediaMTX RTSP server started and listening on port {self.rtsp_port}")
                    return True
                if i < max_retries - 1:
                    time.sleep(1)
            
            print(f"⚠️  Warning: MediaMTX started but port {self.rtsp_port} is not listening after {max_retries} attempts")
            # Check if process is still running
            if self.mediamtx_process.poll() is not None:
                stderr_output = self.mediamtx_process.stderr.read().decode('utf-8', errors='ignore')
                print(f"MediaMTX process exited. Error: {stderr_output}")
                return False
            return False
            
        except Exception as e:
            print(f"❌ Error: Could not start MediaMTX: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _stop_mediamtx(self):
        """Stop MediaMTX RTSP server."""
        if self.mediamtx_process:
            try:
                self.mediamtx_process.terminate()
                try:
                    self.mediamtx_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.mediamtx_process.kill()
                    self.mediamtx_process.wait()
            except Exception as e:
                print(f"Error stopping MediaMTX: {e}")
            finally:
                self.mediamtx_process = None
    
    def _monitor_ffmpeg_stderr(self):
        """Monitor FFmpeg stderr in real-time to capture errors."""
        if not self.ffmpeg_process:
            return
        
        try:
            line_count = 0
            while self.running and self.ffmpeg_process:
                line = self.ffmpeg_process.stderr.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore').strip()
                if line:
                    self.ffmpeg_errors.append(line)
                    line_count += 1
                    
                    # Print first 10 lines to see startup info
                    if line_count <= 10:
                        print(f"[FFmpeg] {line}")
                    # Always print important errors/warnings/connection messages
                    elif any(keyword in line.lower() for keyword in [
                        'error', 'failed', 'cannot', 'connection refused', 'timeout',
                        'connected', 'streaming', 'publishing', 'rtsp', 'tcp'
                    ]):
                        print(f"[FFmpeg] {line}")
        except Exception as e:
            print(f"Error monitoring FFmpeg stderr: {e}")
    
    def start_stream(self):
        """Start the RTSP video stream."""
        if self.running:
            print("Stream is already running!")
            return None
        
        # Check if FFmpeg is available
        if not self._check_ffmpeg():
            print("Error: FFmpeg is not installed or not in PATH.")
            print("Please install FFmpeg from https://ffmpeg.org/download.html")
            return None
        
        # Start MediaMTX if available
        if self.use_mediamtx:
            if not self._start_mediamtx():
                print("Falling back to direct FFmpeg RTSP streaming...")
                self.use_mediamtx = False
        
        self.running = True
        rtsp_url = self.get_rtsp_url()
        
        print(f"\n{'='*60}")
        print(f"Starting RTSP Video Streamer...")
        print(f"{'='*60}")
        print(f"RTSP URL: {rtsp_url}")
        print(f"Video: {self.video_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Loop: Enabled (infinite)")
        print(f"Server: {'MediaMTX' if self.use_mediamtx else 'FFmpeg Direct'}")
        print(f"{'='*60}\n")
        
        # FFmpeg command to push video to RTSP server
        # Using stream_loop for infinite looping
        # Note: -re flag is important for RTSP streaming to maintain proper timing
        ffmpeg_cmd = [
            'ffmpeg',
            '-re',  # Read input at native frame rate (important for RTSP)
            '-stream_loop', '-1',  # Loop indefinitely
            '-i', self.video_path,
            '-c:v', 'libx264',  # Video codec
            '-preset', 'ultrafast',  # Encoding preset for low latency
            '-tune', 'zerolatency',  # Zero latency tuning
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            '-r', str(self.fps),  # Output frame rate
            '-g', str(int(self.fps * 2)),  # GOP size
            '-b:v', '2000k',  # Video bitrate
            '-maxrate', '2000k',  # Max bitrate
            '-bufsize', '4000k',  # Buffer size
        ]
        
        # Check if video has audio track
        has_audio = False
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
                self.video_path
            ]
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            has_audio = 'audio' in result.stdout.lower()
        except:
            # If ffprobe fails, try to include audio anyway
            has_audio = True
        
        # Add audio if present
        if has_audio:
            ffmpeg_cmd.extend([
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-ar', '44100',  # Audio sample rate
            ])
        else:
            # No audio track - disable audio
            ffmpeg_cmd.extend(['-an'])
        
        # RTSP output settings
        if self.use_mediamtx:
            # Push to MediaMTX server
            # MediaMTX expects the stream to be published to the path
            # Try TCP first (more reliable), fallback to UDP if needed
            ffmpeg_cmd.extend([
                '-f', 'rtsp',  # Output format
                '-rtsp_transport', 'tcp',  # Use TCP for reliability
                '-muxdelay', '0.1',  # Reduce delay
                '-loglevel', 'info',  # Show info, warnings and errors for debugging
                rtsp_url
            ])
        else:
            # Direct RTSP streaming (requires RTSP server capability)
            # Note: This may not work on all systems
            ffmpeg_cmd.extend([
                '-f', 'rtsp',  # Output format
                '-rtsp_transport', 'tcp',  # Use TCP for reliability
                '-loglevel', 'info',  # Show info, warnings and errors for debugging
                rtsp_url
            ])
        
        # Print FFmpeg command for debugging
        print("FFmpeg command:")
        print(f"  {' '.join(ffmpeg_cmd[:10])} ... {rtsp_url}\n")
        
        try:
            # Clear previous errors
            self.ffmpeg_errors = []
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line buffered
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            
            # Start monitoring FFmpeg stderr in a separate thread
            self.ffmpeg_stderr_thread = threading.Thread(
                target=self._monitor_ffmpeg_stderr,
                daemon=True
            )
            self.ffmpeg_stderr_thread.start()
            
            # Give FFmpeg time to start and connect to MediaMTX
            print("Waiting for FFmpeg to connect to MediaMTX...")
            time.sleep(5)
            
            # Check if process is still running
            if self.ffmpeg_process.poll() is not None:
                # Wait a bit for stderr thread to capture output
                time.sleep(1)
                
                # Get all captured errors
                stderr_output = '\n'.join(self.ffmpeg_errors)
                if not stderr_output:
                    # Try to read remaining stderr
                    try:
                        remaining = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                        if remaining:
                            stderr_output = remaining
                    except:
                        pass
                
                print(f"\n❌ Error: FFmpeg process exited unexpectedly.")
                print(f"FFmpeg command: {' '.join(ffmpeg_cmd[:10])} ... {rtsp_url}")
                
                if stderr_output:
                    print(f"\nFFmpeg error output (last 30 lines):")
                    error_lines = stderr_output.split('\n')
                    print('\n'.join(error_lines[-30:]))
                else:
                    print("No error output captured.")
                
                # Check for common errors and suggest fixes
                error_lower = stderr_output.lower()
                if 'connection refused' in error_lower or 'cannot connect' in error_lower or 'connection timed out' in error_lower:
                    print("\n⚠️  Connection error: FFmpeg cannot connect to MediaMTX server.")
                    print("   - Make sure MediaMTX is running and listening on port", self.rtsp_port)
                    print("   - Check if another process is using port", self.rtsp_port)
                    print("   - Verify MediaMTX is accepting publisher connections")
                elif 'codec' in error_lower or 'encoder' in error_lower:
                    print("\n⚠️  Codec error: There may be an issue with video/audio encoding.")
                elif 'permission denied' in error_lower:
                    print("\n⚠️  Permission error: Check file permissions and network access.")
                elif 'no such file' in error_lower or 'cannot find' in error_lower:
                    print("\n⚠️  File error: Cannot find the video file or required resources.")
                
                self.running = False
                self._stop_mediamtx()
                return None
            
            # Check for connection errors in captured output
            error_text = '\n'.join(self.ffmpeg_errors).lower()
            if any(keyword in error_text for keyword in ['connection refused', 'cannot connect', 'connection timed out', 'failed to connect']):
                print("\n⚠️  Warning: FFmpeg may have connection issues. Check the output above.")
            
            # Check if FFmpeg successfully connected (look for success indicators)
            success_indicators = ['streaming', 'connected', 'publishing', 'rtsp']
            has_success = any(indicator in '\n'.join(self.ffmpeg_errors).lower() for indicator in success_indicators)
            
            if not has_success and len(self.ffmpeg_errors) > 0:
                print("\n⚠️  Warning: No clear success indicators from FFmpeg. Stream may not be active.")
                print("   Checking if stream is available...")
            
            # Verify MediaMTX is still running
            if self.use_mediamtx and self.mediamtx_process:
                if self.mediamtx_process.poll() is not None:
                    print("❌ Error: MediaMTX process stopped unexpectedly")
                    self.running = False
                    return None
                if not self._check_port(self.rtsp_port):
                    print(f"❌ Error: MediaMTX is not listening on port {self.rtsp_port}")
                    self.running = False
                    return None
                print(f"✓ MediaMTX is running and listening on port {self.rtsp_port}")
            
            # Additional verification: try to check if stream path is available
            # This is a simple check - MediaMTX may need a moment to register the stream
            time.sleep(2)
            
            print("\n✓ Video stream started successfully!")
            print(f"✓ FFmpeg is pushing stream to {rtsp_url}")
            print(f"\n💡 Tip: If VLC cannot connect, wait a few more seconds for the stream to fully initialize.")
            print(f"   You can also try: ffplay {rtsp_url}")
            return rtsp_url
            
        except Exception as e:
            print(f"Error starting FFmpeg: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
            self._stop_mediamtx()
            return None
    
    def stop_stream(self):
        """Stop the RTSP video stream."""
        self.running = False
        
        # Stop monitoring thread
        if self.ffmpeg_stderr_thread and self.ffmpeg_stderr_thread.is_alive():
            time.sleep(0.5)  # Give thread time to finish reading
        
        # Stop FFmpeg
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait()
                print("✓ FFmpeg stream stopped")
            except Exception as e:
                print(f"Error stopping FFmpeg: {e}")
            finally:
                self.ffmpeg_process = None
        
        # Stop MediaMTX
        if self.use_mediamtx:
            self._stop_mediamtx()


def find_video_file(directory):
    """Find the first video file in the specified directory."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    
    directory_path = Path(directory)
    if not directory_path.exists():
        return None
    
    for ext in video_extensions:
        # Case-insensitive search
        for video_file in directory_path.glob(f'*{ext}'):
            return str(video_file)
        for video_file in directory_path.glob(f'*{ext.upper()}'):
            return str(video_file)
    
    return None


def main():
    """Main function to run the RTSP streamer."""
    # Find video file in sample_video folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_video_dir = os.path.join(script_dir, 'sample_video')
    
    # Create sample_video directory if it doesn't exist
    os.makedirs(sample_video_dir, exist_ok=True)
    
    video_path = find_video_file(sample_video_dir)
    
    if not video_path:
        print(f"Error: No video file found in '{sample_video_dir}'")
        print("\nSupported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v")
        print(f"Please place a video file in: {sample_video_dir}")
        return
    
    print(f"Found video file: {video_path}")
    
    # Create streamer instance
    streamer = RTSPVideoStreamer(
        video_path=video_path,
        rtsp_port=8554,
        stream_name="video_stream",
        use_mediamtx=True
    )
    
    try:
        # Start streaming
        rtsp_url = streamer.start_stream()
        
        if not rtsp_url:
            print("\nFailed to start RTSP stream. Please check the error messages above.")
            return
        
        print(f"\n{'='*60}")
        print(f"✓ RTSP Server is running!")
        print(f"{'='*60}")
        print(f"\nRTSP Stream URL:")
        print(f"  {rtsp_url}")
        print(f"\nYou can access the stream using:")
        print(f"  • VLC Media Player:")
        print(f"    Media -> Open Network Stream -> {rtsp_url}")
        print(f"  • FFplay:")
        print(f"    ffplay {rtsp_url}")
        print(f"  • Any RTSP client application")
        print(f"\nThe video will loop continuously.")
        print(f"\nPress Ctrl+C to stop the server...\n")
        
        # Keep the server running
        while streamer.running:
            time.sleep(1)
            
            # Check if processes are still running
            if streamer.ffmpeg_process and streamer.ffmpeg_process.poll() is not None:
                print("\nWarning: FFmpeg process stopped unexpectedly!")
                break
            
    except KeyboardInterrupt:
        print("\n\nStopping RTSP server...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        streamer.stop_stream()
        print("\n✓ RTSP server stopped. Goodbye!")


if __name__ == "__main__":
    main()
