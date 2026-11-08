"""
Video Utilities for Agentic Video Understanding Framework
========================================================

This module contains all video-related utility functions:
- Video metadata loading from JSON dataset files
- File hashing for integrity checks
- Video file path resolution (compressed fallbacks)
- MIME type detection

Note: Video metadata (duration, fps) is loaded from JSON files rather than
extracting from video files directly. This is faster and avoids codec dependencies.
"""

from typing import Dict, Any, Optional, List, Tuple
import hashlib
import os
import json
import subprocess
import shutil
from pathlib import Path
import math


# ======================================================
# Video Metadata Loading
# ======================================================

# Global cache for video metadata (no default path; call set_metadata_source(json_path) before use)
_VIDEO_METADATA_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def load_video_metadata_from_json(json_path: str) -> Dict[str, Dict[str, Any]]:
    """Load video metadata from JSON file.
    
    Args:
        json_path: Path to JSON file containing video metadata
        
    Returns:
        Dictionary mapping video paths to their metadata
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Build index by video path
    metadata_by_path = {}
    for sample in data:
        video_path = sample.get("path", "")
        if video_path:
            metadata_by_path[video_path] = {
                "duration_sec": sample.get("duration", 0.0),
                "video_path": video_path,
                "video_id": sample.get("video_id", ""),
                "sample": sample,  # Keep full sample for additional info
            }
    
    return metadata_by_path


def set_metadata_source(json_path: str = None) -> None:
    """Set the global metadata source from a JSON file.
    
    This loads video metadata into a cache for fast lookups.
    Call this once at the start of your application.
    
    Args:
        json_path: Path to JSON file with video metadata (required)
    """
    global _VIDEO_METADATA_CACHE
    
    if json_path is None:
        raise ValueError("json_path is required for set_metadata_source(); pass the path to your annotation/metadata JSON")
    
    _VIDEO_METADATA_CACHE = load_video_metadata_from_json(json_path)
    print(f"✅ Loaded metadata for {len(_VIDEO_METADATA_CACHE)} videos from {json_path}")


def get_metadata_cache() -> Dict[str, Dict[str, Any]]:
    """Get the global metadata cache, loading default if not initialized.
    
    Returns:
        Video metadata cache
    """
    global _VIDEO_METADATA_CACHE
    
    if _VIDEO_METADATA_CACHE is None:
        _VIDEO_METADATA_CACHE = {}
    
    return _VIDEO_METADATA_CACHE


class VideoMetadataExtractor:
    """Extracts metadata for video files from pre-loaded JSON cache.
    
    This class provides metadata without needing to open video files.
    Metadata is loaded from JSON files containing video information.
    
    Attributes:
        video_path: Path to the video file
        duration: Duration in seconds (from JSON)
    """
    
    def __init__(self, video_path: str, metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize with video path and optional metadata cache.
        
        Args:
            video_path: Path to video file
            metadata_cache: Optional metadata cache (uses global if None)
            
        Raises:
            ValueError: If video not found in metadata cache
        """
        self.video_path = video_path
        
        # Use provided cache or global cache (auto-loads if needed)
        cache = metadata_cache if metadata_cache is not None else get_metadata_cache()
        
        # Look up metadata
        if video_path in cache:
            metadata = cache[video_path]
            self.duration = float(metadata.get("duration_sec", 0.0))
        else:
            # Fallback: try to find original (uncompressed) video path and use its duration
            # Strip compressed prefix/suffix patterns
            original_path = self._get_original_path(video_path)
            
            if original_path != video_path and original_path in cache:
                metadata = cache[original_path]
                self.duration = float(metadata.get("duration_sec", 0.0))
                print(f"ℹ️  Using duration from original video: {original_path} (duration={self.duration:.1f}s)")
            else:
                # Final fallback: video not in cache
                print(f"⚠️  Video not found in metadata cache: {video_path}")
                print(f"   Using default value (duration=0)")
                self.duration = 0.0
    
    def _get_original_path(self, compressed_path: str) -> str:
        """Extract original path from compressed video path.
        
        Args:
            compressed_path: Path to compressed video
            
        Returns:
            Original video path without compression markers
        """
        # Remove common compression prefixes and suffixes
        path = compressed_path
        
        # Remove "compressed_" prefix
        if "/compressed_" in path:
            path = path.replace("/compressed_", "/")
        elif "compressed_" in path:
            path = path.replace("compressed_", "")
        
        # Remove "_compressed" suffix
        if "_compressed" in path:
            path = path.replace("_compressed", "")
        
        # Remove ".compressed" suffix
        if ".compressed" in path:
            path = path.replace(".compressed", "")
        
        # Remove "_comp" suffix
        if "_comp" in path:
            path = path.replace("_comp", "")
        
        # Remove compressed directory structure
        if "/compressed/" in path:
            path = path.replace("/compressed/", "/")
        elif "/_compressed/" in path:
            path = path.replace("/_compressed/", "/")
        
        return path
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return video metadata as a dictionary.
        
        Returns:
            Dictionary with duration_sec and video_path
        """
        return {
            "duration_sec": self.duration,
            "video_path": self.video_path
        }
    
    def __repr__(self) -> str:
        return f"VideoMetadataExtractor(path='{self.video_path}', duration={self.duration:.1f}s)"


# ======================================================
# File Utilities
# ======================================================

def sha256_file(file_path: Path) -> str:
    """Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ======================================================
# Video File Path Resolution
# ======================================================

def get_mime_type(video_path: str) -> str:
    """Detect MIME type from video file extension.
    
    Args:
        video_path: Path to video file
        
    Returns:
        MIME type string (e.g., 'video/mp4')
    """
    file_ext = Path(video_path).suffix.lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
        ".m4v": "video/x-m4v",
    }
    return mime_types.get(file_ext, "video/mp4")


def find_compressed_video_fallback(original_path: str) -> Optional[str]:
    """Search for compressed version of video file.
    
    Looks for compressed versions in multiple locations:
    - Same directory with 'compressed_' prefix
    - Same directory with '_compressed' suffix
    - 'compressed' subdirectory
    - '_compressed' subdirectory
    
    Args:
        original_path: Path to original video file
        
    Returns:
        Path to compressed video if found, None otherwise
    """
    if not original_path:
        return None
    
    original_path_obj = Path(original_path)
    
    # Define search patterns
    compressed_patterns = [
        # Same directory with prefix/suffix
        original_path_obj.parent / f"compressed_{original_path_obj.name}",
        original_path_obj.parent / f"{original_path_obj.stem}_compressed{original_path_obj.suffix}",
        original_path_obj.parent / f"{original_path_obj.stem}.compressed{original_path_obj.suffix}",
        original_path_obj.parent / f"{original_path_obj.stem}_comp{original_path_obj.suffix}",
        
        # In 'compressed' subdirectory
        original_path_obj.parent / "compressed" / original_path_obj.name,
        original_path_obj.parent / "compressed" / f"{original_path_obj.stem}_compressed{original_path_obj.suffix}",
        original_path_obj.parent / "compressed" / f"{original_path_obj.stem}_comp{original_path_obj.suffix}",
        
        # In '_compressed' subdirectory
        original_path_obj.parent / "_compressed" / original_path_obj.name,
        original_path_obj.parent / "_compressed" / f"{original_path_obj.stem}_compressed{original_path_obj.suffix}",
    ]
    
    # Check each pattern
    for pattern in compressed_patterns:
        if os.path.exists(pattern):
            return str(pattern)
    
    return None


def get_video_path(sample: Dict[str, Any], prefer_compressed: bool = True, debug: bool = False) -> str:
    """Get video path, preferring compressed version if available.
    
    Args:
        sample: Dictionary with 'path' or 'video_path' key
        prefer_compressed: Whether to prefer compressed version if available
        debug: Whether to print debug messages
        
    Returns:
        Path to video file (compressed if available, original otherwise)
    """
    original_path = sample.get("path", sample.get("video_path", ""))
    
    if not prefer_compressed:
        return original_path
    
    compressed_path = find_compressed_video_fallback(original_path)
    
    if compressed_path:
        if debug:
            print(f"✅ Using compressed video: {compressed_path}")
        return compressed_path
    
    if debug:
        print(f"ℹ️  No compressed video found, using original: {original_path}")
    
    return original_path


# ======================================================
# Video Information Display
# ======================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1m 23s" or "45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def get_video_info(video_path: str, include_hash: bool = False) -> Dict[str, Any]:
    """Get comprehensive video information.
    
    Args:
        video_path: Path to video file
        include_hash: Whether to compute SHA256 hash (slower)
        
    Returns:
        Dictionary with video metadata and file info
    """
    extractor = VideoMetadataExtractor(video_path)
    path_obj = Path(video_path)
    
    info = {
        "path": video_path,
        "filename": path_obj.name,
        "exists": path_obj.exists(),
        "mime_type": get_mime_type(video_path),
        "duration_sec": extractor.duration,
        "duration_formatted": format_duration(extractor.duration),
    }
    
    if path_obj.exists():
        info["size_mb"] = path_obj.stat().st_size / (1024 * 1024)
        
        if include_hash:
            info["sha256"] = sha256_file(path_obj)
    else:
        info["size_mb"] = 0.0
    
    return info


def print_video_info(video_path: str, include_hash: bool = False) -> None:
    """Print formatted video information to console.
    
    Args:
        video_path: Path to video file
        include_hash: Whether to compute and display SHA256 hash
    """
    try:
        info = get_video_info(video_path, include_hash)
        
        print(f"\n📹 Video Information")
        print(f"{'─' * 50}")
        print(f"File:     {info['filename']}")
        print(f"Path:     {info['path']}")
        print(f"Exists:   {'✅ Yes' if info['exists'] else '❌ No'}")
        
        if info['exists']:
            print(f"Size:     {info['size_mb']:.2f} MB")
        
        print(f"Type:     {info['mime_type']}")
        print(f"Duration: {info['duration_formatted']} ({info['duration_sec']:.2f}s)")
        
        if include_hash and 'sha256' in info:
            print(f"SHA256:   {info['sha256']}")
        
        print(f"{'─' * 50}\n")
        
    except Exception as e:
        print(f"❌ Error reading video info: {e}")


# ======================================================
# Validation
# ======================================================

def validate_video_file(video_path: str) -> tuple[bool, Optional[str]]:
    """Validate that a video file exists and has metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    path_obj = Path(video_path)
    
    # Check file exists
    if not path_obj.exists():
        return False, f"File does not exist: {video_path}"
    
    # Check it's a file
    if not path_obj.is_file():
        return False, f"Path is not a file: {video_path}"
    
    # Check file size
    size = path_obj.stat().st_size
    if size == 0:
        return False, f"File is empty: {video_path}"
    
    # Check metadata available
    try:
        extractor = VideoMetadataExtractor(video_path)
        if extractor.duration <= 0:
            return False, f"Invalid or missing video duration in metadata"
        return True, None
    except Exception as e:
        return False, f"Cannot get video metadata: {e}"


def normalize_spatial_resolution(load_mode: str, spatial_token_rate: str) -> str:
    """Normalize spatial token rate based on load mode.
    
    Rule: If load_mode is "uniform", spatial_token_rate should be "low".
    This ensures efficient processing for uniform video scans.
    
    Args:
        load_mode: Video load mode ("uniform" or "region")
        spatial_token_rate: Current spatial token rate ("low", "medium", or "high")
        
    Returns:
        Normalized spatial token rate (guaranteed to be "low" if load_mode is "uniform")
    """
    load_mode_lower = str(load_mode).strip().lower()
    spatial_rate_lower = str(spatial_token_rate).strip().lower()
    
    # Enforce rule: if uniform, keep spatial resolution as low
    if load_mode_lower == "uniform":
        return "low"
    
    # For region mode, return the provided rate (or default to "low" if invalid)
    valid_rates = {"low", "medium", "high"}
    if spatial_rate_lower in valid_rates:
        return spatial_rate_lower
    
    # Default fallback
    return "low"


# ======================================================
# Rounding Helpers (full-second intervals)
# ======================================================

def round_interval_full_seconds(start: float, end: float, duration: Optional[float] = None) -> Optional[Tuple[int, int]]:
    """Round an interval to full seconds using floor for start and ceil for end.
    
    Args:
        start: Start timestamp in seconds
        end: End timestamp in seconds
        duration: Optional video duration to clamp within [0, duration]
    
    Returns:
        Tuple (start_int, end_int) if valid after rounding and clamping; otherwise None
    """
    if start is None or end is None:
        return None
    s = int(math.floor(float(start)))
    e = int(math.ceil(float(end)))
    if duration is not None and duration > 0:
        s = max(0, s)
        e = min(int(math.ceil(duration)), e)
    if e <= s:
        return None
    return (s, e)


def round_intervals_full_seconds(ranges: List[Tuple[float, float]], duration: Optional[float] = None) -> List[Tuple[int, int]]:
    """Round multiple intervals to full seconds with floor/ceil and clamp to duration.
    
    Deduplicates identical intervals and drops invalid ones.
    """
    seen = set()
    out: List[Tuple[int, int]] = []
    for start, end in ranges:
        rounded = round_interval_full_seconds(start, end, duration)
        if rounded is None:
            continue
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(rounded)
    return out

# ======================================================
# Video Clipping Utilities
# ======================================================

def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available for video clipping.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def ensure_temp_clips_dir(video_path: str, debug: bool = False) -> str:
    """Ensure the temp_clips directory exists for a video file.
    
    Args:
        video_path: Path to the video file
        debug: Whether to print debug messages
        
    Returns:
        Path to the temp_clips directory
    """
    temp_dir = os.path.join(os.path.dirname(video_path), "temp_clips")
    try:
        os.makedirs(temp_dir, exist_ok=True)
        # Verify directory was created and is writable
        if not os.path.exists(temp_dir):
            if debug:
                print(f"❌ Failed to create temp_clips directory: {temp_dir}")
            raise OSError(f"Failed to create temp_clips directory: {temp_dir}")
        if not os.access(temp_dir, os.W_OK):
            if debug:
                print(f"❌ temp_clips directory is not writable: {temp_dir}")
            raise OSError(f"temp_clips directory is not writable: {temp_dir}")
        if debug:
            print(f"✅ Ensured temp_clips directory exists: {temp_dir}")
    except OSError as e:
        if debug:
            print(f"❌ Error ensuring temp_clips directory {temp_dir}: {e}")
        raise
    return temp_dir


def create_video_clip(video_path: str, start_time: float, end_time: float, 
                     clip_name: str = None, temp_dir: str = None, debug: bool = True) -> Optional[str]:
    """Create a video clip using ffmpeg.
    
    Args:
        video_path: Path to the original video file
        start_time: Start time in seconds
        end_time: End time in seconds
        clip_name: Optional name for the clip file
        temp_dir: Directory to store the clip (creates temp_clips subdirectory if None)
        debug: Whether to print debug messages
        
    Returns:
        Path to the created clip file, or None if creation failed
    """
    if not check_ffmpeg_available():
        if debug:
            print("⚠️  ffmpeg not available, falling back to metadata-based approach")
        return None
    
    if not os.path.exists(video_path):
        if debug:
            print(f"❌ Video file not found: {video_path}")
        return None
    
    if not temp_dir:
        try:
            temp_dir = ensure_temp_clips_dir(video_path, debug=debug)
        except OSError:
            return None
    else:
        # Even if temp_dir is provided, ensure it exists
        try:
            os.makedirs(temp_dir, exist_ok=True)
            if not os.path.exists(temp_dir):
                if debug:
                    print(f"❌ Failed to create temp_clips directory: {temp_dir}")
                return None
            if not os.access(temp_dir, os.W_OK):
                if debug:
                    print(f"❌ temp_clips directory is not writable: {temp_dir}")
                return None
        except OSError as e:
            if debug:
                print(f"❌ Error creating temp_clips directory {temp_dir}: {e}")
            return None
    
    # Generate clip filename
    if clip_name is None:
        video_name = Path(video_path).stem
        clip_name = f"{video_name}_clip_{start_time:.1f}s_{end_time:.1f}s.mp4"
    else:
        # Ensure clip_name has a file extension
        if not clip_name.endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm')):
            # Add .mp4 extension if no extension is present
            clip_name = f"{clip_name}.mp4"
    
    clip_path = os.path.join(temp_dir, clip_name)
    
    # Skip if clip already exists
    if os.path.exists(clip_path):
        if debug:
            print(f"ℹ️  Using existing clip: {clip_path}")
        return clip_path
    
    try:
        # Create ffmpeg command
        # Format time as HH:MM:SS.mmm
        start_hours = int(start_time // 3600)
        start_minutes = int((start_time % 3600) // 60)
        start_seconds = int(start_time % 60)
        start_milliseconds = int((start_time % 1) * 1000)
        start_str = f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}.{start_milliseconds:03d}"
        
        end_hours = int(end_time // 3600)
        end_minutes = int((end_time % 3600) // 60)
        end_seconds = int(end_time % 60)
        end_milliseconds = int((end_time % 1) * 1000)
        end_str = f"{end_hours:02d}:{end_minutes:02d}:{end_seconds:02d}.{end_milliseconds:03d}"
        
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'error',
            '-ss', start_str,
            '-to', end_str,
            '-i', video_path,
            '-c', 'copy',  # Stream copy for speed (no re-encoding)
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite if exists
            clip_path
        ]
        
        if debug:
            print(f"🎬 Creating video clip: {start_time:.1f}s - {end_time:.1f}s")
            print(f"   Output: {clip_path}")
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(clip_path):
            clip_size = os.path.getsize(clip_path) / (1024 * 1024)  # MB
            if debug:
                print(f"✅ Created video clip: {clip_path} ({clip_size:.2f} MB)")
            return clip_path
        else:
            if debug:
                print(f"❌ Failed to create video clip: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        if debug:
            print(f"⏰ ffmpeg timeout while creating clip {start_time:.1f}s - {end_time:.1f}s")
        return None
    except Exception as e:
        if debug:
            print(f"❌ Error creating video clip: {e}")
        return None


def cleanup_video_clips(video_path: str = None, clip_paths: List[str] = None, debug: bool = True) -> None:
    """Clean up specific video clips that were created.
    
    Args:
        video_path: Path to the original video file (legacy parameter, for backward compatibility)
        clip_paths: List of specific clip file paths to delete. If provided, only these will be deleted.
        debug: Whether to print debug messages
    """
    # If specific clip paths are provided, delete only those
    if clip_paths:
        files_removed = 0
        for clip_path in clip_paths:
            if clip_path and os.path.exists(clip_path):
                try:
                    if os.path.isfile(clip_path):
                        os.remove(clip_path)
                        files_removed += 1
                        if debug:
                            print(f"🗑️  Deleted clip: {os.path.basename(clip_path)}")
                except Exception as e:
                    if debug:
                        print(f"⚠️  Could not remove clip file {clip_path}: {e}")
        
        if debug and files_removed > 0:
            print(f"🧹 Cleaned up {files_removed} clip(s) (directory kept)")
        elif debug and len(clip_paths) > 0:
            print(f"🧹 No clips to clean up (already removed or not found)")
        return
    
    # Legacy behavior: if only video_path is provided, clean entire directory
    # (kept for backward compatibility but not recommended)
    if video_path:
        temp_clips_dir = os.path.join(os.path.dirname(video_path), "temp_clips")
        
        if not os.path.exists(temp_clips_dir):
            # Directory doesn't exist, nothing to clean
            return
        
        try:
            # Remove all files in temp_clips directory (but keep the directory itself)
            if os.path.isdir(temp_clips_dir):
                files_removed = 0
                for filename in os.listdir(temp_clips_dir):
                    file_path = os.path.join(temp_clips_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            files_removed += 1
                        elif os.path.isdir(file_path):
                            # Also remove subdirectories if any
                            shutil.rmtree(file_path)
                            files_removed += 1
                    except Exception as e:
                        if debug:
                            print(f"⚠️  Could not remove clip file/dir {file_path}: {e}")
                
                # Keep the directory itself, just report files removed
                if debug and files_removed > 0:
                    print(f"🧹 Cleaned up {files_removed} clip(s) from temp_clips directory (directory kept)")
                elif debug:
                    print(f"🧹 temp_clips directory is empty (directory kept)")
        except Exception as e:
            if debug:
                print(f"⚠️  Error during clip cleanup: {e}")
