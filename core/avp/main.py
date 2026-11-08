"""
Active Video Perception (AVP) — Plan-Observe-Reflect implementation
===================================================================

Single-action plan -> observe -> reflect loop for long-video understanding,
following Algorithm 1 of the paper. Given a query Q and a video V, AVP runs:

1. Plan (one observation action per round):
   - The Planner drafts one observation configuration this round:
     - load_mode: "uniform" (full video) or "region" (specific time span[s])
     - fps: sampling rate; spatial_token_rate: "low" or "medium"
   - The sub_query equals the original query (including MCQ options if any).

2. Observe:
   - The Observer executes the planned observation and extracts query-relevant
     evidence as timestamped intervals with descriptions (the structured
     evidence list E).

3. Reflect:
   - The Reflector (an MLLM) verifies the cumulative evidence against the query
     and jointly emits (confidence in [0,1], justification, reasoning).
   - If `confidence >= tau_conf` (paper Sec 4.3 default 0.7) AND the reflector's
     own `sufficient` flag is true, EXTRACTANSWER pulls the answer from the
     justification; otherwise the justification is fed to the next REPLAN(Q,H,J).
   - On the final round, FORCEANSWER synthesizes from all accumulated evidence.

Videos are passed to Gemini directly: small clips inline, larger files via the
Gemini File API (avoids the 2 GiB inline cap and 1 GiB request-payload cap).

CLI entry point:
  python -m avp.eval_dataset --ann ANNOTATION.json --out OUT --config CONFIG \
                             --max-turns 3 --confidence-threshold 0.7

See README.md and the scripts/ directory for end-to-end evaluation wrappers.
"""
from __future__ import annotations

import dataclasses
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import pathlib
import re
import time
import hashlib

# Gemini / Vertex AI
from google import genai
from google.genai import types
from google.genai.types import Part, Blob, VideoMetadata, FileData
import os
import time
from pathlib import Path


# Files <= this many bytes go inline (fast, no upload roundtrip); larger files
# are uploaded via the Gemini File API. The inline payload caps on AI Studio
# are ~20 MB practical / 1 GiB request / 2 GiB inline_data; 20 MB is a safe
# headroom that keeps small region-clips fast while routing big videos through
# the File API.
INLINE_VIDEO_SIZE_LIMIT_BYTES = 20 * 1024 * 1024


def _generate_content_for_stage(
    client: Any,
    *,
    model: str,
    contents: Any,
    stage: str,
    config: Any = None,
) -> Any:
    """Expose an optional stage hint to adapters without changing providers."""

    set_stage = getattr(client, "set_generation_stage", None)
    if callable(set_stage):
        set_stage(stage)
    try:
        if config is not None:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        return client.models.generate_content(model=model, contents=contents)
    finally:
        if callable(set_stage):
            set_stage(None)


def _probe_video_duration_seconds(path: str) -> Optional[float]:
    """Last-resort duration probe via OpenCV (we already depend on it).

    Returns None on failure; callers must handle that.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps > 0 and n_frames > 0:
            return float(n_frames / fps)
    except Exception:
        pass
    return None

# Import prompt management
from .prompt import (
    PromptManager,
    parse_json_response,
    validate_against_schema,
    PLAN_SCHEMA,
    EVIDENCE_SCHEMA,
    FINAL_ANSWER_SCHEMA,
    MCQ_SCHEMA,
    REFLECTOR_SCHEMA,
)
from .mcq_candidates import (
    apply_hypothesis_verifier_results,
    initialize_open_option_candidates,
)

# Import video utilities
from .video_utils import (
    VideoMetadataExtractor,
    get_mime_type,
    find_compressed_video_fallback,
    round_intervals_full_seconds,
)

# -----------------------------
# Optional deps
# -----------------------------
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x


# Max frame settings moved to config - use AVPConfig.max_frame_* instead




# ======================================================
# Contracts & Schemas
# ======================================================
class SpatialTokenRate(str, Enum):
    low = "low"
    medium = "medium"


@dataclass
class WatchConfig:
    """Configuration for video observation, specifying region and sampling granularity.
    
    Contains:
    - Region: load_mode ("uniform" for full video, "region" for temporal spans) and regions (list of [start, end] tuples)
    - Sampling granularity: fps (temporal sampling rate) and spatial_token_rate (spatial resolution)
    """
    load_mode: str                      # "uniform" | "region" - specifies whether to scan full video or specific regions
    fps: float                          # Temporal sampling rate (frames per second)
    spatial_token_rate: SpatialTokenRate  # Spatial resolution ("low" or "medium")
    regions: List[Tuple[float, float]] = field(default_factory=list)  # Temporal spans: [(start_sec, end_sec), ...]


@dataclass
class PlanSpec:
    """Single observation action plan.
    
    Specifies one observation with:
    - query: The user's question
    - watch: Observation config (load_mode, fps, spatial_token_rate, regions)
    - description: Goal/reasoning objective for this observation
    """
    plan_version: str
    query: str
    watch: Optional[WatchConfig]  # ASR acquisitions do not have a visual cadence.
    description: str = ""  # Goal/reasoning objective
    completion_criteria: str = ""
    need_id: Optional[str] = None
    need_revision: Optional[int] = None
    final_answer: Optional[str] = None
    complete: bool = False
    acquisition: Optional[Dict[str, Any]] = None


@dataclass
class Evidence:
    """Evidence gathered from one observation round.
    
    Contains:
    - detailed_response: Analysis and observations
    - key_evidence: List of [timestamp_start, timestamp_end, description] intervals
    - reasoning: Explanation of findings
    """
    detailed_response: str = ""  # Detailed analysis and observations
    key_evidence: List[Dict[str, Any]] = field(default_factory=list)  # Timestamp intervals with descriptions
    reasoning: str = ""  # Explanation of findings
    frames_used: List[Dict[str, Any]] = field(default_factory=list)  # Video segments analyzed
    model_call: Dict[str, Any] = field(default_factory=dict)  # API call metadata
    timestamp: str = ""  # When evidence was collected
    round_id: int = 0  # Which round this evidence is from
    
    @property
    def step_id(self) -> str:
        """Compatibility property: returns round_id as string."""
        return str(self.round_id)


def _convert_region_relative_key_evidence(
    key_evidence: List[Dict[str, Any]],
    frames_used: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map model-authored segment-local times onto source-video time.

    The model never receives source offsets for bounded observations. A single
    segment is unambiguous; multiple segments require one-based ``segment_id``.
    Invalid or ambiguous timed items are ignored instead of poisoning valid
    statements from the same response.
    """

    units: List[Tuple[float, float]] = []
    for frame in frames_used:
        if not isinstance(frame, dict):
            continue
        try:
            start = float(frame["start"])
            end = float(frame["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            units.append((start, end))
    if not units:
        return []

    converted: List[Dict[str, Any]] = []
    for raw_item in key_evidence:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        try:
            relative_start = float(item["timestamp_start"])
            relative_end = float(item["timestamp_end"])
        except (KeyError, TypeError, ValueError):
            # A timestamp-free statement can inherit a single submitted scope;
            # it is ambiguous when several independent segments were attached.
            if len(units) == 1:
                converted.append(item)
            continue

        if len(units) == 1:
            unit_index = 0
        else:
            try:
                unit_index = int(item.get("segment_id")) - 1
            except (TypeError, ValueError):
                continue
            if unit_index < 0 or unit_index >= len(units):
                continue

        source_start, source_end = units[unit_index]
        duration = source_end - source_start
        if (
            relative_start < 0
            or relative_end <= relative_start
            or relative_start >= duration
            or relative_end > math.ceil(duration) + 1e-6
        ):
            continue

        absolute_start = source_start + relative_start
        absolute_end = min(source_end, source_start + relative_end)
        if absolute_end <= absolute_start:
            continue
        item.pop("segment_id", None)
        item["timestamp_start"] = absolute_start
        item["timestamp_end"] = absolute_end
        converted.append(item)
    return converted


@dataclass
class Blackboard:
    video_path: str
    duration_sec: Optional[float] = None
    evidences: List[Evidence] = field(default_factory=list)  # List of evidence from each round
    meta: Dict[str, Any] = field(default_factory=dict)
    query_confidence: Optional[float] = None  # Query-level confidence (0.0 to 1.0) for answering the query

    def add_evidence(self, ev: Evidence) -> None:
        """Add evidence from current round."""
        self.evidences.append(ev)

    def get_evidence_list(self) -> List[Evidence]:
        """Get evidence as a list, sorted temporally by earliest timestamp.
        
        Evidence is ordered by the earliest timestamp_start in each evidence's key_evidence.
        If evidence has no timestamps, it's placed at the end (sorted by round_id as tiebreaker).
        
        Returns:
            List of Evidence objects sorted temporally (by timestamp_start)
        """
        def get_earliest_timestamp(ev: Evidence) -> float:
            """Get the earliest timestamp_start from evidence's key_evidence."""
            earliest = None
            for kev in ev.key_evidence:
                if isinstance(kev, dict):
                    ts_start = kev.get("timestamp_start")
                    if ts_start is not None:
                        if earliest is None or ts_start < earliest:
                            earliest = ts_start
            # If no timestamps found, return a large number to place at end
            if earliest is None:
                return 999999.0 + ev.round_id  # Place at end, sorted by round_id
            return earliest
        
        evidence_list = list(self.evidences)
        # Sort by earliest timestamp, then by round_id as tiebreaker
        evidence_list.sort(key=lambda ev: (get_earliest_timestamp(ev), ev.round_id))
        return evidence_list

    def summary_text(self) -> str:
        """Generate a condensed text summary of all accumulated evidence.
        
        Returns a formatted string with evidence from each round, including:
        - The detailed response/analysis
        - Key evidence with timestamps and descriptions
        """
        lines = []
        for idx, e in enumerate(self.evidences, 1):
            # Start with the detailed response
            main_text = e.detailed_response
            
            # Format key evidence with descriptions
            evidence_lines = []
            for ev in e.key_evidence:
                if isinstance(ev, dict):
                    ts_start = ev.get("timestamp_start")
                    ts_end = ev.get("timestamp_end")
                    desc = ev.get("description", "")
                    if ts_start is not None and ts_end is not None:
                        if desc:
                            evidence_lines.append(f"  • {ts_start:.1f}s - {ts_end:.1f}s: {desc}")
                        else:
                            evidence_lines.append(f"  • {ts_start:.1f}s - {ts_end:.1f}s")
            
            # Build the full entry
            entry = f"[Round {idx}] {main_text}"
            if evidence_lines:
                entry += "\nKey observations:\n" + "\n".join(evidence_lines)
            
            lines.append(entry)
        
        # Add query-level confidence if available
        if self.query_confidence is not None:
            lines.append(f"\n[Overall Query Confidence: {self.query_confidence:.2f}]")
        
        return "\n\n".join(lines)


# ======================================================
# Persistence Layer
# ======================================================
class Store:
    def __init__(self, run_dir: str):
        self.root = pathlib.Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "evidence").mkdir(exist_ok=True)

    # Files
    @property
    def plan_initial(self) -> pathlib.Path:
        return self.root / "plan.initial.json"

    def plan_updated(self, k: int) -> pathlib.Path:
        return self.root / f"plan.updated.{k}.json"

    @property
    def history(self) -> pathlib.Path:
        return self.root / "history.jsonl"

    def evidence_dir(self, round_id: int) -> pathlib.Path:
        """Get evidence directory for a round."""
        p = self.root / "evidence" / f"round_{round_id}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def evidence_json(self, round_id: int) -> pathlib.Path:
        """Get evidence JSON path for a round."""
        return self.evidence_dir(round_id) / "evidence.json"

    @property
    def meta(self) -> pathlib.Path:
        return self.root / "meta.json"

    @property
    def final_answer(self) -> pathlib.Path:
        return self.root / "final_answer.json"
    
    @property
    def conversation_history(self) -> pathlib.Path:
        return self.root / "conversation_history.json"

    # IO helpers
    def write_json(self, path: pathlib.Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

    def append_history(self, event: Dict[str, Any]) -> None:
        with open(self.history, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """Get interaction history as a structured list.
        
        This provides explicit access to interaction history for the Reflector.
        
        Returns:
            List of interaction history entries (dicts) from history.jsonl
        """
        if not self.history.exists():
            return []
        
        history = []
        with open(self.history, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        history.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return history
    
    def save_conversation_history(self, query: str, plan: PlanSpec, bb: Blackboard, final_answer: Dict[str, Any]) -> None:
        """Save complete conversation history including all interactions."""
        history = {
            "query": query,
            "plan": {
                "reasoning": plan.completion_criteria,
                "description": plan.description,
                "watch_config": {
                    "load_mode": plan.watch.load_mode,
                    "fps": plan.watch.fps,
                    "spatial_token_rate": plan.watch.spatial_token_rate.value if hasattr(plan.watch.spatial_token_rate, 'value') else str(plan.watch.spatial_token_rate),
                    "regions": plan.watch.regions
                } if plan.watch is not None else None,
                "acquisition": plan.acquisition,
            },
            "execution_history": [],
            "final_answer": final_answer,
            "summary": {
                "total_rounds": len(bb.evidences),
                "total_evidence_items": sum(len(ev.key_evidence) for ev in bb.evidences),
                "video_duration_sec": bb.duration_sec,
                "query_confidence": bb.query_confidence
            }
        }
        
        # Add execution results from each round
        for round_idx, evidence in enumerate(bb.evidences, 1):
            exec_entry = {
                "round_id": round_idx,
                "evidence": {
                    "detailed_response": evidence.detailed_response,
                    "reasoning": evidence.reasoning,
                    "key_evidence": evidence.key_evidence,
                    "frames_used": evidence.frames_used
                },
                "timestamp": evidence.timestamp
            }
            history["execution_history"].append(exec_entry)
        
        self.write_json(self.conversation_history, history)


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        plan_replan_model: Optional[str] = None,
        execute_model: Optional[str] = None,
        project: str = "my-project",
        location: str = "us-central1",
        api_key: Optional[str] = None,
        debug: bool = False,
        max_frame_low: int = 512,
        max_frame_medium: int = 128,
        max_frame_high: int = 128,
        confidence_threshold: float = 0.7,
    ):
        # Support separate models for plan/replan vs execute
        # If not specified, use the legacy 'model' parameter for both
        self.plan_replan_model = plan_replan_model if plan_replan_model is not None else model
        self.execute_model = execute_model if execute_model is not None else model
        # Keep legacy model for backward compatibility
        self.model = model
        self.project = project
        self.location = location
        self.api_key = api_key or ""
        self.client = None
        self.debug = debug
        self.max_frame_low = max_frame_low
        self.max_frame_medium = max_frame_medium
        self.max_frame_high = max_frame_high
        # Confidence threshold tau_conf for the reflector (paper Sec 4.3 uses 0.7)
        self.confidence_threshold = confidence_threshold
        # File-API cache: abs_path -> File object (avoid re-uploading the same
        # video across rounds / observations).
        self._uploaded_files: Dict[str, Any] = {}

    def initialize_client(self):
        """Initialize the Gemini client (API key or Vertex AI)."""
        try:
            if self.api_key:
                self.client = genai.Client(api_key=self.api_key)
                if self.debug:
                    print("✅ Initialized Gemini client (API key)")
            else:
                self.client = genai.Client(vertexai=True, project=self.project, location=self.location)
                if self.debug:
                    print(f"✅ Initialized Vertex AI client for project: {self.project} @ {self.location}")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini client: {e}")
            raise
    
    # ---------- File API ----------
    def cleanup_uploaded_files(self) -> None:
        """Delete every video uploaded via the File API on this client.

        AI Studio retains uploaded files for ~48h and enforces a per-project
        storage quota; on multi-hundred-sample evals (each video ~100s of MB)
        we must release them after each sample, not at the end of the run.
        Safe to call even if no files were uploaded.
        """
        if self.client is None or not self._uploaded_files:
            return
        for path, f in list(self._uploaded_files.items()):
            try:
                self.client.files.delete(name=f.name)
                if self.debug:
                    print(f"🗑️  Deleted uploaded file: {f.name}  ({path})")
            except Exception as e:
                if self.debug:
                    print(f"⚠️  Could not delete uploaded file {f.name}: {e}")
        self._uploaded_files.clear()

    def _get_or_upload_file(self, video_path: str, mime_type: str) -> Any:
        """Upload `video_path` via the Gemini File API (or return a cached File).

        Used for videos larger than INLINE_VIDEO_SIZE_LIMIT_BYTES to avoid the
        2 GiB inline-data cap and 1 GiB request-payload cap. Polls until the
        File reaches ACTIVE state. Cached per absolute path on the client so the
        same video isn't re-uploaded across rounds / observations.
        """
        if self.client is None:
            self.initialize_client()
        abs_path = os.path.abspath(video_path)
        cached = self._uploaded_files.get(abs_path)
        if cached is not None:
            # If a previous upload finished long ago, the File may have expired
            # (Gemini File API retains files for ~48h). Try to refresh state;
            # on any error, fall through and re-upload.
            try:
                refreshed = self.client.files.get(name=cached.name)
                if getattr(refreshed.state, "name", "") == "ACTIVE":
                    return refreshed
            except Exception:
                pass

        if self.debug:
            size_mb = os.path.getsize(abs_path) / (1024 * 1024)
            print(f"📤 Uploading via File API: {abs_path} ({size_mb:.1f} MB)")
        f = self.client.files.upload(file=abs_path, config={"mime_type": mime_type})
        # Poll until ACTIVE (small videos take seconds; big ones tens of seconds).
        deadline = time.time() + 600  # 10 min safety
        while time.time() < deadline:
            state = getattr(f.state, "name", str(f.state))
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError(f"File API upload FAILED for {abs_path}")
            time.sleep(2)
            try:
                f = self.client.files.get(name=f.name)
            except Exception:
                # transient; keep polling
                pass
        else:
            raise TimeoutError(f"File API upload did not become ACTIVE in time: {abs_path}")
        self._uploaded_files[abs_path] = f
        if self.debug:
            print(f"📤 File API ready: {f.uri}")
        return f

    # ---------- Video helpers ----------
    # Note: These methods now delegate to video_utils for cleaner separation

    def create_video_part(
        self,
        video_path: str,
        fps: Optional[float] = None,
        start_offset: Optional[str] = None,
        end_offset: Optional[str] = None,
        media_resolution: Optional[str] = None,
        duration_sec: Optional[float] = None,
    ) -> Part:
        """Create a Gemini Part object for video with metadata.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to sample
            start_offset: Start time offset (e.g., "10.5s")
            end_offset: End time offset (e.g., "60s")
            media_resolution: Resolution level (MEDIA_RESOLUTION_LOW/MEDIUM/HIGH)
            duration_sec: Optional explicit duration in seconds (useful for clips not in metadata cache)
            
        Returns:
            Gemini Part object with video blob and metadata
        """
        # Try to find compressed version first
        compressed_path = find_compressed_video_fallback(video_path)
        actual_video_path = compressed_path if compressed_path else video_path
        
        if not os.path.exists(actual_video_path):
            raise FileNotFoundError(f"Video file not found: {actual_video_path}")
        
        if self.debug and compressed_path:
            print(f"📹 Using compressed video: {actual_video_path}")
        
        # Use video_utils to get MIME type
        mime_type = get_mime_type(actual_video_path)

        video_metadata = None
        if fps or start_offset or end_offset:
            kwargs = {}
            if fps:
                # Select max frame based on media resolution (from config)
                if media_resolution == "low":
                    max_frame = self.max_frame_low
                elif media_resolution == "medium":
                    max_frame = self.max_frame_medium
                elif media_resolution == "high":
                    max_frame = self.max_frame_high
                else:
                    max_frame = self.max_frame_medium

                # Calculate duration in seconds from offsets or explicit duration
                duration = duration_sec  # Use explicit duration if provided

                if duration is None or duration <= 0:
                    # Try to calculate from offsets
                    if start_offset and end_offset:
                        try:
                            start_sec = float(start_offset.rstrip("s"))
                            end_sec = float(end_offset.rstrip("s"))
                            duration = max(0, end_sec - start_sec)  # Segment duration
                        except ValueError:
                            duration = None
                    elif end_offset:
                        try:
                            duration = float(end_offset.rstrip("s"))  # Duration from start
                        except ValueError:
                            duration = None

                # If we still don't have duration, try to get it from video metadata
                if duration is None or duration <= 0:
                    meta_extractor = VideoMetadataExtractor(actual_video_path)
                    video_duration = meta_extractor.duration
                    if start_offset:
                        try:
                            start_sec = float(start_offset.rstrip("s"))
                            duration = max(0, video_duration - start_sec)
                        except ValueError:
                            duration = video_duration
                    else:
                        duration = video_duration

                # Last-resort probe via OpenCV; without it, the frame clamp
                # below is silently bypassed -> token / image-count errors.
                if duration is None or duration <= 0:
                    probed = _probe_video_duration_seconds(actual_video_path)
                    if probed and probed > 0:
                        if start_offset:
                            try:
                                start_sec = float(start_offset.rstrip("s"))
                                duration = max(0.0, probed - start_sec)
                            except ValueError:
                                duration = probed
                        else:
                            duration = probed
                        if self.debug:
                            print(f"📐 Probed duration via OpenCV: {duration:.1f}s")

                # Always clamp: if duration is still unknown, assume a conservative
                # worst case so we never request more frames than the API allows
                # (10,800 image cap / 1M-token budget).
                effective_duration = duration if (duration and duration > 0) else 1800.0
                expected_frames = fps * effective_duration
                if expected_frames > max_frame:
                    adjusted_fps = max_frame / effective_duration
                    if self.debug:
                        src = "duration" if (duration and duration > 0) else "worst-case 1800s"
                        print(f"⚠️  Adjusting FPS: {fps:.2f} -> {adjusted_fps:.4f} (max {max_frame} frames over {src})")
                    fps = adjusted_fps

                kwargs["fps"] = fps

            if start_offset:
                kwargs["startOffset"] = start_offset
            if end_offset:
                kwargs["endOffset"] = end_offset

            if kwargs:
                video_metadata = VideoMetadata(**kwargs)
                if self.debug:
                    print(f"📹 Video metadata: fps={fps}, startOffset={start_offset}, endOffset={end_offset}")

        # Note: mediaResolution should be passed via GenerateContentConfig, not Part
        # The media_resolution parameter is still used above for FPS adjustment calculations
        if self.debug and media_resolution:
            print(f"📹 Media resolution: {media_resolution} (will be set in GenerateContentConfig)")

        # Transport choice: tiny clips go inline (no upload roundtrip), bigger
        # videos go through the Gemini File API (avoids the 2 GiB inline cap
        # and the 1 GiB request-payload cap).
        try:
            file_size = os.path.getsize(actual_video_path)
        except OSError:
            file_size = 0
        try:
            if file_size > INLINE_VIDEO_SIZE_LIMIT_BYTES:
                uploaded = self._get_or_upload_file(actual_video_path, mime_type)
                return Part(file_data=FileData(file_uri=uploaded.uri, mime_type=mime_type),
                            video_metadata=video_metadata)
        except Exception as e:
            print(f"Failed to upload video file to Gemini's temporary file storage: {e}")
            print(f"Will use the inline data mode...")

        with open(actual_video_path, "rb") as f:
            video_data = f.read()
        blob = Blob(mime_type=mime_type, data=video_data)
        return Part(inlineData=blob, videoMetadata=video_metadata)

    # ---------- Core calls ----------
    def _map_rate_to_media_res(self, rate: SpatialTokenRate | str) -> str:
        """Map SpatialTokenRate or plain string to simple tokens: low|medium|high."""
        if isinstance(rate, SpatialTokenRate):
            return rate.value
        # assume already a string like "low" | "medium" | "high"
        return str(rate).strip().lower()


    def plan(
        self,
        query: str,
        video_meta: Dict[str, Any] = None,
        prior: Optional[Blackboard] = None,
        options: Optional[List[str]] = None,
        justification: Optional[str] = None,
        option_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanSpec:
        """Generate observation action plan using LLM.

        For initial planning: generates a single observation action based on query and video metadata.
        For replanning: incorporates all evidence and the reflector's justification
        to plan the next observation -- PLANNER.REPLAN(Q, H, J).

        Args:
            query: User's question about the video
            video_meta: Video metadata (duration, fps, etc.)
            prior: Optional prior blackboard state (for replanning - contains evidence and history)
            options: Optional list of options for MCQ questions
            justification: Reflector's justification J of what is still missing/uncertain

        Returns:
            PlanSpec with single observation action (query + watch config)
        """
        if self.client is None:
            self.initialize_client()
        
        if video_meta is None:
            video_meta = {}
        
        # Check if this is a replan (has prior evidence)
        is_replan = prior is not None and len(prior.evidences) > 0
        
        # A tool-enabled run uses one exclusive prompt contract.
        tool_prompt = getattr(self, "build_observation_tool_prompt", None)
        evidence_summary = prior.summary_text() if is_replan else ""
        if tool_prompt is not None:
            prompt = tool_prompt(query=query, video_meta=video_meta, options=options,
                evidence_summary=evidence_summary, justification=justification,
                option_candidates=option_candidates)
        elif is_replan:
            # Replanning: include evidence summary and the reflector's justification
            evidence_summary = prior.summary_text()
            prompt = PromptManager.get_replanning_prompt(
                query,
                video_meta,
                evidence_summary,
                options,
                justification=justification,
                option_candidates=option_candidates,
            )
        else:
            # Initial planning
            prompt = PromptManager.get_planning_prompt(
                query,
                video_meta,
                options,
                option_candidates=option_candidates,
            )
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"🎯 PLANNER INPUT {'(REPLAN)' if is_replan else '(INITIAL)'}")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Video Duration: {video_meta.get('duration_sec', 'unknown')}s")
            if options:
                print(f"Options: {options}")
            if is_replan:
                print(f"Prior Evidence Rounds: {len(prior.evidences)}")
                print(f"Evidence Summary: {evidence_summary[:200]}...")
            print(f"{'='*80}\n")
        
        sampling_policy = getattr(self, "planning_sampling_policy", None)
        if sampling_policy and tool_prompt is None:
            prompt += "\n\nRun sampling policy (overrides earlier sampling examples):\n" + sampling_policy

        # Call LLM using plan_replan_model for planning
        resp = self.client.models.generate_content(
            model=self.plan_replan_model,
            contents=prompt,
        )
        response_text = getattr(resp, "text", str(resp))
        recorder = getattr(self, "record_planner_response", None)
        if recorder is not None:
            recorder(prompt, response_text)
        
        if self.debug:
            print(f"\n{'='*80}")
            print(f"📋 PLANNER OUTPUT (Raw)")
            print(f"{'='*80}")
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            print(f"{'='*80}\n")
        
        # Parse JSON response
        plan_data = parse_json_response(response_text)
        tool_decoder = getattr(self, "decode_observation_tool", None)
        if tool_decoder is not None:
            try:
                plan_data = tool_decoder(plan_data)
            except ValueError as exc:
                if not getattr(self, "repair_planner_format", False):
                    raise
                repair_prompt = (prompt + "\nPrevious response:\n" + response_text
                    + "\nRepair this response's format/parameters only: " + str(exc)
                    + ". Return one complete corrected response using supplied facts. Do not invent observations.")
                repaired = self.client.models.generate_content(model=self.plan_replan_model, contents=repair_prompt)
                response_text = getattr(repaired, "text", str(repaired))
                if recorder is not None:
                    recorder(repair_prompt, response_text)
                plan_data = tool_decoder(parse_json_response(response_text))
            if isinstance(plan_data, PlanSpec):
                return plan_data
        
        if plan_data is None or not validate_against_schema(plan_data, PLAN_SCHEMA):
            # Fallback to default plan if parsing fails
            if self.debug:
                print("⚠️  Failed to parse LLM plan, using fallback")
            return self._get_fallback_plan(query)
        
        # Convert parsed JSON to PlanSpec (take first step only)
        try:
            steps_list = plan_data.get("steps", [])
            if not steps_list:
                if self.debug:
                    print("⚠️  No steps in plan, using fallback")
                return self._get_fallback_plan(query)
            
            # Take only the first step
            s = steps_list[0]
            
            # Handle case-insensitive spatial_token_rate
            spatial_rate_str = str(s["spatial_token_rate"]).strip().lower()
            try:
                spatial_token_rate = SpatialTokenRate(spatial_rate_str)
            except ValueError as e:
                if self.debug:
                    print(f"⚠️  Invalid spatial_token_rate: '{s['spatial_token_rate']}', using 'low' as default")
                spatial_token_rate = SpatialTokenRate.low
            
            # Parse regions with validation
            regions = []
            raw_regions = s.get("regions", [])
            if raw_regions:
                for r in raw_regions:
                    try:
                        # Validate region format
                        if not isinstance(r, (list, tuple)) or len(r) != 2:
                            if self.debug:
                                print(f"⚠️  Invalid region format (expected [start, end]): {r}, skipping")
                            continue
                        # Try to convert to floats
                        start_val = r[0]
                        end_val = r[1]
                        if isinstance(start_val, str) and isinstance(end_val, str):
                            # Both are strings - might be malformed
                            if self.debug:
                                print(f"⚠️  Region contains non-numeric values: {r}, skipping")
                            continue
                        start = float(start_val)
                        end = float(end_val)
                        regions.append((start, end))
                    except (ValueError, TypeError) as e:
                        if self.debug:
                            print(f"⚠️  Error parsing region {r}: {e}, skipping")
                        continue
            
            watch = WatchConfig(
                load_mode=s["load_mode"],
                fps=float(s["fps"]),
                spatial_token_rate=spatial_token_rate,
                regions=regions
            )
            
            plan = PlanSpec(
                plan_version="v1",
                query=query,
                watch=watch,
                description=s.get("description", ""),
                completion_criteria=plan_data.get("completion_criteria", ""),
            )
            
            if self.debug:
                print(f"\n{'='*80}")
                print(f"✅ PLANNER OUTPUT (Parsed)")
                print(f"{'='*80}")
                print(f"Generated single observation action")
                try:
                    print(f"Load Mode: {plan.watch.load_mode}")
                    print(f"FPS: {plan.watch.fps}")
                    print(f"Spatial Resolution: {plan.watch.spatial_token_rate.value if hasattr(plan.watch.spatial_token_rate, 'value') else plan.watch.spatial_token_rate}")
                    print(f"Regions: {plan.watch.regions}")
                    print(f"Description: {plan.description}")
                    print(f"\nFull Plan JSON:")
                    print(json.dumps(dataclasses.asdict(plan), indent=2, ensure_ascii=False))
                except Exception as e:
                    print(f"Error printing plan: {e}")
                print(f"{'='*80}\n")
            
            return plan
            
        except Exception as e:
            if self.debug:
                print(f"⚠️  Error converting plan: {e}, using fallback")
            return self._get_fallback_plan(query)
    
    def _get_fallback_plan(self, query: str) -> PlanSpec:
        """Fallback plan if LLM generation fails - single observation action."""
        # A damaged/unparseable Planner response must not silently authorize an
        # expensive full-video replay.  The fallback remains a single COARSE
        # locator pass; denser full coverage requires an explicit Planner plan.
        watch = WatchConfig(load_mode="uniform", fps=0.1, spatial_token_rate=SpatialTokenRate.low)
        return PlanSpec(
            plan_version="v1",
            query=query,
            watch=watch,
            description="COARSE locator pass after Planner output was unavailable"
        )

    def infer_on_video(
        self,
        video_path: str,
        duration_sec: float,
        sub_query: str,
        context: str,
        start_sec: float,
        end_sec: float,
        watch_cfg: WatchConfig,
        step_id: str,
        original_query: str = "",
        observation_reason: str = "",
    ) -> Evidence:
        """Query Gemini with video file directly (no frame extraction needed).
        
        Args:
            video_path: Path to video file
            sub_query: Question to answer about the video
            context: Accumulated evidence from previous steps
            start_sec: Start time offset in seconds (used when regions are empty)
            end_sec: End time offset in seconds (used when regions are empty)
            watch_cfg: Watch configuration (fps, spatial resolution)
            step_id: Current step identifier
            original_query: Original user query for context
            observation_reason: Planner's concrete objective for this observation
            
        Returns:
            Evidence object with model response
        """
        if self.client is None:
            self.initialize_client()

        # Get fps and media resolution from watch config
        fps = watch_cfg.fps if watch_cfg.fps > 0 else None
        # Use simple "low|medium|high" tokens directly
        media_res = self._map_rate_to_media_res(watch_cfg.spatial_token_rate)
        
        # Get video duration for prompt
        meta_extractor = VideoMetadataExtractor(video_path)
        video_duration = meta_extractor.duration

        # Every observation part references the original video. Region bounds
        # remain absolute offsets on that source; no intermediate video clip is
        # materialized and no relative clip timeline is introduced.
        parts = []
        frames_used = []
        
        if watch_cfg.load_mode == "region" and watch_cfg.regions:
            valid_regions = [
                (max(0.0, float(reg_start)), min(duration_sec, float(reg_end)))
                for reg_start, reg_end in watch_cfg.regions
            ]
            valid_regions = [
                (reg_start, reg_end)
                for reg_start, reg_end in valid_regions
                if reg_start < reg_end
            ]
            if not valid_regions:
                raise ValueError("no_valid_direct_observation_region")
            request_frame_cap = getattr(self, "max_video_frames_per_request", None)
            if fps and request_frame_cap:
                total_region_duration = sum(
                    reg_end - reg_start for reg_start, reg_end in valid_regions
                )
                if total_region_duration > 0:
                    # ffmpeg's fps filter may emit one endpoint frame per
                    # segment, so reserve one frame per attached segment.
                    usable_frame_budget = max(
                        1.0, float(request_frame_cap) - len(valid_regions)
                    )
                    fps = min(
                        fps,
                        usable_frame_budget / total_region_duration,
                    )
            
            for region_idx, (reg_start, reg_end) in enumerate(valid_regions):
                region_duration = reg_end - reg_start
                part = self.create_video_part(
                    video_path=video_path,
                    fps=fps,
                    start_offset=f"{reg_start:.3f}s",
                    end_offset=f"{reg_end:.3f}s",
                    media_resolution=media_res,
                    duration_sec=region_duration,
                )
                parts.append(part)
                frames_used.append({"start": reg_start, "end": reg_end, "fps": fps})
                if self.debug:
                    print(
                        f"📹 Direct source region {region_idx}: {video_path} "
                        f"({reg_start:.1f}s - {reg_end:.1f}s)"
                    )
            
            start_sec = min(frame["start"] for frame in frames_used)
            end_sec = max(frame["end"] for frame in frames_used)
        elif watch_cfg.load_mode == "region" and start_sec is not None and end_sec is not None:
            # Single region mode: keep the original source and absolute bounds.
            if start_sec >= end_sec:
                raise ValueError(
                    f"Invalid direct observation range: {start_sec:.1f}s >= {end_sec:.1f}s"
                )
            part = self.create_video_part(
                video_path=video_path,
                fps=fps,
                start_offset=f"{start_sec:.3f}s",
                end_offset=f"{end_sec:.3f}s",
                media_resolution=media_res,
                duration_sec=end_sec - start_sec,
            )
            parts = [part]
            frames_used = [{"start": start_sec, "end": end_sec, "fps": fps}]
        else:
            # Uniform mode reads the complete original source. Do not attach an
            # artificial bounded range: the transport records the true full
            # source scope after probing the original file.
            part = self.create_video_part(
                video_path=video_path,
                fps=fps,
                media_resolution=media_res,
            )
            parts = [part]
            frames_used = [{"start": start_sec, "end": end_sec, "fps": fps}]
        
        # Determine if this is a region mode
        is_region = watch_cfg.load_mode == "region"
        
        # Build prompt using PromptManager
        # Pass exact source-range ordering when multiple segments are sent.
        regions_for_prompt = None
        if is_region and len(frames_used) > 1:
            regions_for_prompt = [
                (frame["start"], frame["end"]) for frame in frames_used
            ]
        
        prompt = PromptManager.get_inference_prompt(
            sub_query=sub_query,
            context=context,
            start_sec=start_sec,
            end_sec=end_sec,
            original_query=original_query,
            video_duration_sec=video_duration,
            is_region=is_region,
            regions=regions_for_prompt,
            relative_region_timestamps=is_region,
        )
        
        if self.debug:
            if len(parts) > 1:
                print(f"🎬 Querying video with {len(parts)} regions:")
                for i, (reg_start, reg_end) in enumerate(watch_cfg.regions):
                    print(f"   Region {i}: {reg_start:.1f}s - {reg_end:.1f}s")
            else:
                print(f"🎬 Querying original video: {video_path}")
                print(f"   Absolute time range: {start_sec:.1f}s - {end_sec:.1f}s")
            print(f"   FPS: {fps}, Resolution: {media_res}")
        
        # Convert media_resolution to config format (e.g., "MEDIA_RESOLUTION_MEDIUM")
        media_resolution_config = None
        if media_res:
            mr = str(media_res).strip().lower()
            if mr == "low":
                media_resolution_config = "MEDIA_RESOLUTION_LOW"
            elif mr == "medium":
                media_resolution_config = "MEDIA_RESOLUTION_MEDIUM"
            elif mr == "high":
                media_resolution_config = "MEDIA_RESOLUTION_HIGH"
        
        # Create GenerateContentConfig with media_resolution
        config_kwargs = {}
        if media_resolution_config:
            config_kwargs["media_resolution"] = media_resolution_config
        
        generate_content_config = None
        if config_kwargs:
            generate_content_config = types.GenerateContentConfig(**config_kwargs)
        
        # Call the provider with one original-video part per absolute source range.
        contents = [prompt] + parts
        for frame_scope, video_part in zip(frames_used, parts):
            metadata = getattr(video_part, "video_metadata", None)
            submitted_fps = getattr(metadata, "fps", None) if metadata is not None else None
            if submitted_fps is not None:
                frame_scope["fps"] = float(submitted_fps)
        set_source_units = getattr(
            self.client, "set_observation_source_units", None
        )
        if callable(set_source_units):
            set_source_units(video_path=video_path, units=frames_used)
        if generate_content_config:
            resp = self.client.models.generate_content(
                model=self.execute_model, 
                contents=contents,
                config=generate_content_config
            )
        else:
            resp = self.client.models.generate_content(
                model=self.execute_model, 
                contents=contents
            )
        response_text = getattr(resp, "text", str(resp))
        
        # Try to parse structured JSON response
        evidence_data = parse_json_response(response_text)
        prose_timestamp_recovery_count: Optional[int] = None
        
        if evidence_data and validate_against_schema(evidence_data, EVIDENCE_SCHEMA):
            # Use structured response - support both old and new formats
            summary = evidence_data.get("summary") or evidence_data.get("detailed_response", response_text)
            detailed_response = evidence_data.get("detailed_response") or evidence_data.get("summary", "")
            
            # Extract key_evidence (new format only)
            key_evidence = evidence_data.get("key_evidence", [])
            if not key_evidence and detailed_response:
                recovered_key_evidence = self._extract_key_evidence_from_prose(
                    detailed_response
                )
                recovered_key_evidence = (
                    self._filter_prose_recovery_to_observed_scope(
                        recovered_key_evidence,
                        frames_used=frames_used,
                        timestamps_are_relative=is_region,
                    )
                )
                prose_timestamp_recovery_count = len(recovered_key_evidence)
                key_evidence = recovered_key_evidence
            
            if self.debug:
                print(f"✅ Parsed structured evidence: {len(key_evidence)} items")
        else:
            # Fallback: Try to extract fields from malformed JSON using regex
            detailed_response = self._extract_json_field(response_text, "detailed_response")
            
            # If detailed_response contains nested JSON with "description", extract it
            if detailed_response and "description" in detailed_response:
                description_value = self._extract_json_field(detailed_response, "description")
                if description_value:
                    detailed_response = description_value
            
            summary = self._extract_json_field(response_text, "summary") or detailed_response or response_text
            key_evidence = self._extract_key_evidence(response_text)
            
            # If we extracted valid data, use it
            if detailed_response and detailed_response != response_text:
                if self.debug:
                    print(f"✅ Extracted fields from malformed JSON: {len(key_evidence)} items")
            else:
                # Final fallback to regex extraction
                summary = response_text
                detailed_response = response_text
                time_anchors = self._extract_timestamps(response_text)
                # Convert single timestamps to ranges (use ±1 second window)
                key_evidence = [{"timestamp_start": max(0.0, t - 1.0), "timestamp_end": t + 1.0, "description": ""} for t in time_anchors]
                if self.debug:
                    print(f"⚠️  Using fallback parsing: {len(key_evidence)} items")
        
        relative_timestamp_item_count = 0
        dropped_relative_timestamp_items = 0
        if is_region:
            relative_timestamp_item_count = len(key_evidence)
            key_evidence = _convert_region_relative_key_evidence(
                key_evidence,
                frames_used,
            )
            dropped_relative_timestamp_items = (
                relative_timestamp_item_count - len(key_evidence)
            )

        # Normalize key_evidence to full-second intervals using floor for start and ceil for end
        # Clamp to [0, duration_sec] and drop invalid/zero-length intervals; deduplicate
        try:
            raw_ranges = []
            descs = []
            for ev_item in key_evidence:
                if isinstance(ev_item, dict):
                    ts_start = ev_item.get("timestamp_start")
                    ts_end = ev_item.get("timestamp_end")
                    desc = ev_item.get("description", "")
                    if ts_start is not None and ts_end is not None:
                        raw_ranges.append((float(ts_start), float(ts_end)))
                        descs.append(str(desc))
            rounded = round_intervals_full_seconds(raw_ranges, duration=duration_sec)
            # Map rounded intervals to a representative description (prefer first non-empty)
            interval_to_desc: Dict[Tuple[int, int], str] = {}
            for idx, (rs, re) in enumerate(rounded):
                # Find a source description from the original list that rounds to this interval
                chosen_desc = ""
                for jdx, (orig_s, orig_e) in enumerate(raw_ranges):
                    from math import floor, ceil
                    if (floor(orig_s), ceil(orig_e)) == (rs, re):
                        cand = descs[jdx]
                        if cand and not interval_to_desc.get((rs, re)):
                            chosen_desc = cand
                            break
                interval_to_desc[(rs, re)] = chosen_desc
            key_evidence = [
                {"timestamp_start": int(rs), "timestamp_end": int(re), "description": interval_to_desc.get((rs, re), "")}
                for (rs, re) in rounded
            ]
        except Exception:
            # If rounding fails for any reason, leave key_evidence as-is
            pass
        
        # Build model_call metadata
        model_call_metadata = {
            "model": self.execute_model,
            "fps": fps,
            "media_resolution": media_res,
            "prompt_version": "v2_structured"
        }
        if prose_timestamp_recovery_count is not None:
            model_call_metadata["prose_timestamp_recovery_count"] = (
                prose_timestamp_recovery_count
            )
        if is_region:
            model_call_metadata.update(
                {
                    "observer_timestamp_reference": "segment_relative",
                    "timestamp_offset_applied_by_code": True,
                    "relative_timestamp_item_count": relative_timestamp_item_count,
                    "dropped_relative_timestamp_items": dropped_relative_timestamp_items,
                }
            )
        observation_receipt = getattr(resp, "observation_receipt", None)
        if isinstance(observation_receipt, dict):
            # The transport creates this only after a successful video call;
            # none of its fields are parsed from model-authored text.
            model_call_metadata["observation_receipt"] = observation_receipt
        
        # Add region information to metadata
        if len(frames_used) > 1:
            model_call_metadata["regions"] = frames_used
            model_call_metadata["num_regions"] = len(frames_used)
        else:
            if frames_used:
                model_call_metadata["start_offset"] = f"{frames_used[0]['start']}s" if frames_used[0]['start'] > 0 else None
                model_call_metadata["end_offset"] = f"{frames_used[0]['end']}s" if frames_used[0]['end'] > 0 else None
        
        return Evidence(
            detailed_response=detailed_response,
            key_evidence=key_evidence,
            reasoning=summary,  # Use summary as reasoning
            frames_used=frames_used,
            model_call=model_call_metadata,
            timestamp=now_iso(),
            round_id=0  # Will be set by Controller
        )


    def _extract_timestamps(self, text: str) -> List[float]:
        """Extract timestamps from model response text."""
        import re
        # Look for patterns like "10.5s", "at 23 seconds", "1:30", etc.
        patterns = [
            r'(\d+\.?\d*)\s*(?:seconds?|secs?|s\b)',  # "10.5 seconds" or "10s"
            r'at\s+(\d+\.?\d*)',  # "at 10.5"
            r'(\d+):(\d+)',  # "1:30" (minutes:seconds)
        ]
        timestamps = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if ':' in match.group(0):
                    # Handle mm:ss format
                    mins, secs = match.groups()
                    timestamps.append(float(mins) * 60 + float(secs))
                else:
                    timestamps.append(float(match.group(1)))
        return sorted(list(set(timestamps)))  # Remove duplicates and sort

    def _extract_key_evidence_from_prose(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """Recover explicitly timed claims from otherwise valid prose.

        This is a narrow structured-output repair. It recognizes only explicit
        clock or seconds expressions, keeps the containing sentence as the
        claim, and leaves scope validation to the existing observation path.
        """
        import re

        prose = str(text or "").strip()
        if not prose:
            return []

        time_token = (
            r"(?:\d{1,3}:\d{2}|\d+(?:\.\d+)?\s*"
            r"(?:seconds?|secs?|sec|s)\b)"
        )
        range_pattern = re.compile(
            rf"(?P<start>{time_token})\s*"
            rf"(?:to|through|until|[-–—~])\s*"
            rf"(?P<end>{time_token})",
            re.IGNORECASE,
        )
        point_pattern = re.compile(time_token, re.IGNORECASE)

        sentence_spans: List[Tuple[int, int]] = []
        sentence_start = 0
        for boundary in re.finditer(r"(?<=[.!?])\s+|\n+", prose):
            if boundary.start() > sentence_start:
                sentence_spans.append((sentence_start, boundary.start()))
            sentence_start = boundary.end()
        if sentence_start < len(prose):
            sentence_spans.append((sentence_start, len(prose)))

        def parse_time_token(token: str) -> Optional[float]:
            value = token.strip()
            if ":" in value:
                minute_text, second_text = value.split(":", 1)
                minutes = int(minute_text)
                seconds = int(second_text)
                if seconds >= 60:
                    return None
                return float(minutes * 60 + seconds)
            match = re.match(r"\d+(?:\.\d+)?", value)
            return float(match.group(0)) if match else None

        def containing_sentence(start: int, end: int) -> str:
            for sentence_start, sentence_end in sentence_spans:
                if sentence_start <= start and end <= sentence_end:
                    return prose[sentence_start:sentence_end].strip()
            return prose

        recovered: List[Dict[str, Any]] = []
        occupied_spans: List[Tuple[int, int]] = []
        seen: set[Tuple[float, float, str]] = set()

        for match in range_pattern.finditer(prose):
            start = parse_time_token(match.group("start"))
            end = parse_time_token(match.group("end"))
            if start is None or end is None or start < 0 or end <= start:
                continue
            description = containing_sentence(match.start(), match.end())
            identity = (start, end, description)
            if identity not in seen:
                recovered.append(
                    {
                        "timestamp_start": start,
                        "timestamp_end": end,
                        "description": description,
                    }
                )
                seen.add(identity)
            occupied_spans.append(match.span())

        for match in point_pattern.finditer(prose):
            if any(
                occupied_start <= match.start() and match.end() <= occupied_end
                for occupied_start, occupied_end in occupied_spans
            ):
                continue
            timestamp = parse_time_token(match.group(0))
            if timestamp is None or timestamp < 0:
                continue
            description = containing_sentence(match.start(), match.end())
            start = max(0.0, timestamp - 1.0)
            end = timestamp + 1.0
            identity = (start, end, description)
            if identity not in seen:
                recovered.append(
                    {
                        "timestamp_start": start,
                        "timestamp_end": end,
                        "description": description,
                    }
                )
                seen.add(identity)

        return recovered

    def _filter_prose_recovery_to_observed_scope(
        self,
        recovered: List[Dict[str, Any]],
        *,
        frames_used: List[Dict[str, Any]],
        timestamps_are_relative: bool,
    ) -> List[Dict[str, Any]]:
        """Reject recovered prose times that cannot bind to the submitted scope."""
        if not recovered:
            return []

        scopes: List[Tuple[float, float]] = []
        if timestamps_are_relative:
            if len(frames_used) != 1:
                return []
            try:
                scope_duration = float(frames_used[0]["end"]) - float(
                    frames_used[0]["start"]
                )
            except (KeyError, TypeError, ValueError):
                return []
            if scope_duration > 0:
                scopes.append((0.0, scope_duration))
        else:
            for frame in frames_used:
                try:
                    scope_start = float(frame["start"])
                    scope_end = float(frame["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if scope_end > scope_start:
                    scopes.append((scope_start, scope_end))
        if not scopes:
            return []

        tolerance = 1e-3
        kept: List[Dict[str, Any]] = []
        for item in recovered:
            try:
                item_start = float(item["timestamp_start"])
                item_end = float(item["timestamp_end"])
            except (KeyError, TypeError, ValueError):
                continue
            if any(
                item_start >= scope_start - tolerance
                and item_end <= scope_end + tolerance
                for scope_start, scope_end in scopes
            ):
                kept.append(item)
        return kept
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from model response text."""
        import re
        # Look for patterns like "confidence: 0.8", "0.7 confidence", etc.
        patterns = [
            r'confidence[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s+confidence',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conf = float(match.group(1))
                return conf if conf <= 1.0 else conf / 100.0  # Handle percentage
        return 0.5  # Default medium confidence
    
    def _extract_json_field(self, text: str, field_name: str) -> str:
        """Extract a specific field value from malformed JSON using regex.
        
        Args:
            text: The response text (may contain malformed JSON)
            field_name: The field name to extract (e.g., "detailed_response", "summary", "description")
            
        Returns:
            The extracted field value, or empty string if not found
        """
        import re
        
        # Try to find the field value in the text
        # Pattern: "field_name": "value"
        # Handle escaped quotes and multiline strings
        # Define patterns (avoid raw f-string issues with braces)
        nested_pattern = f'"{field_name}"' + r'\s*:\s*\{' + r'\s*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        direct_pattern = rf'"{field_name}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        single_quote_pattern = rf'"{field_name}"\s*:\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\''
        nested_direct_pattern = f'"{field_name}"' + r'[^}]*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        
        patterns = [
            # Nested format: "field_name": { "description": "value" }
            nested_pattern,
            # Direct format: "field_name": "value"
            direct_pattern,
            # Single quotes: "field_name": 'value'
            single_quote_pattern,
            # Also try to extract nested description field directly
            nested_direct_pattern,
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                value = match.group(1)
                # Unescape common escape sequences
                value = value.replace('\\"', '"').replace("\\'", "'").replace('\\n', '\n').replace('\\t', '\t')
                # Clean up any remaining escape sequences
                value = value.replace('\\', '')
                return value
        
        return ""
    
    def _extract_key_evidence(self, text: str) -> List[Dict[str, Any]]:
        """Extract key_evidence array from malformed JSON.
        
        Args:
            text: The response text (may contain malformed JSON)
            
        Returns:
            List of key_evidence items (timestamp_start, timestamp_end, description)
        """
        import re
        
        # Try to extract key_evidence array
        # First try to match new format with timestamp ranges
        key_evidence = []
        
        # Pattern to match new format: "timestamp_start": X, "timestamp_end": Y, "description": "text"
        pattern_range = r'"timestamp_start"\s*:\s*(\d+\.?\d*)\s*,\s*"timestamp_end"\s*:\s*(\d+\.?\d*)\s*,\s*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        
        matches = re.finditer(pattern_range, text, re.DOTALL)
        for match in matches:
            ts_start = float(match.group(1))
            ts_end = float(match.group(2))
            description = match.group(3)
            # Unescape common escape sequences
            description = description.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            
            key_evidence.append({
                "timestamp_start": ts_start,
                "timestamp_end": ts_end,
                "description": description
            })
        
        # If no matches found, try legacy format (single timestamp)
        if not key_evidence:
            pattern_legacy = r'"timestamp_sec"\s*:\s*(\d+\.?\d*)\s*,\s*"description"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
            matches = re.finditer(pattern_legacy, text, re.DOTALL)
            for match in matches:
                timestamp = float(match.group(1))
                description = match.group(2)
                # Unescape common escape sequences
                description = description.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                # Convert single timestamp to range (use ±1 second window)
                key_evidence.append({
                    "timestamp_start": max(0.0, timestamp - 1.0),
                    "timestamp_end": timestamp + 1.0,
                    "description": description
                })
        
        return key_evidence

    # Legacy replan helpers removed in simplified loop

    def synthesize_final_answer(self, plan: PlanSpec, bb: Blackboard) -> Dict[str, Any]:
        """Synthesize final answer from all evidence.
        
        Args:
            plan: Executed plan
            bb: Blackboard with all evidence
            
        Returns:
            Dictionary with final answer and metadata (MCQ format)
        """
        if self.client is None:
            self.initialize_client()
        
        # Get options from blackboard meta if available, normalize to empty list if None
        options = bb.meta.get("options", None)
        options_list = options if options else []
        
        # Generate prompt using PromptManager (always uses MCQ format)
        from experiments.semantic_delta_pilot import upper_observation_only as upper_view
        upper_runtime = getattr(self, "_observe_only_runtime", None)
        if upper_view.enabled(upper_runtime):
            bb = copy.copy(bb)
            bb.evidences = upper_view.evidence_list(upper_runtime, bb.evidences)
        context_text = bb.summary_text()
        context = copy.deepcopy(getattr(plan, "decision_context", None) or {})
        if upper_view.enabled(upper_runtime):
            context = upper_view.decision_context(context)
        if context:
            context_text += "\nAdditional decision context:\n" + json.dumps(context, ensure_ascii=False)
        prompt = PromptManager.get_synthesis_prompt(
            original_query=plan.query,
            all_evidence=context_text,
            video_duration=bb.duration_sec or 0.0,
            options=options_list,
            option_candidates=bb.meta.get("option_candidates"),
            force_best_effort=bool(bb.meta.get("force_best_effort")),
            budget_forced_context=(None if upper_view.enabled(upper_runtime)
                                   else bb.meta.get("budget_forced_context")),
        )
        
        if self.debug:
            print(f"🔄 Synthesizing final answer...")
            if options_list:
                print(f"📝 MCQ format with {len(options_list)} options")
            else:
                print(f"📝 MCQ format (open-ended question)")
        
        # Call LLM using plan_replan_model for final answer synthesis (reasoning task)
        resp = _generate_content_for_stage(
            self.client,
            model=self.plan_replan_model,
            contents=prompt,
            stage="answer",
        )
        response_text = getattr(resp, "text", str(resp))
        
        # Try to parse structured JSON response
        answer_data = parse_json_response(response_text)
        
        # Always use MCQ_SCHEMA for validation
        if answer_data and validate_against_schema(answer_data, MCQ_SCHEMA):
            # Use structured response
            # Add query-level confidence if available from Blackboard
            if bb.query_confidence is not None:
                answer_data["query_confidence"] = bb.query_confidence
            if self.debug:
                print(f"✅ Parsed structured final answer (MCQ format)")
            return answer_data
        else:
            # Fallback: return text as-is in MCQ format
            if self.debug:
                print(f"⚠️  Using unstructured final answer (converted to MCQ format)")
            # Use query-level confidence from Blackboard if available
            query_confidence = bb.query_confidence if bb.query_confidence is not None else 0.5
            return {
                "selected_option": "",
                "confidence": query_confidence,
                "query_confidence": query_confidence,
                "reasoning": response_text[:500],
                "selected_option_text": response_text[:200] if response_text else ""
            }


# ======================================================
# Validation Utilities
# ======================================================
def clamp_regions(regions: List[List[float]], duration: float) -> List[List[float]]:
    out = []
    for start, end in regions:
        s = max(0.0, float(start))
        e = min(duration, float(end))
        if e > s:
            # Take the largest region: floor start, ceil end
            s_rounded = int(s)  # floor
            e_rounded = int(e) + 1 if e > int(e) else int(e)  # ceil
            out.append([s_rounded, e_rounded])
    return out


# ======================================================
# Planner / Observer / Reflector / Controller
# ======================================================
class Planner:
    """Planner component of the plan-observe-reflect framework.
    
    The Planner generates multi-step action plans that guide the agent
    to perceive key query-related evidence from the video.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def initial_plan(
        self,
        query: str,
        video_meta: Dict[str, Any] = None,
        options: Optional[List[str]] = None,
        option_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> PlanSpec:
        """Generate initial observation action plan.
        
        This is the "plan" phase of the plan-observe-reflect framework.
        Creates a single observation action plan (query + watch config).
        """
        plan_kwargs: Dict[str, Any] = {"options": options}
        if option_candidates is not None:
            plan_kwargs["option_candidates"] = option_candidates
        return self.client.plan(query, video_meta, **plan_kwargs)

    # Legacy update_plan removed in simplified loop


class Observer:
    """Observer component of the plan-observe-reflect framework.
    
    The Observer executes actions by observing the video and gathering
    query-related evidence into the evidence list.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def _compute_time_range(self, watch: WatchConfig, duration: float, bb: Blackboard = None) -> Tuple[Optional[float], Optional[float]]:
        """Compute start and end time offsets based on watch configuration.
        
        For region mode with multiple regions, this returns the overall range (min start, max end)
        for prompt context. The actual region processing happens in infer_on_video.
        
        Args:
            watch: Watch configuration
            duration: Video duration in seconds
            bb: Blackboard with accumulated evidence (unused in simplified loop)
        
        Returns:
            (start_sec, end_sec) tuple. None values mean use full video.
        """
        if watch.load_mode == "uniform":
            # Full video scan
            return (0.0, duration)
        elif watch.load_mode == "region":
            # If multiple regions exist, return overall range for prompt context
            # The actual processing will handle each region separately in infer_on_video
            if watch.regions:
                regions = [(float(s), float(e)) for s, e in watch.regions]
                # Return overall range (min start, max end) for prompt context
                start = max(0.0, min(r[0] for r in regions))
                end = min(duration, max(r[1] for r in regions))
                return (start, end)
            else:
                # No region info available, scan full video
                return (0.0, duration)
        else:
            raise ValueError(f"Unknown load_mode: {watch.load_mode}")

    def observe(self, plan: PlanSpec, bb: Blackboard) -> Evidence:
        """Observe the video segment specified by the plan and gather evidence.

        This is the "observe" phase of the plan-observe-reflect framework.
        Executes the observation action and collects query-related evidence.

        Pure: does NOT mutate `bb`. The Controller is the sole owner of the
        blackboard append and does so after assigning `ev.round_id`.

        Args:
            plan: Plan with query and watch config
            bb: Blackboard with video path (used for video metadata only)

        Returns:
            Evidence from this observation (round_id left at default 0)
        """
        # Get video metadata
        meta_extractor = VideoMetadataExtractor(bb.video_path)
        duration = meta_extractor.duration
        
        # Compute time range
        start_sec, end_sec = self._compute_time_range(plan.watch, duration, bb)
        
        # If querying full video, automatically use uniform mode instead of region mode
        is_full_video = (start_sec is not None and end_sec is not None and 
                        start_sec <= 1.0 and abs(end_sec - duration) <= 1.0)
        
        preserve_region_scope = bool(
            bb.meta.get("preserve_observation_regions", False)
        )

        # Create a copy of watch_cfg to modify if needed
        watch_cfg = plan.watch
        if (
            is_full_video
            and watch_cfg.load_mode == "region"
            and not preserve_region_scope
        ):
            # Switch to uniform mode for full video queries
            watch_cfg = dataclasses.replace(
                watch_cfg,
                load_mode="uniform",
                regions=[]  # Clear regions when using uniform mode
            )
            if self.client.debug:
                print(f"🔄 Detected full video query (0.0s - {duration:.1f}s), switching to uniform mode")
        
        # Get accumulated context
        context = bb.summary_text()
        
        # Build the complete query with options if available
        query_to_use = plan.query
        options = bb.meta.get("options", [])
        if options:
            options_text = "\n".join([f"- {opt}" for opt in options])
            query_to_use = f"{plan.query}\n\nOptions:\n{options_text}"
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"👁️  OBSERVER INPUT")
            print(f"{'='*80}")
            print(f"Query: {query_to_use[:200]}...")
            print(f"Video: {bb.video_path}")
            print(f"Time Range: {start_sec:.1f}s - {end_sec:.1f}s")
            print(f"Load Mode: {watch_cfg.load_mode}")
            print(f"FPS: {watch_cfg.fps}")
            print(f"Spatial Resolution: {watch_cfg.spatial_token_rate}")
            if watch_cfg.regions:
                print(f"Regions: {watch_cfg.regions}")
            print(f"Context: {context[:200]}..." if context else "Context: None")
            print(f"{'='*80}\n")
        
        ev = self.client.infer_on_video(
                video_path=bb.video_path,
                duration_sec=duration,
                sub_query=query_to_use,
                context=context,
                start_sec=start_sec,
                end_sec=end_sec,
                watch_cfg=watch_cfg,
                step_id="1",  # Always "1" in single-action mode
                original_query=plan.query,
                observation_reason=plan.description,
            )
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"📊 OBSERVER OUTPUT")
            print(f"{'='*80}")
            print(f"Detailed Response: {ev.detailed_response[:300]}...")
            print(f"Key Evidence Count: {len(ev.key_evidence)}")
            for i, kev in enumerate(ev.key_evidence[:5], 1):  # Show first 5
                if isinstance(kev, dict):
                    ts_start = kev.get("timestamp_start", 0)
                    ts_end = kev.get("timestamp_end", 0)
                    desc = kev.get("description", "")[:100]
                    print(f"  {i}. [{int(ts_start)}, {int(ts_end)}]s: {desc}")
            if len(ev.key_evidence) > 5:
                print(f"  ... and {len(ev.key_evidence) - 5} more")
            print(f"Reasoning: {ev.reasoning[:200]}...")
            print(f"{'='*80}\n")

        # Observer is pure: the Controller owns blackboard mutation and is the
        # only place that appends the returned Evidence (after setting round_id).
        return ev


class Reflector:
    """Reflector component of the plan-observe-reflect framework.
    
    The Reflector reflects over the query, evidence list, and interaction history
    to decide whether the evidence is sufficient to answer the query or whether
    additional actions need to be generated.
    """
    def __init__(self, client: GeminiClient):
        self.client = client

    def reflect(
        self,
        query: str,
        plan: PlanSpec,
        evidence_list: List[Evidence],
        interaction_history: List[Dict[str, Any]] = None,
        video_path: str = "",
        duration_sec: Optional[float] = None,
        is_last_round: bool = False,
        options: Optional[List[str]] = None,
        option_candidates: Optional[List[Dict[str, Any]]] = None,
        initial_visual_content: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Reflect on the current state and decide on next actions.
        
        This is the "reflect" phase of the plan-observe-reflect framework.
        Analyzes the query, gathered evidence, and interaction history to determine
        if evidence is sufficient or if new actions are needed.
        
        Args:
            query: Original user query
            plan: Current plan
            evidence_list: List of evidence gathered so far
            interaction_history: Optional list of interaction history entries
            video_path: Path to video file
            duration_sec: Video duration in seconds
            is_last_round: If True, directly generate final answer using synthesis prompt
            options: Optional list of multiple choice options
            
        Returns:
            Dictionary with:
            - sufficient: bool - whether evidence is sufficient to answer
            - should_update: bool - whether plan should be updated
            - updates: List[Dict] - proposed plan updates
            - reasoning: str - explanation of the reflection decision
            - confidence: float - confidence in the reflection decision
            - final_answer: Optional[Dict] - final answer data (if is_last_round=True)
        """
        from experiments.semantic_delta_pilot import upper_observation_only as upper_view
        upper_runtime = getattr(self.client, "_observe_only_runtime", None)
        evidence_list = upper_view.evidence_list(upper_runtime, evidence_list)
        # Convert evidence list to blackboard format for compatibility and summary text
        bb = Blackboard(video_path=video_path or "")
        for ev in evidence_list:
            bb.add_evidence(ev)
        context_text = bb.summary_text()
        decision_context = copy.deepcopy(getattr(plan, "decision_context", None) or {})
        if upper_view.enabled(upper_runtime):
            decision_context = upper_view.decision_context(decision_context)
        asr_text = decision_context.pop("asr_transcript", "")
        if isinstance(asr_text, str) and asr_text.strip():
            context_text += "\nASR transcript (speech text, not visual observation):\n" + asr_text
        if initial_visual_content:
            context_text += ("\nThe attached frames are a uniform sample of the original video. "
                "Assess the question using these images and any ASR above. Frame times are "
                "sampling estimates; discrete images do not establish continuous action coverage. "
                "Explain the answer and any remaining information gap using the existing response fields.")
        if decision_context:
            context_text += "\nDecision state (not new evidence):\n" + json.dumps(decision_context, ensure_ascii=False)
        
        def generate_decision(prompt, schema):
            if self.client.client is None:
                self.client.initialize_client()
            response = _generate_content_for_stage(self.client.client,
                model=self.client.plan_replan_model,
                contents=[prompt, *initial_visual_content] if initial_visual_content else prompt,
                stage="answer")
            text = getattr(response, "text", str(response))
            if getattr(plan, "decision_context", None) and not validate_against_schema(parse_json_response(text) or {}, schema):
                prompt += ("\nPrevious response:\n" + text +
                    "\nRepair only the response format to the requested fields. Use existing facts; do not request new video.")
                response = _generate_content_for_stage(self.client.client,
                    model=self.client.plan_replan_model, contents=prompt, stage="answer")
                text = getattr(response, "text", str(response))
                if not validate_against_schema(parse_json_response(text) or {}, schema):
                    raise RuntimeError("decision_format_repair_failed")
            return text

        # Last round: REFLECTOR.FORCEANSWER(Q, E) -- force an answer from all
        # accumulated evidence (Algorithm 1, line 10).
        if is_last_round:
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"🎯 LAST ROUND - FORCEANSWER(Q, E)")
                print(f"{'='*80}")
                print(f"Query: {query}")
                print(f"Evidence Count: {len(evidence_list)}")
                print(f"Video Duration: {duration_sec:.1f}s" if duration_sec else "Duration: unknown")
                if options:
                    print(f"Options: {options}")
                print(f"{'='*80}\n")
            
            # Normalize options to empty list if None
            options_list = options if options else []
            
            # Generate synthesis prompt (always uses MCQ format for QA)
            prompt = PromptManager.get_synthesis_prompt(
                original_query=query,
                all_evidence=context_text,
                video_duration=duration_sec or 0.0,
                options=options_list,
                option_candidates=option_candidates,
                force_best_effort=True,
            )
            
            # Call LLM using plan_replan_model for final answer synthesis
            if self.client.client is None:
                self.client.initialize_client()
            
            response_text = generate_decision(prompt, MCQ_SCHEMA)

            # Try to parse structured JSON response
            answer_data = parse_json_response(response_text)
            
            # Always use MCQ_SCHEMA for validation
            if answer_data and validate_against_schema(answer_data, MCQ_SCHEMA):
                # Use structured response
                query_confidence = answer_data.get("confidence", 0.5)
                if self.client.debug:
                    print(f"✅ Parsed structured final answer (MCQ format)")
            else:
                # Fallback: return text as-is in MCQ format
                if self.client.debug:
                    print(f"⚠️  Using unstructured final answer (converted to MCQ format)")
                query_confidence = 0.5
                answer_data = {
                    "selected_option": "",
                    "confidence": query_confidence,
                    "reasoning": response_text[:500],
                    "selected_option_text": response_text[:200] if response_text else ""
                }
            
            # Add query_confidence to answer_data
            answer_data["query_confidence"] = query_confidence
            
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"✅ FINAL ANSWER GENERATED IN REFLECTION")
                print(f"{'='*80}")
                print(f"Selected Option: {answer_data.get('selected_option', 'N/A')}")
                print(f"Option Text: {answer_data.get('selected_option_text', '')}")
                print(f"Reasoning: {answer_data.get('reasoning', '')[:300]}...")
                print(f"Confidence: {answer_data.get('confidence', 0.0):.2f}")
                print(f"{'='*80}\n")
            
            return {
                "sufficient": True,
                "should_update": False,
                "updates": [],
                "reasoning": f"Final round: Generated answer directly from all evidence.",
                "confidence": 0.9,
                "query_confidence": query_confidence,
                "event": "FINAL_ANSWER_GENERATED",
                "final_answer": answer_data
            }
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"🔍 REFLECTOR INPUT  (C, J) = REFLECTOR(Q, E)")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Evidence Count: {len(evidence_list)}")
            print(f"Video: {video_path}")
            print(f"Duration: {duration_sec:.1f}s" if duration_sec else "Duration: unknown")
            print(f"{'='*80}\n")

        if self.client.debug:
            total_items = sum(len(ev.key_evidence) for ev in evidence_list)
            print(f"Cumulative evidence: {len(evidence_list)} round(s), {total_items} key-evidence item(s)")

        # Confidence threshold tau_conf (paper Sec 4.3 default 0.7)
        tau_conf = getattr(self.client, "confidence_threshold", 0.7)

        # Short-circuit: with no usable evidence at all, the reflector cannot
        # verify anything -- report low confidence and request another round.
        has_any_evidence = (any(ev.detailed_response for ev in evidence_list)
            or bool(isinstance(asr_text, str) and asr_text.strip()) or bool(initial_visual_content))
        if not has_any_evidence:
            if self.client.debug:
                print(f"\n⚠️  No evidence gathered yet - insufficient, requesting another observation")
            return {
                "sufficient": False,
                "should_update": True,
                "updates": [],
                "reasoning": "No evidence gathered yet; the reflector cannot verify the query.",
                "justification": (
                    "No usable evidence has been gathered yet. The next observation "
                    "should scan the video to locate cues relevant to the query."
                ),
                "confidence": 0.5,
                "query_confidence": 0.0,
                "event": "REFLECTION"
            }

        # REFLECTOR(Q, E): an LLM verifies the cumulative evidence against the
        # query and jointly produces a confidence score and a justification.
        options_list = options if options else []
        prompt = PromptManager.get_reflection_prompt(
            query=query,
            evidence_summary=context_text,
            video_duration=duration_sec or 0.0,
            options=options_list,
            option_candidates=option_candidates,
        )

        if self.client.client is None:
            self.client.initialize_client()

        try:
            response_text = generate_decision(prompt, REFLECTOR_SCHEMA)
            parsed = parse_json_response(response_text)
        except Exception as e:
            if getattr(plan, "decision_context", None):
                raise
            if self.client.debug:
                print(f"⚠️  Reflector LLM call failed: {e}")
            parsed = None

        if parsed and validate_against_schema(parsed, REFLECTOR_SCHEMA):
            try:
                query_confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                query_confidence = 0.0
            query_confidence = max(0.0, min(1.0, query_confidence))
            llm_sufficient = bool(parsed.get("sufficient", False))
            justification = str(parsed.get("justification", "")).strip()
            llm_reasoning = str(parsed.get("reasoning", "")).strip()
            selected_option = str(parsed.get("selected_option", "")).strip()
            selected_option_text = str(parsed.get("selected_option_text", "")).strip()
        else:
            # Robust fallback: keep the loop conservative rather than reverting
            # to a heuristic. Low confidence -> another observation round.
            if self.client.debug:
                print(f"⚠️  Failed to parse reflector output; defaulting to low confidence")
            query_confidence = 0.3
            llm_sufficient = False
            justification = (
                "Reflector output could not be parsed. Treating evidence as "
                "insufficient; the next observation should gather clearer, "
                "query-relevant cues."
            )
            llm_reasoning = "Reflector output unparseable."
            selected_option = ""
            selected_option_text = ""

        # Halting decision (paper Algorithm 1 + Table 12): C >= tau_conf.
        # We additionally require the LLM's own `sufficient` flag (appendix
        # prompt's primary output) to agree, so a confident-but-explicitly-
        # insufficient reflection still triggers another observation round.
        sufficient = (query_confidence >= tau_conf) and llm_sufficient

        reasoning = (
            f"Reflector verified {len(evidence_list)} round(s) of evidence "
            f"against the query. sufficient={llm_sufficient}, "
            f"confidence={query_confidence:.2f} (tau_conf={tau_conf:.2f}). "
            f"{llm_reasoning or justification}"
        )

        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"✅ REFLECTOR OUTPUT")
            print(f"{'='*80}")
            print(f"Sufficient (LLM): {llm_sufficient}")
            print(f"Confidence: {query_confidence:.2f}  (tau_conf={tau_conf:.2f})")
            print(f"Justification: {justification[:300]}")
            print(f"Reasoning: {llm_reasoning[:300]}")
            print(f"Decision: {'✅ SUFFICIENT - EXTRACTANSWER(J)' if sufficient else '❌ INSUFFICIENT - REPLAN(Q, H, J)'}")
            print(f"{'='*80}\n")

        # Preserve the decision even below the continuation confidence threshold.
        # A later no-more-evidence stop must not discard an existing answer.
        answer_data = {
            "selected_option": selected_option,
            "confidence": query_confidence,
            "reasoning": justification,
            "selected_option_text": selected_option_text,
            "query_confidence": query_confidence,
        }
        if sufficient:
            return {
                "sufficient": True,
                "should_update": False,
                "updates": [],
                "reasoning": reasoning,
                "confidence": query_confidence,
                "query_confidence": query_confidence,
                "justification": justification,
                "event": "REFLECTION_ANSWER_EXTRACTED",
                "final_answer": answer_data,
            }

        # Insufficient: justification J feeds the next PLANNER.REPLAN(Q, H, J).
        return {
            "sufficient": False,
            "should_update": True,
            "updates": [],
            "reasoning": reasoning,
            "confidence": query_confidence,
            "query_confidence": query_confidence,
            "justification": justification,
            "candidate_answer": answer_data,
            "event": "REFLECTION"
        }

    # Legacy confidence/decision helpers removed in simplified loop


class Controller:
    """Controller orchestrates the plan-observe-reflect framework.
    
    The Controller manages the iterative loop:
    1. Plan: Generate action plans
    2. Observe: Execute actions and gather evidence
    3. Reflect: Assess evidence sufficiency and decide on next actions
    """
    def __init__(
        self,
        run_dir: str,
        video_path: str,
        client: GeminiClient,
        options: Optional[List[str]] = None,
        admission_runtime: Optional[Any] = None,
        event_assist_runtime: Optional[Any] = None,
    ):
        self.store = Store(run_dir)
        self.client = client
        self.admission_runtime = admission_runtime
        self.client._observe_only_runtime = admission_runtime
        self.event_assist_runtime = event_assist_runtime
        configure_receipts = getattr(
            self.client, "set_observation_receipts_enabled", None
        )
        if callable(configure_receipts):
            configure_receipts(
                admission_runtime is not None or event_assist_runtime is not None
            )
        # ASR timestamps and source identity refer to the original video.
        compressed_path = (None if (getattr(admission_runtime, "asr_context", None) is not None
                           or getattr(admission_runtime, "uniform_seed_manifest", None))
                           else find_compressed_video_fallback(video_path))
        actual_video_path = compressed_path if compressed_path else video_path
        if client.debug and compressed_path:
            print(f"✅ Using compressed video: {compressed_path}")
        self.bb = Blackboard(video_path=actual_video_path)
        if options:
            self.bb.meta["options"] = list(options)
            hypothesis_verifier = getattr(
                self.admission_runtime, "verify_option_hypotheses", None
            )
            if callable(hypothesis_verifier):
                option_candidates = initialize_open_option_candidates(options)
                self.bb.meta["option_candidates"] = option_candidates
                self.store.write_json(
                    self.store.root / "option_candidates.initial.json",
                    option_candidates,
                )
        self._init_meta()

    # ---------- meta ----------
    def _init_meta(self) -> None:
        # Use VideoMetadataExtractor to get video info (from JSON cache - only duration needed!)
        meta_extractor = VideoMetadataExtractor(self.bb.video_path)
        self.bb.duration_sec = meta_extractor.duration
        # Fallback: probe the actual video file when the annotation cache has no
        # usable duration. Without this, the planner / reflector / synthesis
        # prompts all see "Video Duration: 0.0s" and the per-Part frame clamp is
        # silently bypassed (root cause of the token / image-count 400s in our
        # MINERVA run).
        if not self.bb.duration_sec or self.bb.duration_sec <= 0:
            probed = _probe_video_duration_seconds(self.bb.video_path)
            if probed and probed > 0:
                self.bb.duration_sec = probed
                if self.client.debug:
                    print(f"📐 Probed duration via OpenCV: {probed:.1f}s "
                          f"(annotation metadata had none)")
        
        meta = {
            "video_path": self.bb.video_path,
            "duration_sec": self.bb.duration_sec,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self.client.model,  # Legacy field
            "plan_replan_model": self.client.plan_replan_model,
            "execute_model": self.client.execute_model,
            "prompt_versions": {"plan": "v2_json_metadata", "infer": "v2_structured", "update": "v2_structured", "synthesize": "v2_structured"}
        }
        if self.bb.meta.get("option_candidates"):
            meta["option_candidates_initial"] = copy.deepcopy(
                self.bb.meta["option_candidates"]
            )
        self.store.write_json(self.store.meta, meta)

    def _prepare_event_assisted_observation(
        self,
        plan: PlanSpec,
    ) -> Tuple[PlanSpec, Optional[Dict[str, Any]]]:
        if self.event_assist_runtime is None:
            return plan, None
        prepared = self.event_assist_runtime.prepare_observation(
            plan=plan,
            blackboard=self.bb,
        )
        if (
            not isinstance(prepared, tuple)
            or len(prepared) != 2
            or not isinstance(prepared[0], PlanSpec)
            or not isinstance(prepared[1], dict)
        ):
            raise TypeError("event_assist_prepare_observation_contract_invalid")
        prepared_plan, metadata = prepared
        self.store.append_history(
            {
                "ts": now_iso(),
                "event": "EVENT_ASSIST_OBSERVATION_PLAN",
                "need_id": prepared_plan.need_id,
                "event_assist": copy.deepcopy(metadata),
                "watch": dataclasses.asdict(prepared_plan.watch),
            }
        )
        return prepared_plan, metadata

    def _annotate_event_assisted_observation(
        self,
        evidence: Evidence,
        preparation: Optional[Dict[str, Any]],
    ) -> None:
        if self.event_assist_runtime is None or preparation is None:
            return
        self.event_assist_runtime.annotate_observation(
            evidence=evidence,
            preparation=preparation,
        )

    def _run_initial_event_overview(self, query: str) -> Optional[Dict[str, Any]]:
        if self.event_assist_runtime is None:
            return None
        run_overview = getattr(
            self.event_assist_runtime,
            "run_initial_overview",
            None,
        )
        if not callable(run_overview):
            return None
        overview = run_overview(
            client=self.client,
            query=query,
            options=self.bb.meta.get("options", []),
        )
        if not isinstance(overview, dict):
            raise TypeError("event_overview_contract_invalid")
        self.bb.meta["event_overview_navigation"] = copy.deepcopy(overview)
        overview_path = self.store.root / "event_overview.navigation.json"
        self.store.write_json(overview_path, overview)
        self.store.append_history(
            {
                "ts": now_iso(),
                "event": "EVENT_OVERVIEW_COMPLETED",
                "mode": "NAVIGATION_ONLY",
                "overview_path": str(overview_path),
                "event_count": overview.get("event_count"),
                "frame_count": overview.get("frame_count"),
                "physical_request_count": overview.get("physical_request_count"),
                "audit_executed": False,
                "answer_eligible": False,
            }
        )
        return overview

    def _event_overview_planning_context(self) -> Optional[str]:
        if self.event_assist_runtime is None:
            return None
        context_builder = getattr(
            self.event_assist_runtime,
            "planning_context_text",
            None,
        )
        overview_complete = bool(
            getattr(self.event_assist_runtime, "overview_complete", False)
        )
        if not overview_complete or not callable(context_builder):
            return None
        context = str(context_builder()).strip()
        return context or None

    @staticmethod
    def _with_event_overview_planning_context(
        planning_blackboard: Blackboard,
        overview_context: str,
    ) -> Blackboard:
        planning_prior = copy.deepcopy(planning_blackboard)
        planning_prior.evidences.insert(
            0,
            Evidence(
                detailed_response=overview_context,
                key_evidence=[],
                reasoning=(
                    "Navigation-only Event overview; dense rereading required."
                ),
                model_call={
                    "producer": "event_overview_navigation",
                    "audit_executed": False,
                    "answer_eligible": False,
                },
                round_id=0,
            ),
        )
        return planning_prior

    # ---------- planning ----------
    def plan(self, query: str) -> PlanSpec:
        """Generate initial plan with video metadata."""
        planner = Planner(self.client)
        # Prepare video metadata for planning (only duration is needed!)
        video_meta = {
            "duration_sec": self.bb.duration_sec,
        }
        # Get options from blackboard if available
        options = self.bb.meta.get("options", None)
        overview_context = self._event_overview_planning_context()
        if overview_context:
            planning_prior = self._with_event_overview_planning_context(
                self.bb,
                overview_context,
            )
            plan = self.client.plan(
                query,
                video_meta=video_meta,
                prior=planning_prior,
                options=options,
                justification=(
                    "Use the navigation-only Event overview to choose explicit "
                    "time region(s) for dense raw-video rereading."
                ),
                option_candidates=self.bb.meta.get("option_candidates"),
            )
        else:
            plan = planner.initial_plan(
                query,
                video_meta,
                options=options,
                option_candidates=self.bb.meta.get("option_candidates"),
            )
        
        self.store.write_json(self.store.plan_initial, dataclasses.asdict(plan))
        self.store.append_history({
            "ts": now_iso(),
            "event": "PLAN_INITIAL",
            "plan_path": str(self.store.plan_initial),
            "query": query,
            "planning_trigger": (
                "EVENT_OVERVIEW" if overview_context else "RAW_QUESTION"
            ),
            "need_id": None,
        })
        return plan

    # ---------- execute observation ----------
    def execute_observation(self, plan: PlanSpec) -> Evidence:
        """Execute the observation action from the plan."""
        # Use round_id = len(evidences) + 1
        round_id = len(self.bb.evidences) + 1
        self.store.append_history({"ts": now_iso(), "event": "EXEC_OBSERVATION_START", "round_id": round_id})
        observation_plan, event_assist_preparation = (
            self._prepare_event_assisted_observation(plan)
        )
        observer = Observer(self.client)
        ev = observer.observe(observation_plan, self._observation_blackboard())
        ev.round_id = round_id
        self._annotate_event_assisted_observation(ev, event_assist_preparation)
        admission_event = self._admit_observation(
            plan.query, observation_plan, ev
        )
        self.bb.add_evidence(ev)
        # Persist evidence with interval_map for readability
        ev_dict = dataclasses.asdict(ev)
        try:
            interval_map = {}
            for item in ev.key_evidence:
                if isinstance(item, dict):
                    s = item.get("timestamp_start")
                    e = item.get("timestamp_end")
                    d = item.get("description", "")
                    if s is not None and e is not None:
                        interval_map[f"[{int(s)},{int(e)}]"] = d or ""
            ev_dict["interval_map"] = interval_map
        except Exception:
            pass
        self.store.write_json(self.store.evidence_json(round_id), ev_dict)
        # Extract timestamps from key_evidence (use midpoint for logging)
        timestamps = []
        for e in ev.key_evidence:
            if isinstance(e, dict):
                ts_start = e.get("timestamp_start")
                ts_end = e.get("timestamp_end")
                if ts_start is not None and ts_end is not None:
                    timestamps.append((ts_start + ts_end) / 2.0)
                elif ts_start is not None:
                    timestamps.append(ts_start)
        history_event = {
            "ts": now_iso(), "event": "EXEC_OBSERVATION_END", "round_id": round_id,
            "evidence_path": str(self.store.evidence_json(round_id)),
            "timestamps": timestamps,
        }
        if admission_event is not None:
            history_event["admission"] = admission_event
        self.store.append_history(history_event)
        return ev

    def _admit_observation(
        self,
        query: str,
        plan: PlanSpec,
        evidence: Evidence,
    ) -> Optional[Dict[str, Any]]:
        if self.admission_runtime is None:
            return None
        admission_event = self.admission_runtime.admit_observation(
            query=query,
            plan=plan,
            evidence=evidence,
            interaction_history=self.store.get_interaction_history(),
            current_confidence=self.bb.query_confidence,
            video_path=self.bb.video_path,
        )
        # Preserve the completed observation even when the text Verifier fails.
        self.store.write_json(self.store.evidence_json(evidence.round_id), dataclasses.asdict(evidence))
        if getattr(self.admission_runtime, "need_local_audit_enabled", False):
            return admission_event
        return self._verify_admitted_observation(query, plan, evidence, admission_event)

    def _verify_admitted_observation(
        self, query: str, plan: PlanSpec, evidence: Optional[Evidence],
        admission_event: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # Reuse has no new Evidence or Audit; citations retain their original rounds.
        current_candidates = self.bb.meta.get("option_candidates")
        hypothesis_verifier = getattr(
            self.admission_runtime, "verify_option_hypotheses", None
        )
        if not current_candidates or not callable(hypothesis_verifier):
            return admission_event

        # Hypothesis status is updated only after the observation itself has
        # passed through Evidence Audit.  The Hypothesis Verifier receives a
        # copy so it cannot mutate the frozen pre-verification state in place.
        verifier_results = hypothesis_verifier(
            query=query,
            option_candidates=copy.deepcopy(current_candidates),
            plan=plan,
            evidence=evidence,
            admission_event=copy.deepcopy(admission_event),
            video_path=self.bb.video_path,
        )
        check_failure = getattr(self.admission_runtime, "raise_for_verifier_failure", None)
        if callable(check_failure):
            check_failure()
        if verifier_results is None:
            return admission_event

        verifier_results = list(verifier_results)
        updated_candidates = apply_hypothesis_verifier_results(
            current_candidates,
            verifier_results,
        )
        self.bb.meta["option_candidates"] = updated_candidates
        candidate_event = {
            "round_id": getattr(evidence, "round_id", None),
            "acquisition": copy.deepcopy(getattr(plan, "acquisition", None)),
            "before": copy.deepcopy(current_candidates),
            "verifier_results": copy.deepcopy(verifier_results),
            "after": copy.deepcopy(updated_candidates),
        }
        if evidence is None:
            artifact = f"option_candidates.{plan.acquisition['action']}_{plan.acquisition['decision_id']}.json"
        else:
            artifact = f"option_candidates.round_{evidence.round_id}.json"
        self.store.write_json(self.store.root / artifact, candidate_event)
        if isinstance(admission_event, dict):
            combined_event = copy.deepcopy(admission_event)
            combined_event["hypothesis_verifier"] = candidate_event
            return combined_event
        return {"hypothesis_verifier": candidate_event}

    def _reasoning_evidence_list(self) -> List[Evidence]:
        evidence_list = self.bb.get_evidence_list()
        if self.admission_runtime is None:
            return evidence_list
        return self.admission_runtime.project_evidence_list(evidence_list)

    def _reasoning_blackboard(self) -> Blackboard:
        if self.admission_runtime is None:
            return self.bb
        return self.admission_runtime.project_blackboard(self.bb)

    def _planning_blackboard(self) -> Blackboard:
        if self.admission_runtime is None:
            return self.bb
        projector = getattr(
            self.admission_runtime, "project_planning_blackboard", None
        )
        if callable(projector):
            return projector(self.bb)
        return self.admission_runtime.project_blackboard(self.bb)

    def _observation_blackboard(self) -> Blackboard:
        if self.admission_runtime is None:
            return self.bb
        return self.admission_runtime.observation_blackboard(self.bb)

    def _replan(
        self,
        query: str,
        *,
        options: Optional[List[str]],
        justification: Optional[str],
        event: str,
    ) -> PlanSpec:
        video_meta = {"duration_sec": self.bb.duration_sec}
        planning_blackboard = self._planning_blackboard()
        overview_context = self._event_overview_planning_context()
        if overview_context:
            planning_blackboard = self._with_event_overview_planning_context(
                planning_blackboard,
                overview_context,
            )
        active_need = planning_blackboard.meta.get("active_observation_need")
        planner_justification = justification
        if isinstance(active_need, dict):
            need_id = str(active_need.get("need_id") or "").strip()
            formatter = getattr(
                self.admission_runtime,
                "observation_need_justification",
                None,
            )
            if (
                need_id
                and callable(formatter)
                and f"Active observation need {need_id}" not in str(planner_justification or "")
            ):
                planner_justification = formatter(planner_justification)
        plan_kwargs: Dict[str, Any] = {
            "video_meta": video_meta,
            "prior": planning_blackboard,
            "options": options,
            "justification": planner_justification,
        }
        option_candidates = self.bb.meta.get("option_candidates")
        if option_candidates is not None:
            plan_kwargs["option_candidates"] = option_candidates
        plan = self.client.plan(query, **plan_kwargs)
        if isinstance(active_need, dict):
            plan.need_id = str(active_need.get("need_id") or "").strip() or None
            try:
                plan.need_revision = int(active_need.get("revision"))
            except (TypeError, ValueError):
                plan.need_revision = None
        plan_index = next_plan_index(self.store)
        plan_path = self.store.plan_updated(plan_index)
        self.store.write_json(plan_path, dataclasses.asdict(plan))
        self.store.append_history(
            {
                "ts": now_iso(),
                "event": event,
                "plan_path": str(plan_path),
                "justification": planner_justification,
                "planning_trigger": (
                    "OBSERVATION_NEED" if plan.need_id else "REFLECTOR_JUSTIFICATION"
                ),
                "need_id": plan.need_id,
                "need_revision": plan.need_revision,
            }
        )
        return plan

    # ---------- reflection ----------
    # Replanning is driven by the Planner; verifier only verifies evidence and signals sufficiency.
    # Legacy placeholder autofill removed in simplified loop
    
    # Legacy region-merging helpers removed in simplified loop

    # ---------- run loop (plan-observe-reflect framework) ----------
    def run(
        self, query: str, max_rounds: Optional[int] = 3, *,
        initial_audited_observation: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the iterative plan-observe-reflect loop.
        
        The framework follows this process:
        1. Plan (single observation action)
        2. Observe (gather evidence)
        3. Reflect & Verify (re-watch regions). If sufficient → answer; else → replan.
        
        Args:
            query: User's question
            max_rounds: Maximum number of plan-observe-reflect cycles
        """
        if initial_audited_observation is not None:
            if self.admission_runtime is None or (max_rounds is not None and max_rounds < 1):
                raise ValueError("audited_prefill_requires_runtime_and_positive_budget")
            if self._load_latest_plan() is not None or self.event_assist_runtime is not None:
                raise ValueError("audited_prefill_requires_fresh_unassisted_run")
            plan, initial_evidence, initial_admission = initial_audited_observation
            if initial_evidence.round_id < 1 or initial_evidence not in self.bb.evidences:
                raise ValueError("audited_prefill_evidence_not_in_blackboard")
            self.store.write_json(self.store.plan_initial, dataclasses.asdict(plan))
            overview_round_offset = initial_evidence.round_id - 1
            # The initial decision does not consume the local observation budget.
            observation_round_limit = float("inf") if max_rounds is None else int(max_rounds) + 1
        else:
            # Plan phase: Generate initial plan if not present
            latest_plan = self._load_latest_plan()
            overview_record = None
            if latest_plan is None:
                overview_record = self._run_initial_event_overview(query)
            plan = latest_plan or self.plan(query)
            overview_round_offset = 1 if overview_record is not None else 0
            observation_round_limit = (float("inf") if max_rounds is None else max(
                1, int(max_rounds) - overview_round_offset))
        
        # Track final answer if generated in reflection
        final_answer_from_reflection = None
        final_answer_admitted = False
        asr_initial = getattr(self.admission_runtime, "asr_initial_answer", None)
        decision_candidate = copy.deepcopy(asr_initial)
        budget_force_decision: Optional[Dict[str, Any]] = None
        observation_need_stopped = bool(getattr(self.admission_runtime, "asr_initial_decision", None)
                                        and plan.complete and not plan.need_id)
        observation_stop_reason = ("asr_seed_decision_no_observable_need" if observation_need_stopped
                                   else "mechanical_observation_repeat_rejected")
        if observation_need_stopped:
            observation_round_limit = 0

        def authorize_budget_forced_answer(
            reason: str,
            *,
            route: Optional[str] = None,
        ) -> Dict[str, Any]:
            decision = {
                "decision": "FORCE_BEST_EFFORT",
                "source": "planner_budget_policy",
                "reason": reason,
                "route": route,
                "max_rounds": max_rounds,
            }
            self.store.append_history(
                {
                    "ts": now_iso(),
                    "event": "PLANNER_BUDGET_DECISION",
                    **decision,
                }
            )
            return decision

        unavailable_evidence_counts = set()
        from itertools import count
        rounds = count() if max_rounds is None else range(observation_round_limit + 1)
        for round_idx in rounds:
            if round_idx >= observation_round_limit:
                break
            if self.client.debug:
                print(f"\n{'#'*80}")
                print(f"🔄 ROUND {round_idx + 1} / {max_rounds}")
                print(f"{'#'*80}\n")

            acquisition = getattr(plan, "acquisition", None) or {}
            current_evidence = None
            if acquisition.get("action") == "unavailable":
                evidence_count = len(self.bb.evidences)
                self.store.append_history({"ts": now_iso(), "event": "OBSERVATION_NEED_UNAVAILABLE",
                    "acquisition": acquisition, "need_id": plan.need_id, "need_revision": plan.need_revision})
                if (not getattr(self.admission_runtime, "need_local_audit_enabled", False)
                        and evidence_count in unavailable_evidence_counts):
                    observation_need_stopped = True
                    observation_stop_reason = "acquisition_no_progress"
                    break
                unavailable_evidence_counts.add(evidence_count)
                admission_event = {"status": "ACQUISITION_UNAVAILABLE", "reason": acquisition["reason"]}
                if not getattr(self.admission_runtime, "need_local_audit_enabled", False):
                    admission_event = self._verify_admitted_observation(query, plan, None, admission_event)
            elif acquisition.get("action") == "reuse":
                admission_event = {"status": "EXISTING_EVIDENCE_REUSED", "reused_claim_ids": acquisition["claim_ids"]}
                if not getattr(self.admission_runtime, "need_local_audit_enabled", False):
                    admission_event = self._verify_admitted_observation(query, plan, None, admission_event)
                self.store.append_history({"ts": now_iso(), "event": "OBSERVATION_REUSED",
                    "acquisition": acquisition, "admission": admission_event})
            elif initial_audited_observation is not None and round_idx == 0:
                ev = initial_evidence
                admission_event = self._verify_admitted_observation(
                    query, plan, ev, initial_admission
                )
                self.store.append_history({
                    "ts": now_iso(), "event": "EVENT_PREFILL_DECISION",
                    "round_id": ev.round_id, "admission": admission_event,
                })
            else:
                observation_plan, event_assist_preparation = (
                    self._prepare_event_assisted_observation(plan)
                )
                repeat_check = getattr(
                    self.admission_runtime,
                    "observation_plan_is_mechanical_repeat",
                    None,
                ) if self.admission_runtime is not None else None
                if (not getattr(self.admission_runtime, "need_acquisition_enabled", False)
                        and callable(repeat_check) and repeat_check(observation_plan)):
                    reject_repeat = getattr(
                        self.admission_runtime,
                        "reject_mechanical_observation_repeat",
                        None,
                    )
                    if callable(reject_repeat):
                        reject_repeat(plan)
                    self.store.append_history(
                        {
                            "ts": now_iso(),
                            "event": "OBSERVATION_SKIPPED_MECHANICAL_REPEAT",
                            "need_id": getattr(plan, "need_id", None),
                            "need_revision": getattr(plan, "need_revision", None),
                        }
                    )
                    observation_need_stopped = True
                    break

                # Both modalities use this action budget and the same admission/Audit loop.
                if acquisition.get("action") == "asr":
                    asr = getattr(self.admission_runtime, "asr_context", None)
                    if asr is None:
                        raise ValueError("asr_mode_not_enabled")
                    ev = asr.acquire(observation_plan)
                else:
                    if observation_plan.watch is None:
                        raise ValueError("visual_observation_watch_required")
                    observer = Observer(self.client)
                    ev = observer.observe(
                        observation_plan, self._observation_blackboard()
                    )
                ev.round_id = round_idx + 1 + overview_round_offset
                self._annotate_event_assisted_observation(
                    ev, event_assist_preparation
                )
                admission_event = self._admit_observation(
                    query, observation_plan, ev
                )
                current_evidence = ev
                record_plan = getattr(
                    self.admission_runtime,
                    "record_completed_observation_plan",
                    None,
                ) if self.admission_runtime is not None else None
                if callable(record_plan):
                    record_plan(observation_plan, ev, admission_event)
                self.bb.add_evidence(ev)
                ev_dict = dataclasses.asdict(ev)
                try:
                    interval_map = {}
                    for item in ev.key_evidence:
                        if isinstance(item, dict):
                            s = item.get("timestamp_start")
                            e = item.get("timestamp_end")
                            d = item.get("description", "")
                            if s is not None and e is not None:
                                interval_map[f"[{int(s)},{int(e)}]"] = d or ""
                    ev_dict["interval_map"] = interval_map
                except Exception:
                    pass
                self.store.write_json(self.store.evidence_json(ev.round_id), ev_dict)
                # Log timestamps (midpoints)
                timestamps = []
                for e in ev.key_evidence:
                    if isinstance(e, dict):
                        ts_start = e.get("timestamp_start")
                        ts_end = e.get("timestamp_end")
                        if ts_start is not None and ts_end is not None:
                            timestamps.append((ts_start + ts_end) / 2.0)
                        elif ts_start is not None:
                            timestamps.append(ts_start)
                history_event = {
                    "ts": now_iso(), "event": "OBSERVE_ROUND_END", "round_id": ev.round_id,
                    "evidence_path": str(self.store.evidence_json(ev.round_id)),
                    "timestamps": timestamps,
                }
                if admission_event is not None:
                    history_event["admission"] = admission_event
                self.store.append_history(history_event)

            is_normal_last_round = round_idx == observation_round_limit - 1
            is_last_round = (
                is_normal_last_round or round_idx >= observation_round_limit
            )
            options = self.bb.meta.get("options", None)
            if getattr(self.admission_runtime, "need_local_audit_enabled", False):
                from experiments.semantic_delta_pilot import need_answer_audit
                need_answer = need_answer_audit.audit(self.admission_runtime, plan,
                    limited_reason="Total observation action budget exhausted." if is_last_round else None)
                self.store.append_history({"ts": now_iso(), "event": "NEED_LOCAL_RESULT", "need_answer": need_answer})
                if self.admission_runtime.can_dispatch_repair():
                    repair_need = self.admission_runtime.repair_justification()
                    plan = self._replan(query, options=options, justification=repair_need,
                                        event="NEED_REPAIR_PLAN")
                    self.admission_runtime.mark_repair_dispatched()
                    continue
                admission_event = self._verify_admitted_observation(query, plan, current_evidence, admission_event)
            if self.admission_runtime is not None and self.admission_runtime.has_blocking_delta():
                if not is_last_round and self.admission_runtime.can_dispatch_repair():
                    repair_need = self.admission_runtime.repair_justification()
                    plan = self._replan(
                        query,
                        options=options,
                        justification=repair_need,
                        event="ADMISSION_REPAIR_PLAN",
                    )
                    self.admission_runtime.mark_repair_dispatched()
                    continue
                blocked_event = (
                    "ADMISSION_REPAIR_BUDGET_EXHAUSTED"
                    if self.admission_runtime.repair_budget_exhausted()
                    else "ADMISSION_BLOCKED_NO_CONTINUATION"
                )
                self.store.append_history(
                    {
                        "ts": now_iso(),
                        "event": blocked_event,
                        "admission": self.admission_runtime.summary(),
                    }
                )
                if is_last_round:
                    budget_force_decision = authorize_budget_forced_answer(
                        "Observation budget exhausted with an unresolved evidence "
                        "obligation.",
                        route="ADMISSION_BLOCKED",
                    )
                break

            answer_route_fn = getattr(
                self.admission_runtime,
                "answer_route",
                None,
            ) if self.admission_runtime is not None else None
            answer_route = answer_route_fn() if callable(answer_route_fn) else None
            if answer_route in {"SEARCH", "VERIFIER_GAP"}:
                route_reason_fn = getattr(
                    self.admission_runtime,
                    "answer_route_justification",
                    None,
                )
                route_reason = (
                    route_reason_fn()
                    if callable(route_reason_fn)
                    else "Audited hypotheses are not ready for Answer."
                )
                self.store.append_history(
                    {
                        "ts": now_iso(),
                        "event": "ANSWER_SKIPPED_BY_HYPOTHESIS_ROUTE",
                        "route": answer_route,
                        "justification": route_reason,
                    }
                )
                if is_last_round:
                    budget_force_decision = authorize_budget_forced_answer(
                        route_reason,
                        route=answer_route,
                    )
                    break
                plan = self._replan(
                    query,
                    options=options,
                    justification=route_reason,
                    event="HYPOTHESIS_ROUTE_REPLAN",
                )
                continue
            
            # Reflect & verify
            reflector = Reflector(self.client)
            evidence_list = self._reasoning_evidence_list()
            interaction_history = self.store.get_interaction_history()
            
            decision_plan = copy.deepcopy(plan)
            if getattr(self.admission_runtime, "need_acquisition_enabled", False):
                decision_plan.decision_context = {
                    "hypothesis_states": self.admission_runtime._last_hypothesis_results,
                    "active_need": self.admission_runtime._active_observation_need,
                    "need_answers": getattr(self.admission_runtime, "_need_answers", []),
                    "acquisition": acquisition}
                if getattr(self.admission_runtime, "asr_initial_analysis", ""):
                    decision_plan.decision_context["initial_asr_analysis"] = self.admission_runtime.asr_initial_analysis
            reflection = reflector.reflect(
                query=query,
                plan=decision_plan,
                evidence_list=evidence_list,
                interaction_history=interaction_history,
                video_path=self.bb.video_path,
                duration_sec=self.bb.duration_sec,
                is_last_round=is_last_round,
                options=options,
                option_candidates=self.bb.meta.get("option_candidates"),
            )
            
            if reflection.get("candidate_answer") is not None or asr_initial is None:
                decision_candidate = copy.deepcopy(reflection.get("candidate_answer"))

            # Record the reflector's confidence and justification BEFORE any
            # early exit, so EXTRACTANSWER / FORCEANSWER runs still persist them.
            query_confidence = reflection.get("query_confidence")
            if query_confidence is not None:
                self.bb.query_confidence = query_confidence
                if self.client.debug:
                    print(f"📊 Query-level confidence: {query_confidence:.2f}")
            justification = reflection.get("justification")
            self.store.append_history({
                "ts": now_iso(),
                "event": reflection.get("event", "REFLECTION"),
                "reflection": reflection,
                "justification": justification,
                "query_confidence": query_confidence
            })

            # Final answer produced by the reflector (EXTRACTANSWER when
            # confident, or FORCEANSWER on the last round).
            if reflection.get("final_answer"):
                candidate_final = reflection.get("final_answer")
                if getattr(self.admission_runtime, "need_acquisition_enabled", False):
                    forced = bool(is_last_round and reflection.get("event") == "FINAL_ANSWER_GENERATED")
                    candidate_final["forced_at_budget"] = forced
                    if forced:
                        candidate_final["planner_budget_decision"] = authorize_budget_forced_answer(
                            "Final action round reached; Reflector used FORCEANSWER.", route=answer_route)
                if self.admission_runtime is None:
                    final_answer_from_reflection = candidate_final
                    final_answer_admitted = True
                elif self.admission_runtime.audit_answer_candidate(
                    candidate_final,
                    question=query,
                    options=options,
                ):
                    final_answer_from_reflection = candidate_final
                    final_answer_admitted = True
                else:
                    if not is_last_round and self.admission_runtime.can_dispatch_repair():
                        repair_need = self.admission_runtime.repair_justification(justification)
                        plan = self._replan(
                            query,
                            options=options,
                            justification=repair_need,
                            event="ANSWER_COMMITMENT_REPAIR_PLAN",
                        )
                        self.admission_runtime.mark_repair_dispatched()
                        continue
                    if is_last_round:
                        budget_force_decision = authorize_budget_forced_answer(
                            "The final Answer candidate was not admitted and no "
                            "observation budget remains.",
                            route="ANSWER_COMMITMENT_REJECTED",
                        )
                if final_answer_admitted:
                    if self.client.debug:
                        print(f"✅ Final answer produced by reflector "
                              f"({reflection.get('event', 'REFLECTION')}).")
                    plan.complete = True
                break

            if reflection.get("sufficient", False):
                if self.client.debug:
                    print(f"✅ Evidence sufficient after verification (query_confidence={query_confidence:.2f}).")
                plan.complete = True
                break
            
            if (not is_last_round and getattr(self.admission_runtime, "need_acquisition_enabled", False)
                    and self.admission_runtime.active_observation_need() is None):
                next_need = self.admission_runtime.prepare_decision_observation_need(
                    question=query, option_candidates=self.bb.meta.get("option_candidates") or [],
                    feedback=justification)
                if next_need is None:
                    observation_need_stopped = True
                    observation_stop_reason = "decision_no_observable_need"
                    break

            # Not sufficient → replan with all evidence and history (only if not last round)
            if not is_last_round:
                if self.client.debug:
                    print(f"❌ Evidence insufficient. Replanning with all evidence and history...")
                
                plan = self._replan(
                    query,
                    options=options,
                    justification=justification,
                    event="REFLECTION_REPLAN",
                )

        # Preserve ASR input for the existing synthesis fallback, without giving it to Observe.
        if getattr(self.admission_runtime, "asr_initial_decision", None):
            plan.decision_context = copy.deepcopy(getattr(plan, "decision_context", None) or {})
            plan.decision_context["initial_asr_analysis"] = self.admission_runtime.asr_initial_analysis
            if not self.bb.evidences:
                plan.decision_context["asr_transcript"] = self.admission_runtime.asr_seed_text

        # final answer
        if final_answer_from_reflection is not None:
            # Use final answer generated in last round reflection
            final = final_answer_from_reflection
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"✅ USING FINAL ANSWER FROM LAST ROUND REFLECTION")
                print(f"{'='*80}")
                print(f"Total Rounds: {len(self.bb.evidences)}")
                print(f"Total Evidence Items: {sum(len(ev.key_evidence) for ev in self.bb.evidences)}")
                print(f"{'='*80}\n")
        elif (observation_need_stopped
                and getattr(self.admission_runtime, "need_acquisition_enabled", False)
                and self.bb.meta.get("options")):
            # Stopping acquisition is a decision to answer with current evidence,
            # not a reason to erase a low-confidence candidate.
            option_texts = {chr(ord("A") + i): text
                            for i, text in enumerate(self.bb.meta["options"])}
            final = copy.deepcopy(decision_candidate or {})
            selected = str(final.get("selected_option") or "").strip().upper()
            if selected not in option_texts:
                synthesis_bb = self._planning_blackboard()
                synthesis_bb.meta = copy.deepcopy(synthesis_bb.meta)
                synthesis_bb.meta["force_best_effort"] = True
                final = self.client.synthesize_final_answer(plan, synthesis_bb)
                selected = str(final.get("selected_option") or "").strip().upper()
            if selected not in option_texts:
                # Explicit user policy: a marked placeholder beats an empty MCQ.
                selected = next(iter(option_texts))
                final = {"selected_option": selected, "confidence": 0.0,
                         "query_confidence": self.bb.query_confidence,
                         "answer_placeholder": True,
                         "reasoning": "No legal option was produced after acquisition stopped; "
                                      "the first supplied option is a placeholder, not an evidence-supported answer."}
            final["selected_option"] = selected
            final["selected_option_text"] = option_texts[selected]
            self.admission_runtime.audit_answer_candidate(
                final, question=query, options=self.bb.meta["options"])
            final.update(status="ANSWERED", answer_mode="best_effort",
                         forced_at_budget=False, admission_unresolved=True,
                         observation_stop_reason=observation_stop_reason)
            self.store.append_history({"ts": now_iso(), "event": "STOPPED_NEED_ANSWER",
                "reason": observation_stop_reason, "selected_option": selected,
                "answer_placeholder": bool(final.get("answer_placeholder"))})
        elif observation_need_stopped and self.admission_runtime is not None:
            final = self.admission_runtime.unresolved_final()
            final["error_code"] = observation_stop_reason
            if observation_stop_reason == "observation_need_unavailable":
                final["reasoning"] = acquisition["reason"]
                final["observation_need"] = {"need_id": plan.need_id,
                    "status": "UNSUPPORTED", "missing_proposition": plan.query}
        elif budget_force_decision is not None:
            synthesis_bb = self._planning_blackboard()
            synthesis_bb.meta = copy.deepcopy(synthesis_bb.meta)
            synthesis_bb.meta["force_best_effort"] = True
            admission_summary = (
                self.admission_runtime.summary()
                if self.admission_runtime is not None
                else None
            )
            synthesis_bb.meta["budget_forced_context"] = {
                "planner_decision": copy.deepcopy(budget_force_decision),
                "admission_summary": admission_summary,
            }
            final = self.client.synthesize_final_answer(plan, synthesis_bb)
            selected_option = str(final.get("selected_option") or "").strip().upper()
            valid_options = {
                chr(ord("A") + index)
                for index, _ in enumerate(self.bb.meta.get("options") or [])
            }
            if valid_options and selected_option not in valid_options:
                self.store.append_history({"ts": now_iso(), "event": "BUDGET_ANSWER_NO_LEGAL_OPTION",
                                           "candidate": copy.deepcopy(final)})
                incumbent = copy.deepcopy(decision_candidate or {})
                incumbent_option = str(incumbent.get("selected_option") or "").strip().upper()
                if incumbent_option in valid_options:
                    final = incumbent
                    selected_option = incumbent_option
                else:
                    # Same explicit nonempty-MCQ policy as stopped-need answers.
                    selected_option = "A"
                    final["answer_placeholder"] = True
                    final["confidence"] = 0.0
                    final["reasoning"] = (
                        "No legal option was produced after final synthesis; the first supplied "
                        "option is a placeholder, not an evidence-supported answer. "
                        + str(final.get("reasoning") or ""))
            if valid_options:
                final["selected_option"] = selected_option
                final["selected_option_text"] = self.bb.meta["options"][ord(selected_option) - ord("A")]
                final["status"] = "ANSWERED"
            else:
                final["status"] = "FAILED"
                final["error_code"] = "budget_forced_answer_invalid_selection"
            final["answer_mode"] = "budget_forced"
            final["forced_at_budget"] = True
            final["admission_unresolved"] = True
            final["planner_budget_decision"] = copy.deepcopy(budget_force_decision)
        elif self.admission_runtime is not None and self.admission_runtime.has_blocking_delta():
            final = self.admission_runtime.unresolved_final()
        else:
            # Generate final answer using synthesis
            if self.client.debug:
                print(f"\n{'='*80}")
                print(f"🎯 SYNTHESIZING FINAL ANSWER")
                print(f"{'='*80}")
                print(f"Total Rounds: {len(self.bb.evidences)}")
                print(f"Total Evidence Items: {sum(len(ev.key_evidence) for ev in self.bb.evidences)}")
                print(f"{'='*80}\n")
            
            synthesis_bb = self._reasoning_blackboard()
            candidate_final = self.client.synthesize_final_answer(plan, synthesis_bb)
            if self.admission_runtime is None:
                final = candidate_final
            elif self.admission_runtime.audit_answer_candidate(
                candidate_final,
                question=query,
                options=self.bb.meta.get("options", None),
            ):
                final = candidate_final
                final_answer_admitted = True
            else:
                final = self.admission_runtime.unresolved_final()

        if self.admission_runtime is not None:
            final["admission_summary"] = self.admission_runtime.summary()
        
        # Extract final answer text from MCQ format
        # Prefer selected_option_text, fallback to reasoning
        final_answer_text = final.get("selected_option_text", "") or final.get("reasoning", "")
        plan.final_answer = final_answer_text
        final["query"] = query  # Add query to final result
        self.store.write_json(self.store.final_answer, final)
        self.store.append_history({"ts": now_iso(), "event": "SYNTHESIZE_ANSWER_END", "final_path": str(self.store.final_answer)})
        
        if self.client.debug:
            print(f"\n{'='*80}")
            print(f"✅ FINAL ANSWER")
            print(f"{'='*80}")
            print(f"Selected Option: {final.get('selected_option', 'N/A')}")
            print(f"Option Text: {final.get('selected_option_text', '')}")
            print(f"Reasoning: {final.get('reasoning', '')[:300]}...")
            print(f"Confidence: {final.get('confidence', 0.0):.2f}")
            # Print evidence timestamps if available
            evidence_timestamps = final.get('evidence_timestamps', [])
            if evidence_timestamps:
                print(f"Evidence Timestamps ({len(evidence_timestamps)} range(s)):")
                for i, ts in enumerate(evidence_timestamps, 1):
                    ts_start = ts.get('timestamp_start', 'N/A')
                    ts_end = ts.get('timestamp_end', 'N/A')
                    desc = ts.get('description', '')
                    if isinstance(ts_start, (int, float)) and isinstance(ts_end, (int, float)):
                        if desc:
                            print(f"  {i}. [{ts_start:.1f}s - {ts_end:.1f}s]: {desc}")
                        else:
                            print(f"  {i}. [{ts_start:.1f}s - {ts_end:.1f}s]")
            print(f"{'='*80}\n")
        
        # Save full conversation history
        self.store.save_conversation_history(query, plan, self.bb, final)

        return {"plan": dataclasses.asdict(plan), "final": final}

    # ---------- helpers ----------
    def _load_latest_plan(self) -> Optional[PlanSpec]:
        if self.store.plan_initial.exists():
            data = json.loads(self.store.plan_initial.read_text())
            plan = plan_from_dict(data)
            # apply the last updated snapshot if present
            k = last_plan_index(self.store)
            if k is not None:
                up = json.loads(self.store.plan_updated(k).read_text())
                plan = plan_from_dict(up)
            return plan
        return None


# ======================================================
# Helpers & Builders
# ======================================================
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def plan_from_dict(d: Dict[str, Any]) -> PlanSpec:
    """Parse PlanSpec from dict (supports both old steps format and new watch format)."""
    if d.get("watch", {}) is None:
        acquisition = d.get("acquisition") or {}
        if acquisition.get("action") != "asr" and not (d.get("complete") is True and not acquisition):
            raise ValueError("missing_watch_requires_asr_acquisition")
        return PlanSpec(plan_version=d.get("plan_version", "v1"), query=d.get("query", ""),
            watch=None, description=d.get("description", ""),
            completion_criteria=d.get("completion_criteria", ""),
            need_id=d.get("need_id"), need_revision=d.get("need_revision"),
            final_answer=d.get("final_answer"), complete=bool(d.get("complete", False)),
            acquisition=copy.deepcopy(acquisition))
    # New format: watch directly in plan
    if "watch" in d:
        watch_data = d["watch"]
        spatial_rate_str = str(watch_data["spatial_token_rate"]).strip().lower()
        try:
            spatial_token_rate = SpatialTokenRate(spatial_rate_str)
        except ValueError:
            spatial_token_rate = SpatialTokenRate.low
        
        watch = WatchConfig(
            load_mode=watch_data["load_mode"],
            fps=float(watch_data["fps"]),
            spatial_token_rate=spatial_token_rate,
            regions=[(float(a), float(b)) for a, b in watch_data.get("regions", [])]
        )
        
        return PlanSpec(
            plan_version=d.get("plan_version", "v1"),
            query=d.get("query", ""),
            watch=watch,
            description=d.get("description", ""),
            completion_criteria=d.get("completion_criteria", ""),
            need_id=d.get("need_id"),
            need_revision=d.get("need_revision"),
            final_answer=d.get("final_answer"),
            complete=bool(d.get("complete", False)),
            acquisition=copy.deepcopy(d.get("acquisition")),
        )
    
    # Legacy format: steps array (take first step)
    elif "steps" in d and d["steps"]:
        s = d["steps"][0]  # Take first step only
        spatial_rate_str = str(s["watch"]["spatial_token_rate"]).strip().lower()
        try:
            spatial_token_rate = SpatialTokenRate(spatial_rate_str)
        except ValueError:
            spatial_token_rate = SpatialTokenRate.low
        
        watch = WatchConfig(
            load_mode=s["watch"]["load_mode"],
            fps=float(s["watch"]["fps"]),
            spatial_token_rate=spatial_token_rate,
            regions=[(float(a), float(b)) for a, b in s["watch"].get("regions", [])]
        )
        
        return PlanSpec(
            plan_version=d.get("plan_version", "v1"),
            query=d.get("query", ""),
            watch=watch,
            description=s.get("description", ""),
            completion_criteria=d.get("completion_criteria", ""),
            need_id=d.get("need_id"),
            need_revision=d.get("need_revision"),
            final_answer=d.get("final_answer"),
            complete=bool(d.get("complete", False)),
            acquisition=copy.deepcopy(d.get("acquisition")),
        )
    
    # Fallback: empty plan
    return PlanSpec(
        plan_version="v1",
        query=d.get("query", ""),
        watch=WatchConfig(load_mode="uniform", fps=0.5, spatial_token_rate=SpatialTokenRate.low),
        description=""
    )


def last_plan_index(store: Store) -> Optional[int]:
    pattern = re.compile(r"plan\.updated\.(\d+)\.json$")
    max_k = None
    for p in store.root.glob("plan.updated.*.json"):
        m = pattern.search(p.name)
        if m:
            k = int(m.group(1))
            max_k = k if (max_k is None or k > max_k) else max_k
    return max_k


def next_plan_index(store: Store) -> int:
    k = last_plan_index(store)
    return 1 if k is None else k + 1


# ======================================================
# CLI
# ======================================================
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Agentic Video Understanding (Separated Loop)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("plan")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--video", required=True)
    a.add_argument("--query", required=True)
    a.add_argument("--model", default="gemini-2.0-flash-exp")

    b = sub.add_parser("run")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--video", required=True)
    b.add_argument("--query", required=True)
    b.add_argument("--max-rounds", type=int, default=3)

    e = sub.add_parser("show")
    e.add_argument("--run-dir", required=True)

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "plan":
        client = GeminiClient()
        ctl = Controller(run_dir=args.run_dir, video_path=args.video, client=client)
        plan = ctl.plan(args.query)
        print(json.dumps(dataclasses.asdict(plan), indent=2))
        return

    if args.cmd == "run":
        client = GeminiClient()
        ctl = Controller(run_dir=args.run_dir, video_path=args.video, client=client)
        result = ctl.run(query=args.query, max_rounds=args.max_rounds)
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "show":
        store = Store(args.run_dir)
        if not store.meta.exists():
            raise SystemExit("Run not initialized.")
        meta = json.loads(store.meta.read_text())
        client = GeminiClient()
        ctl = Controller(run_dir=args.run_dir, video_path=meta["video_path"], client=client)
        plan = ctl._load_latest_plan()
        plan_dict = dataclasses.asdict(plan) if plan else None
        ev_summary = [dataclasses.asdict(v) for v in ctl.bb.evidences]
        print(json.dumps({"meta": meta, "plan": plan_dict, "evidences": ev_summary}, indent=2))
        return


if __name__ == "__main__":
    main()
