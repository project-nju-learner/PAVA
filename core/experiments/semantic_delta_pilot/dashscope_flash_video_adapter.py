"""Experiment-local Qwen3.8-Flash video transport for AVP.

AVP still chooses the temporal scope and creates its normal video parts. A
COARSE full-video locator uses 0.1 fps.  When the Planner gives an explicit
answer-critical whole-video coverage reason, a denser uniform observation is
split into bounded source ranges and executed concurrently.  No temporary
video clips are created.  It never reads a credential file directly.
"""

from __future__ import annotations

import base64
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any, Callable, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google.genai.types import FileData, Part, VideoMetadata

from avp.main import Evidence, GeminiClient, WatchConfig
from avp.video_utils import get_mime_type


OPENAI_COMPATIBLE_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
REQUIRED_MODEL = os.environ.get("PAVA_MODEL", "qwen3.8-flash")
FIXED_VIDEO_FPS = 0.1
MIN_VIDEO_FRAMES = 4
MAX_VIDEO_FRAMES = 250
MAX_VIDEO_DURATION_SEC = 40 * 60
TARGET_VIDEO_FRAMES_PER_PARTITION = 200
MAX_PARALLEL_VIDEO_REQUESTS = 8
SOURCE_SCOPE_TIME_TOLERANCE_SEC = 1e-3


def assert_dashscope_flash_backend(*, base_url: str, model: str) -> None:
    if str(model).strip().lower() != REQUIRED_MODEL.lower():
        raise ValueError("qwen3.8_flash_required")
    if not str(base_url).strip().startswith(("https://", "http://")):
        raise ValueError("configured_http_base_url_required")


@dataclass(frozen=True)
class _GenerateResponse:
    text: str
    observation_receipt: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class _RegisteredFile:
    uri: str
    name: str


@dataclass(frozen=True)
class _SubmittedVideo:
    content_part: dict[str, Any]
    source_path: str
    frame_count: int
    # The Planner-owned request remains frozen separately from the effective
    # sampling rate used to satisfy the provider's four-frame minimum.
    fps: float
    submitted_fps: float
    source_start_sec: float
    source_end_sec: float


@dataclass(frozen=True)
class _ExplicitFrameSequence:
    jpeg_frames: tuple[bytes, ...]
    playback_fps: float = 1.0


class _DashScopeModels:
    def __init__(self, transport: "DashScopeFlashTransport") -> None:
        self._transport = transport

    def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any = None,
    ) -> _GenerateResponse:
        del config
        if str(model).strip().lower() != REQUIRED_MODEL.lower():
            raise ValueError("qwen3.8_flash_required")
        content_parts, videos = self._transport.translate_contents(contents)
        payload: dict[str, Any] = {
            "model": REQUIRED_MODEL,
            "messages": [{"role": "user", "content": content_parts}],
            "stream": False,
        }
        payload.update(self._transport.generation_overrides)
        payload["enable_thinking"] = self._transport.generation_stage == "answer"
        forbidden = {"mm_processor_kwargs", "media_io_kwargs"} & set(payload)
        if forbidden:
            raise ValueError(f"vllm_only_payload_fields_forbidden:{sorted(forbidden)}")
        pending = self._transport.begin_observation_receipt(videos)
        response_text = self._transport.post_chat_completion(payload)
        receipt = self._transport.finish_observation_receipt(pending)
        return _GenerateResponse(response_text, observation_receipt=receipt)


class DashScopeFlashTransport:
    """OpenAI-compatible transport using at most 250 base64 JPEG frames."""

    def __init__(
        self,
        *,
        base_url: str = OPENAI_COMPATIBLE_BASE_URL,
        generation_overrides: Optional[Mapping[str, Any]] = None,
        request_opener: Callable[..., Any] = urlopen,
        duration_probe: Optional[Callable[[Path], float]] = None,
        frame_extractor: Optional[
            Callable[[Path, float, float, float], list[bytes]]
        ] = None,
    ) -> None:
        assert_dashscope_flash_backend(base_url=base_url, model=REQUIRED_MODEL)
        self.base_url = str(base_url).strip().rstrip("/")
        self.generation_overrides = dict(generation_overrides or {})
        if {"mm_processor_kwargs", "media_io_kwargs"} & set(
            self.generation_overrides
        ):
            raise ValueError("vllm_only_generation_override_forbidden")
        self._request_opener = request_opener
        self._duration_probe = duration_probe or self._probe_duration
        self._frame_extractor = frame_extractor or self._extract_jpeg_frames
        self._temporary_media = tempfile.TemporaryDirectory(
            prefix="avp_dashscope_video_"
        )
        self._registered_paths: dict[str, Path] = {}
        self._registered_ids: dict[Path, str] = {}
        self._pending_source_units: Optional[list[dict[str, Any]]] = None
        self._request_count = 0
        self._frame_batch_count = 0
        self._generation_stage: Optional[str] = None
        self.observation_receipts_enabled = False
        self.models = _DashScopeModels(self)

    @property
    def generation_stage(self) -> Optional[str]:
        return self._generation_stage

    def set_generation_stage(self, stage: Optional[str]) -> None:
        normalized = str(stage or "").strip().lower() or None
        if normalized not in {None, "answer"}:
            raise ValueError("unsupported_generation_stage")
        self._generation_stage = normalized

    def register_media(self, path: str | Path) -> str:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("registered_media_must_be_a_file")
        media_id = self._registered_ids.get(resolved)
        if media_id is None:
            media_id = f"media_{len(self._registered_paths) + 1}"
            self._registered_ids[resolved] = media_id
            self._registered_paths[media_id] = resolved
        return f"avp-dashscope-local://{media_id}"

    def _resolve_registered(self, uri: str) -> Path:
        prefix = "avp-dashscope-local://"
        if not str(uri).startswith(prefix):
            raise ValueError("unregistered_media_uri")
        path = self._registered_paths.get(str(uri)[len(prefix) :])
        if path is None:
            raise ValueError("unknown_registered_media")
        return path.resolve(strict=True)

    def set_observation_source_units(
        self,
        *,
        video_path: str,
        units: list[dict[str, Any]],
    ) -> None:
        if not self.observation_receipts_enabled:
            return
        if self._pending_source_units is not None:
            raise RuntimeError("observation_source_units_already_pending")
        source_video = str(Path(video_path).resolve(strict=True))
        normalized: list[dict[str, Any]] = []
        for unit in units:
            start = float(unit["start"])
            end = float(unit["end"])
            fps = float(unit["fps"])
            if start < 0 or end <= start or fps <= 0:
                raise ValueError("invalid_observation_source_unit")
            normalized.append(
                {
                    "source_video_path": source_video,
                    "source_start_sec": start,
                    "source_end_sec": end,
                    "fps": fps,
                }
            )
        if not normalized:
            raise ValueError("observation_source_units_required")
        self._pending_source_units = normalized

    @staticmethod
    def _video_metadata_fps(item: Any) -> float:
        metadata = getattr(item, "video_metadata", None)
        raw = getattr(metadata, "fps", None) if metadata is not None else None
        if raw is None:
            raise ValueError("dashscope_video_requires_explicit_fps")
        fps = float(raw)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("dashscope_video_requires_positive_fps")
        return fps

    def _path_from_part(self, item: Any) -> Path:
        file_data = getattr(item, "file_data", None)
        inline_data = getattr(item, "inline_data", None)
        if file_data is not None and getattr(file_data, "file_uri", None):
            return self._resolve_registered(str(file_data.file_uri))
        if inline_data is not None and getattr(inline_data, "data", None) is not None:
            raise ValueError("dashscope_direct_range_requires_registered_source_path")
        raise TypeError(f"unsupported_video_part:{type(item).__name__}")

    @staticmethod
    def _duration_seconds(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        total_seconds = getattr(value, "total_seconds", None)
        if callable(total_seconds):
            return float(total_seconds())
        seconds = getattr(value, "seconds", None)
        nanos = getattr(value, "nanos", None)
        if seconds is not None:
            return float(seconds) + float(nanos or 0) / 1_000_000_000.0
        text = str(value).strip()
        if text.endswith("s"):
            text = text[:-1]
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"invalid_video_offset:{value!r}") from exc

    @classmethod
    def _metadata_offsets(cls, item: Any) -> tuple[Optional[float], Optional[float]]:
        metadata = getattr(item, "video_metadata", None)
        if metadata is None:
            return None, None
        start = getattr(metadata, "start_offset", None)
        end = getattr(metadata, "end_offset", None)
        if start is None:
            start = getattr(metadata, "startOffset", None)
        if end is None:
            end = getattr(metadata, "endOffset", None)
        return cls._duration_seconds(start), cls._duration_seconds(end)

    @staticmethod
    def _probe_duration(path: Path) -> float:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError("ffprobe_video_duration_failed")
        duration = float(completed.stdout.strip())
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("ffprobe_video_duration_invalid")
        return duration

    def _source_scope(self, path: Path, item: Any) -> tuple[float, float]:
        duration = self._duration_probe(path)
        start, end = self._metadata_offsets(item)
        start = max(0.0, float(start or 0.0))
        end = min(duration, float(end if end is not None else duration))
        if end <= start:
            raise ValueError("invalid_submitted_video_scope")
        if end - start > MAX_VIDEO_DURATION_SEC + 1e-6:
            raise ValueError("submitted_video_scope_exceeds_40_minutes")
        return start, end

    def _extract_jpeg_frames(
        self,
        path: Path,
        start_sec: float,
        end_sec: float,
        fps: float,
    ) -> list[bytes]:
        self._frame_batch_count += 1
        frame_dir = Path(self._temporary_media.name) / (
            f"frames_{self._frame_batch_count:06d}"
        )
        frame_dir.mkdir(parents=True, exist_ok=False)
        pattern = frame_dir / "frame_%06d.jpg"
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_sec:.6f}",
                "-i",
                str(path),
                "-t",
                f"{end_sec - start_sec:.6f}",
                "-map",
                "0:v:0",
                "-vf",
                (
                    f"fps={fps:.12g},"
                    "scale=512:512:force_original_aspect_ratio=decrease:"
                    "force_divisible_by=2"
                ),
                "-frames:v",
                str(MAX_VIDEO_FRAMES + 1),
                "-q:v",
                "5",
                "-y",
                str(pattern),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise RuntimeError("ffmpeg_direct_range_frame_extraction_failed")
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        return [frame.read_bytes() for frame in frames]

    def prepare_uniform_seed(self, video_path: Path, duration: float, output_dir: Path):
        """Reuse the tested uniform250 extraction; retain frame/time/source mapping."""
        video_path = video_path.resolve()
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("uniform_seed_duration_invalid")
        count = int(os.environ.get('PAVA_INITIAL_FRAMES', '250'))
        if not 4 <= count <= MAX_VIDEO_FRAMES:
            raise ValueError('initial_frames_must_be_4_to_250')
        fps = count / duration
        source_stat = video_path.stat()
        binding = dict(video_path=str(video_path), source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns, frame_count=count,
            start_sec=0.0, end_sec=duration, fps=fps,
            method="existing_ffmpeg_uniform_fps_filter",
            frame_time_semantics="regular sampling grid, not measured source-frame PTS",
            estimated_sample_times_sec=[i / fps for i in range(count)],
            pixel_size="existing adapter: long side at most 512")
        manifest_path = output_dir / "manifest.json"
        paths = [output_dir / f"frame_{i+1:03d}.jpg" for i in range(count)]
        reused = manifest_path.is_file()
        if reused:
            if json.loads(manifest_path.read_text()) != binding:
                raise ValueError("uniform_seed_cache_source_mismatch")
            frames = [path.read_bytes() for path in paths]
        else:
            frames = self._frame_extractor(video_path, 0.0, duration, fps)
            if len(frames) != count:
                raise RuntimeError(f"uniform_expected_{count}_frames_received_{len(frames)}")
            output_dir.mkdir(parents=True, exist_ok=True)
            for path, frame in zip(paths, frames):
                path.write_bytes(frame)
            manifest_path.write_text(json.dumps(binding, indent=2) + "\n")
        content = []
        for i, frame in enumerate(frames):
            content.extend([f"Frame {i+1}, approximate video time {i / fps:.3f} seconds.",
                Part.from_bytes(data=frame, mime_type="image/jpeg")])
        return content, dict(binding, manifest_path=str(manifest_path),
            frames_reused=reused, ground_truth_loaded=False)

    def translate_contents(
        self,
        contents: Any,
    ) -> tuple[list[dict[str, Any]], list[_SubmittedVideo]]:
        items = contents if isinstance(contents, (list, tuple)) else [contents]
        translated: list[dict[str, Any]] = []
        videos: list[_SubmittedVideo] = []
        for item in items:
            if isinstance(item, _ExplicitFrameSequence):
                frame_count = len(item.jpeg_frames)
                if frame_count < MIN_VIDEO_FRAMES or frame_count > MAX_VIDEO_FRAMES:
                    raise ValueError(
                        f"video_frame_count_outside_official_bounds:{frame_count}"
                    )
                if not math.isfinite(float(item.playback_fps)) or item.playback_fps <= 0:
                    raise ValueError("explicit_frame_sequence_fps_invalid")
                frame_uris = [
                    "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")
                    for frame in item.jpeg_frames
                ]
                translated.append(
                    {
                        "type": "video",
                        "video": frame_uris,
                        "fps": float(item.playback_fps),
                    }
                )
                continue
            if isinstance(item, str):
                translated.append({"type": "text", "text": item})
                continue
            text = getattr(item, "text", None)
            if text is not None:
                translated.append({"type": "text", "text": str(text)})
                continue
            inline_data = getattr(item, "inline_data", None)
            if inline_data is not None and inline_data.mime_type == "image/jpeg":
                translated.append({"type": "image_url", "image_url": {"url":
                    "data:image/jpeg;base64," + base64.b64encode(inline_data.data).decode("ascii")}})
                continue
            fps = self._video_metadata_fps(item)
            source_path = self._path_from_part(item)
            source_start, source_end = self._source_scope(source_path, item)
            submitted_fps = fps
            frame_bytes = self._frame_extractor(
                source_path,
                source_start,
                source_end,
                submitted_fps,
            )
            frame_count = len(frame_bytes)
            if 0 < frame_count < MIN_VIDEO_FRAMES:
                # One local re-extraction for this range only; never retry an empty
                # extraction or change the Planner's frozen requested cadence.
                submitted_fps = fps * 4
                frame_bytes = self._frame_extractor(
                    source_path, source_start, source_end, submitted_fps,
                )
                frame_count = len(frame_bytes)
            if frame_count < MIN_VIDEO_FRAMES or frame_count > MAX_VIDEO_FRAMES:
                raise ValueError(
                    f"video_frame_count_outside_official_bounds:{frame_count}"
                )
            frame_uris = [
                "data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii")
                for frame in frame_bytes
            ]
            part = {
                "type": "video",
                "video": frame_uris,
                "fps": submitted_fps,
            }
            translated.append(part)
            videos.append(
                _SubmittedVideo(
                    content_part=part,
                    source_path=str(source_path.resolve(strict=True)),
                    frame_count=frame_count,
                    fps=fps,
                    submitted_fps=submitted_fps,
                    source_start_sec=source_start,
                    source_end_sec=source_end,
                )
            )
        return translated, videos

    def begin_observation_receipt(
        self,
        videos: list[_SubmittedVideo],
    ) -> Optional[dict[str, Any]]:
        if not videos or not self.observation_receipts_enabled:
            return None
        source_units = self._pending_source_units
        self._pending_source_units = None
        if source_units is None or len(source_units) != len(videos):
            raise ValueError("authority_owned_observation_source_units_required")
        for source, video in zip(source_units, videos):
            if Path(source["source_video_path"]).resolve(strict=True) != Path(
                video.source_path
            ).resolve(strict=True):
                raise ValueError("observation_source_path_does_not_match_video_part")
            if (
                abs(float(source["source_start_sec"]) - video.source_start_sec)
                > SOURCE_SCOPE_TIME_TOLERANCE_SEC
                or abs(float(source["source_end_sec"]) - video.source_end_sec)
                > SOURCE_SCOPE_TIME_TOLERANCE_SEC
                or abs(float(source["fps"]) - video.fps) > 1e-12
            ):
                raise ValueError("observation_source_scope_does_not_match_video_part")
        self._request_count += 1
        return {
            "request_id": f"dashscope_video_request_{self._request_count:06d}",
            "units": [
                {
                    **source,
                    "submitted_media_path": video.source_path,
                    "submitted_start_sec": video.source_start_sec,
                    "submitted_end_sec": video.source_end_sec,
                    "requested_fps": video.fps,
                    "submitted_fps": video.submitted_fps,
                    "submitted_frame_count": video.frame_count,
                }
                for source, video in zip(source_units, videos)
            ],
        }

    @staticmethod
    def finish_observation_receipt(
        pending: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if pending is None:
            return None
        return {
            "producer": "qwen_video_transport",
            "backend": "dashscope_base64_jpeg_frames",
            "request_id": pending["request_id"],
            "status": "COMPLETED",
            "response_received": True,
            "source_units": [
                {
                    **unit,
                    "transport_evidence": {
                        "kind": "direct_bounded_ffmpeg_base64_jpeg_frames",
                        "frame_count": unit["submitted_frame_count"],
                    },
                }
                for unit in pending["units"]
            ],
        }

    def post_chat_completion(self, payload: Mapping[str, Any]) -> str:
        if str(payload.get("model") or "").strip().lower() != REQUIRED_MODEL.lower():
            raise ValueError("qwen3.8_flash_required")
        if not isinstance(payload.get("enable_thinking"), bool):
            raise ValueError("enable_thinking_boolean_required")
        if {"mm_processor_kwargs", "media_io_kwargs"} & set(payload):
            raise ValueError("vllm_only_payload_fields_forbidden")
        key = str(os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "").strip()
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {key}"} if key else {}),
            },
            method="POST",
        )
        try:
            with self._request_opener(request, timeout=1800) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"DashScope endpoint unavailable: {exc.reason}") from exc
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("malformed_dashscope_response") from exc
        if not isinstance(content, str):
            raise RuntimeError("dashscope_response_content_must_be_text")
        return content

    def close(self) -> None:
        self._registered_paths.clear()
        self._registered_ids.clear()
        self._temporary_media.cleanup()


class DashScopeFlashClient(GeminiClient):
    """GeminiClient-compatible provider swap; AVP logic remains inherited."""

    def __init__(
        self,
        *args: Any,
        base_url: str = OPENAI_COMPATIBLE_BASE_URL,
        generation_overrides: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        for model in (self.model, self.plan_replan_model, self.execute_model):
            assert_dashscope_flash_backend(base_url=base_url, model=model)
        self.base_url = str(base_url).strip().rstrip("/")
        self.generation_overrides = dict(generation_overrides or {})
        self._observation_receipts_enabled = True
        self.max_video_frames_per_request = MAX_VIDEO_FRAMES
        self._uniform_partition_batch_count = 0
        self._partition_request_count = 0
        self._partition_request_count_lock = threading.Lock()

    def initialize_client(self) -> None:
        if self.client is None:
            self.client = DashScopeFlashTransport(
                base_url=self.base_url,
                generation_overrides=self.generation_overrides,
            )
            self.client.observation_receipts_enabled = (
                self._observation_receipts_enabled
            )

    def set_observation_receipts_enabled(self, enabled: bool) -> None:
        self._observation_receipts_enabled = bool(enabled)
        if self.client is not None:
            self.client.observation_receipts_enabled = bool(enabled)

    def infer_on_event_overview(
        self,
        *,
        prompt: str,
        jpeg_frames: tuple[bytes, ...],
    ) -> str:
        """Run one navigation-only representative-frame packet without receipt."""

        if self.client is None:
            self.initialize_client()
        response = self.client.models.generate_content(
            model=self.execute_model,
            contents=[
                str(prompt),
                _ExplicitFrameSequence(tuple(bytes(frame) for frame in jpeg_frames)),
            ],
        )
        return str(getattr(response, "text", response))

    def create_video_part(
        self,
        video_path: str,
        fps: Optional[float] = None,
        start_offset: Optional[str] = None,
        end_offset: Optional[str] = None,
        media_resolution: Optional[str] = None,
        duration_sec: Optional[float] = None,
    ) -> Any:
        is_bounded = start_offset is not None or end_offset is not None
        submitted_fps = (
            float(fps)
            if is_bounded and fps is not None and float(fps) > 0
            else FIXED_VIDEO_FPS
        )
        if duration_sec is not None and float(duration_sec) > 0:
            # ffmpeg's fps filter may include an endpoint frame.  Reserve one
            # slot and lower only over-budget replay requests, preserving a
            # uniform sample instead of failing after materialization.
            submitted_fps = min(
                submitted_fps,
                (MAX_VIDEO_FRAMES - 1) / float(duration_sec),
            )
        del media_resolution
        resolved = str(Path(video_path).resolve(strict=True))
        mime_type = get_mime_type(resolved)
        registered = self._get_or_upload_file(resolved, mime_type)
        metadata_kwargs: dict[str, Any] = {"fps": submitted_fps}
        if start_offset is not None:
            metadata_kwargs["startOffset"] = start_offset
        if end_offset is not None:
            metadata_kwargs["endOffset"] = end_offset
        return Part(
            file_data=FileData(file_uri=registered.uri, mime_type=mime_type),
            video_metadata=VideoMetadata(**metadata_kwargs),
        )

    @staticmethod
    def _uniform_partition_ranges(
        *,
        duration_sec: float,
        fps: float,
    ) -> list[tuple[float, float]]:
        duration = float(duration_sec)
        requested_fps = float(fps)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("uniform_partition_requires_positive_duration")
        if not math.isfinite(requested_fps) or requested_fps <= 0:
            raise ValueError("uniform_partition_requires_positive_fps")
        # Keep normal partitions near 200 frames while retaining the provider's
        # independent hard ceiling of 250.  Reserve one slot because ffmpeg's
        # fps filter may emit an endpoint frame.
        max_duration = (TARGET_VIDEO_FRAMES_PER_PARTITION - 1) / requested_fps
        if max_duration <= 0:
            raise ValueError("uniform_partition_duration_is_invalid")
        ranges: list[tuple[float, float]] = []
        start = 0.0
        while start < duration - 1e-9:
            end = min(duration, start + max_duration)
            ranges.append((start, end))
            start = end
        return ranges

    @staticmethod
    def _complete_whole_video_coverage(
        source_units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Describe one completed dense uniform pass without semantic claims."""

        if not source_units:
            raise ValueError("whole_video_coverage_requires_source_units")
        ordered = sorted(
            source_units,
            key=lambda item: (
                float(item["source_start_sec"]),
                float(item["source_end_sec"]),
            ),
        )
        first_fps = float(ordered[0]["fps"])
        if first_fps <= FIXED_VIDEO_FPS:
            raise ValueError("whole_video_coverage_requires_dense_sampling")
        expected_start = 0.0
        source_path = str(ordered[0]["source_video_path"])
        for unit in ordered:
            start = float(unit["source_start_sec"])
            end = float(unit["source_end_sec"])
            fps = float(unit["fps"])
            if abs(start - expected_start) > 1e-6:
                raise ValueError("whole_video_coverage_has_gap_or_overlap")
            if end <= start:
                raise ValueError("whole_video_coverage_has_invalid_unit")
            if abs(fps - first_fps) > 1e-12:
                raise ValueError("whole_video_coverage_has_inconsistent_fps")
            if str(unit["source_video_path"]) != source_path:
                raise ValueError("whole_video_coverage_has_mixed_sources")
            expected_start = end
        return {
            "scope": "WHOLE_VIDEO",
            "sampling_fps": first_fps,
            "start_sec": 0.0,
            "end_sec": expected_start,
            "coverage_complete": True,
            "source_unit_count": len(ordered),
        }

    def _new_partition_worker(self) -> "DashScopeFlashClient":
        worker = DashScopeFlashClient(
            model=self.model,
            plan_replan_model=self.plan_replan_model,
            execute_model=self.execute_model,
            project=self.project,
            location=self.location,
            api_key=None,
            max_frame_low=self.max_frame_low,
            max_frame_medium=self.max_frame_medium,
            max_frame_high=self.max_frame_high,
            confidence_threshold=self.confidence_threshold,
            debug=self.debug,
            base_url=self.base_url,
            generation_overrides=self.generation_overrides,
        )
        worker.set_observation_receipts_enabled(
            self._observation_receipts_enabled
        )
        worker.initialize_client()
        return self._attach_partition_request_counter(worker)

    def _attach_partition_request_counter(
        self,
        worker: "DashScopeFlashClient",
    ) -> "DashScopeFlashClient":
        original = worker.client.post_chat_completion

        def counted(payload: Mapping[str, Any]) -> str:
            with self._partition_request_count_lock:
                self._partition_request_count += 1
            return original(payload)

        worker.client.post_chat_completion = counted
        return worker

    def partition_request_count(self) -> int:
        with self._partition_request_count_lock:
            return int(self._partition_request_count)

    def _merge_uniform_partition_evidence(
        self,
        *,
        partition_results: list[tuple[int, tuple[float, float], Evidence]],
        requested_fps: float,
        observation_reason: str,
        include_coverage: bool = True,
    ) -> Evidence:
        ordered = sorted(partition_results, key=lambda item: item[0])
        frames_used: list[dict[str, Any]] = []
        key_evidence: list[dict[str, Any]] = []
        detailed_parts: list[str] = []
        partition_responses: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        source_units: list[dict[str, Any]] = []
        physical_requests: list[dict[str, Any]] = []
        for index, (start_sec, end_sec), evidence in ordered:
            frames_used.extend(evidence.frames_used)
            for item in evidence.key_evidence:
                if not isinstance(item, dict):
                    continue
                enriched = dict(item)
                enriched["partition_index"] = index
                enriched["source_start_sec"] = start_sec
                enriched["source_end_sec"] = end_sec
                key_evidence.append(enriched)
            scope = (
                f"partition {index:02d} | "
                f"source[{start_sec:.3f},{end_sec:.3f})"
            )
            if str(evidence.detailed_response or "").strip():
                detailed_parts.append(f"[{scope}] {evidence.detailed_response}")
            if str(evidence.reasoning or "").strip():
                reasoning_parts.append(f"[{scope}] {evidence.reasoning}")
            receipt = evidence.model_call.get("observation_receipt")
            if not isinstance(receipt, dict):
                raise ValueError("uniform_partition_receipt_required")
            units = receipt.get("source_units")
            if not isinstance(units, list) or len(units) != 1:
                raise ValueError("uniform_partition_requires_one_unit_per_request")
            source_units.extend(units)
            partition_responses.append({"partition_index": index,
                "detailed_response": evidence.detailed_response,
                "source_units": copy.deepcopy(units)})
            physical_requests.append(
                {
                    "partition_index": index,
                    "request_id": str(receipt.get("request_id") or ""),
                    "source_start_sec": start_sec,
                    "source_end_sec": end_sec,
                }
            )

        frames_used.sort(
            key=lambda item: (
                float(item.get("start", 0.0)),
                float(item.get("end", 0.0)),
            )
        )
        key_evidence.sort(
            key=lambda item: (
                float(item.get("timestamp_start", float("inf"))),
                float(item.get("timestamp_end", float("inf"))),
                int(item.get("partition_index", 0)),
            )
        )
        source_units.sort(
            key=lambda item: (
                float(item.get("source_start_sec", 0.0)),
                float(item.get("source_end_sec", 0.0)),
            )
        )
        observation_coverage = (
            self._complete_whole_video_coverage(source_units) if include_coverage else None
        )

        self._uniform_partition_batch_count += 1
        logical_request_id = (
            "dashscope_uniform_partition_batch_"
            f"{self._uniform_partition_batch_count:06d}"
        )
        first = ordered[0][2]
        model_call = {
            "model": self.execute_model,
            "fps": requested_fps,
            "media_resolution": first.model_call.get("media_resolution"),
            "prompt_version": first.model_call.get("prompt_version"),
            "observer_timestamp_reference": "segment_relative",
            "timestamp_offset_applied_by_code": True,
            "uniform_partitioned": True,
            "partition_responses": partition_responses,
            "full_video_coverage_reason": observation_reason,
            "observation_coverage": observation_coverage,
            "physical_request_count": len(physical_requests),
            "physical_requests": physical_requests,
            "observation_receipt": {
                "producer": "qwen_video_transport",
                "backend": "dashscope_base64_jpeg_frames",
                "request_id": logical_request_id,
                "status": "COMPLETED",
                "response_received": True,
                "source_units": source_units,
                "physical_requests": physical_requests,
            },
        }
        return Evidence(
            detailed_response="\n\n".join(detailed_parts),
            key_evidence=key_evidence,
            reasoning="\n\n".join(reasoning_parts),
            frames_used=frames_used,
            model_call=model_call,
            timestamp=first.timestamp,
            round_id=0,
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
        requested_fps = float(watch_cfg.fps)
        if watch_cfg.load_mode != "uniform" or requested_fps <= FIXED_VIDEO_FPS:
            return super().infer_on_video(
                video_path=video_path,
                duration_sec=duration_sec,
                sub_query=sub_query,
                context=context,
                start_sec=start_sec,
                end_sec=end_sec,
                watch_cfg=watch_cfg,
                step_id=step_id,
                original_query=original_query,
                observation_reason=observation_reason,
            )

        reason = str(observation_reason or "").strip()
        if not reason:
            raise ValueError("full_video_partition_reason_required")
        ranges = self._uniform_partition_ranges(
            duration_sec=duration_sec,
            fps=requested_fps,
        )

        def run_partition(
            index: int,
            bounds: tuple[float, float],
        ) -> tuple[int, tuple[float, float], Evidence]:
            worker = self._new_partition_worker()
            try:
                region_watch = WatchConfig(
                    load_mode="region",
                    fps=requested_fps,
                    spatial_token_rate=watch_cfg.spatial_token_rate,
                    regions=[bounds],
                )
                evidence = GeminiClient.infer_on_video(
                    worker,
                    video_path=video_path,
                    duration_sec=duration_sec,
                    sub_query=sub_query,
                    context=context,
                    start_sec=bounds[0],
                    end_sec=bounds[1],
                    watch_cfg=region_watch,
                    step_id=f"{step_id}_partition_{index:03d}",
                    original_query=original_query,
                    observation_reason=reason,
                )
                return index, bounds, evidence
            finally:
                worker.cleanup_uploaded_files()

        if len(ranges) == 1:
            index, bounds, evidence = run_partition(1, ranges[0])
            evidence.model_call["uniform_partitioned"] = False
            evidence.model_call["full_video_coverage_reason"] = reason
            evidence.model_call["physical_request_count"] = 1
            receipt = evidence.model_call.get("observation_receipt")
            if not isinstance(receipt, dict) or not isinstance(
                receipt.get("source_units"), list
            ):
                raise ValueError("whole_video_coverage_receipt_required")
            evidence.model_call["observation_coverage"] = (
                self._complete_whole_video_coverage(receipt["source_units"])
            )
            return evidence

        results: list[tuple[int, tuple[float, float], Evidence]] = []
        max_workers = min(MAX_PARALLEL_VIDEO_REQUESTS, len(ranges))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_partition, index, bounds): index
                for index, bounds in enumerate(ranges, 1)
            }
            for future in as_completed(futures):
                results.append(future.result())
        return self._merge_uniform_partition_evidence(
            partition_results=results,
            requested_fps=requested_fps,
            observation_reason=reason,
        )

    def _get_or_upload_file(self, video_path: str, mime_type: str) -> _RegisteredFile:
        del mime_type
        if self.client is None:
            self.initialize_client()
        absolute = os.path.abspath(video_path)
        cached = self._uploaded_files.get(absolute)
        if cached is not None:
            return cached
        uri = self.client.register_media(absolute)
        registered = _RegisteredFile(uri=uri, name=Path(absolute).name)
        self._uploaded_files[absolute] = registered
        return registered

    def cleanup_uploaded_files(self) -> None:
        self._uploaded_files.clear()
        if self.client is not None:
            self.client.close()
            self.client = None


def make_dashscope_flash_client(
    *,
    base_url: str,
    model: str,
    temperature: float,
    seed: int,
) -> DashScopeFlashClient:
    assert_dashscope_flash_backend(base_url=base_url, model=model)
    client = DashScopeFlashClient(
        model=REQUIRED_MODEL,
        plan_replan_model=REQUIRED_MODEL,
        execute_model=REQUIRED_MODEL,
        project="dashscope",
        location="global",
        api_key=None,
        max_frame_low=MAX_VIDEO_FRAMES,
        max_frame_medium=MAX_VIDEO_FRAMES,
        max_frame_high=MAX_VIDEO_FRAMES,
        confidence_threshold=0.7,
        debug=False,
        base_url=base_url,
        generation_overrides={
            "temperature": float(temperature),
            "seed": int(seed),
        },
    )
    client.initialize_client()
    return client
