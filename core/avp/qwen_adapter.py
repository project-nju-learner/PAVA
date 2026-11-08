"""OpenAI-compatible Qwen transport for the unchanged AVP algorithm.

This module deliberately adapts only the provider boundary.  ``QwenClient``
inherits every planning, observation, reflection, parsing, and fallback method
from ``GeminiClient``.  Prompts remain ordinary strings produced by the
existing ``PromptManager``; Gemini video ``Part`` objects are translated at the
last moment to vLLM ``video_url`` content parts.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock, Thread
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .main import GeminiClient
from .video_utils import VideoMetadataExtractor


_QWEN_PIXELS_BY_MEDIA_RESOLUTION = {
    # Qwen3-VL uses 16x16 patches with 2x2 spatial merging: these correspond
    # to 64 / 256 / 1024 visual tokens per frame.  Together with AVP's
    # existing 2048 / 512 / 128 frame caps, each tier keeps the same total
    # 131072-token ceiling while preserving low < medium < high.
    "LOW": 65_536,
    "MEDIUM": 262_144,
    "HIGH": 1_048_576,
}


@dataclass(frozen=True)
class _GenerateResponse:
    text: str
    observation_receipt: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class _RegisteredFile:
    uri: str
    name: str


class _LocalMediaServer:
    """Expose only explicitly registered local files to localhost vLLM."""

    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}
        self._ids_by_path: dict[Path, str] = {}
        self._transfer_stats: dict[str, dict[str, int]] = {}
        self._lock = Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
                owner._serve(self, include_body=False)

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                owner._serve(self, include_body=True)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def origin(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def register(self, path: str | Path) -> str:
        resolved = Path(path).resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Qwen media path is not a file: {resolved}")
        with self._lock:
            media_id = self._ids_by_path.get(resolved)
            if media_id is None:
                media_id = f"media_{len(self._paths) + 1}"
                self._paths[media_id] = resolved
                self._ids_by_path[resolved] = media_id
                self._transfer_stats[media_id] = {
                    "head_count": 0,
                    "get_count": 0,
                    "bytes_served": 0,
                }
        return f"{self.origin}/media/{media_id}"

    def resolve(self, url: str) -> Optional[Path]:
        parsed = urlsplit(url)
        if f"{parsed.scheme}://{parsed.netloc}" != self.origin:
            return None
        prefix = "/media/"
        if not parsed.path.startswith(prefix):
            return None
        with self._lock:
            return self._paths.get(parsed.path[len(prefix) :])

    def transfer_stats(self, url: str) -> dict[str, int]:
        parsed = urlsplit(url)
        if f"{parsed.scheme}://{parsed.netloc}" != self.origin:
            raise ValueError("media_url_is_not_owned_by_local_transport")
        prefix = "/media/"
        if not parsed.path.startswith(prefix):
            raise ValueError("media_url_has_no_registered_media_id")
        media_id = parsed.path[len(prefix) :]
        with self._lock:
            if media_id not in self._paths:
                raise ValueError("media_url_is_not_registered")
            return dict(self._transfer_stats[media_id])

    def _serve(self, handler: BaseHTTPRequestHandler, include_body: bool) -> None:
        parsed = urlsplit(handler.path)
        prefix = "/media/"
        media_id = parsed.path[len(prefix) :] if parsed.path.startswith(prefix) else ""
        with self._lock:
            path = self._paths.get(media_id)
        if path is None:
            handler.send_error(404)
            return

        with self._lock:
            counter = "get_count" if include_body else "head_count"
            self._transfer_stats[media_id][counter] += 1

        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = handler.headers.get("Range", "")
        partial = False
        if range_header.startswith("bytes="):
            raw_start, _, raw_end = range_header[6:].partition("-")
            try:
                start = int(raw_start) if raw_start else 0
                end = int(raw_end) if raw_end else size - 1
                end = min(end, size - 1)
                if start < 0 or start > end:
                    raise ValueError
                partial = True
            except ValueError:
                handler.send_error(416)
                return

        handler.send_response(206 if partial else 200)
        handler.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        handler.send_header("Accept-Ranges", "bytes")
        handler.send_header("Content-Length", str(end - start + 1))
        if partial:
            handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.end_headers()
        if not include_body:
            return
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            served = 0
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
                served += len(chunk)
        with self._lock:
            self._transfer_stats[media_id]["bytes_served"] += served

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _QwenModels:
    def __init__(self, transport: "OpenAICompatibleQwen") -> None:
        self._transport = transport

    def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any = None,
    ) -> _GenerateResponse:
        max_pixels = self._transport.max_pixels_for_config(config)
        content_parts, fps = self._transport.translate_contents(
            contents, max_pixels=max_pixels
        )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content_parts}],
            "stream": False,
        }
        payload.update(self._transport.generation_overrides)
        has_video = any(
            isinstance(part, dict) and part.get("type") == "video_url"
            for part in content_parts
        )
        processor_kwargs = self._transport.processor_kwargs_for(
            fps=fps, max_pixels=max_pixels, has_video=has_video
        )
        if processor_kwargs:
            payload["mm_processor_kwargs"] = processor_kwargs
        if fps is not None:
            # vLLM otherwise imposes its own 32-frame media-loader default
            # before the Qwen processor sees AVP's sampling rate.  Removing
            # that cap makes the loader honor the exact AVP fps request.
            payload["media_io_kwargs"] = {
                "video": {
                    "num_frames": self._transport.video_num_frames(),
                    "fps": fps,
                }
            }
        pending = self._transport.begin_video_observation_receipt(
            contents=contents,
            translated_parts=content_parts,
            fps=fps,
        )
        response_text = self._transport.post_chat_completion(payload)
        receipt = self._transport.finish_video_observation_receipt(pending)
        return _GenerateResponse(response_text, observation_receipt=receipt)


class OpenAICompatibleQwen:
    """Minimal ``client.models.generate_content`` facade over vLLM chat API."""

    def __init__(
        self,
        base_url: str,
        debug: bool = False,
        generation_overrides: Optional[dict[str, Any]] = None,
    ) -> None:
        normalized = str(base_url).strip().rstrip("/")
        if not normalized:
            raise ValueError("Qwen base_url is required")
        self.base_url = normalized
        self.debug = debug
        self.generation_overrides = dict(generation_overrides or {})
        self.models = _QwenModels(self)
        self.media = _LocalMediaServer()
        self._temporary_media = tempfile.TemporaryDirectory(prefix="avp_qwen_media_")
        self._temporary_index = 0
        self._prepared_media: dict[
            tuple[Path, Optional[float], Optional[int]], Path
        ] = {}
        self._video_request_count = 0
        self._pending_observation_source_units: Optional[list[dict[str, Any]]] = None
        self.observation_receipts_enabled = False

    @staticmethod
    def video_num_frames() -> int:
        return -1

    @staticmethod
    def processor_kwargs_for(
        *, fps: Optional[float], max_pixels: Optional[int], has_video: bool
    ) -> dict[str, Any]:
        del has_video
        kwargs: dict[str, Any] = {}
        if fps is not None:
            kwargs["fps"] = fps
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        return kwargs

    def set_observation_source_units(
        self,
        *,
        video_path: str,
        units: list[dict[str, Any]],
    ) -> None:
        """Freeze raw-video source ranges before the next video request.

        The request transport consumes this exactly once.  It is deliberately
        separate from model output so an Evidence response cannot invent a
        source file, range, sampling rate, or transaction identity.
        """
        if not self.observation_receipts_enabled:
            return
        if self._pending_observation_source_units is not None:
            raise RuntimeError("observation_source_units_already_pending")
        resolved_video = str(Path(video_path).resolve(strict=True))
        normalized: list[dict[str, Any]] = []
        for unit in units:
            if not isinstance(unit, dict):
                raise TypeError("observation_source_unit_must_be_an_object")
            start = float(unit["start"])
            end = float(unit["end"])
            raw_fps = unit.get("fps")
            unit_fps = float(raw_fps) if raw_fps is not None else None
            if start < 0 or end <= start or (unit_fps is not None and unit_fps <= 0):
                raise ValueError("invalid_observation_source_unit")
            normalized.append(
                {
                    "source_video_path": resolved_video,
                    "source_start_sec": start,
                    "source_end_sec": end,
                    "fps": unit_fps,
                }
            )
        if not normalized:
            raise ValueError("observation_source_units_required")
        self._pending_observation_source_units = normalized

    @staticmethod
    def _video_inputs(contents: Any) -> list[Any]:
        items = contents if isinstance(contents, (list, tuple)) else [contents]
        videos: list[Any] = []
        for item in items:
            if isinstance(item, str) or getattr(item, "text", None) is not None:
                continue
            if getattr(item, "file_data", None) is not None or getattr(
                item, "inline_data", None
            ) is not None:
                videos.append(item)
        return videos

    @staticmethod
    def _duration_seconds(path: Path, fallback: float) -> float:
        try:
            duration = float(VideoMetadataExtractor(str(path)).duration)
            if duration > 0:
                return duration
        except Exception:
            pass
        return fallback

    def begin_video_observation_receipt(
        self,
        *,
        contents: Any,
        translated_parts: list[dict[str, Any]],
        fps: Optional[float],
    ) -> Optional[dict[str, Any]]:
        video_inputs = self._video_inputs(contents)
        video_parts = [
            part for part in translated_parts if part.get("type") == "video_url"
        ]
        if not video_parts:
            return None
        if not self.observation_receipts_enabled:
            return None
        if len(video_inputs) != len(video_parts):
            raise ValueError("video_input_translation_count_mismatch")

        source_units = self._pending_observation_source_units
        self._pending_observation_source_units = None
        if source_units is None or len(source_units) != len(video_parts):
            raise ValueError("authority_owned_observation_source_units_required")

        self._video_request_count += 1
        request_id = f"qwen_video_request_{self._video_request_count:06d}"
        submissions: list[dict[str, Any]] = []
        for source_unit, part in zip(source_units, video_parts):
            url = str(part["video_url"]["url"])
            submitted_path = self.media.resolve(url)
            if submitted_path is None:
                raise ValueError("raw_video_receipt_requires_registered_local_media")
            before = self.media.transfer_stats(url)
            source_duration = (
                float(source_unit["source_end_sec"])
                - float(source_unit["source_start_sec"])
            )
            submitted_duration = self._duration_seconds(
                submitted_path, fallback=source_duration
            )
            submissions.append(
                {
                    **source_unit,
                    "submitted_media_path": str(submitted_path.resolve(strict=True)),
                    "submitted_start_sec": 0.0,
                    "submitted_end_sec": submitted_duration,
                    "submitted_fps": float(fps) if fps is not None else None,
                    "media_url": url,
                    "transfer_before": before,
                }
            )
        return {"request_id": request_id, "submissions": submissions}

    def finish_video_observation_receipt(
        self, pending: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        if pending is None:
            return None
        units: list[dict[str, Any]] = []
        for submission in pending["submissions"]:
            after = self.media.transfer_stats(submission["media_url"])
            before = submission["transfer_before"]
            get_count = after["get_count"] - before["get_count"]
            bytes_served = after["bytes_served"] - before["bytes_served"]
            units.append(
                {
                    key: value
                    for key, value in submission.items()
                    if key not in {"media_url", "transfer_before"}
                }
                | {
                    "transport_evidence": {
                        "kind": "local_http_get",
                        "get_count": get_count,
                        "bytes_served": bytes_served,
                    }
                }
            )
        return {
            "producer": "qwen_video_transport",
            "request_id": pending["request_id"],
            "status": "COMPLETED",
            "response_received": True,
            "source_units": units,
        }

    def register_media(self, path: str | Path) -> str:
        return self.media.register(path)

    def translate_contents(
        self, contents: Any, max_pixels: Optional[int] = None
    ) -> tuple[list[dict[str, Any]], Optional[float]]:
        items = contents if isinstance(contents, (list, tuple)) else [contents]
        translated: list[dict[str, Any]] = []
        requested_fps: list[float] = []
        for item in items:
            if isinstance(item, str):
                translated.append({"type": "text", "text": item})
                continue
            text = getattr(item, "text", None)
            if text is not None:
                translated.append({"type": "text", "text": text})
                continue

            metadata = getattr(item, "video_metadata", None)
            fps = getattr(metadata, "fps", None) if metadata is not None else None
            if fps is not None:
                requested_fps.append(float(fps))
            video_url = self._video_url(item, metadata, max_pixels=max_pixels)
            translated.append({"type": "video_url", "video_url": {"url": video_url}})

        distinct_fps = {round(value, 12) for value in requested_fps}
        if len(distinct_fps) > 1:
            raise ValueError("vLLM accepts one video fps per request; AVP supplied different values")
        return translated, (requested_fps[0] if requested_fps else None)

    def _video_url(
        self, part: Any, metadata: Any, max_pixels: Optional[int]
    ) -> str:
        file_data = getattr(part, "file_data", None)
        inline_data = getattr(part, "inline_data", None)
        source_path: Optional[Path] = None
        source_url: Optional[str] = None

        if file_data is not None and getattr(file_data, "file_uri", None):
            source_url = str(file_data.file_uri)
            source_path = self.media.resolve(source_url)
        elif inline_data is not None and getattr(inline_data, "data", None) is not None:
            mime_type = getattr(inline_data, "mime_type", None) or "video/mp4"
            raw = bytes(inline_data.data)
            if not self.observation_receipts_enabled and (
                metadata is None
                or (
                    getattr(metadata, "start_offset", None) is None
                    and getattr(metadata, "end_offset", None) is None
                )
            ):
                encoded = base64.b64encode(raw).decode("ascii")
                return f"data:{mime_type};base64,{encoded}"
            suffix = mimetypes.guess_extension(mime_type) or ".mp4"
            self._temporary_index += 1
            source_path = (
                Path(self._temporary_media.name)
                / f"inline_source_{self._temporary_index}{suffix}"
            )
            source_path.write_bytes(raw)
            source_url = self.media.register(source_path)
        else:
            raise TypeError(f"Unsupported Gemini content part for Qwen: {type(part).__name__}")

        if source_path is None or metadata is None:
            if source_url is None:
                raise ValueError("Qwen video part has no usable media source")
            return source_url

        start = self._offset_seconds(getattr(metadata, "start_offset", None), default=0.0)
        end = self._offset_seconds(getattr(metadata, "end_offset", None), default=None)
        if self._is_bounded_range(source_path, start, end):
            if end is None or end <= start:
                raise ValueError(
                    f"Invalid AVP video offsets for Qwen: start={start}, end={end}"
                )
            raise ValueError(
                "qwen_bounded_observation_requires_direct_frame_transport"
            )

        sampled_fps = getattr(metadata, "fps", None) if metadata is not None else None
        prepared_path = self._prepare_video_for_qwen(
            source_path,
            fps=float(sampled_fps) if sampled_fps is not None else None,
            max_pixels=max_pixels,
        )
        return self.media.register(prepared_path)

    def _prepare_video_for_qwen(
        self,
        path: Path,
        fps: Optional[float],
        max_pixels: Optional[int],
    ) -> Path:
        """Materialize unsupported codecs without changing AVP's sample request."""
        resolved = path.resolve(strict=True)
        codec, width, height = self._probe_video_stream(resolved)
        if codec == "h264":
            return resolved

        key = (resolved, fps, max_pixels)
        cached = self._prepared_media.get(key)
        if cached is not None and cached.exists():
            return cached

        self._temporary_index += 1
        output = Path(self._temporary_media.name) / f"qwen_transport_{self._temporary_index}.mp4"
        filters: list[str] = []
        if fps is not None and fps > 0:
            filters.append(f"fps={fps:.12g}")
        target_width, target_height = self._target_dimensions(
            width, height, max_pixels
        )
        if (target_width, target_height) != (width, height):
            filters.append(f"scale={target_width}:{target_height}")

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(resolved),
            "-map",
            "0:v:0",
        ]
        if filters:
            command.extend(["-vf", ",".join(filters)])
        command.extend(
            [
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(output),
            ]
        )
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=1800
        )
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(
                "Could not prepare video codec for Qwen transport: "
                + completed.stderr.strip()
            )
        self._prepared_media[key] = output
        return output

    @staticmethod
    def _probe_video_stream(path: Path) -> tuple[str, int, int]:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Could not inspect video for Qwen transport: "
                + completed.stderr.strip()
            )
        try:
            stream = json.loads(completed.stdout)["streams"][0]
            return str(stream["codec_name"]), int(stream["width"]), int(stream["height"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("ffprobe returned no usable video stream") from exc

    @staticmethod
    def _target_dimensions(
        width: int, height: int, max_pixels: Optional[int]
    ) -> tuple[int, int]:
        if max_pixels is None or width * height <= max_pixels:
            return width, height
        scale = math.sqrt(max_pixels / float(width * height))
        target_width = max(2, int(width * scale) // 2 * 2)
        target_height = max(2, int(height * scale) // 2 * 2)
        while target_width * target_height > max_pixels:
            if target_width >= target_height and target_width > 2:
                target_width -= 2
            elif target_height > 2:
                target_height -= 2
            else:
                break
        return target_width, target_height

    @staticmethod
    def _offset_seconds(value: Any, default: Optional[float]) -> Optional[float]:
        if value is None:
            return default
        text = str(value).strip()
        if text.endswith("s"):
            text = text[:-1]
        return float(text)

    @staticmethod
    def _is_bounded_range(path: Path, start: float, end: Optional[float]) -> bool:
        if start > 0.001:
            return True
        if end is None:
            return False
        try:
            duration = float(VideoMetadataExtractor(str(path)).duration)
        except Exception:
            duration = 0.0
        if duration <= 0:
            return True
        return end < duration - 0.05

    @staticmethod
    def max_pixels_for_config(config: Any) -> Optional[int]:
        if config is None:
            return None
        resolution = getattr(config, "media_resolution", None)
        if resolution is None:
            return None
        raw = getattr(resolution, "value", resolution)
        name = str(raw).upper()
        for tier, max_pixels in _QWEN_PIXELS_BY_MEDIA_RESOLUTION.items():
            if name.endswith(tier):
                return max_pixels
        raise ValueError(f"Unsupported AVP media resolution for Qwen: {resolution!r}")

    def post_chat_completion(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=1800) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Qwen endpoint unavailable: {exc.reason}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Malformed Qwen response: {result!r}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Qwen response content must be text, got {type(content).__name__}")
        return content

    def close(self) -> None:
        self.media.close()
        self._temporary_media.cleanup()


class QwenClient(GeminiClient):
    """GeminiClient-compatible provider swap; AVP algorithm stays inherited."""

    def __init__(
        self,
        *args: Any,
        base_url: str = "http://127.0.0.1:18000/v1",
        generation_overrides: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.generation_overrides = dict(generation_overrides or {})
        self._observation_receipts_enabled = False

    def initialize_client(self) -> None:
        if self.client is None:
            self.client = OpenAICompatibleQwen(
                self.base_url,
                debug=self.debug,
                generation_overrides=self.generation_overrides,
            )
            self.client.observation_receipts_enabled = (
                self._observation_receipts_enabled
            )
            if self.debug:
                print(f"Initialized Qwen OpenAI-compatible client: {self.base_url}")

    def set_observation_receipts_enabled(self, enabled: bool) -> None:
        self._observation_receipts_enabled = bool(enabled)
        if self.client is not None:
            self.client.observation_receipts_enabled = bool(enabled)

    def _get_or_upload_file(self, video_path: str, mime_type: str) -> _RegisteredFile:
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
