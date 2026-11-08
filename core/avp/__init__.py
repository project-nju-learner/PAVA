"""
Active Video Perception Framework (AVP)
================================================

A multi-step video analysis framework using Gemini API with:
- Intelligent planning and replanning
- Coarse-to-fine progressive exploration
- Structured JSON outputs with validation
- Centralized prompt management
"""

from .main import (
    # Data structures
    PlanSpec,
    Evidence,
    Blackboard,
    WatchConfig,
    SpatialTokenRate,
    
    # Core components
    GeminiClient,
    Planner,
    Observer,
    Reflector,
    Controller,
    VideoMetadataExtractor,
    
    # Storage
    Store,
)

from .prompt import (
    PromptManager,
    parse_json_response,
    validate_against_schema,
    PLAN_SCHEMA,
    EVIDENCE_SCHEMA,
    FINAL_ANSWER_SCHEMA,
)

from .video_utils import (
    VideoMetadataExtractor,
    sha256_file,
    get_mime_type,
    find_compressed_video_fallback,
    get_video_path,
    get_video_info,
    print_video_info,
    validate_video_file,
    format_duration,
    set_metadata_source,
    load_video_metadata_from_json,
)

from .config import (
    AVPConfig,
    load_config,
)

__version__ = "1.0.0"
__all__ = [
    # Data structures
    "PlanSpec",
    "Evidence",
    "Blackboard",
    "WatchConfig",
    "SpatialTokenRate",
    
    # Core components
    "GeminiClient",
    "Planner",
    "Observer",
    "Reflector",
    "Controller",
    "Store",
    
    # Video utilities
    "VideoMetadataExtractor",
    "sha256_file",
    "get_mime_type",
    "find_compressed_video_fallback",
    "get_video_path",
    "get_video_info",
    "print_video_info",
    "validate_video_file",
    "format_duration",
    "set_metadata_source",
    "load_video_metadata_from_json",
    
    # Prompt management
    "PromptManager",
    "parse_json_response",
    "validate_against_schema",
    "PLAN_SCHEMA",
    "EVIDENCE_SCHEMA",
    "FINAL_ANSWER_SCHEMA",
    
    # Configuration
    "AVPConfig",
    "load_config",
]

