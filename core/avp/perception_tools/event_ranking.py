from __future__ import annotations

import numpy as np

from .contracts import EventMatch, PerceptionMap


def rank_events(
    perception_map: PerceptionMap,
    query_embedding: np.ndarray,
) -> tuple[EventMatch, ...]:
    """Rank every Event; relevance never discards part of the timeline."""
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1 or not query.size or not np.isfinite(query).all():
        raise ValueError("query_embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(query))
    if norm <= 1e-12:
        raise ValueError("query_embedding must not be a zero vector")
    query = query / norm
    matches: list[EventMatch] = []
    for event in perception_map.events:
        event_embedding = np.asarray(event.embedding, dtype=np.float32)
        if event_embedding.size != query.size:
            raise ValueError("query and event embedding dimensions must match")
        representative_embeddings = event_embedding
        score = float(representative_embeddings @ query)
        matches.append(
            EventMatch(
                event_index=event.event_index,
                start_sec=event.start_sec,
                end_sec=event.end_sec,
                event_score=score,
                best_frame_score=score,
                best_timestamp_sec=event.representative_timestamps[
                    len(event.representative_timestamps) // 2
                ],
                representative_timestamps=event.representative_timestamps,
            )
        )
    matches.sort(key=lambda match: (-match.event_score, match.event_index))
    return tuple(matches)
