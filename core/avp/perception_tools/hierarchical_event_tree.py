"""Transient recursive hierarchy for isolated Event perception experiments.

The hierarchy is query-independent. It groups the existing flat
PerceptionEvent timeline and never stores captions, answers, or Evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import PerceptionEvent, PerceptionMap


MIN_NODE_FRAMES = 4
MAX_LEAF_REFINEMENT_FRAMES = 16


@dataclass(frozen=True)
class HierarchicalEventNode:
    node_id: str
    start_sec: float
    end_sec: float
    child_ids: tuple[str, ...]
    source_event_indexes: tuple[int, ...]
    embedding: tuple[float, ...]
    level_from_leaves: int

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids

    def to_dict(self, *, depth_from_root: int) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "start_sec": round(self.start_sec, 6),
            "end_sec": round(self.end_sec, 6),
            "child_ids": list(self.child_ids),
            "source_event_ids": [
                f"event_{index:06d}" for index in self.source_event_indexes
            ],
            "is_leaf": self.is_leaf,
            "depth_from_root": depth_from_root,
            "level_from_leaves": self.level_from_leaves,
        }


@dataclass(frozen=True)
class HierarchicalEventTree:
    root_id: str
    nodes: tuple[HierarchicalEventNode, ...]
    branch_factor: int

    def node(self, node_id: str) -> HierarchicalEventNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown hierarchy node: {node_id}")

    @property
    def root(self) -> HierarchicalEventNode:
        return self.node(self.root_id)

    def depths(self) -> dict[str, int]:
        result = {self.root_id: 0}
        pending = [self.root]
        while pending:
            parent = pending.pop(0)
            for child_id in parent.child_ids:
                result[child_id] = result[parent.node_id] + 1
                pending.append(self.node(child_id))
        return result

    def to_dict(self) -> dict[str, Any]:
        depths = self.depths()
        return {
            "schema_version": "transient_hierarchical_event_tree_v1",
            "root_id": self.root_id,
            "branch_factor": self.branch_factor,
            "tree_depth": max(depths.values(), default=0),
            "node_count": len(self.nodes),
            "leaf_count": sum(node.is_leaf for node in self.nodes),
            "nodes": [
                node.to_dict(depth_from_root=depths[node.node_id])
                for node in self.nodes
            ],
        }


@dataclass(frozen=True)
class ScheduledTreeFrame:
    node_id: str
    child_id: str
    source_event_id: str
    timestamp_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "child_id": self.child_id,
            "source_event_id": self.source_event_id,
            "timestamp_sec": round(self.timestamp_sec, 6),
        }


@dataclass(frozen=True)
class RefinementChoice:
    target_node_id: str
    requested_frame_count: int
    reason: str


def _normalized(values: np.ndarray) -> tuple[float, ...]:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not vector.size or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("hierarchy_embedding_must_be_finite_and_nonzero")
    return tuple(float(value) for value in vector / norm)


def _balanced_groups(
    nodes: Sequence[HierarchicalEventNode], branch_factor: int
) -> tuple[tuple[HierarchicalEventNode, ...], ...]:
    """Split contiguous nodes into the fewest duration-balanced legal groups."""
    count = len(nodes)
    group_count = int(math.ceil(count / branch_factor))
    if group_count <= 1:
        return (tuple(nodes),)
    groups: list[tuple[HierarchicalEventNode, ...]] = []
    start = 0
    for group_index in range(group_count):
        groups_left = group_count - group_index
        maximum_end = min(start + branch_factor, count - (groups_left - 1))
        minimum_end = max(start + 1, count - branch_factor * (groups_left - 1))
        remaining = sum(node.end_sec - node.start_sec for node in nodes[start:])
        target = remaining / groups_left
        end = min(
            range(minimum_end, maximum_end + 1),
            key=lambda value: (
                abs(
                    sum(
                        node.end_sec - node.start_sec
                        for node in nodes[start:value]
                    )
                    - target
                ),
                value,
            ),
        )
        groups.append(tuple(nodes[start:end]))
        start = end
    if start != count or any(not 1 <= len(group) <= branch_factor for group in groups):
        raise RuntimeError("hierarchy_grouping_contract_failed")
    return tuple(groups)


def _parent_node(
    node_id: str,
    children: Sequence[HierarchicalEventNode],
    level_from_leaves: int,
) -> HierarchicalEventNode:
    if len({len(child.embedding) for child in children}) != 1:
        raise ValueError("source_event_embedding_dimensions_do_not_match")
    weights = np.asarray(
        [child.end_sec - child.start_sec for child in children], dtype=np.float32
    )
    vectors = np.asarray([child.embedding for child in children], dtype=np.float32)
    aggregate = np.average(vectors, axis=0, weights=weights)
    return HierarchicalEventNode(
        node_id=node_id,
        start_sec=children[0].start_sec,
        end_sec=children[-1].end_sec,
        child_ids=tuple(child.node_id for child in children),
        source_event_indexes=tuple(
            index for child in children for index in child.source_event_indexes
        ),
        embedding=_normalized(aggregate),
        level_from_leaves=level_from_leaves,
    )


def build_transient_event_tree(
    perception_map: PerceptionMap, *, branch_factor: int = 4
) -> HierarchicalEventTree:
    """Build a balanced recursive hierarchy without modifying the source map."""
    if branch_factor < 2:
        raise ValueError("branch_factor_must_be_at_least_two")
    if not perception_map.events:
        raise ValueError("perception_map_requires_events")
    leaves = tuple(
        HierarchicalEventNode(
            node_id=event.event_id,
            start_sec=float(event.start_sec),
            end_sec=float(event.end_sec),
            child_ids=(),
            source_event_indexes=(event.event_index,),
            embedding=_normalized(np.asarray(event.embedding, dtype=np.float32)),
            level_from_leaves=0,
        )
        for event in perception_map.events
    )
    all_nodes: list[HierarchicalEventNode] = list(leaves)
    current = leaves
    level = 1
    while len(current) > 1:
        groups = _balanced_groups(current, branch_factor)
        is_root_level = len(groups) == 1
        parents = tuple(
            _parent_node(
                "root" if is_root_level else f"level_{level:02d}_node_{index:03d}",
                group,
                level,
            )
            for index, group in enumerate(groups)
        )
        all_nodes.extend(parents)
        current = parents
        level += 1
    if len(leaves) == 1:
        root = _parent_node("root", leaves, 1)
        all_nodes.append(root)
    else:
        root = current[0]
    tree = HierarchicalEventTree("root", tuple(all_nodes), branch_factor)
    _validate_tree(tree, perception_map)
    return tree


def _validate_tree(tree: HierarchicalEventTree, perception_map: PerceptionMap) -> None:
    identifiers = [node.node_id for node in tree.nodes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("hierarchy_node_ids_must_be_unique")
    expected_events = tuple(event.event_index for event in perception_map.events)
    if tree.root.source_event_indexes != expected_events:
        raise ValueError("hierarchy_root_must_cover_every_source_event")
    for node in tree.nodes:
        if len(node.child_ids) > tree.branch_factor:
            raise ValueError("hierarchy_branch_factor_exceeded")
        children = [tree.node(child_id) for child_id in node.child_ids]
        if not children:
            continue
        if children[0].start_sec != node.start_sec or children[-1].end_sec != node.end_sec:
            raise ValueError("hierarchy_child_coverage_mismatch")
        for left, right in zip(children, children[1:]):
            if not math.isclose(left.end_sec, right.start_sec, abs_tol=1e-6):
                raise ValueError("hierarchy_children_must_be_contiguous")


def _event_for_timestamp(
    perception_map: PerceptionMap, timestamp_sec: float
) -> PerceptionEvent:
    for event in perception_map.events:
        if event.start_sec <= timestamp_sec < event.end_sec:
            return event
    if math.isclose(timestamp_sec, perception_map.events[-1].end_sec, abs_tol=1e-6):
        return perception_map.events[-1]
    raise ValueError("scheduled_timestamp_outside_source_events")


def _child_for_timestamp(
    tree: HierarchicalEventTree,
    node: HierarchicalEventNode,
    timestamp_sec: float,
) -> HierarchicalEventNode:
    if node.is_leaf:
        return node
    for child_id in node.child_ids:
        child = tree.node(child_id)
        if child.start_sec <= timestamp_sec < child.end_sec:
            return child
    if math.isclose(timestamp_sec, node.end_sec, abs_tol=1e-6):
        return tree.node(node.child_ids[-1])
    raise ValueError("scheduled_timestamp_outside_direct_children")


def _representative_for_child(
    child: HierarchicalEventNode, perception_map: PerceptionMap
) -> float:
    midpoint = (child.start_sec + child.end_sec) / 2.0
    candidates = [
        timestamp
        for index in child.source_event_indexes
        for timestamp in perception_map.events[index].representative_timestamps
    ]
    return float(
        min(candidates, key=lambda value: (abs(value - midpoint), value))
        if candidates
        else midpoint
    )


def schedule_node_frames(
    tree: HierarchicalEventTree,
    perception_map: PerceptionMap,
    node_id: str,
    *,
    frame_count: int = MIN_NODE_FRAMES,
) -> tuple[ScheduledTreeFrame, ...]:
    """Schedule distinct frames with direct-child and source-Event lineage."""
    node = tree.node(node_id)
    maximum = (
        MAX_LEAF_REFINEMENT_FRAMES
        if node.is_leaf
        else max(MIN_NODE_FRAMES, len(node.child_ids))
    )
    if not MIN_NODE_FRAMES <= frame_count <= maximum:
        raise ValueError("node_frame_count_outside_contract")
    selected: list[tuple[float, str]] = []
    used: set[float] = set()

    def add(timestamp: float, child_id: str) -> None:
        value = min(
            max(float(timestamp), node.start_sec),
            math.nextafter(node.end_sec, node.start_sec),
        )
        key = round(value, 6)
        if key not in used:
            used.add(key)
            selected.append((value, child_id))

    if not node.is_leaf:
        for child_id in node.child_ids:
            child = tree.node(child_id)
            add(_representative_for_child(child, perception_map), child_id)
    if len(selected) > frame_count:
        raise RuntimeError("direct_child_count_exceeds_node_frame_budget")
    for multiplier in (1, 2, 4, 8):
        if len(selected) >= frame_count:
            break
        slots = frame_count * multiplier
        for index in range(slots):
            timestamp = (
                node.start_sec
                + (index + 0.5) * (node.end_sec - node.start_sec) / slots
            )
            child = _child_for_timestamp(tree, node, timestamp)
            add(timestamp, child.node_id)
            if len(selected) >= frame_count:
                break
        if len(selected) >= frame_count:
            break
    if len(selected) != frame_count:
        raise RuntimeError("unable_to_schedule_distinct_node_frames")
    return tuple(
        ScheduledTreeFrame(
            node_id=node.node_id,
            child_id=child_id,
            source_event_id=_event_for_timestamp(perception_map, timestamp).event_id,
            timestamp_sec=timestamp,
        )
        for timestamp, child_id in sorted(selected, key=lambda item: item[0])
    )


def normalize_refinement_requests(
    tree: HierarchicalEventTree,
    observed_node_id: str,
    requests: Iterable[Mapping[str, Any]],
    *,
    already_observed: Iterable[str] = (),
    dense_leaf_ids: Iterable[str] = (),
) -> tuple[tuple[RefinementChoice, ...], tuple[str, ...]]:
    """Accept only direct-child expansion or one denser revisit of this leaf."""
    observed = tree.node(observed_node_id)
    allowed = set(observed.child_ids) if not observed.is_leaf else {observed.node_id}
    seen = set(already_observed)
    dense = set(dense_leaf_ids)
    accepted: list[RefinementChoice] = []
    rejected: list[str] = []
    accepted_ids: set[str] = set()
    for index, raw in enumerate(requests):
        target = str(raw.get("target_node_id") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        try:
            count = int(raw.get("requested_frame_count", MIN_NODE_FRAMES))
        except (TypeError, ValueError):
            count = 0
        if target not in allowed:
            rejected.append(f"request_{index}:target_not_directly_expandable")
            continue
        target_node = tree.node(target)
        if not reason:
            rejected.append(f"request_{index}:reason_required")
            continue
        if target_node.is_leaf:
            if target == observed.node_id:
                if target in dense or not MIN_NODE_FRAMES <= count <= MAX_LEAF_REFINEMENT_FRAMES:
                    rejected.append(
                        f"request_{index}:invalid_or_repeated_leaf_refinement"
                    )
                    continue
            else:
                count = MIN_NODE_FRAMES
        else:
            count = MIN_NODE_FRAMES
        if target in accepted_ids or (target in seen and target != observed.node_id):
            rejected.append(f"request_{index}:duplicate_or_already_observed")
            continue
        accepted_ids.add(target)
        accepted.append(RefinementChoice(target, count, reason))
    return tuple(accepted), tuple(rejected)


def select_embedding_fallback(
    tree: HierarchicalEventTree,
    query_embedding: Sequence[float],
    *,
    observed_node_ids: Iterable[str],
    dense_leaf_ids: Iterable[str] = (),
) -> RefinementChoice | None:
    """Choose one unobserved child, or one not-yet-dense observed leaf."""
    query = np.asarray(query_embedding, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if query.ndim != 1 or not query.size or not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("fallback_query_embedding_must_be_finite_and_nonzero")
    query = query / norm
    observed = set(observed_node_ids)
    dense = set(dense_leaf_ids)
    candidates: dict[str, HierarchicalEventNode] = {}
    for node_id in observed:
        node = tree.node(node_id)
        if node.is_leaf and node_id not in dense:
            candidates[node_id] = node
        for child_id in node.child_ids:
            if child_id not in observed:
                candidates[child_id] = tree.node(child_id)
    if not candidates:
        return None
    if {len(node.embedding) for node in candidates.values()} != {query.size}:
        raise ValueError("fallback_query_and_node_embedding_dimensions_must_match")
    target = max(
        candidates.values(),
        key=lambda node: (
            float(np.asarray(node.embedding, dtype=np.float32) @ query),
            -node.start_sec,
            node.node_id,
        ),
    )
    count = 8 if target.is_leaf and target.node_id in observed else MIN_NODE_FRAMES
    return RefinementChoice(
        target_node_id=target.node_id,
        requested_frame_count=count,
        reason="embedding_ranked_single_fallback",
    )
