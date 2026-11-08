"""Text-only need answers over source-checked observations; no media client."""
import copy
import json


AUDIT_PROMPT = """Answer and audit the neutral search need using only the supplied observations.
Check whether the observed entities, events, ordering, scope and relationships actually
support the answer. Explain the supported part and missing links in plain language.
Use available_observation_tools to judge feasibility: if the only remedy requires a
frame rate or modality outside those tools, return limited, not repair. Do not propose
an unavailable higher frame rate conditionally as though it were an executable remedy.
Navigation ranking does not establish event order. Failure to locate a target is not
proof it is absent. JPEG frames provide no audio; sparse unseen intervals remain unknown.
Source checks certify origin, not the truth of model descriptions. Preserve uncertainty
and contradictions, including those in full responses. Do not infer missing premises.
Return one JSON object: action (deliver, repair, or limited), answer, and references
(a list of supplied ref_id strings). Explain any missing fact in the answer;
an optional gap may specify the next fact to check. Do not copy source quotations.
Use repair when a concrete further observable fact could resolve the gap.
An unlocated target or unexplored interval calls for repair, not limited: ask Planner
to locate or distinguish it. Use limited only when the required capability/modality or
essential task constraint is unavailable, and explain which. Never label a retrieved
segment as the requested ordinal event unless observations establish that identity.
Deliver means this need is answered,
not that a global question or option is proved. Cite support for substantive conclusions.
Leave references empty when no supplied material supports the response.
Do not decide options, plan execution parameters, or claim an independent visual review."""


def initialize(runtime):
    runtime.need_local_audit_enabled = True
    runtime._need_observations = []
    runtime._need_answers = []
    runtime._need_audit_current = None
    runtime._need_audit_consumed = {}
    runtime._need_audit_calls = 0


def record_observation(runtime, plan, evidence, event):
    if event.get("status") == "AUDIT_UNAVAILABLE":
        raise RuntimeError("need_source_audit_failed:" + str(event.get("error_message")))
    receipts = runtime._round_receipts.get(evidence.round_id, ())
    call = evidence.model_call or {}
    if call.get("modality") == "asr_transcript":
        packet = call["asr"]
        if not packet.get("source_verified"):
            raise RuntimeError("asr_source_admission_required")
        runtime._need_observations.append({"need_id": plan.need_id, "round_id": evidence.round_id,
            "task": plan.description, "modality": "asr_transcript",
            "responses": copy.deepcopy(packet["records"]), "source_audit": copy.deepcopy(event),
            "asr_source": copy.deepcopy(packet["source"]),
            "acquisition_reason": (plan.acquisition or {}).get("reason", ""),
            "requested": copy.deepcopy(packet["arguments"]),
            "full_transcript_read": packet["full_transcript_read"],
            **({"need_response": copy.deepcopy(packet["need_response"])}
               if "need_response" in packet else {}),
            "diagnostic": packet["diagnostic"]})
        return
    packets = call.get("partition_responses")
    if packets is None:
        packets = [{"detailed_response": evidence.detailed_response,
                    "source_units": call.get("observation_receipt", {}).get("source_units", [])}]
    responses = []
    for index, packet in enumerate(packets, 1):
        units = packet.get("source_units", [])
        matched = [r for r in receipts if any(
            abs(r.start_sec - u["source_start_sec"]) <= .001
            and abs(r.end_sec - u["source_end_sec"]) <= .001 for u in units)]
        responses.append({"ref_id": f"observe_{evidence.round_id}_response_{index}",
            "text": packet.get("detailed_response", ""),
            "source_ids": list(dict.fromkeys(r.record.source_id for r in matched)),
            "source_qualified": bool(matched) and all(any(
                abs(r.start_sec-u["source_start_sec"]) <= .001
                and abs(r.end_sec-u["source_end_sec"]) <= .001 for r in matched) for u in units),
            "source_scopes": [{k: u[k] for k in (
                "source_video_path", "source_start_sec", "source_end_sec", "submitted_fps") if k in u}
                for u in units], "modality": "sampled_JPEG_frames_no_audio",
            "semantic_verification": "NOT_PERFORMED",
            **({"input_asr_guidance": copy.deepcopy(call["input_asr_guidance"])}
               if call.get("input_asr_guidance") else {})})
    runtime._need_observations.append({"need_id": plan.need_id,
        "round_id": evidence.round_id, "task": plan.description,
        **({"acquisition_reason": (plan.acquisition or {}).get("reason", "")}
           if getattr(runtime, "asr_context", None) is not None else {}),
        "responses": responses, "source_audit": copy.deepcopy(event)})


def context(runtime, plan=None):
    from experiments.event_partition_scan import PLANNER_FPS
    active = runtime._active_observation_need or {}
    current = [o for o in runtime._need_observations if o["need_id"] == active.get("need_id")]
    rounds = {o["round_id"] for o in current}
    acquisition = getattr(plan, "acquisition", None) or {}
    selected = set(acquisition.get("claim_ids", [])) if acquisition.get("action") == "reuse" else set()
    # Explicit reuse brings original full responses alongside the selected facts.
    states = [s for s in runtime._evidence_state_snapshot() if s.get("evidence_kind") != "need_answer"]
    selected_rounds = {s.get("round_id") for s in states if s["claim_id"] in selected}
    observations = [o for o in runtime._need_observations
                    if o in current or o["round_id"] in selected_rounds]
    facts = [{k: copy.deepcopy(s[k]) for k in ("claim_id", "claim", "source_alignment",
        "source_ids", "round_id", "latest", "claim_scope", "source_scopes", "observation_uncertainty", "modality", "evidence_kind") if k in s}
        for s in states if s.get("round_id") in rounds or s["claim_id"] in selected]
    # Persist the selected context within this need without copying QA state.
    previous = [a for a in runtime._need_answers if a["need_id"] == active.get("need_id")]
    for answer in previous:
        for observation in answer["attempts"]:
            if observation not in observations:
                observations.append(observation)
    refs = {r["ref_id"]: r for o in observations for r in o["responses"]}
    for s in facts:
        if s.get("evidence_kind") == "asr_segment" and s["claim_id"] in refs:
            continue
        refs[s["claim_id"]] = {"ref_id": s["claim_id"], "text": s["claim"],
            "source_ids": s.get("source_ids", []), "source_scopes": s.get("source_scopes", []),
            "source_qualified": s.get("latest") is not False and s.get("source_alignment") == "ALIGNED"}
    for answer in previous:
        for ref_id, ref in answer["materials"].items():
            # Never revive a superseded fact via an older answer.
            state = next((s for s in states if s["claim_id"] == ref_id), None)
            if state is not None and (state.get("latest") is False or state.get("source_alignment") != "ALIGNED"):
                ref = dict(ref, source_qualified=False)
            refs.setdefault(ref_id, ref)
    result = {"search_need": active.get("missing_proposition"),
        "available_observation_tools": {"tools": ["full_video_coverage", "observe_region"],
            "modalities": ["sampled_JPEG_frames"], "planner_fps": list(PLANNER_FPS),
            "audio_or_full_rate_video": False}, "observations": observations,
        "facts": facts, "materials": refs,
        "previous_attempts": [{k: a[k] for k in ("action", "answer", "gap", "references",
                              "reference_diagnostics", "acquisition_diagnostic") if k in a}
                              for a in previous]}
    asr = getattr(runtime, "asr_context", None)
    if asr is not None:
        # A visual action may explicitly reuse ASR from an earlier need. Include
        # precisely that supplied text, not the entire historical transcript read.
        native_refs = {f"asr_segment_{s['id']}": s for s in asr.records}
        state_refs = {s["claim_id"]: s for s in states if s.get("evidence_kind") == "asr_segment"}
        for observation in observations:
            for response in observation["responses"]:
                guidance = response.get("input_asr_guidance", {})
                if guidance.get("source") != asr.source:
                    continue
                for ref_id in guidance.get("claim_ids", []):
                    if ref_id in refs or ref_id not in native_refs or ref_id not in state_refs:
                        continue
                    ref = asr.material(native_refs[ref_id], "selected_guidance")
                    state = state_refs[ref_id]
                    ref["source_qualified"] = state.get("latest") is not False and state.get("source_alignment") == "ALIGNED"
                    refs[ref_id] = ref
        result["available_observation_tools"]["tools"] += ["read_asr", "search_asr"]
        result["available_observation_tools"]["modalities"].append("asr_transcript")
        result["available_observation_tools"]["audio_retranscription"] = False
        result["asr_source"] = copy.deepcopy(asr.source)
        result["facts"] = [s for s in facts if s.get("evidence_kind") != "asr_segment"]
    return result


def model_context(runtime, payload):
    """Retain original material once, with observation attempts referring to its IDs."""
    if getattr(runtime, "asr_context", None) is None:
        return payload
    result = copy.deepcopy(payload)
    for observation in result["observations"]:
        if observation.get("modality") == "asr_transcript":
            observation["response_ref_ids"] = [r["ref_id"] for r in observation.pop("responses")]
            observation.pop("asr_source", None)
    # Only the ASR model view changes; stored materials and audit validation use payload.
    result["asr_materials"] = runtime.asr_context.model_materials(result["materials"])
    result["materials"] = {k: v for k, v in result["materials"].items()
                           if v.get("modality") != "asr_transcript"}
    result.pop("asr_source", None)  # Same source is carried once in asr_materials.
    return result


def audit(runtime, plan, *, limited_reason=None):
    from .avp_integration import _response_text, _json_object_with_minor_repair, _estimate_request_input_tokens
    active = runtime._active_observation_need
    if active is None or plan.need_id != active["need_id"]:
        raise RuntimeError("need_audit_lineage_mismatch")
    payload = context(runtime, plan)
    acquisition = getattr(plan, "acquisition", None) or {}
    # Subsets/reorderings of already considered facts are not new information.
    eligible = {k for k, v in payload["materials"].items() if v["source_qualified"]}
    consumed = runtime._need_audit_consumed.setdefault(active["need_id"], set())
    previous = next((a for a in reversed(runtime._need_answers) if a["need_id"] == active["need_id"]), None)
    # A different whole-text inference over the same cited passages still needs
    # review. Identical cached responses retain the existing no-progress reuse.
    new_asr_response = any(o.get("need_response") and not any(
        p.get("need_response") == o["need_response"] for p in (previous or {}).get("attempts", []))
        for o in payload["observations"])
    if acquisition.get("blocked_reason"):
        result = dict(previous or {}, action=(previous or {}).get("action", "repair"))
        result.setdefault("answer", acquisition["blocked_reason"])
        result.setdefault("gap", acquisition["blocked_reason"])
        result.setdefault("references", [])
        result["acquisition_diagnostic"] = acquisition["blocked_reason"]
    elif acquisition.get("action") == "unavailable":
        result = dict(previous or {}, action="limited", gap=acquisition["reason"])
        result.setdefault("answer", "No supported answer to this need was obtained.")
        result.setdefault("references", [])
    elif (previous is not None and not new_asr_response and eligible <= consumed and all(
            r["ref_id"] in eligible for r in previous["references"])):
        result = dict(previous, acquisition_diagnostic="No new eligible evidence; reusing the previous audit and diagnosis.")
    else:
        prompt = AUDIT_PROMPT
        if getattr(runtime, "asr_context", None) is not None:
            prompt += (" An ASR observation may include need_response: a whole-transcript draft "
                "answer with its supporting explanation and unresolved parts. Audit that reasoning "
                "against the original cited text in asr_materials and the original search_need. "
                "The draft is not an observed fact or independent evidence, and source admission "
                "does not approve its conclusions. Preserve its useful cross-passage reasoning "
                "while correcting unsupported links or changed scope. Its author read the whole "
                "transcript; you see the supplied original passages, not necessarily the whole text. "
                "Do not mistake a citation subset for evidence of an exhaustive comparison. "
                "If needed support, competing instances or context is missing, use the existing "
                "repair gap to request it, including whole-text reasoning or a full read. Cite "
                "original material ref_ids, not the draft, when delivering your audited answer. "
                "ASR text may directly answer a language need, including whole-transcript "
                "and cross-segment reasoning. Its source check proves the quoted record, not that "
                "Whisper heard correctly. Preserve unknown speaker identity; "
                "do not treat utterance time as described-event time, speech as visible action, "
                "or same-time faces as identified speakers. Read all supplied relevant text, including "
                "later corrections; a visual report marked input_asr_guidance may repeat that text "
                "and is not independent audio confirmation. Preserve "
                "negations and later corrections. Full transcript reading is not proof of perfect "
                "recognition; search misses do not establish absence. Original ASR text is in "
                "asr_materials rows, other text in materials; response_ref_ids cite those records. "
                "For complementary observations, check the specific missing relation described "
                "in task/acquisition_reason and previous_attempts. State what is now supported "
                "and what remains unconnected: shared time is not shared object, speaker, event "
                "or purpose. A visual report echoing supplied ASR is not a second confirmation. "
                "Do not repair toward a preferred conclusion. If the target does not match, "
                "request a concrete distinguishing observation or relocation through the existing "
                "repair gap; Planner chooses scope and modality. If a single modality already "
                "answers this need, deliver without requiring cross-modal confirmation.")
        request = {"response_mode": "text", "max_tokens": 4096, "enable_thinking": False,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(model_context(runtime, payload), ensure_ascii=False)}]}
        request["estimated_input_tokens"] = _estimate_request_input_tokens(request["messages"])
        if getattr(runtime, "asr_context", None) is not None:
            from .asr_mode import MAX_INPUT_TOKENS
            if request["estimated_input_tokens"] > MAX_INPUT_TOKENS:
                raise ValueError("asr_need_audit_input_too_large; no material was truncated")
        for attempt in range(2):
            runtime._need_audit_calls += 1
            raw = runtime.backend.call("need_answer_audit", request)
            text = _response_text(raw)
            result = _json_object_with_minor_repair(text)
            try:
                if not isinstance(result, dict) or result.get("action") not in ("deliver", "repair", "limited"):
                    raise ValueError("action must be deliver, repair or limited")
                answer = result.get("answer")
                gap = result.get("gap")
                answer = answer.strip() if isinstance(answer, str) else ""
                gap = gap.strip() if isinstance(gap, str) else ""
                if not answer and not gap:
                    raise ValueError("explain the finding or next fact to check in answer")
                result["answer"] = answer or gap
                result["gap"] = gap or (answer if result["action"] != "deliver" else "")
                break
            except ValueError as exc:
                if attempt:
                    raise RuntimeError("need_answer_audit_format_failed:" + str(exc)) from exc
                runtime._write_state("NEED_AUDIT_FORMAT_REPAIR", reason=str(exc))
                request["messages"].extend([{"role": "assistant", "content": text},
                    {"role": "user", "content": "Repair this text only: " + str(exc)}])
        # IDs establish provenance; the model need not reproduce source text.
        references, diagnostics = [], []
        supplied = result.get("references", [])
        if not isinstance(supplied, list):
            diagnostics.append("references is not a list; no support inferred")
            supplied = []
        for citation in supplied:
            ref_id = citation.get("ref_id") if isinstance(citation, dict) else citation
            ref = payload["materials"].get(ref_id) if isinstance(ref_id, str) else None
            if ref is None or not ref["source_qualified"]:
                diagnostics.append("Unusable reference: " + str(ref_id))
                continue
            references.append({"ref_id": ref_id, "quote": ref["text"]})
        if not references:
            diagnostics.append("No eligible support references; response remains diagnostic.")
        result.update(references=references, reference_diagnostics=diagnostics)
        consumed.update(eligible)
    result = {k: copy.deepcopy(result[k]) for k in ("action", "answer", "references", "gap",
        "reference_diagnostics", "acquisition_diagnostic") if k in result}
    if limited_reason and result["action"] == "repair":
        result.update(action="limited", gap=result["gap"] + " " + limited_reason)
    result.update(need_id=active["need_id"], search_need=active["missing_proposition"],
                  materials=payload["materials"], attempts=payload["observations"])
    runtime._need_answers.append(result)
    runtime._need_audit_current = result
    if acquisition.get("action") == "reuse":
        runtime._need_reuse_attempts.add((" ".join(active["missing_proposition"].casefold().split()),
                                       tuple(sorted(acquisition["claim_ids"]))))
    runtime._write_state("NEED_ANSWER_AUDITED", need_answer=result)
    if result["action"] != "repair":
        runtime._close_active_observation_need("RESOLVED" if result["action"] == "deliver" else "UNRESOLVED",
                                               reason=result["gap"] or result["answer"])
    return copy.deepcopy(result)


def answer_states(runtime):
    """Keep uncertain conclusions visible without certifying their unsupported parts."""
    latest = {a["need_id"]: a for a in runtime._need_answers}
    result = []
    for need_id, answer in latest.items():
        refs = answer["references"]
        valid = bool(refs) and not answer.get("reference_diagnostics") and all(answer["materials"][r["ref_id"]]["source_qualified"] and
            (r["ref_id"] not in runtime._evidence_states or
             (runtime._evidence_states[r["ref_id"]].get("latest") is not False and
              runtime._evidence_states[r["ref_id"]].get("source_alignment") == "ALIGNED")) for r in refs)
        result.append({"claim_id": need_id + "_answer", "claim": answer["answer"], "evidence_kind": "need_answer",
            "round_id": max((o["round_id"] for o in answer["attempts"]), default=None),
            "source_alignment": "ALIGNED" if valid else "UNCERTAIN", "latest": True,
            "source_ids": list(dict.fromkeys(s for r in refs for s in answer["materials"][r["ref_id"]]["source_ids"])),
            "audit_mode": "receipt_only", "semantic_verification": "TEXT_SUPPORT_REVIEW_ONLY",
            "missing_premises": ([answer["gap"]] if answer["gap"] else []) +
                answer.get("reference_diagnostics", []) + ([] if valid else ["Composite support is incomplete or no longer current; original valid facts remain usable."]),
            "need_answer": copy.deepcopy(answer)})
    return result
