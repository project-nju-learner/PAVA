"""Small shared I/O and actual HTTP accounting."""
import base64
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parent
DATASETS = ('LVBench', 'LongVideoBench', 'Video-MME', 'MLVU')
QA_KEYS = ('question_id', 'video_id', 'question', 'options')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def first_text(message):
    value = message.get('content', '')
    if isinstance(value, str):
        return value
    return next((x.get('text', '') for x in value if isinstance(x, dict) and x.get('type') == 'text'), '')

def stage_of(payload):
    messages = payload.get('messages', [])
    system = next((first_text(m) for m in messages if m.get('role') == 'system'), '')
    user = next((first_text(m) for m in messages if m.get('role') == 'user'), '')
    rules = (
        ('Answer and audit the neutral search need', 'local_audit'),
        ('Read the complete supplied ASR transcript', 'asr_initial_understanding'),
        ('Use the complete supplied ASR transcript', 'local_asr_search'),
        ('For receipt_only evidence, ALIGNED', 'global_candidate_assessment'),
        ('As Verifier,', 'global_need_generation'),
        ('Recover exactly one missing observation need', 'global_need_generation'),
        ('**Reflector prompt', 'global_reflector'),
        ('Choose one acquisition for the neutral search_need', 'local_planner'),
        ('You are analyzing a video segment', 'visual_observe'),
        ('You are synthesizing the final answer', 'final_answer'),
    )
    text = system or user
    return next((label for prefix, label in rules if text.startswith(prefix)), 'unclassified')

class BudgetExit(RuntimeError):
    pass

class Recorder:
    """Serial HTTP boundary shared by all calls/packet threads of one question."""
    def __init__(self, directory, config, question_id, shared_stop):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.question_id = question_id
        self.shared_stop = Path(shared_stop)
        self.lock = threading.Lock()
        self.calls = []
        self.images = 0
        self.budget_exhausted = False

    def clean(self, value):
        text = str(value)
        for key in ('OPENAI_API_KEY', 'DASHSCOPE_API_KEY'):
            secret = os.environ.get(key)
            if secret:
                text = text.replace(secret, '[REDACTED]')
        return re.sub(r'sk-[A-Za-z0-9_-]{8,}', '[REDACTED]', text)

    def append(self, value):
        with (self.directory / 'calls.jsonl').open('a') as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + '\n')

    def payload_view(self, value, folder, attachments):
        if isinstance(value, str) and value.startswith('data:image/'):
            header, encoded = value.split(',', 1)
            ext = 'png' if 'image/png' in header else 'jpg'
            path = folder / f'image_{len(attachments)+1:04d}.{ext}'
            path.write_bytes(base64.b64decode(encoded, validate=True))
            attachments.append(str(path))
            return {'local_attachment': str(path), 'order': len(attachments)}
        if isinstance(value, dict):
            return {k: self.payload_view(v, folder, attachments) for k, v in value.items()}
        if isinstance(value, list):
            return [self.payload_view(v, folder, attachments) for v in value]
        return self.clean(value) if isinstance(value, str) else value

    @staticmethod
    def image_count(value):
        if isinstance(value, str):
            return int(value.startswith('data:image/'))
        if isinstance(value, dict):
            return sum(Recorder.image_count(v) for v in value.values())
        if isinstance(value, list):
            return sum(Recorder.image_count(v) for v in value)
        return 0

    def wrap(self, opener):
        def call(request, *args, **kwargs):
            with self.lock:
                if self.shared_stop.exists():
                    raise RuntimeError('shared_service_stop_no_new_request')
                if request.full_url != self.config['base_url'].rstrip('/') + '/chat/completions':
                    raise ValueError('unexpected_model_endpoint')
                payload = json.loads(request.data)
                if payload.get('model') != self.config['model']:
                    raise ValueError('unexpected_model')
                count = self.image_count(payload)
                if len(self.calls) >= self.config['max_requests_per_question'] or self.images + count > self.config['max_images_per_question']:
                    self.budget_exhausted = True
                    raise BudgetExit('question_request_or_image_budget_exhausted')
                number = len(self.calls) + 1
                folder = self.directory / f'call_{number:04d}'
                folder.mkdir()
                attachments = []
                save(folder / 'request.json', self.payload_view(payload, folder, attachments))
                row = dict(question_id=self.question_id, call=number, stage=stage_of(payload),
                           started_at=now(), images=count, prompt_tokens=None,
                           completion_tokens=None, total_tokens=None, status='sent',
                           request_path=str(folder / 'request.json'), automatic_retries=0)
                self.calls.append(row)
                self.images += count
                self.append(dict(event='SEND', **row))
                save(self.directory / 'usage.json', self.summary())
                started = time.monotonic()
                try:
                    with opener(request, *args, **kwargs) as response:
                        body = response.read()
                        row['http_status'] = getattr(response, 'status', 200)
                    envelope = json.loads(body)
                    safe_response = json.loads(self.clean(json.dumps(envelope, ensure_ascii=False)))
                    save(folder / 'response.json', safe_response)
                    usage = envelope.get('usage') or {}
                    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        if isinstance(usage.get(field), int):
                            row[field] = usage[field]
                    row.update(envelope.get('_local_timing') or {})
                    row['context_tokens'] = row['prompt_tokens']
                    row['finish_reason'] = (envelope.get('choices') or [{}])[0].get('finish_reason')
                    row['status'] = 'received'
                    return io.BytesIO(body)
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__, error=self.clean(exc)[:600])
                    if isinstance(exc, HTTPError):
                        row['http_status'] = exc.code
                        # Consume and preserve a safe error body, then restore it for the original adapter.
                        raw = exc.read()
                        save(folder / 'error.json', {'http_status': exc.code, 'body': self.clean(raw.decode('utf-8', errors='replace'))[:4000]})
                        if exc.code in (401, 402, 403, 429) or exc.code >= 500:
                            save(self.shared_stop, {'http_status': exc.code, 'at': now(), 'question_id': self.question_id})
                        raise HTTPError(exc.url, exc.code, exc.reason, exc.headers, io.BytesIO(raw)) from None
                    raise
                finally:
                    row.update(finished_at=now(), elapsed_seconds=round(time.monotonic()-started, 6))
                    self.append(dict(event='END', **row))
                    save(self.directory / 'usage.json', self.summary())
        return call

    def summary(self):
        def total(rows):
            complete = [r for r in rows if r['prompt_tokens'] is not None and r['completion_tokens'] is not None]
            return dict(requests=len(rows), responses=sum(r['status']=='received' for r in rows),
                        calls_without_usage=len(rows)-len(complete),
                        known_prompt_tokens=sum(r['prompt_tokens'] for r in rows if r['prompt_tokens'] is not None),
                        known_completion_tokens=sum(r['completion_tokens'] for r in rows if r['completion_tokens'] is not None),
                        images=sum(r['images'] for r in rows),
                        call_seconds=sum(r.get('elapsed_seconds',0) for r in rows))
        stages = defaultdict(list)
        for row in self.calls:
            stages[row['stage']].append(row)
        return dict(total=total(self.calls), stages={s:total(v) for s,v in stages.items()},
                    budget_exhausted=self.budget_exhausted, raw_usage='response.json',
                    failed_request_usage_is_unknown=True)
