"""Configured video/image transport and streaming timing; PAVA prompts stay intact."""
import copy
import io
import json
import time
from urllib.request import Request


def local_payload(payload, config):
    payload = copy.deepcopy(payload)
    assert payload['model'] == config['model']
    if config['input_format'] == 'dashscope_video':
        thinking = payload.pop('chat_template_kwargs', {}).get('enable_thinking')
        if thinking is not None:
            payload['enable_thinking'] = thinking
    elif 'enable_thinking' in payload:
        payload.setdefault('chat_template_kwargs', {})['enable_thinking'] = payload.pop('enable_thinking')
    for message in payload.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        converted = []
        for part in content:
            if config['input_format'] == 'openai_images' and isinstance(part, dict) and part.get('type') == 'video':
                frames, fps = part['video'], float(part['fps'])
                if not 0 < fps or not isinstance(frames, list):
                    raise ValueError('invalid_frame_sequence')
                for i, uri in enumerate(frames):
                    converted.append({'type': 'text', 'text': f'Frame {i+1}, approximate clip-relative time {i/fps:.3f} seconds.'})
                    converted.append({'type': 'image_url', 'image_url': {'url': uri}})
            else:
                converted.append(part)
        message['content'] = converted
    payload.setdefault('max_tokens', config['default_max_output_tokens'])
    payload['stream'] = config.get('stream', True)
    if payload['stream']:
        payload['stream_options'] = {'include_usage': True}
    return payload


def read_stream(opener, request, *args, **kwargs):
    begin = time.monotonic()
    first_token = last_token = None
    content, reasoning = [], []
    usage, finish, envelope_id, model = None, None, None, None
    done = False
    chunks = 0
    with opener(request, *args, **kwargs) as response:
        for raw in response:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            packet = json.loads(data)
            if packet.get('error'):
                raise RuntimeError('vllm_stream_error:' + str(packet['error']))
            envelope_id = packet.get('id', envelope_id)
            model = packet.get('model', model)
            if packet.get('usage'):
                usage = packet['usage']
            for choice in packet.get('choices', []):
                if choice.get('index', 0) != 0:
                    raise ValueError('unexpected_multiple_completions')
                delta = choice.get('delta') or {}
                text = delta.get('content') or ''
                thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if delta.get('tool_calls'):
                    raise ValueError('unexpected_native_tool_calls')
                if text or thought:
                    instant = time.monotonic()
                    first_token = first_token if first_token is not None else instant
                    last_token = instant
                    chunks += 1
                content.append(text)
                reasoning.append(thought)
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
    elapsed = time.monotonic() - begin
    if not done or finish is None:
        raise RuntimeError('incomplete_completion_stream')
    usage = usage or {}
    output = usage.get('completion_tokens')
    generation = last_token-first_token if first_token is not None and last_token is not None else None
    timing = dict(http_seconds=elapsed,
        ttft_seconds=first_token-begin if first_token is not None else None,
        generation_seconds=generation, nonempty_stream_chunks=chunks,
        output_tokens_per_http_second=output/elapsed if isinstance(output, int) and elapsed > 0 else None,
        approximate_decode_tokens_per_second=(output-1)/generation if isinstance(output, int) and output > 1 and generation and generation > 0 else None)
    message = {'role': 'assistant', 'content': ''.join(content)}
    if any(reasoning):
        message['reasoning_content'] = ''.join(reasoning)
    envelope = dict(id=envelope_id, model=model, object='chat.completion',
        choices=[{'index': 0, 'message': message, 'finish_reason': finish}],
        usage=usage, _local_timing=timing)
    return io.BytesIO(json.dumps(envelope, ensure_ascii=False).encode())


def recorded_opener(recorder, opener):
    receive = (lambda request, *a, **kw: read_stream(opener, request, *a, **kw)) if recorder.config.get('stream', True) else opener
    wrapped = recorder.wrap(receive)
    def call(request, *args, **kwargs):
        endpoint = recorder.config['base_url'].rstrip('/') + '/chat/completions'
        if request.full_url != endpoint:
            raise ValueError('configured_endpoint_required')
        payload = local_payload(json.loads(request.data), recorder.config)
        converted = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                            headers=dict(request.header_items()), method='POST')
        return wrapped(converted, *args, **kwargs)
    return call
