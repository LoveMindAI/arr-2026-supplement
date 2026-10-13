#!/usr/bin/env python3
"""Run the BESSI college cohort, both directions (prepared by bessi_prepare.py). Mirrors the original frozen Panorama external-cohort runner
(04_SEPTEMBER_ARCHIVE/external_execution_20260916/run.py): same model routes, providers, quantization, temperatures, reasoning
settings, JSON mode, delivery checks (model id + provider), finish_reason check, two attempts, cost receipts and a spend cap.
Phases: portrait (Gemma draft -> Gemma edit with source scores -> terminology repair of flagged paragraphs) -> Luna continuous readback
-> predictions by Sol, GLM, Flash for numeric / structured / portrait, plus a no-person baseline per direction and route.
One deliberate difference: max_tokens is 16384 for every prediction call (Sept: 4096 Sol / 2048 others). The 192-item answer needs the
room, and in the 5-person smoke GLM's reasoning truncated even the 60-item answer at 2048. Flash spent 0 reasoning tokens and Sol's
reasoning stayed small at every limit, so the change only removes truncation. OpenRouter reports $0 for GLM on this route, so the spend counter here excludes GLM.
Usage: bessi_run.py --smoke 5     (first 5 evaluation people, both directions, plus the no-person baselines)
       bessi_run.py               (all 200)"""
import argparse, concurrent.futures as cf, fcntl, hashlib, json, os, re, threading, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "bessi_run"
CFG = {'gemma': ('google/gemma-4-31b-it', 'Novita', 'novita/bf16', .14e-6, .4e-6), 'luna': ('openai/gpt-5.6-luna', 'OpenAI', 'openai', .2e-6, 1.2e-6),
       'sol': ('openai/gpt-5.6-sol', 'OpenAI', 'openai', 2e-6, 10e-6), 'glm': ('z-ai/glm-5.2', 'Mistral', 'mistral', 1.4e-6, 4.4e-6),
       'flash': ('google/gemini-3.5-flash-lite', 'Google AI Studio', 'google-ai-studio', .3e-6, 2.5e-6)}
DATED = {'gemma': '-20260402', 'luna': '-20260709', 'sol': '-20260709', 'glm': '-20260616', 'flash': '-20260721'}
PAT = re.compile(r'\b(?:openness|open-mindedness|conscientiousness|extraversion|extroversion|introversion|agreeableness|neuroticism|negative emotionality|'
                 r'self-management|social engagement|cooperation skills?|emotional resilience|innovation skills?|Big Five|BFI(?:-44|-2)?|BESSI|'
                 r'percentile|psychometric|questionnaire)\b', re.I)
DIRS = {"bfi2_to_bessi": {"source": "bfi2", "target": "bessi", "reader_dom": ["E", "A", "C", "N", "O"]},
        "bessi_to_bfi2": {"source": "bessi", "target": "bfi2", "reader_dom": ["SM", "SE", "CO", "ER", "IN"]}}
INST_NAME = {"bfi2": "BFI-2", "bessi": "BESSI"}
LOCK = threading.Lock(); SPEND = 0.0; RESERVED = 0.0; CAP = 90.0; HOLDS = {}; FAILURES = {}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(x): return hashlib.sha256(x if isinstance(x, bytes) else x.encode()).hexdigest()
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); tmp.replace(p)
def request(path, body=None):
    req = urllib.request.Request('https://openrouter.ai/api/v1/' + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=240) as f: return json.load(f)
def json_parse(text):
    text = text.strip()
    if text.startswith('```'):
        m = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', text); assert m, 'Non-JSON explanation'; text = m[1]
    return json.loads(text)

def call(key, route, messages, validator, kind='json', max_tokens=2048):
    global SPEND, RESERVED
    dest = ROOT / 'calls' / (key + '.json'); h = sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    if dest.exists():
        old = json.loads(dest.read_text()); assert old['prompt_sha256'] == h, 'Prompt changed: ' + key; return old
    model, provider, tag, ir, orr = CFG[route]
    body = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'provider': {'only': [tag], 'allow_fallbacks': False, 'require_parameters': True}}
    if route == 'gemma': body.update(temperature=.65, reasoning={'enabled': False}); body['provider']['quantizations'] = ['bf16']
    elif route == 'luna': body['reasoning'] = {'effort': 'none'}
    else:
        body['reasoning'] = {'effort': 'minimal' if route == 'flash' else 'low'}
        if route != 'sol': body['temperature'] = .7
    if kind == 'json' and route != 'gemma': body['response_format'] = {'type': 'json_object'}
    reserve = (len(json.dumps(messages, ensure_ascii=False).encode()) + 1024) * ir + max_tokens * orr
    for attempt in range(1, 3):
        ap = ROOT / 'attempts' / (key + f'__attempt{attempt}.json')
        if ap.exists(): continue
        with LOCK:
            if route in HOLDS: return {'valid': False, 'held': HOLDS[route], 'key': key}
            if SPEND + RESERVED + reserve > CAP: return {'valid': False, 'error': 'budget capacity', 'key': key}
            RESERVED += reserve
        save(ROOT / 'requests' / (key + f'__attempt{attempt}.json'), body)
        record = {'key': key, 'route': route, 'attempt': attempt, 'valid': False, 'prompt_sha256': h, 'started_at': now()}
        try:
            data = request('chat/completions', body); save(ROOT / 'responses' / (key + f'__attempt{attempt}.json'), data)
            cost = (data.get('usage') or {}).get('cost'); assert type(cost) in (int, float), 'Missing cost receipt'; record['cost_usd'] = cost
            assert data.get('model') in {model, model + DATED[route]} and data.get('provider') == provider, f"DELIVERY_MISMATCH {data.get('model')} {data.get('provider')}"
            choice = data['choices'][0]; assert choice['finish_reason'] == 'stop', 'Incomplete output: ' + str(choice.get('finish_reason'))
            rt = (data['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
            if route in ['gemma', 'luna']: assert rt in (0, None), 'REASONING_MISMATCH'
            text = choice['message']['content']; assert isinstance(text, str) and text.strip()
            parsed = validator(text); record.update(valid=True, text=text, parsed=parsed, usage=data['usage'], delivered_model=data['model'], provider=data['provider'])
        except urllib.error.HTTPError as e:
            record.update(error='HTTP ' + str(e.code), detail=e.read().decode(errors='replace')[:1000])
            if e.code in [400, 401, 402, 403, 404]:
                with LOCK: HOLDS[route] = record['error']
            if e.code in [400, 401, 402, 403, 404, 429]: record['cost_usd'] = 0
        except Exception as e:
            record['error'] = type(e).__name__ + ': ' + str(e)[:300]
            if any(k in str(e) for k in ['MISMATCH', 'Missing cost']):
                with LOCK: HOLDS[route] = record['error']
        record['finished_at'] = now(); save(ap, record)
        with LOCK:
            SPEND += record.get('cost_usd', reserve); RESERVED -= reserve
            if not record['valid']:
                FAILURES[route] = FAILURES.get(route, 0) + 1
                if FAILURES[route] >= 10: HOLDS[route] = 'Ten failed attempts; inspect before further dispatch'
        if record['valid']: save(dest, record); return record
        time.sleep(attempt)
    return {'key': key, 'valid': False, 'error': 'two attempts exhausted'}

def portrait(p, d):
    pid = p['id']; tag = f"{pid}__{d}"; dest = ROOT / 'portraits' / (tag + '.txt'); rp = ROOT / 'portrait_receipts' / (tag + '.json')
    if rp.exists():
        r = json.loads(rp.read_text()); assert sha(dest.read_bytes()) == r['sha256']; return True
    pk = p[d]
    a = call(tag + '__writer', 'gemma', [{'role': 'user', 'content': pk['writer']}], lambda t: t, 'text', 6000)
    if not a['valid']: return False
    b = call(tag + '__edit', 'gemma', [{'role': 'user', 'content': pk['editor_template'].replace('{{DRAFT}}', a['text'])}], lambda t: t, 'text', 6000)
    if not b['valid']: return False
    text = b['text']; record = {'writer_key': a['key'], 'edit_key': b['key']}
    parts = re.split(r'(\n[ \t]*\n)', text); flagged = {str(i): v for i, v in enumerate(parts) if i % 2 == 0 and PAT.search(v)}
    if flagged:
        prompt = (ROOT / 'templates' / 'REPAIR.txt').read_text().replace('{{PORTRAIT}}', text).replace('{{PARAGRAPHS}}', json.dumps(flagged, ensure_ascii=False))
        def valid_patch(t):
            q = json_parse(t); assert isinstance(q, dict) and set(q) == set(flagged), 'Wrong paragraph keys'
            for v in q.values(): assert isinstance(v, str) and v.strip() and '\n' not in v and not PAT.search(v), 'Invalid or prohibited replacement'
            return q
        z = call(tag + '__terminology', 'gemma', [{'role': 'user', 'content': prompt}], valid_patch, 'json', 6000)
        if not z['valid']: return False
        for k, v in z['parsed'].items(): parts[int(k)] = v
        text = ''.join(parts); record['terminology_key'] = z['key']; record['repaired_paragraphs'] = len(flagged)
    assert not PAT.search(text); dest.parent.mkdir(exist_ok=True); dest.write_text(text)
    save(rp, {'id': pid, 'direction': d, 'sha256': sha(text), 'words': len(text.split()), **record}); return True

def readback(p, d):
    src = DIRS[d]['source']; dom = DIRS[d]['reader_dom']; tag = f"{p['id']}__{d}"
    text = (ROOT / 'portraits' / (tag + '.txt')).read_text()
    system = (ROOT / 'templates' / f"READER_{src.upper()}.txt").read_text()
    target = json.loads((ROOT / 'READER_TARGETS.json').read_text())[src]
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps({'text': text, 'instrument': INST_NAME[src], 'target': target}, ensure_ascii=False)}]
    def v(t):
        x = json_parse(t); assert set(x) == {'domains'} and set(x['domains']) == set(dom), 'Wrong readback keys'
        assert all(type(y) in (int, float) and 1 <= y <= 5 for y in x['domains'].values()), 'Readback out of range'; return x['domains']
    r = call(tag + '__readback', 'luna', messages, v)
    if r['valid']: save(ROOT / 'readbacks' / (tag + '.json'), {'id': p['id'], 'direction': d, 'domains': r['parsed'], 'portrait_sha256': sha(text)})
    return r

def prediction(p, d, condition, route):
    tgt = DIRS[d]['target']; task = json.loads((ROOT / 'TARGETS.json').read_text())[tgt]
    allowed = {k: set(int(s) for s in q['scale']) for q in task for k in q['items']}
    def validate(t):
        x = json_parse(t); assert isinstance(x, dict) and set(x) == set(allowed), 'Missing or extra item keys'
        # Flash sometimes quotes integers ("3"). Accept an exact integer string, unchanged in value; the record is flagged
        # (coerced_string_ints) in predictions/. Anything else (decimals, words, out of range) stays invalid.
        x = {k: (int(v) if isinstance(v, str) and re.fullmatch(r'[1-9]', v.strip()) else v) for k, v in x.items()}
        assert all(type(v) == int and v in allowed[k] for k, v in x.items()), 'Non-integer or out-of-range answer'; return x
    if condition == 'no_person':
        text = 'No individual personality information is available. Predict the responses of a typical adult to the questions below. Return a JSON object mapping exactly each item ID to one integer on its supplied response scale.\n' + json.dumps(task, ensure_ascii=False); pid = 'no_person'
    else:
        pid = p['id']; packet = (ROOT / 'portraits' / f"{pid}__{d}.txt").read_text() if condition == 'portrait' else p[d][condition]
        text = 'You will answer questions as a specific real person. Stay consistent with the supplied personality information.\n\n' + packet + '\n\nAnswer the following as this person. Return a JSON object mapping exactly each item ID to one integer on its supplied response scale. Do not add explanations.\n' + json.dumps(task, ensure_ascii=False)
    key = f"{pid}__{d}__{condition}__{route}"
    mt = 16384   # smoke (Oct 9): GLM's reasoning (up to ~4.9k tokens) truncated the 60-item reply at 2048; Flash used 0 reasoning tokens
                 # and Sol's stayed small at every limit, so the larger limit only removes truncation.
    r = call(key, route, [{'role': 'user', 'content': text}], validate, max_tokens=mt)
    if r['valid']: save(ROOT / 'predictions' / (key + '.json'), {'id': None if condition == 'no_person' else pid, 'direction': d, 'condition': condition, 'route': route, 'model': CFG[route][0], 'answers': r['parsed'], 'key': key, 'prompt_sha256': r['prompt_sha256'], 'coerced_string_ints': bool(re.search(r'"\s*[1-9]\s*"', r.get('text') or ''))})
    return r

def main():
    global SPEND, CAP
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', type=int, default=0); ap.add_argument('--cap', type=float, default=90.0); ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    lock = (ROOT / 'RUN.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    SPEND = sum(json.loads(x.read_text()).get('cost_usd', 0) for sub in ('attempts', 'superseded_maxtokens/attempts') if (ROOT / sub).exists() for x in (ROOT / sub).glob('*.json'))
    w = request('credits')['data']; bal = w['total_credits'] - w['total_usage']; CAP = min(a.cap, SPEND + max(0, bal - 2))
    people = json.loads((ROOT / 'PEOPLE.json').read_text()); people = people[:a.smoke] if a.smoke else people
    state = ROOT / ('SMOKE_STATE.json' if a.smoke else 'RUN_STATE.json')
    save(state, {'status': 'running', 'started_at': now(), 'people': len(people), 'cap_usd': CAP, 'prior_spend': SPEND, 'wallet_balance': bal})
    print(json.dumps({'phase': 'start', 'people': len(people), 'cap': CAP, 'prior_spend': SPEND}), flush=True)
    with cf.ThreadPoolExecutor(a.workers) as pool:
        res = [f.result() for f in cf.as_completed([pool.submit(portrait, p, d) for p in people for d in DIRS])]
    print(json.dumps({'phase': 'portraits', 'valid': sum(res), 'expected': len(res), 'spend': round(SPEND, 4), 'holds': HOLDS}), flush=True)
    ok = {(p['id'], d) for p in people for d in DIRS if (ROOT / 'portrait_receipts' / f"{p['id']}__{d}.json").exists()}
    jobs = [(readback, p, d) for p in people for d in DIRS if (p['id'], d) in ok]
    jobs += [(prediction, None, d, 'no_person', r) for d in DIRS for r in ['sol', 'glm', 'flash']]
    jobs += [(prediction, p, d, c, r) for p in people for d in DIRS if (p['id'], d) in ok for c in ['numeric', 'structured', 'portrait'] for r in ['sol', 'glm', 'flash']]
    done = valid = 0
    with cf.ThreadPoolExecutor(a.workers) as pool:
        for f in cf.as_completed([pool.submit(fn, *args) for fn, *args in jobs]):
            x = f.result(); done += 1; valid += bool(x.get('valid'))
            if done % 100 == 0: print(json.dumps({'phase': 'downstream', 'done': done, 'of': len(jobs), 'valid': valid, 'spend': round(SPEND, 4)}), flush=True)
    actual = sum(json.loads(x.read_text()).get('cost_usd', 0) for x in (ROOT / 'attempts').glob('*.json'))
    counts = {'portraits': len(ok), 'readbacks': sum(1 for p in people for d in DIRS if (ROOT / 'readbacks' / f"{p['id']}__{d}.json").exists()),
              'predictions': sum(1 for p in people for d in DIRS for c in ['numeric', 'structured', 'portrait'] for r in ['sol', 'glm', 'flash'] if (ROOT / 'predictions' / f"{p['id']}__{d}__{c}__{r}.json").exists()),
              'no_person': sum(1 for d in DIRS for r in ['sol', 'glm', 'flash'] if (ROOT / 'predictions' / f"no_person__{d}__no_person__{r}.json").exists())}
    exp = {'portraits': 2 * len(people), 'readbacks': 2 * len(people), 'predictions': 18 * len(people), 'no_person': 6}
    save(state, {'status': 'complete' if counts == exp else 'incomplete', 'counts': counts, 'expected': exp, 'reported_cost_total': actual, 'holds': HOLDS, 'finished_at': now()})
    print(state.read_text(), flush=True); print('FINISHED', flush=True)

if __name__ == '__main__':
    main()
