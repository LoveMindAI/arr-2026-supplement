"""Recompute all matched reader results; persist aggregates, never person records."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
SOURCE=Path('<PROJECT_ROOT>/experiments_rescue/expC_haiku_reader')
tasks=[json.loads(l) for l in (SOURCE/'tasks.jsonl').read_text().splitlines()]
key=lambda r:(r['set'],r['pid'],r['mode'])
target={key(t) for t in tasks}
assert len(tasks)==len(target)==1096
saved=json.loads((SOURCE/'RESULTS_C3.json').read_text())
groups={'Twin (150)':['twin_sept','twin_oct'],'Harmonies (148)':['harmonies_sept'],
        'Indonesia (100)':['indonesia_oct'],'College BFI-2 (200)':['college_bfi2'],
        'College skills (200)':['college_bessi']}
out={}
for reader,file,model in [('haiku','haiku.jsonl','anthropic/claude-haiku-5.5'),
                          ('mistral','mistral.jsonl','mistralai/mistral-large-4-0')]:
    rows=sorted([json.loads(l) for l in (SOURCE/file).read_text().splitlines()],key=key)
    assert len(rows)==len(set(map(key,rows)))==1096
    assert set(map(key,rows))==target
    assert all(r['model']==model for r in rows)
    assert all(set(r['haiku'])==set(r['truth'])==set(r['luna']) for r in rows)
    assert all(1<=v<=5 for r in rows for v in r['haiku'].values())
    for name,sets in groups.items():
        for mode in ['continuous','questionnaire','readback']:
            rr=[r for r in rows if r['set'] in sets and r['mode']==mode]
            if not rr:continue
            panel=f'{name} {mode}'; ds=sorted(rr[0]['truth'])
            out.setdefault(panel,{'n':len(rr)})
            # The generic execution script stores both new models under "haiku".
            for label,field in [(reader,'haiku'),('luna','luna')]:
                ranks={d:float(spearmanr([r[field][d] for r in rr],[r['truth'][d] for r in rr]).statistic) for d in ds}
                mean=float(np.mean(list(ranks.values())))
                mae=float(np.mean([abs(r[field][d]-r['truth'][d]) for r in rr for d in ds]))
                bias={d:float(np.mean([r[field][d]-r['truth'][d] for r in rr])) for d in ds}
                assert np.allclose([mean,mae,len(rr)],saved[label][panel],rtol=0,atol=1e-12)
                rec={'mean_rank_r':mean,'mae':mae,'domain_rank_r':ranks,'bias':bias}
                if label in out[panel]: assert rec==out[panel][label]
                out[panel][label]=rec
    print(reader, len(rows), dict(Counter(r.get('provider') for r in rows)),flush=True)
assert all(r['haiku']['mae']<r['luna']['mae'] and r['haiku']['mae']<r['mistral']['mae'] for r in out.values())
for name in ['Twin (150) questionnaire','Harmonies (148) questionnaire']:
    assert out[name]['haiku']['bias']['A']>0 and out[name]['haiku']['bias']['C']>0
(HERE/'READERS_READY.json').write_text(json.dumps(out,indent=2)+'\n')
(HERE/'THREE_READER_SOURCE_HASHES.json').write_text(json.dumps({name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['tasks.jsonl','haiku.jsonl','mistral.jsonl','RESULTS_C3.json','run_reader.py']},indent=2)+'\n')
print('All 21 reader-panel rank and MAE results match the three-reader source; biases recomputed.')
