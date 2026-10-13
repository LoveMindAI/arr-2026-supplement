"""Check newly posted reader and 32B record results; aggregate output only."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent
EXP=Path('<PROJECT_ROOT>/experiments_rescue')
spec=importlib.util.spec_from_file_location('C',EXP/'common.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
rows=[json.loads(l) for l in (EXP/'expC_haiku_reader/haiku.jsonl').read_text().splitlines()]
tasks=[json.loads(l) for l in (EXP/'expC_haiku_reader/tasks.jsonl').read_text().splitlines()]
key=lambda r:(r['set'],r['pid'],r['mode'])
assert len(rows)==len(set(map(key,rows)))==len(tasks)==1096
assert set(map(key,rows))==set(map(key,tasks))
assert all(set(r['haiku'])==set(r['truth'])==set(r['luna']) for r in rows)
assert all(1<=v<=5 for r in rows for v in r['haiku'].values())
assert all('haiku-5.5' in r['model'] for r in rows)
groups={'Twin (150)':['twin_sept','twin_oct'],'Harmonies (148)':['harmonies_sept'],
        'Indonesia (100)':['indonesia_oct'],'College BFI-2 (200)':['college_bfi2'],
        'College skills (200)':['college_bessi']}
saved=json.loads((EXP/'expC_haiku_reader/RESULTS_C.json').read_text())
out={}
for g,sets in groups.items():
    for mode in ['continuous','questionnaire','readback']:
        rr=[r for r in rows if r['set'] in sets and r['mode']==mode]
        if not rr:continue
        name=f'{g} {mode}';ds=sorted(rr[0]['truth']);out[name]={'n':len(rr)}
        for reader in ['luna','haiku']:
            rho={d:float(spearmanr([r[reader][d] for r in rr],[r['truth'][d] for r in rr]).statistic) for d in ds}
            mae=float(np.mean([abs(r[reader][d]-r['truth'][d]) for r in rr for d in ds]))
            bias={d:float(np.mean([r[reader][d]-r['truth'][d] for r in rr])) for d in ds}
            mean=float(np.mean(list(rho.values())))
            assert np.allclose([mean,mae,len(rr)],saved[reader][name],atol=1e-12,rtol=0)
            out[name][reader]={'mean_rank_r':mean,'mae':mae,'domain_rank_r':rho,'bias':bias}
print('Reader cells verified:',len(out),'requests:',len(rows),'models:',dict(Counter(r['model'] for r in rows)),flush=True)
(ROOT/'READERS_READY.json').write_text(json.dumps(out,indent=2)+'\n')
late_paths=['RESULTS.md','expC_haiku_reader/RESULTS_C.json','expC_haiku_reader/haiku.jsonl',
            'expC_haiku_reader/tasks.jsonl','expC_haiku_reader/collect.py','expC_haiku_reader/run_reader.py']
rec={}
for tag in ['allenai_Olmo-3-1125-32B','allenai_Olmo-3.1-32B-Instruct']:
    path=f'expB/out_record_local/{tag}.jsonl';late_paths.append(path)
    d=pd.read_json(EXP/path,lines=True)
    p=d.pivot(index='pid',columns='item',values='ev').astype(float)
    assert p.shape==(467,109) and not p.isna().any().any()
    rec[tag]={'n':len(p),'amplitude':C.jackknife_amp(p,B=100),'direction':C.amplitude(p)[1],
              'distinctive':C.boot_mean(C.distinctive(p)),'digit_mass':float(d['mass'].mean())}
    print(tag,json.dumps(rec[tag]),flush=True)
(ROOT/'RECORD_32B_READY.json').write_text(json.dumps(rec,indent=2)+'\n')
(ROOT/'LATE_SOURCE_HASHES.json').write_text(json.dumps({p:hashlib.sha256((EXP/p).read_bytes()).hexdigest() for p in late_paths},indent=2)+'\n')
