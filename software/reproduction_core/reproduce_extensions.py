"""Run inside the staged packet; paths resolve relative to this script only."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];D=R/'data';E=R/'expected';checks=[]
def read(n):return json.loads((E/n).read_text())
def corr(a,b):
 a,b=np.broadcast_arrays(a,b);ok=np.isfinite(a)&np.isfinite(b);aa=np.where(ok,rankdata(np.where(ok,a,np.inf),axis=-1),np.nan);bb=np.where(ok,rankdata(np.where(ok,b,np.inf),axis=-1),np.nan);aa-=np.nanmean(aa,axis=-1,keepdims=True);bb-=np.nanmean(bb,axis=-1,keepdims=True);den=np.sqrt(np.nansum(aa*aa,-1)*np.nansum(bb*bb,-1));return np.divide(np.nansum(aa*bb,-1),den,out=np.full(den.shape,np.nan),where=(den>0)&(ok.sum(-1)>=3))
def metrics(p,y,norm,minimum=3):
 mask=np.isfinite(p)&np.isfinite(y);ok=mask.sum(-1)>=minimum
 return {k:np.where(ok,v,np.nan) for k,v in {'raw':corr(p,y),'distinctive':corr(p-norm,y-norm),'mae':np.nanmean(np.where(mask,abs(p-y),np.nan),axis=-1)}.items()}
def check(label,v,expected):
 actual=float(np.nanmean(v));assert abs(actual-expected)<1e-10,(label,actual,expected);checks.append({'label':label,'mean':actual,'expected':expected})
inference_checks=[]
def check_inference(label,v,groups,expected):
 groups=np.asarray(groups);v=np.asarray(v);ok=np.isfinite(v);labels=np.unique(groups[ok]);g=[v[(groups==k)&ok] for k in labels]
 if not len(g):assert expected['mean'] is None;return
 totals=np.array([a.sum() for a in g]);counts=np.array([len(a) for a in g]);rng=np.random.default_rng(20260916);draw=rng.integers(0,len(g),size=(5000,len(g)));boot=totals[draw].sum(1)/counts[draw].sum(1);ci=np.quantile(boot,[.025,.975]);assert np.max(abs(ci-expected['ci95']))<1e-10,label
 assert int(ok.sum())==expected['valid_people'] and len(g)==expected['groups']
 if 'p' in expected:
  exceed=0
  for _ in range(20):
   vals=((rng.integers(0,2,size=(1000,len(g)))*2-1)*totals).sum(1)/counts.sum();exceed+=np.count_nonzero(abs(vals)>=abs(v[ok].mean())-1e-15)
  assert abs((1+exceed)/20001-expected['p'])<1e-12,label
 inference_checks.append(label)
for n,h in json.loads((R/'MANIFEST.json').read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
a=np.load(D/'frontier.npz')
for row in read('frontier.json')['cells']:
 for k,v in metrics(a[row['cell']],a['truth'],a['norm'],60).items():check('frontier/'+row['cell']+'/'+k,v,row[k]['mean'])
a=np.load(D/'repeat.npz')
for row in read('repeat.json')['results']:
 p=a[row['model']];z=p-(p.sum(1,keepdims=True)-p)/99;v=np.mean([corr(z[i],z[j]) for i,j in itertools.combinations(range(3),2)],0);check('repeat/'+row['model'],v,row['shared_item_removed_same_person']['mean'])
ref=np.load(D/'external_reference.npz');a=np.load(D/'external_predictions.npz')
for block,b in read('external.json').items():
 for c,row in b['models'].items():
  if c.endswith('no_person'):continue
  p=a[block+'__'+c.replace('|','__')];met=metrics(p,ref[block+'__truth'],ref[block+'__population'])
  for k,v in met.items():check('external/'+block+'/'+c+'/'+k,v,row['matched'][{'raw':'raw_spearman','distinctive':'distinctive_spearman','mae':'mae'}[k]]['mean'])
a=np.load(D/'distortion.npz')
for row in read('distortion.json')['rows']:
 block=row['block'];before=metrics(a[block+'__original'],ref[block+'__truth'],ref[block+'__population']);after=metrics(a[block+'__'+row['decoder']],ref[block+'__truth'],ref[block+'__population'])
 for k in before:check('distortion/'+block+'/'+row['decoder']+'/'+k,after[k]-before[k],row['difference_recovered_minus_original'][{'raw':'raw_spearman','distinctive':'distinctive_spearman','mae':'mae'}[k]]['mean'])
a=np.load(D/'quantity.npz')
for row in read('quantity.json')['summary']:
 d=row['direction'];p=a[d+'__'+row['model']+'__'+row['condition']];met=metrics(p,a[d+'__truth'],a[d+'__norm'],40 if d=='taste_to_values' else 60)
 for k,v in met.items():check('quantity/'+d+'/'+row['model']+'/'+row['condition']+'/'+k,v,row['metrics'][k]['mean'])
a=np.load(D/'quantity_comparator.npz')
for row in read('quantity_comparator.json')['rows']:
 d=row['direction'];met=metrics(a[d+'__'+str(row['quantity'])],a[d+'__truth'],a[d+'__norm'],40 if d=='taste_to_values' else 60)
 for k,v in met.items():check('quantity_comparator/'+d+'/'+str(row['quantity'])+'/'+k,v,row['metrics'][k]['mean'])
a=np.load(D/'controlled.npz')
controlled={}
for label,row in read('controlled.json')['results'].items():
 model,block,condition=label.split('|');met=metrics(a[model+'__'+block+'__'+condition].mean(0),a[block+'__truth'],a[block+'__norm']);controlled[label]=met
 for k,v in met.items():check('controlled/'+label+'/'+k,v,row['metrics'][{'raw':'raw_spearman','distinctive':'distinctive_spearman','mae':'mae'}[k]]['mean'])
for row in read('controlled.json')['contrasts']:
 model,block=row['model'],row['block'];condition=row['comparison'].split(' minus ')[0];difference=controlled[model+'|'+block+'|'+condition]['distinctive']-controlled[model+'|'+block+'|numeric']['distinctive'];check('controlled_difference/'+model+'/'+block+'/'+condition,difference,row['distinctive']['mean'])
 check_inference('controlled/'+model+'/'+block+'/'+condition,difference,a[block+'__groups'],row['distinctive'])
expanded=np.load(D/'expanded_predictions.npz');ref=np.load(D/'expanded_reference.npz');ind=np.load(D/'indonesia_reference.npz');meters={}
for block,b in read('expanded.json')['blocks'].items():
 norm=expanded[block+'__norm'];truth=expanded[block+'__truth'];groups=ind['groups'] if block.startswith('indonesia') else ref[block+'__groups']
 for label,row in b['cells'].items():
  model,condition=label.split('|');met=metrics(expanded[block+'__'+model+'__'+condition],truth,norm);meters[block+'|'+label]=met
  for k,v in met.items():
   expected=row['matched'][{'raw':'raw_spearman','distinctive':'distinctive_spearman','mae':'mae'}[k]];check('expanded/'+block+'/'+label+'/'+k,v,expected['mean']);check_inference('expanded/'+block+'/'+label+'/'+k,v,groups,expected)
for row in read('expanded.json')['contrasts']:
 block,model=row['block'],row['model'];b,a=row['comparison'].split(' minus ');v=meters[block+'|'+model+'|'+b]['distinctive']-meters[block+'|'+model+'|'+a]['distinctive'];groups=ind['groups'] if block.startswith('indonesia') else ref[block+'__groups'];check('expanded_difference/'+block+'/'+model+'/'+b+'-'+a,v,row['distinctive']['mean']);check_inference('expanded_difference/'+block+'/'+model+'/'+b+'-'+a,v,groups,row['distinctive'])
for row in read('expanded.json')['readbacks']:
 prefix='readback__'+row['cohort']+'__'+row['mode']+'__';y=expanded[prefix+'truth'];z=expanded[prefix+'estimate'];mask=np.ones(len(y),bool) if row['subset']=='all' else expanded[prefix+row['subset']];check('readback/'+prefix+row['subset'],corr(y[mask].T,z[mask].T),row['mean_domain_rank']);check('readback_mae/'+prefix+row['subset'],abs(y[mask]-z[mask]),row['mae'])
dist=np.load(D/'expanded_distortion.npz')
for row in read('expanded_distortion.json')['rows']:
 block=row['block'];before=metrics(dist[block+'__original'],ref[block+'__truth'],ref[block+'__population']);after=metrics(dist[block+'__'+row['decoder']],ref[block+'__truth'],ref[block+'__population'])
 for k in before:
  expected=row['difference_recovered_minus_original'][{'raw':'raw_spearman','distinctive':'distinctive_spearman','mae':'mae'}[k]];v=after[k]-before[k];check('expanded_distortion/'+block+'/'+row['decoder']+'/'+k,v,expected['mean']);check_inference('expanded_distortion/'+block+'/'+row['decoder']+'/'+k,v,ref[block+'__groups'],expected)
assign=read('expanded_assignments.json')
for row in read('expanded_controls.json')['rows']:
 block='__'.join(row['cell'].split('__')[:2]);p=expanded[row['cell']];y=expanded[block+'__truth'];norm=expanded[block+'__norm'];donor=np.array(assign[block]['donor_indices']);clusters=np.array(assign[block]['clusters']);keep=donor>=0;v=corr(p[keep]-norm[keep],y[keep]-norm[keep])-corr(p[donor[keep]]-norm[keep],y[keep]-norm[keep]);check('expanded_near/'+row['cell'],v,row['near_swap_gain']['mean']);check_inference('expanded_near/'+row['cell'],v,clusters[keep],row['near_swap_gain'])
out=R/'output';out.mkdir(exist_ok=True);(out/'EXTENSION_VERIFICATION.json').write_text(json.dumps({'status':'PASS','compared_means':len(checks),'grouped_interval_tests_checked':len(inference_checks),'max_absolute_difference':max(abs(x['mean']-x['expected']) for x in checks),'checks':checks,'inference_checks':inference_checks},indent=2));print('PASS:',len(checks),'independently recomputed means;',len(inference_checks),'grouped intervals/tests')
