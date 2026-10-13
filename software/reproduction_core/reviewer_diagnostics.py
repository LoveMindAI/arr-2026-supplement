"""Portable, offline diagnostics on saved Panorama item predictions."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ('flash','glm52','minimaxm3','inkling','qwen37','mimopro','terra','sonnet5')
CONDITIONS = ('C-fac','C-cardBF','C-cardFull')
SEED = 20260915

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def row_mean(a):
    n=np.isfinite(a).sum(axis=1)
    return np.divide(np.nansum(a,axis=1),n,out=np.full(len(a),np.nan),where=n>0)

def correlations(a,b):
    valid=np.isfinite(a)&np.isfinite(b)
    aa=rankdata(np.where(valid,a,np.nan),axis=1,nan_policy='omit')
    bb=rankdata(np.where(valid,b,np.nan),axis=1,nan_policy='omit')
    aa=aa-row_mean(aa)[:,None]
    bb=bb-row_mean(bb)[:,None]
    den=np.sqrt(np.nansum(aa*aa,axis=1)*np.nansum(bb*bb,axis=1))
    result=np.full(len(a),np.nan)
    np.divide(np.nansum(aa*bb,axis=1),den,out=result,where=(den>0)&(valid.sum(axis=1)>=3))
    return result

def finite_mean(a):
    return float(np.nanmean(a)) if np.isfinite(a).any() else None

def paired_inference(d,rng):
    boots=[]
    for _ in range(20):
        indices=rng.integers(0,len(d),size=(1000,len(d)))
        boots.extend(d[indices].mean(axis=1))
    observed=abs(float(d.mean()))
    exceed=0
    for _ in range(50):
        signs=rng.choice(np.array([-1.0,1.0]),size=(1000,len(d)))
        exceed+=int((np.abs((signs*d).mean(axis=1))>=observed-1e-15).sum())
    return np.quantile(boots,[.025,.975]).tolist(),(exceed+1)/50001

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis-dir',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    base=args.analysis_dir
    paths={
      'items':base/'MEDIA469_ITEM_PREDICTIONS.csv',
      'scores':base/'MEDIA469_PERSON_SCORES.csv',
      'ridge_items':base/'MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PREDICTIONS_LONG.csv',
      'ridge_scores':base/'MEDIA469_ECOSYSTEM/HUMAN_RIDGE_PERSON_SCORES.csv'}
    items=pd.read_csv(paths['items'])
    scores=pd.read_csv(paths['scores'])
    ri=pd.read_csv(paths['ridge_items'])
    rs=pd.read_csv(paths['ridge_scores']).set_index('escs_id')
    assert not items.duplicated(['escs_id','model','condition','item']).any()
    assert not scores.duplicated(['escs_id','model','condition']).any()
    assert not ri.duplicated(['escs_id','item']).any()
    assert set(items.model)==set(MODELS) and set(items.condition)==set(CONDITIONS)
    assert len(scores)==467*8*3
    assert items.groupby(['escs_id','item']).truth.nunique().max()==1
    assert items.groupby('item').disjoint_norm.nunique().max()==1
    coverage=items.groupby(['escs_id','item']).size()
    assert coverage.le(24).all()
    truth=items[['escs_id','item','truth','disjoint_norm']].drop_duplicates()
    joined=truth.merge(ri,on=['escs_id','item'],suffixes=('_model','_ridge'),validate='one_to_one')
    assert len(joined)==len(truth)
    assert np.allclose(joined.truth_model,joined.truth_ridge)
    assert np.allclose(joined.disjoint_norm_model,joined.disjoint_norm_ridge)
    ids=sorted(scores.escs_id.unique())
    keys=sorted(truth.item.unique())
    assert len(ids)==467 and len(keys)==109
    def wide(frame,column):
        return frame.pivot(index='escs_id',columns='item',values=column).reindex(index=ids,columns=keys).to_numpy(float)
    y=wide(truth,'truth'); norm=wide(truth,'disjoint_norm'); ridge=wide(ri,'prediction')
    ridge_dist=correlations(ridge-norm,y-norm)
    assert np.allclose(ridge_dist,rs.loc[ids].distinctive_spearman,equal_nan=True,atol=1e-12)
    assert np.allclose(correlations(ridge,y),rs.loc[ids].raw_spearman,equal_nan=True,atol=1e-12)
    groups={'all':np.ones(len(keys),dtype=bool)}
    for prefix,n in [('music',22),('books',35),('movies',18),('tv',34)]:
        groups[prefix]=np.array([k.startswith(prefix) for k in keys]);assert groups[prefix].sum()==n
    def summarize(p,model,condition):
        result=[]
        for domain,mask in groups.items():
            a=p[:,mask];b=y[:,mask];n=norm[:,mask]
            valid=np.isfinite(a)&np.isfinite(b)
            minimum=60 if domain=='all' else max(3,math.ceil(mask.sum()/2))
            eligible=valid.sum(axis=1)>=minimum
            raw=correlations(a,b); dist=correlations(a-n,b-n)
            raw[~eligible]=np.nan;dist[~eligible]=np.nan
            errors=np.where(valid,a-b,np.nan)
            mae=row_mean(np.abs(errors));rmse=np.sqrt(row_mean(errors**2))
            mae[~eligible]=np.nan;rmse[~eligible]=np.nan
            result.append({'model':model,'condition':condition,'domain':domain,'n_items':int(mask.sum()),'n_eligible_people':int(eligible.sum()),'n_raw_correlations':int(np.isfinite(raw).sum()),'n_distinctive_correlations':int(np.isfinite(dist).sum()),'raw_spearman':finite_mean(raw),'distinctive_spearman':finite_mean(dist),'mae':finite_mean(mae),'rmse':finite_mean(rmse)})
        return result
    paired=[];domain_rows=[];rng=np.random.default_rng(SEED)
    for model in MODELS:
        for condition in CONDITIONS:
            block=items[(items.model==model)&(items.condition==condition)]
            pred=wide(block,'prediction')
            assert not (np.isfinite(pred)&~np.isfinite(y)).any()
            valid=np.isfinite(pred)&np.isfinite(y)
            stored=scores[(scores.model==model)&(scores.condition==condition)].set_index('escs_id').loc[ids]
            assert np.array_equal(valid.sum(axis=1),stored.n_items)
            assert np.allclose(correlations(pred,y),stored.raw_spearman,equal_nan=True,atol=1e-12)
            assert np.allclose(correlations(pred-norm,y-norm),stored.distinctive_spearman,equal_nan=True,atol=1e-12)
            domain_rows.extend(summarize(pred,model,condition))
            if condition=='C-fac':
                d=correlations(pred-norm,y-norm)
                matched_ridge=correlations(np.where(valid,ridge-norm,np.nan),y-norm)
                delta=matched_ridge-d
                assert np.isfinite(delta).all()
                ci,p=paired_inference(delta,rng)
                paired.append({'model':model,'n':len(ids),'ridge_mean':float(matched_ridge.mean()),'model_mean':float(d.mean()),'ridge_minus_model':float(delta.mean()),'ci95':ci,'sign_flip_p':p,'model_missing_observed_items':int((np.isfinite(y)&~valid).sum()),'comparison':'Both systems scored on each participant-model shared observed-item mask'})
    order=np.argsort([r['sign_flip_p'] for r in paired]);last=0.0
    for rank,index in enumerate(order):
        last=max(last,min(1.0,(len(paired)-rank)*paired[index]['sign_flip_p']))
        paired[index]['holm_p']=last
    domain_rows.extend(summarize(ridge,'ridge35','C-fac'))
    domain_rows.extend(summarize(norm,'population_mean','none'))
    result={'analysis':'post-submission reviewer-responsive analysis of fixed saved predictions','seed':SEED,'paired_comparisons':paired,'domain_summaries':domain_rows,'coverage':{'unique_observed_person_items':len(truth),'saved_item_predictions':len(items),'missing_model_ratings_on_observed_targets':int(24*len(truth)-len(items))},'validation':'Unique keys, shared targets/norms, original correlations for all 24 model-condition cells and ridge verified. Paired comparisons use identical item masks per model and participant. No raw model-output or source-data audit performed here.','inference_scope':'Fixed items, fitted model outputs and normative estimates. Participant bootstrap and sign-flip test; Holm family of eight distinctive-accuracy comparisons.','input_hashes':{k:{'file':str(p.relative_to(base)),'sha256':sha(p)} for k,p in paths.items()}}
    (args.out_dir/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    report=['# First local reviewer-responsive results','',result['analysis'],'','## Ridge versus facet-conditioned models','','| System | Model | Ridge | Difference | Paired 95% CI | Holm p |','| --- | ---: | ---: | ---: | --- | ---: |']
    for r in paired:report.append(f"| {r['model']} | {r['model_mean']:.4f} | {r['ridge_mean']:.4f} | {r['ridge_minus_model']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['holm_p']:.5f} |")
    report.extend(['',result['inference_scope'],'',result['validation'],'','Domain results are descriptive and stored in RESULTS.json. Correlation denominators are reported separately from error denominators. Undefined population-residual correlations remain missing.'])
    (args.out_dir/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))

if __name__=='__main__':main()
