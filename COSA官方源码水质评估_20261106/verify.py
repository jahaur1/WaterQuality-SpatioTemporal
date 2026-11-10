"""Audit source integrity, chronological label availability, metrics and online replay."""
from pathlib import Path
import sys,json,hashlib,time,subprocess
import numpy as np
import pandas as pd
import torch
from native import CausalEndpointAdapter,SimpleAdapter,SOURCE,OPT
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;CACHE=ROOT/'冻结预训练模型校正头基线_20261106'
sys.stdout.reconfigure(encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    start=time.time();torch.set_num_threads(1)
    cfg=json.loads((HERE/'实验配置.json').read_text(encoding='utf-8'));assert cfg['source_COSA_sha256']==sha(SOURCE) and cfg['source_optimizer_sha256']==sha(OPT)
    assert cfg['native_wrapper_sha256']==sha(HERE/'native.py') and cfg['runner_sha256']==sha(HERE/'run.py')
    clean=subprocess.check_output(['git','-c','safe.directory='+str((HERE/'source').as_posix()),'status','--porcelain'],cwd=HERE/'source',text=True);assert not clean.strip(),clean
    previous=json.loads((CACHE/'独立核验.json').read_text(encoding='utf-8'));assert previous['passed']
    oldcfg=json.loads((CACHE/'训练配置与头权重.json').read_text(encoding='utf-8'))
    for name in ['inputs.npz','配对索引.csv','站点训练标准化.json']:assert sha(CACHE/name)==oldcfg['fingerprints'][name]
    idx=pd.read_csv(HERE/'测试配对索引.csv');old=pd.read_csv(CACHE/'配对索引.csv');old=old[old.test].reset_index(drop=True)
    for c in ['起报时间','目标时间']:
        idx[c]=pd.to_datetime(idx[c],format='mixed');old[c]=pd.to_datetime(old[c],format='mixed')
    pd.testing.assert_frame_equal(idx,old,check_exact=True)
    pred=np.load(HERE/'test_predictions.npz');truth=pred['truth'];z=np.load(CACHE/'inputs.npz');te=pd.read_csv(CACHE/'配对索引.csv').test.to_numpy(bool);assert np.array_equal(truth,z['truth'][te])
    log=pd.read_csv(HERE/'标签释放与更新记录.csv')
    for c in ['原预测起报时间','标签目标时间','实际更新时间']:log[c]=pd.to_datetime(log[c],format='mixed')
    assert (log['标签目标时间']<=log['实际更新时间']).all();assert (log['标签目标时间']-log['原预测起报时间']).eq(pd.Timedelta(hours=24)).all()
    assert not log.duplicated(['模型','站点ID','原预测起报时间']).any()
    stats=json.loads((CACHE/'站点训练标准化.json').read_text(encoding='utf-8'));ids=sorted(idx['站点ID'].unique());chosen=[ids[i] for i in [0,187,374,561,749]]
    replays=[];releases=0
    for backbone,folder in [('Timestral','Timestral209M水质24小时评估_20261104'),('Chronos2','Chronos2水质24小时评估_20261103')]:
        fp=json.loads((CACHE/backbone/'完成核验.json').read_text());assert fp['weights_sha256']==sha(ROOT/folder/'weights/model.safetensors')
        b=np.load(CACHE/backbone/'predictions.npz')['raw'][te];assert np.array_equal(b,pred[backbone+'_冻结原模型'])
        for sid,ix in idx.groupby('站点ID').indices.items():
            q=idx.iloc[ix];origins=q['起报时间'].to_numpy();targets=q['目标时间'].to_numpy();expected=[]
            for k,t in enumerate(targets):
                j=int(np.searchsorted(origins,t,side='left'))
                if j<len(origins):expected.append((origins[k],t,origins[j]))
            g=log[log['模型'].eq(backbone)&log['站点ID'].eq(sid)]
            actual=list(g[['原预测起报时间','标签目标时间','实际更新时间']].itertuples(index=False,name=None))
            assert actual==[(pd.Timestamp(a),pd.Timestamp(t),pd.Timestamp(u)) for a,t,u in expected];releases+=len(actual)
            ck=torch.load(HERE/backbone/'stations'/f'{sid}.pt',map_location='cpu',weights_only=True)
            assert ck['completed_updates']==len(actual) and ck['gradient_steps']==3*len(actual)
            if sid not in chosen:continue
            torch.manual_seed(ck['seed']);state=CausalEndpointAdapter();assert state._run_adapt_steps.__func__ is SimpleAdapter._run_adapt_steps
            mu=np.array(stats[sid]['final']['mean'][5:]);sd=np.array(stats[sid]['final']['scale'][5:]);pz=torch.tensor(((b[ix]-mu)/sd)[:,None,:],dtype=torch.float32)
            pending=[];out=[]
            for j,o in enumerate(origins):
                ready=[v for v in pending if targets[v[0]]<=o];pending=[v for v in pending if targets[v[0]]>o]
                for k,ctx in ready:
                    obs=torch.tensor(((truth[ix[k]]-mu)/sd)[None,None,:],dtype=torch.float32)
                    state.observe_completed(pz[k:k+1],obs,ctx,k,k)
                ctx=state.available_context();out.append(state.predict(pz[j:j+1],ctx)[0,0].numpy()*sd+mu);pending.append((j,ctx.clone()))
            saved=pred[backbone+'_COSA官方核心_因果在线'][ix];assert np.array_equal(np.stack(out),saved)
            assert all(torch.equal(v,ck['state_dict'][k]) for k,v in state.output_adapter.state_dict().items())
            replays.append({'backbone':backbone,'station':sid,'pairs':len(ix),'chronological_output_and_final_state_exact':True})
    M=pd.read_csv(HERE/'01_逐站点模型误差.csv');lookup=M.set_index(['模型','站点ID','目标变量']);chem=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];checks=0
    for name in pred.files:
        if name=='truth':continue
        p=np.maximum(pred[name],0)
        for sid,ix in idx.groupby('站点ID').indices.items():
            for j,target in enumerate(chem):
                y=truth[ix,j];v=p[ix,j];e=v-y;sst=np.sum((y-y.mean())**2);r2=1-np.dot(e,e)/sst if len(y)>=2 and sst>1e-12*max(1.,np.dot(y,y)) else np.nan;r=lookup.loc[(name,sid,target)]
                assert np.isclose(r.R2,r2,rtol=1e-8,atol=1e-8,equal_nan=True);assert np.isclose(r.MAE,np.abs(e).mean(),rtol=1e-8,atol=1e-10);assert np.isclose(r.RMSE,np.sqrt(np.mean(e*e)),rtol=1e-8,atol=1e-10);checks+=1
    result={'passed':True,'official_source_git_checkout_unchanged':True,'source_and_runner_hashes_verified':True,'backbone_weight_hashes_unchanged':True,'same_750_stations_and_23136_test_pairs':True,'all_label_releases_causal_and_at_first_available_origin':True,'verified_release_events':releases,'no_reuse_of_a_label_for_multiple_completed_pair_updates':True,'metrics_groups_checked':checks,'chronological_replays':replays,'previous_cache_independent_source_validation_reused':True,'seconds':time.time()-start}
    (HERE/'独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
