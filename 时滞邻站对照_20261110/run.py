"""Chronos-2 LoRA with matched exact-time, lag-aware spatial residual heads."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('matched_framework',ROOT/'河网三图预测_20261110/run.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
sys.stdout.reconfigure(encoding='utf-8')
LAGS=np.array([0,4,8,12,24])
OFFSETS=np.array(sorted(set((LAGS[:,None]+LAGS[None,:]).ravel())))
LAG_INDEX=np.array([int(np.flatnonzero(OFFSETS==x)[0]) for x in LAGS])
SUM_INDEX=np.array([[int(np.flatnonzero(OFFSETS==x+y)[0]) for y in LAGS] for x in LAGS])

class LagGraph(ref.LatentGraph):
    def __init__(self,a,mode):
        super().__init__(a,'三图_单位对照')
        self.raw_projection=nn.Linear(1,64)
        z=np.load(HERE/'final_时滞图.npz')
        if mode=='不使用邻站':
            support=np.zeros_like(z['zero']);support[0]=np.eye(750)[None,:,:]
        else:support=z['zero'] if mode=='邻站当前值' else z['lagged']
        self.register_buffer('temporal_support',torch.tensor(support,dtype=torch.float32,device='cuda'))
        self.register_buffer('lag_index',torch.tensor(LAG_INDEX,device='cuda'))
        self.register_buffer('sum_index',torch.tensor(SUM_INDEX,device='cuda'))
    def forward(self,h,available,raw):
        # raw[offset,node,chemical] contains only actual observations at t-offset.
        valid=torch.isfinite(raw)
        projected=self.raw_projection(torch.nan_to_num(raw)[...,None])*valid[...,None]
        current=self.down(h)*available[:,None,None]
        first=torch.einsum('lcij,ljcd->icd',self.temporal_support,projected[self.lag_index])
        # For a two-hop path, delays accumulate: x(t-lag1-lag2), never x(t+lag).
        previous_first=torch.einsum('kcij,lkjcd->licd',self.temporal_support,projected[self.sum_index])
        second=torch.einsum('lcij,ljcd->icd',self.temporal_support,previous_first)
        # Preserve the original 7*64 input shape and author 1x1 graph fusion.
        blocks=[current,first,second,current,current,current,current]
        x=torch.cat(blocks,dim=-1).permute(2,0,1).unsqueeze(0)
        message=nn.functional.dropout(self.gcn.mlp(x),self.gcn.dropout,training=self.training)[0].permute(1,2,0)
        return h+torch.tanh(self.gate)[None,:,None]*self.up(message)

def raw_inputs(times,stations):
    raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间'])
    stats=json.loads((ref.CACHE/'站点训练标准化.json').read_text(encoding='utf-8'))
    mean=np.array([stats[s]['final']['mean'][5:] for s in stations],np.float32)
    scale=np.array([stats[s]['final']['scale'][5:] for s in stations],np.float32)
    requested=pd.DatetimeIndex((times[:,None]-OFFSETS[None,:]*np.timedelta64(1,'h')).ravel())
    arrays=[]
    for j,c in enumerate(ref.CHEM):
        frame=raw.pivot(index='监测时间',columns='站点ID',values=c).reindex(index=requested,columns=stations)
        values=frame.to_numpy(np.float32).reshape(len(times),len(OFFSETS),750)
        arrays.append((values-mean[None,None,:,j])/scale[None,None,:,j])
    return np.stack(arrays,axis=-1)

def forward(base,graph,h,f,available,raw):
    hidden=[];locations=[]
    for start in range(0,750,50):
        stop=min(start+50,750)
        context=torch.tensor(h[start:stop].reshape(-1,32),device='cuda')
        future=torch.tensor(f[start:stop].reshape(-1,16),device='cuda')
        groups=torch.arange(stop-start,device='cuda').repeat_interleave(9)
        enc,loc,_,_=base.encode(context=context,future_covariates=future,group_ids=groups,num_output_patches=1)
        hidden.append(enc[0][:,-1].reshape(stop-start,9,768)[:,:4])
        locations.append(tuple(v.reshape(stop-start,9,-1)[:,:4].reshape(-1,1) for v in loc))
    loc=tuple(torch.cat([v[k] for v in locations]) for k in range(2))
    with torch.autocast('cuda',enabled=False):
        h_all=graph(torch.cat(hidden).float(),torch.tensor(available,device='cuda'),torch.tensor(raw,device='cuda'))
        q=base.output_patch_embedding(h_all.reshape(-1,1,768)).reshape(3000,21,16)
    return base.instance_norm.inverse(q.reshape(3000,-1),loc).reshape(750,4,21,16)[:,:,10,5]

def main():
    torch.set_num_threads(2)
    idx,h,f,available,y,masks,a,scale,ii,nodes=ref.data()
    times=np.sort(pd.to_datetime(idx['起报时间'],format='mixed').unique())
    stations=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids']
    raw=raw_inputs(times,stations)
    np.savez_compressed(HERE/'时间输入索引.npz',origin_times=times,offsets_hours=OFFSETS,station_ids=stations)
    train_times=np.flatnonzero(masks['refit_train'].any(1));test_times=np.flatnonzero(masks['test'].any(1))
    results={};records={};started=time.time();initial_hash=None
    for mode in ['不使用邻站','邻站当前值','邻站学习时滞']:
        ref.local.set_seed(20261109);base=ref.backbone(mode);graph=LagGraph(a,mode).to('cuda')
        digest=hashlib.sha256()
        for name,p in list(base.named_parameters())+list(graph.named_parameters()):
            if p.requires_grad:digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
        own_hash=digest.hexdigest()
        if initial_hash is None:initial_hash=own_hash
        assert own_hash==initial_hash
        parameters=[p for p in list(base.parameters())+list(graph.parameters()) if p.requires_grad]
        optimizer=torch.optim.AdamW([dict(params=graph.parameters(),lr=.001),dict(params=[p for p in base.parameters() if p.requires_grad],lr=1e-5)],weight_decay=.0001)
        base.eval();graph.train();rng=np.random.default_rng(20261109);trace=[]
        for step in range(24):
            t=int(rng.choice(train_times));optimizer.zero_grad()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=forward(base,graph,h[t],f[t],available[t],raw[t])
                mask=torch.tensor(masks['refit_train'][t],device='cuda')
                target=torch.tensor(y[t],device='cuda');s=torch.tensor(scale,device='cuda')
                loss=nn.functional.smooth_l1_loss((pred[mask]-target[mask])/s[mask],torch.zeros_like(target[mask]))
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();trace.append(float(loss.detach()))
            if step%4==0:print(mode,'step',step+1,'loss',trace[-1],'seconds',round(time.time()-started,1),flush=True)
        graph.eval();predictions=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for j,t in enumerate(test_times):
                predictions.append(forward(base,graph,h[t],f[t],available[t],raw[t]).float().cpu().numpy())
                if j%20==0:print(mode,'test',j+1,'/',len(test_times),flush=True)
        positions={t:i for i,t in enumerate(test_times)};selected=idx.test.to_numpy(bool)
        results[mode]=np.stack(predictions)[np.array([positions[t] for t in ii[selected]]),nodes[selected]]
        records[mode]=dict(steps=24,initial_trainable_parameter_sha256=own_hash,trainable=sum(p.numel() for p in parameters),lora_trainable=sum(p.numel() for p in base.parameters() if p.requires_grad),loss=trace)
        torch.save(dict(graph=graph.state_dict(),lora={k:v.cpu() for k,v in base.state_dict().items() if 'lora_' in k}),HERE/f'{mode}.pt')
        np.savez_compressed(HERE/'predictions.npz',truth=np.load(ref.CACHE/'inputs.npz')['truth'][selected],**results)
        (HERE/'训练记录.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        del base,graph,optimizer,parameters;torch.cuda.empty_cache()

if __name__=='__main__':main()
