"""Joint 750-node latent graph with a pretrained Chronos-2 decoder."""
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
spec=importlib.util.spec_from_file_location('water_lora',ROOT/'Chronos2_LoRA水质24小时评估_20261109/run.py')
local=importlib.util.module_from_spec(spec);spec.loader.exec_module(local)
CACHE=local.CACHE
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
COLS=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']+CHEM

def data():
    idx=pd.read_csv(CACHE/'配对索引.csv');raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间'])
    matrix=np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz');stations=matrix['station_ids'].tolist()
    times=np.sort(pd.to_datetime(idx['起报时间'],format='mixed').unique());h=np.full((len(times),750,9,32),np.nan,np.float32)
    f=np.full((len(times),750,9,16),np.nan,np.float32);valid=np.zeros((len(times),750),np.float32)
    for n,sid in enumerate(stations):
        g=raw[raw['站点ID'].eq(sid)].sort_values('监测时间');ts=g['监测时间'].to_numpy();v=g[COLS].to_numpy(np.float32);v=np.concatenate((v[:,5:],v[:,:5]),axis=1)
        for i,t in enumerate(times):
            end=np.searchsorted(ts,t,side='right')-1
            if end>=5 and t-ts[end]<=np.timedelta64(24,'h'):
                grid=t-np.arange(31,-1,-1)*np.timedelta64(4,'h');take=np.searchsorted(ts[end-5:end+1],grid,side='right')-1;ok=take>=0
                h[i,n,:,ok]=v[end-5+take[ok]];valid[i,n]=1
            j=np.searchsorted(ts,t+np.timedelta64(24,'h'))
            if j<len(ts) and ts[j]==t+np.timedelta64(24,'h'):f[i,n,4:,5]=v[j,4:]
    time_map={pd.Timestamp(t):i for i,t in enumerate(times)};nodes={s:i for i,s in enumerate(stations)}
    ii=np.array([time_map[pd.Timestamp(t)] for t in idx['起报时间']]);nnodes=np.array([nodes[s] for s in idx['站点ID']])
    truth=np.load(CACHE/'inputs.npz')['truth'];y=np.zeros((len(times),750,4),np.float32);y[ii,nnodes]=truth
    masks={k:np.zeros((len(times),750),bool) for k in ['refit_train','test']}
    for k in masks:masks[k][ii,nnodes]=idx[k]
    stats=json.loads((CACHE/'站点训练标准化.json').read_text(encoding='utf-8'));scale=np.array([stats[s]['final']['scale'][5:] for s in stations],np.float32)
    assert valid[masks['test']].all();assert pd.to_datetime(idx.loc[idx.refit_train,'目标时间'],format='mixed').max()<pd.to_datetime(idx.loc[idx.test,'起报时间'],format='mixed').min()
    return idx,h,f,valid,y,masks,matrix['final'],scale,ii,nnodes

class LatentGraph(nn.Module):
    def __init__(self,adjacency):
        super().__init__();self.register_buffer('adjacency',torch.tensor(adjacency,dtype=torch.float32,device='cuda'))
        self.down=nn.Linear(768,64);self.up=nn.Linear(64,768);self.gate=nn.Parameter(torch.zeros(4))
    def forward(self,h,available):
        # h [750,4,768], with all stations connected before final decoding.
        a=self.adjacency*available[None,:];a=a/a.sum(-1,keepdim=True).clamp_min(1e-6)
        x=torch.relu(self.down(h));message=torch.einsum('ij,jcd->icd',a.to(x.dtype),x)
        return h+torch.tanh(self.gate)[None,:,None]*self.up(message)

def backbone(mode):
    pipe=local.base_model();base=pipe.model
    for p in base.parameters():p.requires_grad_(False)
    if mode!='冻结骨干加图':
        config=local.peft.LoraConfig(r=8,lora_alpha=16,target_modules=['self_attention.q','self_attention.k','self_attention.v','self_attention.o'],layers_to_transform=[10,11],layers_pattern='block')
        wrapper=local.peft.get_peft_model(base,config);base=wrapper.get_base_model()
    return base

def forward(base,graph,h,f,available):
    hidden=[];locations=[]
    # Independent nine-variable station encoding, then joint 750-node graph.
    for start in range(0,750,50):
        stop=min(start+50,750);context=torch.tensor(h[start:stop].reshape(-1,32),device='cuda')
        future=torch.tensor(f[start:stop].reshape(-1,16),device='cuda');groups=torch.arange(stop-start,device='cuda').repeat_interleave(9)
        enc,loc,_,_=base.encode(context=context,future_covariates=future,group_ids=groups,num_output_patches=1)
        hidden.append(enc[0][:,-1].reshape(stop-start,9,768)[:,:4]);locations.append(tuple(v.reshape(stop-start,9,-1)[:,:4].reshape(-1,1) for v in loc))
    h_all=torch.cat(hidden);loc=tuple(torch.cat([v[k] for v in locations]) for k in range(2))
    with torch.autocast('cuda',enabled=False):
        modified=graph(h_all.float(),torch.tensor(available,device='cuda'))
        q=base.output_patch_embedding(modified.reshape(-1,1,768)).reshape(3000,21,16)
    physical=base.instance_norm.inverse(q.reshape(3000,-1),loc).reshape(750,4,21,16)
    return physical[:,:,10,5]

def main():
    torch.set_num_threads(2)
    idx,h,f,available,y,masks,a,scale,ii,nodes=data();train_times=np.flatnonzero(masks['refit_train'].any(1));test_times=np.flatnonzero(masks['test'].any(1))
    results={};records={};started=time.time()
    for mode in ['LoRA仅自身','冻结骨干加图','LoRA加联合图']:
        local.set_seed(20261109);base=backbone(mode);graph=LatentGraph(np.eye(750) if mode=='LoRA仅自身' else a).to('cuda')
        parameters=[p for p in list(base.parameters())+list(graph.parameters()) if p.requires_grad]
        assert all('lora_' in n for n,p in base.named_parameters() if p.requires_grad)
        groups=[dict(params=graph.parameters(),lr=.001)]
        lora_parameters=[p for p in base.parameters() if p.requires_grad]
        if lora_parameters:groups.append(dict(params=lora_parameters,lr=1e-5))
        optimizer=torch.optim.AdamW(groups,weight_decay=.0001)
        rng=np.random.default_rng(20261109);trace=[]
        base.eval();graph.train()
        for step in range(24):
            t=int(rng.choice(train_times));optimizer.zero_grad()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=forward(base,graph,h[t],f[t],available[t]);mask=torch.tensor(masks['refit_train'][t],device='cuda')
                target=torch.tensor(y[t],device='cuda');s=torch.tensor(scale,device='cuda')
                loss=torch.nn.functional.smooth_l1_loss((pred[mask]-target[mask])/s[mask],torch.zeros_like(target[mask]))
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();trace.append(float(loss.detach()))
            if step%4==0:print(mode,'step',step+1,'loss',trace[-1],'seconds',round(time.time()-started,1),flush=True)
        graph.eval();predictions=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for j,t in enumerate(test_times):
                predictions.append(forward(base,graph,h[t],f[t],available[t]).float().cpu().numpy())
                if j%20==0:print(mode,'test time',j+1,'/',len(test_times),flush=True)
        positions={t:i for i,t in enumerate(test_times)};selected=idx.test.to_numpy(bool)
        results[mode]=np.stack(predictions)[np.array([positions[t] for t in ii[selected]]),nodes[selected]]
        record=dict(steps=24,loss=trace,trainable=sum(p.numel() for p in parameters),lora_trainable=sum(p.numel() for p in base.parameters() if p.requires_grad))
        records[mode]=record
        torch.save(dict(graph=graph.state_dict(),lora={k:v.cpu() for k,v in base.state_dict().items() if 'lora_' in k}),HERE/f'{mode}.pt')
        np.savez_compressed(HERE/'predictions.npz',truth=np.load(CACHE/'inputs.npz')['truth'][selected],**results)
        (HERE/'训练记录.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        del base,graph,optimizer,parameters;torch.cuda.empty_cache()

if __name__=='__main__':main()
