"""Synchronous all-station graph forecasting with exact timestamp target masks."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CACHE=ROOT/'冻结预训练模型校正头基线_20261106'
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
COLS=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']+CHEM

class JointGraph(nn.Module):
    def __init__(self, adjacency):
        super().__init__()
        self.register_buffer('adjacency',torch.tensor(adjacency,dtype=torch.float32))
        self.temporal=nn.Sequential(nn.Linear(54,32),nn.ReLU())
        self.future=nn.Sequential(nn.Linear(5,16),nn.ReLU())
        self.graph1=nn.Linear(32,32)
        self.graph2=nn.Linear(32,32)
        self.readout=nn.Linear(48,4)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, history, future, available):
        # A single call processes every station, with real cross-node dependence.
        h=self.temporal(history.flatten(-2))*available[...,None]
        a=self.adjacency[None]*available[:,None,:]
        a=a/a.sum(-1,keepdim=True).clamp_min(1e-6)
        h=h+torch.relu(self.graph1(torch.bmm(a,h)))
        h=h+torch.relu(self.graph2(torch.bmm(a,h)))
        residual=self.readout(torch.cat((h,self.future(future)),dim=-1))
        return history[:,:,-1,5:]+residual

def prepare(stage,raw,idx,stations,times):
    stats=json.loads((CACHE/'站点训练标准化.json').read_text(encoding='utf-8'))
    mu=np.array([stats[s][stage]['mean'] for s in stations])
    sd=np.array([stats[s][stage]['scale'] for s in stations])
    x=np.zeros((len(times),750,6,9),np.float32)
    f=np.zeros((len(times),750,5),np.float32)
    available=np.zeros((len(times),750),np.float32)
    y=np.zeros((len(times),750,4),np.float32)
    masks={name:np.zeros((len(times),750),bool) for name in ['selection_train','validation','refit_train','test']}
    for n,sid in enumerate(stations):
        g=raw[raw['站点ID'].eq(sid)].sort_values('监测时间')
        ts=g['监测时间'].to_numpy();v=g[COLS].to_numpy(float)
        ends=np.searchsorted(ts,times,side='right')-1
        for t,end in enumerate(ends):
            if end>=5 and times[t]-ts[end]<=np.timedelta64(24,'h'):
                x[t,n]=np.clip((v[end-5:end+1]-mu[n])/sd[n],-10,10)
                available[t,n]=1
            target=times[t]+np.timedelta64(24,'h');j=np.searchsorted(ts,target)
            if j<len(ts) and ts[j]==target:
                f[t,n]=np.clip((v[j,:5]-mu[n,:5])/sd[n,:5],-10,10)
    station_map={s:i for i,s in enumerate(stations)}
    time_map={pd.Timestamp(t):i for i,t in enumerate(times)}
    truths=np.load(CACHE/'inputs.npz')['truth']
    for k,r in enumerate(idx.itertuples(index=False)):
        t=time_map[pd.Timestamp(r.起报时间)];n=station_map[r.站点ID]
        y[t,n]=(truths[k]-mu[n,5:])/sd[n,5:]
        for name in masks:masks[name][t,n]=getattr(r,name)
    edges=pd.read_csv(ROOT/f'水质跨站预测_20261109/{stage}_关联站点.csv')
    a=np.eye(750,dtype=np.float32)
    for r in edges.to_dict('records'):
        a[station_map[r['接收站']],station_map[r['邻站']]]+=r['权重']/4
    a/=a.sum(1,keepdims=True)
    assert all(available[m].all() for m in masks.values())
    return x,f,available,y,masks,a,mu,sd

def train(data,mode,epochs,validation=False):
    x,f,available,y,masks,a,_,_=data
    torch.manual_seed(20261109)
    model=JointGraph(a if mode=='联合拓扑' else np.eye(750)).to('cuda')
    trainmask=masks['selection_train' if validation else 'refit_train']
    selected=np.flatnonzero(trainmask.any(1))
    loader=DataLoader(TensorDataset(*[torch.tensor(v[selected]) for v in [x,f,available,y,trainmask]]),batch_size=8,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf');best_epoch=1;bad=0;trace=[]
    valtimes=np.flatnonzero(masks['validation'].any(1))
    for epoch in range(1,epochs+1):
        model.train()
        for h,c,ok,target,mask in loader:
            h,c,ok,target,mask=[v.to('cuda') for v in [h,c,ok,target,mask]]
            optimizer.zero_grad()
            pred=model(h,c,ok)
            loss=(pred[mask]-target[mask]).square().mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        if validation:
            model.eval();total=0.;count=0
            with torch.no_grad():
                for start in range(0,len(valtimes),8):
                    tt=valtimes[start:start+8];m=masks['validation'][tt]
                    p=model(*[torch.tensor(v[tt],device='cuda') for v in [x,f,available]]).cpu().numpy()
                    total+=np.square(p[m]-y[tt][m]).sum();count+=m.sum()*4
            value=total/count;trace.append(dict(epoch=epoch,validation_MSE=float(value)))
            if value<best:best=value;best_epoch=epoch;bad=0
            else:bad+=1
            if bad>=20:break
    return model,best_epoch,trace

def main():
    torch.set_num_threads(2)
    raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv',parse_dates=['监测时间'])
    idx=pd.read_csv(CACHE/'配对索引.csv');stations=sorted(raw['站点ID'].unique())
    times=np.sort(pd.to_datetime(idx['起报时间'],format='mixed').unique())
    first=prepare('selection',raw,idx,stations,times)
    final=prepare('final',raw,idx,stations,times)
    np.savez_compressed(HERE/'拓扑与站点.npz',selection=first[5],final=final[5],station_ids=stations)
    predictions={};logs={}
    target_times=np.flatnonzero(final[4]['test'].any(1))
    for mode in ['联合拓扑','仅自身连接']:
        _,epochs,trace=train(first,mode,150,True)
        print(mode,'selected epochs',epochs,flush=True)
        model,_,_=train(final,mode,epochs)
        model.eval();parts=[]
        with torch.no_grad():
            for start in range(0,len(target_times),8):
                tt=target_times[start:start+8]
                parts.append(model(*[torch.tensor(v[tt],device='cuda') for v in final[:3]]).cpu().numpy())
        predictions[mode]=np.concatenate(parts)*final[-1][None,:,5:]+final[-2][None,:,5:]
        torch.save(model.state_dict(),HERE/f'{mode}.pt')
        logs[mode]=dict(epochs=epochs,validation=trace,parameters=sum(p.numel() for p in model.parameters()))
    test=idx[idx.test].reset_index(drop=True);time_map={pd.Timestamp(times[t]):i for i,t in enumerate(target_times)};node_map={s:i for i,s in enumerate(stations)}
    pair_t=np.array([time_map[pd.Timestamp(t)] for t in test['起报时间']]);pair_n=np.array([node_map[s] for s in test['站点ID']])
    y=np.load(CACHE/'inputs.npz')['truth'][idx.test.to_numpy(bool)]
    np.savez_compressed(HERE/'predictions.npz',truth=y,**{k:v[pair_t,pair_n] for k,v in predictions.items()})
    (HERE/'训练记录.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf-8')
    print('joint shape',final[0].shape,'test masks',final[4]['test'].sum(),flush=True)

if __name__=='__main__':main()
