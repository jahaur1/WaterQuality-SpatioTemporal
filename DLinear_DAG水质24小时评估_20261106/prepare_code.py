"""Vendor minimal official networks and reuse the previously audited time-split runner."""
from pathlib import Path
import json,hashlib,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit(p):return subprocess.check_output(['git','-c','safe.directory='+p.as_posix(),'rev-parse','HEAD'],cwd=p,text=True).strip()
dl=HERE/'source_DLinear';dg=HERE/'source_DAG';prov={'DLinear_commit':commit(dl),'DAG_commit':commit(dg),'changes':[],'files':[]}
(HERE/'vendor_DLinear.py').write_bytes((dl/'models/DLinear.py').read_bytes())
prov['files'].append({'source':'source_DLinear/models/DLinear.py','source_sha256':sha(dl/'models/DLinear.py'),'vendored':'vendor_DLinear.py','sha256':sha(HERE/'vendor_DLinear.py')})
src=dg/'ts_benchmark/baselines/dag'
for name in ['__init__.py','layers/__init__.py','models/__init__.py']:
    p=HERE/'vendor_dag'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('',encoding='utf-8')
paths=list((src/'layers').glob('*.py'))+[src/'models/dag_model.py']
for p in paths:
    if p.name=='__init__.py':continue
    text=p.read_text(encoding='utf-8');text=text.replace('ts_benchmark.baselines.dag.layers','vendor_dag.layers').replace('ts_benchmark.baselines.time_series_library.utils.masking','vendor_dag.layers.masking')
    if p.name=='SelfAttention_Family.py':
        text=text.replace('from reformer_pytorch import LSHSelfAttention','try:\n    from reformer_pytorch import LSHSelfAttention\nexcept ImportError:\n    LSHSelfAttention = None  # Unused by native DAG FullAttention; no model substitution.')
    if p.name=='TC_EncDec.py':
        needle='attn_alpha.repeat(self.series_dim, 1, 1, 1)';assert text.count(needle)==1
        text=text.replace(needle,'attn_alpha.repeat_interleave(self.series_dim, dim=0)')
    dest=HERE/'vendor_dag'/p.relative_to(src);dest.write_text(text,encoding='utf-8')
    prov['files'].append({'source':str(p.relative_to(HERE)),'source_sha256':sha(p),'vendored':str(dest.relative_to(HERE)),'sha256':sha(dest)})
mask=dg/'ts_benchmark/baselines/time_series_library/utils/masking.py';(HERE/'vendor_dag/layers/masking.py').write_bytes(mask.read_bytes())
prov['files'].append({'source':str(mask.relative_to(HERE)),'source_sha256':sha(mask),'vendored':'vendor_dag/layers/masking.py','sha256':sha(HERE/'vendor_dag/layers/masking.py')})
prov['changes']=['Relative import rewrites to avoid unrelated TFB benchmark framework dependencies','Reformer optional import only; DAG uses FullAttention and never calls Reformer','Temporal gate repeat -> repeat_interleave for series_dim=4, preserving within-window association; original repeat mixes window-specific gates in batched multi-target inference','Adapter: direct exact+24h label horizon, six observations, four targets, five known covariates; native DAG pred_len=1,patch=2,stride=2,d_model=16,d_ff=32,n_heads=4,e_layers=1','No-future DAG adapter uses predicted future exogenous values during both training and testing; never uses real future easy values in that arm','DLinear retains native shared channelwise decomposition, kernel25; adds learnable 9-to4 mixing head and same 5-to16-to4 future correction branch as XLinear']
(HERE/'source_provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf-8')
s=(ROOT/'XLinear水质24小时条件预测_20261102/run_forecast24.py').read_text(encoding='utf-8')
s=s.replace('import argparse,hashlib,json,random,sys,time,platform','import argparse,hashlib,json,random,sys,time,platform\nsys.stdout.reconfigure(encoding="utf-8")')
start=s.index('def model_for(');end=s.index('\ndef score(',start)
s=s[:start]+'''def model_for(conf,use_future):
    from adapter import Conditional24
    return Conditional24(conf['model_name'],use_future=use_future)
'''+s[end:]
s=s.replace("parser=argparse.ArgumentParser();", "parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['DLinear','DAG'],required=True);")
s=s.replace("'d_model':16,'t_ff':16", "'model_name':args.model,'d_model':16,'t_ff':16")
s=s.replace("HERE/'vendor_XLinear.py'", "HERE/'vendor_DLinear.py'")
s=s.replace("'vendor_sha256':sha(HERE/'vendor_DLinear.py') if (HERE/'vendor_DLinear.py').exists() else None", "'vendor_sha256':sha(HERE/'vendor_DLinear.py'), 'provenance_sha256':sha(HERE/'source_provenance.json')")
s=s.replace("loss=criterion(model(htrain,ftrain),ytrain)","prediction=model(htrain,ftrain);loss=criterion(prediction,ytrain)+model.last_aux_loss")
s=s.replace("loss=criterion(model(hr,fr),yr)","prediction=model(hr,fr);loss=criterion(prediction,yr)+model.last_aux_loss")
s=s.replace("'val", "'val")
s=s.replace('torch.set_num_threads(2)','torch.set_num_threads(1)')
s=s.replace('two prespecified XLinear arms','two prespecified DLinear/DAG arms')
s=s.replace("parser.add_argument('--model',choices=['DLinear','DAG'],required=True);", "parser.add_argument('--model',choices=['DLinear','DAG'],required=True);parser.add_argument('--shard-count',type=int,default=1);parser.add_argument('--shard-index',type=int,default=0);")
s=s.replace("groups=list(df.groupby('站点ID'));groups=groups[:args.limit] if args.limit else groups", "groups=list(df.groupby('站点ID'));groups=groups[:args.limit] if args.limit else groups\n    assert 0<=args.shard_index<args.shard_count\n    groups=[g for k,g in enumerate(groups) if k%args.shard_count==args.shard_index]")
s=s.replace("if not args.limit:assert len(M)==6000 and len(coverage)==750", "if not args.limit and args.shard_count==1:assert len(M)==6000 and len(coverage)==750")
s=s.replace("'normalization':'one training-only", "'model_settings':({'kernel_size':25,'individual':False,'history_channel_mixing':'Linear9to4','future_branch':'5to16to4_zero_initial_output'} if args.model=='DLinear' else {'d_model':16,'d_ff':32,'n_heads':4,'e_layers':1,'patch_len':2,'stride':2,'alpha':0.5,'beta':0.1,'loss':'four-target standardized MSE plus native beta-weighted auxiliary losses','validation_selection':'forecast MSE only','no_future_training':'predicted exogenous path at both train and eval'}),'normalization':'one training-only")
(HERE/'run_train.py').write_text(s,encoding='utf-8')
print(json.dumps({'DAG_commit':prov['DAG_commit'],'DLinear_commit':prov['DLinear_commit'],'files':len(prov['files'])},indent=2))
