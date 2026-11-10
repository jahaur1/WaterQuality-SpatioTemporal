"""Demonstrate and audit the multi-target gate-order fix, without changing source files."""
from pathlib import Path
import copy,types,json,torch
from adapter import Conditional24
torch.set_num_threads(1);torch.manual_seed(20261106);HERE=Path(__file__).resolve().parent
native=HERE/'source_DAG/ts_benchmark/baselines/dag/layers/TC_EncDec.py'
text=native.read_text(encoding='utf-8').replace('ts_benchmark.baselines.dag.layers','vendor_dag.layers')
space={};exec(compile(text,str(native),'exec'),space)
fixed=Conditional24('DAG',True).eval();original=copy.deepcopy(fixed)
original.native.temporal_encoder.forward=types.MethodType(space['TemporalCausalityEncoder'].forward,original.native.temporal_encoder)
h=torch.randn(7,6,9);f=torch.randn(7,1,5)
with torch.no_grad():
    a=fixed(h,f);aa=torch.cat([fixed(h[i:i+1],f[i:i+1]) for i in range(7)])
    b=original(h,f);bb=torch.cat([original(h[i:i+1],f[i:i+1]) for i in range(7)])
after=float((a-aa).abs().max());before=float((b-bb).abs().max())
assert torch.allclose(a,aa,rtol=1e-4,atol=1e-5) and before>1e-5
for name in ['DLinear','DAG']:
    hidden=Conditional24(name,False).eval()
    with torch.no_grad():assert torch.equal(hidden(h,None),hidden(h,f+123))
out={'passed':True,'seed':20261106,'synthetic_windows':7,'chemical_targets':4,'original_gate_batch_vs_single_max_abs_difference':before,'corrected_gate_batch_vs_single_max_abs_difference':after,'change':'attn_alpha.repeat(series_dim,1,1,1) -> repeat_interleave(series_dim,dim=0)','both_no_future_adapters_ignore_future_easy':True,'note':'Numerical isolation test, not accuracy evaluation or training.'}
(HERE/'DAG批次隔离修正核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=True,indent=2))
