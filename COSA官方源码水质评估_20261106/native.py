"""Execute original author class/function ASTs without changing their bodies."""
from pathlib import Path
import ast,os,json,time
from copy import deepcopy
from collections import deque,defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'source/tta/cosa.py'
OPT=HERE/'source/models/optimizer.py'
_ns=dict(torch=torch,nn=nn,F=F,np=np,pd=pd,os=os,json=json,time=time,deepcopy=deepcopy,deque=deque,defaultdict=defaultdict,ADAPTER_TYPES=('Linear','MLP'))
for path,names in [(OPT,{'get_optimizer'}),(SOURCE,{'SimpleOutputAdapter','SimpleAdapter'})]:
    tree=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
    selected=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    assert len(selected)==len(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),_ns)
SimpleOutputAdapter=_ns['SimpleOutputAdapter'];SimpleAdapter=_ns['SimpleAdapter'];get_optimizer=_ns['get_optimizer']

class CausalEndpointAdapter(SimpleAdapter):
    """Timestamp orchestration only. Native head, updates, loss, optimizer unchanged."""
    def __init__(self):
        nn.Module.__init__(self)
        self.model=nn.Identity();self.norm_module=None
        self.output_adapter=SimpleOutputAdapter(pred_len=1,buffer_context_size=10,n_vars=4,var_wise_gating=True,adapter_type='Linear',hidden_dim=64)
        solver=SimpleNamespace(OPTIMIZING_METHOD='adam',BASE_LR=.001,WEIGHT_DECAY=.0001)
        self.cfg=SimpleNamespace(TTA=SimpleNamespace(SOLVER=solver))
        self.optimizer=get_optimizer(self.output_adapter.parameters(),self.cfg.TTA)
        self.fast_adaptation=True;self.adaptive_lr=True;self.per_batch_lr_reset=True
        self.max_lr=.005;self.min_lr=.0001;self.adapt_steps=3;self.convergence_threshold=.0001;self.l2_reg=.0001
        self.loss_history=deque(maxlen=5);self.current_lr=.001
        self.n_adapt=0;self.step_count=0;self.sample_history=deque(maxlen=200);self.buffer_context_size=10
        self.time_stats=defaultdict(float);self.time_counts=defaultdict(int)
    def available_context(self):
        # Same ordering and padding as author's _get_context_vector; CPU replaces
        # its hard-coded CUDA tensor allocation. No future observations are read.
        values=[v['target_mean'] for v in list(self.sample_history)[-10:][::-1]]
        values=values+[values[-1]]*(10-len(values)) if values else [0.]*10
        return torch.tensor(values,dtype=torch.float32).unsqueeze(0)
    def observe_completed(self,prediction,truth,context,time_idx,batch_idx):
        self._push_history(float(truth.mean().item()),time_idx)
        self._run_adapt_steps(prediction,truth,context,3,batch_idx,tag='full')
    def predict(self,prediction,context):
        self.switch_model_to_eval()
        with torch.no_grad():return self.output_adapter(prediction,context)
