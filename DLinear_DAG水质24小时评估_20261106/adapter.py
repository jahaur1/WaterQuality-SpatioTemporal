from types import SimpleNamespace
import torch
from torch import nn
from vendor_DLinear import Model as DLinear
from vendor_dag.models.dag_model import DAGModel

class Conditional24(nn.Module):
    def __init__(self,model_name,use_future=True):
        super().__init__();self.model_name=model_name;self.use_future=bool(use_future);self.last_aux_loss=0.
        if model_name=='DLinear':
            self.native=DLinear(SimpleNamespace(seq_len=6,pred_len=1,individual=False,enc_in=9))
            self.mix=nn.Linear(9,4)
            with torch.no_grad():self.mix.weight.zero_();self.mix.bias.zero_();self.mix.weight[:,5:9]=torch.eye(4)
            self.future_branch=nn.Sequential(nn.Linear(5,16),nn.ReLU(),nn.Linear(16,4));nn.init.zeros_(self.future_branch[-1].weight);nn.init.zeros_(self.future_branch[-1].bias)
        elif model_name=='DAG':
            c=SimpleNamespace(seq_len=6,pred_len=1,patch_len=2,stride=2,use_c=True,use_t=True,use_c_exog=True,use_t_exog=True,alpha=.5,beta=.1,series_dim=4,infer_use_future=True,enc_in=9,d_model=16,d_ff=32,n_heads=4,e_layers=1,dropout=0.,factor=1,activation='gelu',criterion=nn.MSELoss())
            self.native=DAGModel(c)
        else:raise ValueError(model_name)
    def forward(self,history,future_easy=None):
        assert history.ndim==3 and history.shape[1:]==(6,9)
        if self.use_future:assert future_easy is not None and future_easy.shape==(len(history),1,5)
        if self.model_name=='DLinear':
            self.last_aux_loss=history.new_zeros(())
            y=self.mix(self.native(history)[:,0,:])
            if self.use_future:y=y+self.future_branch(future_easy[:,0,:])
            return y
        # Native DAG expects target channels first, exogenous channels second.
        x=torch.cat([history[:,:,5:9],history[:,:,:5]],dim=-1)
        if self.use_future:
            y,aux=self.native(x,future_easy)
        else:
            # Original infer_use_future=False only switches at eval. Apply the
            # same predicted-covariate path at train and eval for a clean ablation.
            t,estimate,_=self.native.temporal_encoder(x,None,True)
            c,aux=self.native.cov_encoder(x,estimate,True)
            y=self.native.alpha*t+(1-self.native.alpha)*c
            aux=self.native.beta*aux
        self.last_aux_loss=aux
        return y[:,0,:]
