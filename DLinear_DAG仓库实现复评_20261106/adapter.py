from types import SimpleNamespace
import torch
from torch import nn
from vendor_DLinear import DLinear
from dag_native_layers import MLP

class Conditional24(nn.Module):
    def __init__(self,model_name,use_future=True):
        super().__init__()
        assert model_name=='DLinear'
        self.use_future=bool(use_future)
        cfg=SimpleNamespace(task_name='short_term_forecast',seq_len=6,pred_len=1,moving_avg=25,enc_in=9,input_dim=9,output_dim=4,mlp_hidden_dims=32)
        self.native=DLinear(cfg)
        self.fusion=MLP(cfg) if self.use_future else None
        self.last_aux_loss=0.
    def forward(self,history,future_easy=None):
        assert history.ndim==3 and history.shape[1:]==(6,9)
        # Repository convention: chemical targets first, easy covariates second.
        x=torch.cat((history[:,:,5:9],history[:,:,:5]),dim=-1)
        output=self.native(x,None,None,None)[:,:,:4]
        if self.use_future:
            assert future_easy is not None and future_easy.shape==(len(history),1,5)
            output=self.fusion(future_easy,output)
        self.last_aux_loss=history.new_zeros(())
        return output[:,0,:]
