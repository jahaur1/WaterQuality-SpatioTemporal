"""Small read-only checks of the downloaded author's implementation."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
sys.stdout.reconfigure(encoding='utf-8')
spec=importlib.util.spec_from_file_location('author_tgcn',HERE/'source/models/tgcn.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.manual_seed(1)
model=module.TGCN(adj=np.eye(2),hidden_dim=2).eval()
x=torch.tensor([[[1.,2.],[1.5,3.],[2.,4.]]])
xx=x.clone();xx[:,:,1]+=10
with torch.no_grad():
    a=model(x);b=model(xx)
diff=float(torch.max(torch.abs(a[:,0]-b[:,0])))
assert diff>1e-6
# Check the suspected cause by splitting gates within each node, in memory only.
def corrected_cell(self,inputs,hidden_state):
    batch,nodes=inputs.shape
    gates=torch.sigmoid(self.graph_conv1(inputs,hidden_state)).reshape(batch,nodes,2*self._hidden_dim)
    r,u=torch.chunk(gates,2,dim=-1)
    r=r.reshape(batch,-1);u=u.reshape(batch,-1)
    c=torch.tanh(self.graph_conv2(inputs,r*hidden_state))
    state=u*hidden_state+(1-u)*c
    return state,state
module.TGCNCell.forward=corrected_cell
with torch.no_grad():
    corrected_diff=float(torch.max(torch.abs(model(x)[:,0]-model(xx)[:,0])))
assert corrected_diff<1e-7
from utils.graph_conv import calculate_laplacian_with_self_loop
original=np.array([[0.,1.],[0.,0.]],np.float32)
support=calculate_laplacian_with_self_loop(torch.tensor(original))
message=support@torch.tensor([[7.],[0.]])
assert message[1,0]>0 and support[0,1]==0
spec=importlib.util.spec_from_file_location('author_metrics',HERE/'source/utils/metrics.py')
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
y=torch.tensor([0.,1.,2.]);pred=y+1
r_author=float(metrics.r2(pred,y));r_standard=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum())
assert abs(r_author-r_standard)>.1
report=dict(direction_preserved_by_normalization=True,normalized_two_node_support=support.tolist(),identity_graph_other_node_perturbation_changes_target_hidden_state=diff,identity_graph_after_per_node_gate_split_change=corrected_diff,gate_split_issue_confirmed=True,r2_formula_example=dict(y=y.tolist(),pred=pred.tolist(),author_r2=r_author,standard_r2=r_standard),note='Checks of public code only; not a replication or verdict on the published experiments. The saved source is unchanged. No training performed.')
(HERE/'小例核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
