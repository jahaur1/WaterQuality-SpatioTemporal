"""Matched self-only inference, LoRA-only, and frozen-backbone ablations."""
import ast
from pathlib import Path

here=Path(__file__).resolve().parent
source=(here.parent/'河网三图预测_20261110/run.py').read_text(encoding='utf-8')
old="distance=(np.eye(750) if mode=='三图_单位对照' else np.load(HERE/'河网拓扑_探索.npz')['adjacency'])"
assert source.count(old)==1
source=source.replace(old,'distance=np.eye(750)')
old="return torch.nn.functional.softmax(torch.nn.functional.relu(torch.mm(self.nodevec1,self.nodevec2)),dim=1)"
new="""logits=torch.nn.functional.relu(torch.mm(self.nodevec1,self.nodevec2))
        # Keep the original embeddings and parameter shapes, prohibit all other nodes.
        mask=torch.eye(750,device=logits.device,dtype=torch.bool)
        return torch.nn.functional.softmax(logits.masked_fill(~mask,float('-inf')),dim=1)"""
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace("for mode in ['三图_单位对照','三图_河网探索']:","for mode in ['冻结同框架','本站LoRA','本站LoRA加校正']:")
old="""if mode=='三图_单位对照':control_initial_hash=initial_hash
        else:assert initial_hash==control_initial_hash, 'Non-topology initialization changed'"""
new="""expected=json.loads((ROOT/'河网三图预测_20261110/训练记录.json').read_text(encoding='utf-8'))['三图_单位对照']['initial_graph_hash_without_second_support']
        assert initial_hash==expected, 'Parameter initialization differs from spatial controls'
        # Initialization uses the original Pearson SVD; only the inference buffers change.
        graph.fixed.copy_(torch.eye(750,device='cuda'))
        graph.distance.copy_(torch.eye(750,device='cuda'))
        if mode in ['冻结同框架','本站LoRA']:
            for p in graph.parameters():p.requires_grad_(False)
        if mode=='冻结同框架':
            for p in base.parameters():p.requires_grad_(False)
        steps=0 if mode=='冻结同框架' else 24"""
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace('for step in range(24):','for step in range(steps):')
source=source.replace('initial_graph_hash_without_second_support=initial_hash,steps=24,','initial_graph_hash_before_support_replacement=initial_hash,steps=steps,')
old="""records[mode]=record
        torch.save"""
new="""# Verify inference does not depend on other stations, at encoder and graph levels.
        if mode=='本站LoRA加校正':
            t=int(test_times[0]);observed=np.flatnonzero(masks['test'][t]);target_node=int(observed[0])
            hh=h[t].copy();ff=f[t].copy();aa=available[t].copy()
            other=np.arange(750)!=target_node
            hh[other]=np.nan;ff[other]=np.nan;aa[other]=0
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                own=forward(base,graph,h[t],f[t],available[t])[target_node].float().cpu().numpy()
                perturbed=forward(base,graph,hh,ff,aa)[target_node].float().cpu().numpy()
            np.testing.assert_allclose(own,perturbed,rtol=1e-5,atol=1e-6)
            record['neighbour_perturbation_check']=dict(target_station=str(np.load(ROOT/'全站拓扑联合预测_20261109/拓扑与站点.npz')['station_ids'][target_node]),max_absolute_prediction_change=float(np.max(np.abs(own-perturbed))),passed=True)
        records[mode]=record
        torch.save"""
assert source.count(old)==1
source=source.replace(old,new)
ast.parse(source)
(here/'run.py').write_text(source,encoding='utf-8')
print('Created matched frozen, self-only LoRA, and self-only LoRA+correction arms.')
