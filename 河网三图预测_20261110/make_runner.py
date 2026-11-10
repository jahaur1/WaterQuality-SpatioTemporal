"""Reuse the existing framework, changing only the second spatial support."""
from pathlib import Path
import ast

here=Path(__file__).resolve().parent
source=(here.parent/'三图联合预测_20261110/run.py').read_text(encoding='utf-8')
source=source.replace('import importlib.util','import hashlib\nimport importlib.util',1)
old="distance=np.load(ROOT/'距离拓扑_20261109/距离拓扑.npz')['adjacency']"
new="""distance=(np.eye(750) if mode=='三图_单位对照' else np.load(HERE/'河网拓扑_探索.npz')['adjacency'])"""
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace("for mode in ['三图联合']:","for mode in ['三图_单位对照','三图_河网探索']:")
old="local.set_seed(20261109);base=backbone(mode);graph=LatentGraph(a,mode).to('cuda')"
new=old+"""
        digest=hashlib.sha256()
        for key,value in graph.state_dict().items():
            if key!='distance':digest.update(value.detach().cpu().numpy().tobytes())
        initial_hash=digest.hexdigest()
        if mode=='三图_单位对照':control_initial_hash=initial_hash
        else:assert initial_hash==control_initial_hash, 'Non-topology initialization changed'
        print('common graph initialization',initial_hash,flush=True)"""
assert source.count(old)==1
source=source.replace(old,new)
source=source.replace('record=dict(steps=24,','record=dict(initial_graph_hash_without_second_support=initial_hash,steps=24,')
ast.parse(source)
(here/'run.py').write_text(source,encoding='utf-8')
print('Generated two arms with identical parameter initialization and three-support shapes.')
