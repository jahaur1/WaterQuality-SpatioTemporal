from pathlib import Path
import os,json,sys,hashlib,time
HERE=Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HOME']=str(HERE/'hf_cache')
sys.stdout.reconfigure(encoding='utf-8')
from huggingface_hub import HfApi,snapshot_download
api=HfApi();info=api.model_info('amazon/chronos-2',files_metadata=True)
manifest={'repo':'amazon/chronos-2','revision':info.sha,'files':[{'name':x.rfilename,'size':x.size} for x in info.siblings]}
(HERE/'官方下载清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
path=snapshot_download('amazon/chronos-2',revision=info.sha,local_dir=str(HERE/'weights'),allow_patterns=['*.json','*.safetensors','README.md','LICENSE'],max_workers=2)
for p in Path(path).glob('*.safetensors'):print(p.name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
print('DOWNLOAD_COMPLETE',path,flush=True)
