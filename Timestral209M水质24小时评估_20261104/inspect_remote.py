from pathlib import Path
import requests,json,sys
sys.stdout.reconfigure(encoding='utf-8');HERE=Path(__file__).resolve().parent
r=requests.get('https://huggingface.co/api/models/mistralai/Timestral-209M?blobs=true',timeout=30);r.raise_for_status();data=r.json()
(HERE/'model.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:data.get(k) for k in ['sha','gated','siblings']},ensure_ascii=False,indent=2),flush=True)
