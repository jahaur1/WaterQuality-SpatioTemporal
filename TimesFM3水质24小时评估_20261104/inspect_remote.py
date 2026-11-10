from pathlib import Path
import requests,json,sys
sys.stdout.reconfigure(encoding='utf-8');HERE=Path(__file__).resolve().parent
urls={'github_directory':'https://api.github.com/repos/google-research/timesfm/contents/timesfm3-usage','model':'https://huggingface.co/api/models/google/timesfm-3.0-pytorch?blobs=true'}
for name,url in urls.items():
    r=requests.get(url,timeout=30);print(name,r.status_code,flush=True);r.raise_for_status();data=r.json();(HERE/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    if name=='model':print(json.dumps({k:data.get(k) for k in ['sha','gated','siblings']},ensure_ascii=False,indent=2),flush=True)
    else:print(json.dumps(data,ensure_ascii=False,indent=2),flush=True)
