from pathlib import Path
import requests,json,hashlib,time,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.stdout.reconfigure(encoding='utf-8');HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'model.json').read_text(encoding='utf-8'));rev=m['sha'];blob=next(x for x in m['siblings'] if x['rfilename']=='model.safetensors');size=blob['size'];expected=blob['lfs']['sha256']
weights=HERE/'weights';weights.mkdir(exist_ok=True);parts=HERE/'download_parts';parts.mkdir(exist_ok=True);chunk=8*1024*1024
for name in ['config.json','README.md','LICENSE']:
    r=requests.get(f'https://huggingface.co/mistralai/Timestral-209M/resolve/{rev}/{name}',timeout=40);r.raise_for_status();(weights/name).write_bytes(r.content)
def get(i):
    start=i*chunk;end=min(size,start+chunk)-1;p=parts/f'{i:04d}.part'
    if p.exists() and p.stat().st_size==end-start+1:return i
    for retry in range(4):
        try:
            url=f'https://huggingface.co/mistralai/Timestral-209M/resolve/{rev}/model.safetensors?download=true&water_part={i}'
            with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,60),stream=True) as r:
                r.raise_for_status();assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'),(r.status_code,r.headers.get('Content-Range'))
                with p.open('wb') as out:
                    for b in r.iter_content(1024*1024):out.write(b)
            assert p.stat().st_size==end-start+1;return i
        except Exception:
            if retry==3:raise
            time.sleep(1)
count=(size+chunk-1)//chunk;start=time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
    for n,task in enumerate(as_completed([pool.submit(get,i) for i in range(count)]),1):
        task.result()
        if n%8==0 or n==count:print('PARTS',n,'/',count,'SECONDS',round(time.time()-start,1),flush=True)
temp=weights/'download.safetensors'
with temp.open('wb') as out:
    for i in range(count):
        with (parts/f'{i:04d}.part').open('rb') as inp:
            while b:=inp.read(1024*1024):out.write(b)
assert temp.stat().st_size==size
actual=hashlib.sha256(temp.read_bytes()).hexdigest();assert actual==expected,(actual,expected);temp.replace(weights/'model.safetensors')
(HERE/'下载SHA256核验.json').write_text(json.dumps({'passed':True,'repo':'mistralai/Timestral-209M','revision':rev,'size':size,'sha256':actual,'verified_against':'official Hugging Face LFS sha256'},indent=2),encoding='utf-8')
print('DOWNLOAD_VERIFIED',size,actual,flush=True)
