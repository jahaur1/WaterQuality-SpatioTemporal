from pathlib import Path
import requests,json,hashlib,time,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.stdout.reconfigure(encoding='utf-8');HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'官方下载清单.json').read_text(encoding='utf-8'));rev=m['revision'];size=next(x['size'] for x in m['files'] if x['name']=='model.safetensors')
meta=requests.get('https://huggingface.co/api/models/amazon/chronos-2',params={'revision':rev,'blobs':'true'},timeout=30);meta.raise_for_status()
blob=next(x for x in meta.json()['siblings'] if x['rfilename']=='model.safetensors');expected=blob['lfs']['sha256']
dest=HERE/'weights/model.safetensors';parts=HERE/'download_parts';parts.mkdir(exist_ok=True);chunk=8*1024*1024
def get(i):
    start=i*chunk;end=min(size,start+chunk)-1;p=parts/f'{i:04d}.part'
    if p.exists() and p.stat().st_size==end-start+1:return i
    for retry in range(4):
        try:
            url=f'https://huggingface.co/amazon/chronos-2/resolve/{rev}/model.safetensors?download=true&water_part={i}'
            with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,40),stream=True) as r:
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
        task.result();print('PARTS',n,'/',count,'SECONDS',round(time.time()-start,1),flush=True)
temp=HERE/'weights/range_download.safetensors'
with temp.open('wb') as out:
    for i in range(count):
        with (parts/f'{i:04d}.part').open('rb') as inp:
            while b:=inp.read(1024*1024):out.write(b)
assert temp.stat().st_size==size
actual=hashlib.sha256(temp.read_bytes()).hexdigest();assert actual==expected,(actual,expected)
temp.replace(dest)
(HERE/'下载SHA256核验.json').write_text(json.dumps({'passed':True,'repo':m['repo'],'revision':rev,'size':size,'sha256':actual,'verified_against':'official Hugging Face LFS sha256','method':'HTTP ranges from official immutable revision'},indent=2),encoding='utf-8')
print('DOWNLOAD_VERIFIED',size,actual,flush=True)
