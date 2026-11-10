import concurrent.futures
import pandas as pd
from geocode import HERE,one

frame=pd.read_csv(HERE/'站点地图定位.csv')
failed=frame[frame['定位状态'].str.contains('查询失败')]
print('Retrying transient network failures',len(failed),flush=True)
updates=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(one,r) for r in failed[['站点ID','省份','流域','断面名称']].to_dict('records')]
    for future in concurrent.futures.as_completed(futures):
        updates.append(future.result())
        if len(updates)%25==0:print('Retry',len(updates),'/',len(failed),flush=True)
if updates:
    changed=pd.DataFrame(updates);out=pd.concat([frame[~frame['站点ID'].isin(changed['站点ID'])],changed],ignore_index=True).sort_values('站点ID');assert len(out)==750
    out.to_csv(HERE/'站点地图定位.csv',index=False,encoding='utf-8-sig')
