"""Public map search, with explicit approximate-location and ambiguity flags."""
import concurrent.futures
import json
import re
import sys
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE/'deps'));sys.stdout.reconfigure(encoding='utf-8')
from bd09convertor import convertMC2LL
LOCK=threading.Lock();LAST=[0.]
THREAD=threading.local()

def normalize(value):
    return re.sub(r'[\s（）()·—\-]','',str(value))

def province(value):
    return re.sub('省|市|自治区|壮族|回族|维吾尔','',str(value))

def haversine(a,b):
    lon1,lat1,lon2,lat2=np.radians([a[0],a[1],b[0],b[1]])
    return 6371*2*np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))

def query(sid,query_text,suffix):
    folder=HERE/'map_cache';folder.mkdir(exist_ok=True);path=folder/f'{sid}_{suffix}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    with LOCK:
        wait=max(0,.8-(time.monotonic()-LAST[0]))
        if wait:time.sleep(wait)
        LAST[0]=time.monotonic()
    if not hasattr(THREAD,'session'):
        THREAD.session=requests.Session()
        THREAD.session.mount('https://',HTTPAdapter(max_retries=Retry(total=1,backoff_factor=.6,status_forcelist=[502,503,504])))
    response=THREAD.session.get('https://map.baidu.com/',params={'qt':'s','wd':query_text,'pn':0,'rn':10,'ie':'utf-8'},timeout=8)
    response.raise_for_status();obj=response.json();content=obj.get('content',[])
    if not isinstance(content,list):content=[]
    candidates=[]
    for item in content:
        if not isinstance(item,dict):continue
        admin=item.get('admin_info') or {}
        x=item.get('diPointX',item.get('x'));y=item.get('diPointY',item.get('y'))
        if x is None or y is None:
            bounds=re.findall(r'-?\d+(?:\.\d+)?',str(item.get('geo','')).split('|')[1] if '|' in str(item.get('geo','')) else '')
            if len(bounds)>=2:
                nums=list(map(float,bounds[:4]));x=100*(nums[0]+nums[2])/2 if len(nums)==4 else nums[0]*100;y=100*(nums[1]+nums[3])/2 if len(nums)==4 else nums[1]*100
        candidates.append(dict(name=item.get('name',''),address=item.get('addr',''),tag=item.get('di_tag',''),province=admin.get('province_name',''),city=admin.get('city_name',item.get('city_name','')),district=admin.get('area_name',''),x=x,y=y,uid=item.get('uid','')))
    result=dict(query=query_text,url=response.url,candidates=candidates)
    path.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    return result

def rank_candidates(row,result):
    name=normalize(row['断面名称']);base=re.sub(r'国控|断面|水质|监测站|自动站|出口|入口|入湖口|出水口','',name)
    accepted=[]
    for c in result['candidates']:
        if c['province']:
            if province(row['省份'])!=province(c['province']):continue
        elif row['省份'] not in c['address'] and not (row['省份'] in ['北京市','天津市','上海市','重庆市'] and row['省份'] in c['city']):continue
        title=normalize(c['name']);tag=c['tag'];address=normalize(c['address'])
        if not base:continue
        bridge_stem=re.sub('公路桥|特大桥|大桥|桥','',base)
        alias_match=base not in title and '桥' in base and len(bridge_stem)>=2 and bridge_stem in title and bool(re.search('桥$',title))
        village_match=base.endswith('村') and len(base)>2 and base[:-1] in title and bool(re.search('村委会|村民委员会|村$',title))
        if base not in title and not alias_match and not village_match:continue
        if c['x'] is None or c['y'] is None:continue
        formal=bool(re.search('水质(?:自动)?监测站|水文站|国家生态环境监测',title))
        bridge=bool(re.search(r'(桥|渡口|码头|闸|水坝|大坝)(管理所|管理站|水利站)?$',title)) and not re.search('公司|酒店|宾馆|店|厂|餐饮|公寓|小区|停车|充电|公交|加油|售票',title+tag)
        place=bool(re.search('行政地标|乡镇|村庄|自然地物|河流|湖泊|水库',tag)) or village_match
        exact=title==name or title==base
        if formal:level='同名监测设施候选';radius=.5;score=100
        elif bridge and (exact or len(base)>=2):level='桥闸渡口估算';radius=2. if exact else 10. if len(base)>=3 else 20.;score=80+(10 if exact else 0)
        elif place:
            level='同名地名范围估算';radius=100. if title.endswith('市') else 50. if title.endswith(('区','县')) else 10. if re.search('乡|镇|村|街道',title) else 20.;score=60+(10 if exact else 0)
        elif exact and not re.search('公司|酒店|宾馆|超市|充电|车站|餐饮|学校|医院',title+tag):level='同名POI粗估';radius=10.;score=40
        else:continue
        try:lon,lat=convertMC2LL(float(c['x'])/100,float(c['y'])/100)
        except (TypeError,ValueError):continue
        if not (73<=lon<=136 and 3<=lat<=54):continue
        accepted.append(dict(**c,longitude=lon,latitude=lat,precision=level,uncertainty_km=radius,score=score,source_url=result['url'],query=result['query']))
    return sorted(accepted,key=lambda c:-c['score'])

def one(row):
    sid=row['站点ID'];query1=f"{row['省份']} {row['断面名称']} 水质监测站"
    try:
        candidates=rank_candidates(row,query(sid,query1,'station'))
        if not candidates:
            query2=f"{row['省份']} {row['断面名称']}"
            candidates=rank_candidates(row,query(sid,query2,'place'))
        if not candidates:return dict(**row,定位状态='未取得可靠同名候选',经度=np.nan,纬度=np.nan)
        top=candidates[0];ambiguous=any(c['score']>=top['score']-10 and haversine((top['longitude'],top['latitude']),(c['longitude'],c['latitude']))>20 for c in candidates[1:])
        return dict(**row,定位状态='同省多地同名待核实' if ambiguous else '地图估算待人工核实',经度=top['longitude'],纬度=top['latitude'],坐标系='BD09LL',估算类别=top['precision'],估算半径_km=top['uncertainty_km'],地图地名=top['name'],地图地址=top['address'],地图地市=top['city'],地图区县=top['district'],候选数=len(candidates),可用于初步距离图=not ambiguous and top['uncertainty_km']<=20,来源URL=top['source_url'])
    except Exception as exc:return dict(**row,定位状态='查询失败_'+type(exc).__name__,经度=np.nan,纬度=np.nan,查询错误=str(exc)[:180])

def main():
    raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv')
    stations=raw[['站点ID','省份','流域','断面名称']].drop_duplicates('站点ID').sort_values('站点ID')
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,r) for r in stations.to_dict('records')]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results)%25==0 or len(results)==750:
                frame=pd.DataFrame(results).sort_values('站点ID');frame.to_csv(HERE/'站点地图定位.csv',index=False,encoding='utf-8-sig')
                eligible=int(frame.get('可用于初步距离图',pd.Series(dtype=bool)).fillna(False).sum())
                print(len(results),'/750, usable provisional locations',eligible,flush=True)
    (HERE/'地图定位汇总.json').write_text(json.dumps(pd.DataFrame(results)['定位状态'].value_counts().to_dict(),ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
