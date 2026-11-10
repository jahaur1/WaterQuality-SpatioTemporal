"""Apply the final transparent matching rules to cached public searches only."""
import json
import pandas as pd
from geocode import HERE,ROOT,one
import geocode

def cached(sid,text,suffix):
    p=HERE/'map_cache'/f'{sid}_{suffix}.json'
    if p.exists():return json.loads(p.read_text(encoding='utf-8'))
    return dict(query=text,url='',candidates=[])

geocode.query=cached
before=pd.read_csv(HERE/'站点地图定位.csv').set_index('站点ID')
raw=pd.read_csv(ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv')
stations=raw[['站点ID','省份','流域','断面名称']].drop_duplicates('站点ID').sort_values('站点ID')
rows=[one(r) for r in stations.to_dict('records')]
frame=pd.DataFrame(rows);frame['网络检索状态']=frame['站点ID'].map(before['网络检索状态'] if '网络检索状态' in before else before['定位状态'])
frame['定位规则版本']='同省匹配_地物估算_有歧义排除_v2'
frame['定位日期']='2026-11-09'
frame.to_csv(HERE/'站点地图定位.csv',index=False,encoding='utf-8-sig')
summary=dict(stations=750,located=int(frame['经度'].notna().sum()),eligible=int(frame['可用于初步距离图'].fillna(False).sum()),status=frame['定位状态'].value_counts().to_dict(),classes=frame['估算类别'].value_counts().to_dict(),network_failures=int(frame['网络检索状态'].str.contains('失败').sum()))
(HERE/'地图定位汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(summary)
