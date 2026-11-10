from pathlib import Path
import pandas as pd
import json

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'30天_750站点数据集_20250410-20250510'/'水质数据_750站点_20250410-20250510.csv'
df=pd.read_csv(SOURCE);df['监测时间']=pd.to_datetime(df['监测时间'])
cut_select=pd.Timestamp('2025-04-24');val_start=pd.Timestamp('2025-04-25');cut_refit=pd.Timestamp('2025-04-29');test_start=pd.Timestamp('2025-04-30');end=pd.Timestamp('2025-05-10')
rows=[]
for sid,g in df.groupby('站点ID'):
    g=g.sort_values('监测时间').reset_index(drop=True);lookup={t:i for i,t in enumerate(g['监测时间'])};pairs=[]
    for i in range(5,len(g)):
        origin=g.loc[i,'监测时间'];target=origin+pd.Timedelta(hours=24)
        if target in lookup:pairs.append((origin,target))
    a=sum(t<cut_select for o,t in pairs);b=sum(val_start<=t<cut_refit and o>=cut_select for o,t in pairs);c=sum(t<cut_refit for o,t in pairs);d=sum(test_start<=t<=end and o>=cut_refit for o,t in pairs)
    rows.append({'站点ID':sid,'省份':g.loc[0,'省份'],'流域':g.loc[0,'流域'],'断面':g.loc[0,'断面名称'],'原始记录数':len(g),'精确24小时配对数':len(pairs),'选epoch训练配对数':a,'验证配对数':b,'前19天重拟合配对数':c,'测试配对数':d,'满足训练验证测试最低样本条件':int(a>=3 and b>=1 and c>=3 and d>=1)})
stats=pd.DataFrame(rows);stats.to_csv(OUT/'独立预检_24小时配对与覆盖.csv',index=False,encoding='utf-8-sig')
s={'站点数':len(stats),'精确24小时配对总数':int(stats['精确24小时配对数'].sum()),'测试配对总数':int(stats['测试配对数'].sum()),'可按最低样本条件训练的站点数':int(stats['满足训练验证测试最低样本条件'].sum()),'训练配对最少':int(stats['选epoch训练配对数'].min()),'验证配对最少':int(stats['验证配对数'].min()),'测试配对最少':int(stats['测试配对数'].min()),'测试不足20条站点数':int((stats['测试配对数']<20).sum()),'不满足条件站点':stats[stats['满足训练验证测试最低样本条件']==0].to_dict('records'),'严格时间规则':'选择train label<Apr24；val target[Apr25,Apr29)；refit label<Apr29；test target[Apr30,May10]；任何训练标签与调参所见验证标签均早于最早最终测试预测起点，非仅早于目标时刻'}
(OUT/'独立预检摘要.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(s,ensure_ascii=False,indent=2))
