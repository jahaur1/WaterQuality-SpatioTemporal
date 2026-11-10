"""Effect sizes and exploratory time-block uncertainty; never retrains models."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import json,hashlib,time,math,sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).parent;BASE=HERE/'results';OUT=BASE/'R2提升幅度与显著性';OUT.mkdir(exist_ok=True)
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];START=pd.Timestamp('2025-04-30')
MFILE=BASE/'01_逐站点两分支四目标误差.csv';PFILE=BASE/'02_精确24小时测试实测与预测.csv'
M=pd.read_csv(MFILE);P=pd.read_csv(PFILE);P['目标时间']=pd.to_datetime(P['目标时间'],format='mixed')
A=M[M['分支']=='conditional_future_easy'].set_index(['站点ID','目标变量']);H=M[M['分支']=='history_only'].set_index(['站点ID','目标变量']);assert len(A)==len(H)==3000
AP={sid:g.sort_values('目标时间').reset_index(drop=True) for sid,g in P[P['分支']=='conditional_future_easy'].groupby('站点ID')}
HP={sid:g.sort_values('目标时间').reset_index(drop=True) for sid,g in P[P['分支']=='history_only'].groupby('站点ID')}
def boot_weights(length):
    k=math.ceil(10/length);total=10**k
    starts=(np.arange(total)[:,None]//(10**np.arange(k)))%10
    idx=((starts[:,:,None]+np.arange(length))%10).reshape(total,-1)[:,:10];weights=np.zeros((total,10),dtype=np.uint8)
    for day in range(10):weights[:,day]=(idx==day).sum(axis=1)
    assert np.all(weights.sum(axis=1)==10)
    unique,multiplicity=np.unique(weights,axis=0,return_counts=True)
    assert multiplicity.sum()==total
    return unique.astype(float),multiplicity,total
WEIGHTS={l:boot_weights(l) for l in [2,3]}
def adjustment(ps,dependent=False):
    p=np.where(np.isfinite(ps),ps,1.);n=len(p);order=np.argsort(p,kind='stable');factor=sum(1/k for k in range(1,n+1)) if dependent else 1.
    q=p[order]*n*factor/np.arange(1,n+1);q=np.minimum.accumulate(q[::-1])[::-1];out=np.empty(n);out[order]=np.clip(q,0,1);return out
def uncertainty(d,y,day,length):
    counts=np.bincount(day,minlength=10).astype(float);loss=np.bincount(day,weights=d,minlength=10);sy=np.bincount(day,weights=y,minlength=10);sy2=np.bincount(day,weights=y*y,minlength=10)
    w,multiplicity,total=WEIGHTS[length];nn=w@counts;ds=w@loss;ssy=w@sy;ssy2=w@sy2
    mu=d.mean();null=w@(loss-mu*counts)
    usable=nn>1;nullmeans=null[usable]/nn[usable]
    tail=int(multiplicity[usable][nullmeans>=mu].sum());valid_total=int(multiplicity[usable].sum())
    p=(1+tail)/(valid_total+1)
    sst=np.full(len(w),np.nan);sst[usable]=ssy2[usable]-ssy[usable]**2/nn[usable]
    finite=usable&(sst>1e-12*np.maximum(1.,np.abs(ssy2)))
    delta=ds[finite]/sst[finite]
    fraction=float(multiplicity[finite].sum()/total)
    if fraction>=.9:
        order=np.argsort(delta);values=delta[order];cw=np.cumsum(multiplicity[finite][order]);positions=np.searchsorted(cw,np.array([.025,.975])*cw[-1],side='left');lo,hi=values[positions]
    else:lo=hi=np.nan
    return p,float(lo),float(hi),fraction
rows=[];start=time.time()
for i,sid in enumerate(sorted(AP)):
    a=AP[sid];h=HP[sid];assert a['目标时间'].equals(h['目标时间'])
    days=((a['目标时间']-START).dt.total_seconds()//86400).astype(int).to_numpy();assert days.min()>=0 and days.max()<=10;days=np.minimum(days,9)
    ndays=len(np.unique(days));n=len(a)
    for target in CHEM:
        ma=A.loc[(sid,target)];mh=H.loc[(sid,target)];ra=float(ma['R2']);rh=float(mh['R2']);delta=ra-rh
        y=a[target+'_实测'].to_numpy();assert np.allclose(y,h[target+'_实测'].to_numpy(),atol=1e-10,rtol=0)
        pa=a[target+'_预测'].to_numpy();ph=h[target+'_预测'].to_numpy();d=(y-ph)**2-(y-pa)**2
        sst=np.sum((y-y.mean())**2)
        if np.isfinite(delta):assert np.isclose(d.sum()/sst,delta,rtol=1e-6,atol=1e-7)
        reason=''
        if not np.isfinite(delta):reason='R²不可计算，仍可查看误差差值'
        elif n<20:reason='测试配对不足20组'
        elif ndays<8:reason='非空24小时区间不足8个'
        r={'站点ID':sid,'省份':ma['省份'],'流域':ma['流域'],'断面':ma['断面'],'化学指标':target,'测试配对数':n,'非空24小时区间数':ndays,'仅历史R2':rh,'加入未来易测R2':ra,'R2提高量':delta,'R2数值提高':int(np.isfinite(delta) and delta>0),'提高后R2为正':int(np.isfinite(delta) and delta>0 and ra>0),'仅历史MAE':float(mh['MAE']),'加入未来易测MAE':float(ma['MAE']),'MAE降低量':float(mh['MAE']-ma['MAE']),'平均平方损失降低量':float(d.mean()),'可进行探索性检验':int(not reason),'未检验原因':reason}
        for l,label in [(2,'48h'),(3,'72h')]:
            if not reason:p,lo,hi,fraction=uncertainty(d,y,days,l)
            else:p=lo=hi=fraction=np.nan
            r.update({label+'_单侧近似p':p,label+'_R2差值95区间下限':lo,label+'_R2差值95区间上限':hi,label+'_自助样本R2可计算比例':fraction})
        rows.append(r)
    if i%150==0:print(f'Computed {i+1}/750 stations; {time.time()-start:.1f}s',flush=True)
R=pd.DataFrame(rows);assert len(R)==3000
for label in ['48h','72h']:
    p=R[label+'_单侧近似p'].to_numpy();R[label+'_BH校正q']=np.where(np.isfinite(p),adjustment(p,False),np.nan);R[label+'_BY校正q']=np.where(np.isfinite(p),adjustment(p,True),np.nan)
    R[label+'_探索性显著改善']=((R['R2数值提高']==1)&(R[label+'_BY校正q']<.05)&(R[label+'_R2差值95区间下限']>0)).astype(int)
def flag(r):
    if not r['可进行探索性检验']:return '证据不足，未检验：'+r['未检验原因']
    if pd.isna(r['48h_R2差值95区间下限']):return '证据不足：重采样时R²不稳定'
    return '有探索性提示（BY q<0.05且区间下限>0）' if r['48h_探索性显著改善'] else '未达到校正后的探索性显著阈值'
R['是否显著改善_主48h']=R.apply(flag,axis=1)
R['是否已确认统计显著']='未确认：仅单个10日窗口的探索性提示'
R['72h敏感性是否也支持']=R['72h_探索性显著改善']
def save(frame,name):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.12g')
save(R,'01_全部750站四指标提升与探索性检验.csv');I=R[R['R2数值提高']==1].copy();save(I,'02_R2提高的逐站逐指标明细.csv')
wide=[]
for sid,g in R.groupby('站点ID'):
    inc=g[g['R2数值提高']==1]
    if inc.empty:continue
    first=g.iloc[0];rr={'站点ID':sid,'省份':first['省份'],'流域':first['流域'],'断面':first['断面'],'提高指标数':len(inc),'提高的指标':'；'.join(inc['化学指标']),'探索性显著提高指标数':int(g['48h_探索性显著改善'].sum()),'探索性显著提高指标':'；'.join(g[g['48h_探索性显著改善']==1]['化学指标']) or '无','已做探索性检验指标数':int(g['可进行探索性检验'].sum()),'是否已有确认性显著证据':'尚不能确认：仅单个10日窗口探索'}
    for row in g.to_dict('records'):
        label=row['化学指标'].split('(')[0]
        for field in ['仅历史R2','加入未来易测R2','R2提高量','是否显著改善_主48h']:rr[label+'_'+field]=row[field]
    wide.append(rr)
W=pd.DataFrame(wide);assert len(W)==613 and len(I)==1378;save(W,'03_613个提高站点一站一行.csv')
save(W[W['探索性显著提高指标数']>0],'07_主方案探索性提示站点名单.csv');save(R[R['48h_探索性显著改善']==1],'08_主方案探索性提示的指标明细.csv')
stats=[]
for target in CHEM:
    g=R[R['化学指标']==target];positive=g[g['R2数值提高']==1];d=positive['R2提高量']
    stats.append({'化学指标':target,'基础站点数':len(g),'R2提高站点数':len(positive),'正提高量中位数':d.median(),'正提高量P25':d.quantile(.25),'正提高量P75':d.quantile(.75),'最大提高量':d.max(),'提高至少0.01站点数':int((g['R2提高量']>=.01).sum()),'提高至少0.05站点数':int((g['R2提高量']>=.05).sum()),'提高至少0.10站点数':int((g['R2提高量']>=.1).sum()),'提高后R2为正站点数':int(g['提高后R2为正'].sum()),'可探索检验站点数':int(g['可进行探索性检验'].sum()),'数值提高但证据不足未检验站点数':int(((g['R2数值提高']==1)&(g['可进行探索性检验']==0)).sum()),'48h_BY校正探索性显著站点数':int(g['48h_探索性显著改善'].sum()),'72h_BY校正探索性显著站点数':int(g['72h_探索性显著改善'].sum()),'48h未校正p小于005站点数':int(((g['48h_单侧近似p']<.05)&(g['R2数值提高']==1)).sum()),'48h仅BH校正q小于005站点数':int(((g['48h_BH校正q']<.05)&(g['R2数值提高']==1)).sum())})
T=pd.DataFrame(stats);save(T,'04_各化学指标改善幅度与显著性汇总.csv')
hist=W['提高指标数'].value_counts().sort_index();distribution=pd.DataFrame({'提高指标数':range(5),'站点数':[750-len(W)]+[int(hist.get(k,0)) for k in range(1,5)]});save(distribution,'05_每站提高指标数量分布.csv')
basin=W.groupby('流域').agg(R2提高站点数=('站点ID','size'),至少一项探索性显著站点数=('探索性显著提高指标数',lambda x:int((x>0).sum()))).reset_index();save(basin,'06_流域提升与显著性站点数.csv')
summary={'全部站点数':750,'站点指标比较总数':3000,'数值提高站点数':len(W),'数值提高站点指标数':len(I),'提高指标数分布':distribution.to_dict('records'),'48h至少一项BY校正探索性显著站点数':int((W['探索性显著提高指标数']>0).sum()),'72h至少一项BY校正探索性显著站点数':int(R[R['72h_探索性显著改善']==1]['站点ID'].nunique()),'可检验站点指标数':int(R['可进行探索性检验'].sum()),'未检验站点指标数':int((R['可进行探索性检验']==0).sum()),'48h全枚举序列数':WEIGHTS[2][2],'72h全枚举序列数':WEIGHTS[3][2],'主块长度小时':48,'敏感性块长度小时':72,'随机抽样':False,'多重比较家族数':3000,'主校正规则':'BY q<0.05，并要求48h配对自助ΔR²区间下限>0；全部3000比较校正，未检验项p=1参与','方法性质':'全枚举消除模拟误差，不能消除10天时间块自助近似误差；不是确认性统计证明','指标汇总':stats,'输入文件SHA256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [MFILE,PFILE]},'耗时秒':round(time.time()-start,2)}
(OUT/'分析摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
report='''# R²提高多少，以及是否有显著改善证据

## 对照与效应量

比较“历史9项＋未来易测真值”与“同历史9项、不使用未来信息”。不是与持久性或此前同期岭回归比较。所有750站、四化学目标共3000组均保留，先作整家族校正，再展示数值提高者，不只检验筛出的613站。

ΔR²＝加入未来易测值模型R²−仅历史模型R²。ΔR²=0.05表示增加0.05个R²单位，不是提高5%；负R²情况下相对百分比没有稳定解释。初始R²特别差时，较大ΔR²仍可能对应负R²，所以同时保留两个原始值。所有分档只作描述，不是验收标准。

## 幅度汇总

正提高量的中位数及四分位数只针对该指标数值提高的站点，不是全部750站。

|指标|提高站点数|正提高量中位数|正提高量P25—P75|ΔR²≥0.05|ΔR²≥0.10|提高后R²>0|探索性显著|
|---|---:|---:|---|---:|---:|---:|---:|
'''+''.join(f"|{r['化学指标']}|{r['R2提高站点数']}|{r['正提高量中位数']:.5g}|{r['正提高量P25']:.5g}—{r['正提高量P75']:.5g}|{r['提高至少0.05站点数']}|{r['提高至少0.10站点数']}|{r['提高后R2为正站点数']}|{r['48h_BY校正探索性显著站点数']}|\n" for r in stats)+f'''
共有613站至少一项数值提高，1378组站点—指标比较为正提高。主48小时设定下，至少一项达到BY校正探索性显著规则的站点为{summary['48h至少一项BY校正探索性显著站点数']}个。72小时敏感性设定对应{summary['72h至少一项BY校正探索性显著站点数']}个。这些是主方案下的探索性提示，尚不能宣布存在确认性的统计显著改善；未达到阈值不等于无改善。

## 显著性方法与重要限制

主方法为按日历天构造48小时圆形移动块的配对自助法，72小时块作敏感性分析。48小时完整枚举100,000种起点组合，72小时完整枚举10,000种；相同重采样权重合并并按出现次数计算，不使用随机蒙特卡洛来估计极小尾概率。4月30日至5月10日00:00的端点并入最后一个日区间，共10个24小时区间。每次抽取相邻2天（或3天）块并圆形衔接，拼回10天；所有观测按原来的点数加权，日区间不是简单等权平均损失。

每个配对损失差d=(仅历史误差)²−(未来信息误差)²。单侧检验关注平均d是否大于0；零假设自助序列先按原平均d中心化。近似p=(1+零假设自助均值≥原均值的枚举次数)/(1+有效枚举总数)，其中加1是保守平滑，避免报告p=0；全枚举的含义只是无蒙特卡洛误差，不是精确有限样本检验。差值区间从配对重采样的ΔR²加权百分位数估计，重算各自助样本的目标方差；若超过10%重采样中R²无法定义，则不输出该区间，不标显著。

主标记要求：原始ΔR²>0、48小时块近似p经全部3000比较的BY校正q<0.05，并且配对ΔR²的95%区间下限>0。同时附BH校正作为较宽松参考；跨站点/指标可能依赖，因此主报告用较保守BY。未检验项在家族校正中设p=1，但其结果表中p、q仍留空，不伪装成真正检验结果。

本次采用的最低筛查为测试配对≥20、至少8个非空日区间、R²可计算。它们只是操作性证据门槛，不是这些数据已满足所有统计假设的证明。证据不足项标“未检验”，而不是“不显著”。

时间块自助法仍依赖弱依赖、局部平稳及有限矩等近似条件；10天最多只有约5个48小时有效块，无法可靠验证这些条件或覆盖新季节。BY校正不能修复原始近似p值的有效性问题。本次自助法只针对已训练模型在这段测试窗口的损失，没有重训神经网络来计入训练不确定性。数据此前已做过探索，所以所有“显著”均明确标为探索性，不作为确认性证明或化学检测替代依据。不能把“未显著”解读为效果不存在；本数据很可能缺乏检验力。

72小时块更少，完整枚举只有10,000种且这里使用保守平滑，最小近似p约0.0001；在3000重比较校正下分辨率和检验力受限。因此72小时不达阈值不能证明效果消失，也不宜据此作正式的稳健性否定。全枚举后没有蒙特卡洛尾部估计误差，但10天近似分布仍不足以可靠校准极小的真实尾概率。

方法参考：[配对预测损失比较](https://doi.org/10.1080/07350015.1995.10524599)、[时间序列圆形块重采样](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data)、[BY多重比较依赖校正](https://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf)。本实现的按日聚合与样本门槛属于本次分析方案，不是上述论文对当前10日数据有效性的保证。

## 文件

- 01：全部750站×4指标的原始R²、提高量、误差差值、区间、近似p与校正q。
- 02：只包含1378组数值提高的站点—指标明细，仍保留未检验原因。
- 03：613个提高站点，一站一行，列出四指标提高量及显著性标记。
- 04：四化学指标的幅度、分档和检验数量汇总。
- 05：每站恰好0、1、2、3、4项提高的分布。
- 06：各流域数值提高站点和探索性显著站点数量。
- 07、08：主48小时方案有提示的站点名单及站点—指标明细，仍全部明确为探索性。
'''
(OUT/'提升幅度与探索性显著性说明.md').write_text(report,encoding='utf-8')
assert int(T['R2提高站点数'].sum())==1378 and distribution['站点数'].sum()==750
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
