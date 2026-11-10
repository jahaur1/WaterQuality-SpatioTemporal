import csv,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'30天_750站点数据集_20250410-20250510').is_dir() else HERE.parent.parent
BASE=ROOT/'30天_750站点数据集_20250410-20250510';SOURCE=BASE/'水质数据_750站点_20250410-20250510.csv'
OUT=BASE/'测5项估4项_时间留出预测基线';OUT.mkdir(exist_ok=True)
XCOL=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']
YCOL=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
ALPHAS=[0.,.1,1.,10.,100.];CUT=pd.Timestamp('2025-04-30 00:00:00');STOP=pd.Timestamp('2025-05-10 00:00:00')
FOLDS=[(pd.Timestamp('2025-04-20'),pd.Timestamp('2025-04-25')),(pd.Timestamp('2025-04-25'),CUT)]
sourcehash=hashlib.sha256(SOURCE.read_bytes()).hexdigest();assert sourcehash=='9267d65df4c67f934339812b9fdea839696594c8c55af6b8f6fefb445d3cb7eb'
df=pd.read_csv(SOURCE);df['监测时间']=pd.to_datetime(df['监测时间']);assert len(df)==79940 and df['站点ID'].nunique()==750
assert np.isfinite(df[XCOL+YCOL].to_numpy(dtype=float)).all()
assert not df.duplicated(['站点ID','监测时间']).any()
def fit(x,y,alpha):
    xm=x.mean(0);xs=x.std(0);xs=np.where(xs>1e-12,xs,1.);ym=y.mean(0);z=(x-xm)/xs;yc=y-ym
    beta=np.linalg.lstsq(z,yc,rcond=None)[0] if alpha==0 else np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@yc)
    return xm,xs,ym,beta
def predict(params,x):
    xm,xs,ym,beta=params;raw=((x-xm)/xs)@beta+ym
    return np.maximum(raw,0),raw
def score(y,p):
    residual=p-y;denom=np.sum((y-y.mean())**2);tol=1e-12*max(1.,float(np.sum(y*y)))
    return {'MAE':float(np.mean(np.abs(residual))),'RMSE':float(np.sqrt(np.mean(residual**2))),'偏差_预测减实测':float(residual.mean()),'R2':float(1-np.sum(residual**2)/denom) if denom>tol else np.nan,'P95绝对误差':float(np.quantile(np.abs(residual),.95))}
metrics=[];predictions=[];models=[];cvdetail=[];siteinfo=[];start=time.time();verified_sites=0
for ix,(sid,g) in enumerate(df.groupby('站点ID')):
    g=g.sort_values('监测时间');tr=g[g['监测时间']<CUT];te=g[(g['监测时间']>=CUT)&(g['监测时间']<=STOP)];assert len(tr)>5 and len(te)>1 and tr['监测时间'].max()<te['监测时间'].min()
    x=tr[XCOL].to_numpy(float);y=tr[YCOL].to_numpy(float);xt=te[XCOL].to_numpy(float);yt=te[YCOL].to_numpy(float)
    sse=np.zeros((len(ALPHAS),4));nv=0;foldcount=0
    for foldnum,(boundary,end) in enumerate(FOLDS,1):
        ft=tr[tr['监测时间']<boundary];fv=tr[(tr['监测时间']>=boundary)&(tr['监测时间']<end)]
        if len(ft)<10 or len(fv)<3:continue
        assert ft['监测时间'].max()<fv['监测时间'].min() and fv['监测时间'].max()<CUT
        xf=ft[XCOL].to_numpy(float);yf=ft[YCOL].to_numpy(float);xv=fv[XCOL].to_numpy(float);yv=fv[YCOL].to_numpy(float)
        for ai,alpha in enumerate(ALPHAS):
            pv,_=predict(fit(xf,yf,alpha),xv);errors=((pv-yv)**2).sum(0);sse[ai]+=errors
            for j,target in enumerate(YCOL):cvdetail.append([sid,target,foldnum,alpha,len(ft),len(fv),float(errors[j]/len(fv)),str(ft['监测时间'].min()),str(ft['监测时间'].max()),str(fv['监测时间'].min()),str(fv['监测时间'].max())])
        nv+=len(fv);foldcount+=1
    chosen=np.argmin(sse,axis=0) if nv else np.full(4,ALPHAS.index(1.))
    p=np.zeros_like(yt);rawp=np.zeros_like(yt)
    for ai,alpha in enumerate(ALPHAS):
        mask=np.flatnonzero(chosen==ai)
        if not len(mask):continue
        params=fit(x,y,alpha);pv,raw=predict(params,xt);p[:,mask]=pv[:,mask];rawp[:,mask]=raw[:,mask]
        xm,xs,ym,beta=params
        # Recover original-unit equation coefficients, and independently compare predictions.
        coef=beta/xs[:,None];intercept=ym-xm@coef
        assert np.allclose(raw,xt@coef+intercept,rtol=1e-8,atol=1e-8)
        for j in mask:
            models.append({'站点ID':sid,'目标变量':YCOL[j],'岭惩罚alpha':alpha,'截距':float(intercept[j]),**{f'{c}_系数':float(coef[k,j]) for k,c in enumerate(XCOL)},'训练记录数':len(tr),'训练内有效验证折数':foldcount,'训练内验证记录数':nv,'训练内验证MSE':float(sse[ai,j]/nv) if nv else np.nan,'无验证时回退参数':int(nv==0),'预测后处理':'负值截为0'})
    meanbase=np.tile(y.mean(0),(len(te),1));lastbase=np.tile(y[-1],(len(te),1));meta=g.iloc[0]
    for j,target in enumerate(YCOL):
        m=score(yt[:,j],p[:,j]);mb=score(yt[:,j],meanbase[:,j]);lb=score(yt[:,j],lastbase[:,j]);tstd=float(yt[:,j].std())
        metrics.append({'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面名称'],'目标变量':target,'训练记录数':len(tr),'测试记录数':len(te),'测试少于20条':int(len(te)<20),'训练少于30条':int(len(tr)<30),'训练内有效验证折数':foldcount,'模型alpha':ALPHAS[int(chosen[j])],**m,'训练均值基线MAE':mb['MAE'],'训练均值基线RMSE':mb['RMSE'],'训练均值基线R2':mb['R2'],'末次训练值基线MAE':lb['MAE'],'末次训练值基线RMSE':lb['RMSE'],'末次训练值基线R2':lb['R2'],'较训练均值基线MAE改进比例':1-m['MAE']/mb['MAE'] if mb['MAE']>0 else np.nan,'较末次训练值基线MAE改进比例':1-m['MAE']/lb['MAE'] if lb['MAE']>0 else np.nan,'测试实测标准差':tstd,'R2可计算':int(np.isfinite(m['R2'])),'原始预测为负次数':int((rawp[:,j]<0).sum()),'训练开始':str(tr['监测时间'].min()),'训练结束':str(tr['监测时间'].max()),'测试开始':str(te['监测时间'].min()),'测试结束':str(te['监测时间'].max())})
    for k,(_,r) in enumerate(te.iterrows()):
        predictions.append({'站点ID':sid,'省份':r['省份'],'流域':r['流域'],'断面':r['断面名称'],'监测时间':str(r['监测时间']),**{c:float(xt[k,j]) for j,c in enumerate(XCOL)},**{f'{target}_{label}':float(values[k,j]) for j,target in enumerate(YCOL) for label,values in [('实测',yt),('预测',p),('未截断预测',rawp),('训练均值基线',meanbase),('末次训练值基线',lastbase)]},})
    siteinfo.append({'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面名称'],'训练记录数':len(tr),'测试记录数':len(te),'训练内有效验证折数':foldcount,'测试少于20条':int(len(te)<20),'训练少于30条':int(len(tr)<30),'训练无变化易测变量':'；'.join(c for j,c in enumerate(XCOL) if x[:,j].std()==0)})
    verified_sites+=1
    if ix%150==0:print(f'Fitted {ix+1}/750 stations; {time.time()-start:.1f}s',flush=True)
M=pd.DataFrame(metrics);P=pd.DataFrame(predictions);S=pd.DataFrame(siteinfo);assert len(M)==3000 and len(S)==750 and len(P)==int(S['测试记录数'].sum())
assert np.isfinite(P.select_dtypes(include=[np.number]).to_numpy()).all()
assert (M['RMSE']+1e-12>=M['MAE']).all()
for target in YCOL:assert (P[target+'_预测']>=0).all()
assert not P.duplicated(['站点ID','监测时间']).any() and pd.to_datetime(P['监测时间']).min()>=CUT
def save(frame,name):frame.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.12g')
save(M,'01_750站点各化学指标预测误差.csv');save(P,'02_测试集实测与预测值.csv');save(S,'03_时间划分与样本量.csv');save(pd.DataFrame(models),'04_模型参数与输入系数.csv')
save(pd.DataFrame(cvdetail,columns=['站点ID','目标变量','训练内验证折','alpha','拟合记录数','验证记录数','验证MSE','拟合起始','拟合终点','验证起始','验证终点']),'05_训练段滚动调参记录.csv')
def aggregate(g,label,target):
    valid=g['R2'].dropna()
    return {'流域':label,'目标变量':target,'站点数':len(g),'测试记录合计':int(g['测试记录数'].sum()),'R2可计算站点数':len(valid),'站点R2中位数':float(valid.median()) if len(valid) else np.nan,'站点R2_P25':float(valid.quantile(.25)) if len(valid) else np.nan,'站点R2_P75':float(valid.quantile(.75)) if len(valid) else np.nan,'站点MAE中位数':float(g['MAE'].median()),'站点RMSE中位数':float(g['RMSE'].median()),'站点偏差中位数':float(g['偏差_预测减实测'].median()),'训练均值基线MAE中位数':float(g['训练均值基线MAE'].median()),'末次训练值基线MAE中位数':float(g['末次训练值基线MAE'].median()),'测试不足20条站点数':int(g['测试少于20条'].sum()),'说明':'各站等权的指标中位数；不代表合格率'}
A=pd.DataFrame([aggregate(g,'全部流域',target) for target,g in M.groupby('目标变量',sort=False)])
B=pd.DataFrame([aggregate(g,basin,target) for (basin,target),g in M.groupby(['流域','目标变量'])]);save(A,'06_总体误差与R2分布.csv');save(B,'07_分流域误差与R2分布.csv')
# Re-read saved predictions and independently recalculate all 3,000 station-target errors.
saved=pd.read_csv(OUT/'02_测试集实测与预测值.csv');lookup=M.set_index(['站点ID','目标变量'])
for sid,g in saved.groupby('站点ID'):
    for target in YCOL:
        sm=score(g[target+'_实测'].to_numpy(),g[target+'_预测'].to_numpy());stored=lookup.loc[(sid,target)]
        for name in ['MAE','RMSE','R2','偏差_预测减实测']:
            assert np.isclose(sm[name],stored[name],rtol=1e-7,atol=1e-8,equal_nan=True),(sid,target,name)
assert SOURCE.exists() and hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sourcehash
summary={'直接测量输入数':5,'估计化学指标数':4,'总指标数':9,'站点数':750,'训练记录数':int(S['训练记录数'].sum()),'测试记录数':len(P),'训练时段':'2025-04-10 00:00至2025-04-30 00:00之前','测试时段':'2025-04-30 00:00至2025-05-10 00:00含端点','站点目标评价行数':len(M),'训练不足30条站点数':int(S['训练少于30条'].sum()),'测试不足20条站点数':int(S['测试少于20条'].sum()),'无有效训练内验证站点数':int((S['训练内有效验证折数']==0).sum()),'常量测试目标导致R2不可计算数量':int(M['R2'].isna().sum()),'预测模型':'每站每目标独立岭回归，alpha含0对应OLS，训练段前推验证选参数；负浓度截0','总体汇总':A.to_dict('records'),'验收阈值':'用户尚未设定，不计算可推导成功率','原始数据SHA256':sourcehash,'验证':{'所有训练监测时间早于相应测试':True,'各折训练早于验证且验证均早于最终测试':True,'标准化仅在各训练折拟合':True,'不以测试成绩选参数或变量':True,'全部站点保留':True,'保存预测独立复算3000组误差一致':True,'原始数据未改写':True},'耗时秒':round(time.time()-start,1)}
(OUT/'实验摘要与校验.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
report='''# 测5项、估4项：时间留出预测基线

## 已明确的研究口径

用户已确认“共9项”指直接测5项、估计另外4项；目前没有业务允许误差标准，先报告预测误差与R²，不把相关比例当作成功率。2500站为设想规模，本次实测研究仍为全部750站，不使用533、406或85站替代基础数据集。

输入：同一监测时刻的水温、pH、溶解氧、电导率、浊度。目标：同一时刻的高锰酸盐指数、氨氮、总磷、总氮。这是软测量/同步估计，不是提前若干小时预测未来水质。

## 划分与模型

逐站独立建模，时间固定划分：4月10日至4月30日之前训练，4月30日至5月10日00:00（含）测试。使用实际记录，不插值、不构造统一采样网格，不随机混合训练和测试。

模型为标准化输入的岭回归，候选alpha为0、0.1、1、10、100；0对应普通最小二乘。每个化学目标分别在训练段内按前推验证MSE选alpha：先用4月20日之前拟合、4月20日至25日之前验证；再用4月25日之前拟合、4月25日至30日之前验证。每折至少10条拟合、3条验证；若不足则回退alpha=1并标记。最终只用整个前20天重拟合。均值、标准差、参数选择均不使用最终测试段。

浓度不应为负，预测负值统一截为0，调参与测试使用同一规则，同时保存未截断预测供核对。模型输入没有任何测试段化学实测值。对比基线为训练段化学均值、训练段最后一次化学值；后者在整个测试段固定，不使用测试期间新测的化学值。

## 总体结果

下表是各站指标的中位数，每站等权。四项化学指标的MAE、RMSE及基线MAE单位均为mg/L。R²为各站测试段单独计算后的中位数，不是全站混合R²。

|目标|MAE中位数|RMSE中位数|R²中位数|训练均值基线MAE中位数|末次训练值基线MAE中位数|
|---|---:|---:|---:|---:|---:|
'''+''.join(f"|{r['目标变量']}|{r['站点MAE中位数']:.5g}|{r['站点RMSE中位数']:.5g}|{r['站点R2中位数']:.4g}|{r['训练均值基线MAE中位数']:.5g}|{r['末次训练值基线MAE中位数']:.5g}|\n" for r in A.to_dict('records'))+f'''
训练观测{summary['训练记录数']:,}条、测试观测{len(P):,}条；测试观测少于20条的站点{summary['测试不足20条站点数']}个，训练观测少于30条的站点{summary['训练不足30条站点数']}个。未因此删除站点，明细含样本量标记。测试目标几乎无变异导致R²不可计算的站点—目标组合{summary['常量测试目标导致R2不可计算数量']}个，保留误差但R²留空。

## 如何解读

- MAE是平均绝对误差；RMSE对较大误差更敏感；偏差为预测减实测的平均值。
- R²=1表示测试观测完美拟合；负值表示平方误差大于用该站测试段实测均值作为常数的误差。这不等于相对可部署的训练均值或末次训练值基线一定更差，须同时比较实际基线误差。
- 测试目标变化很小时，即使绝对误差不大，R²也可能很低或极端负值。按浓度单位和业务需求解释，不能仅凭R²定义合格，也没有将R²>0或≥0.7当作验收条件。
- 本轮只有一个简单模型族，表现不好不能证明变量不可推导；后续可以在严格保持测试隔离的条件下比较非线性方法。
- 30天数据此前已做相关探索，这次属于回溯性时间留出基线，不是完全未经探索的前瞻外部验证。275站测试观测少于20条，且存在时间自相关，不能凭这次小测试段作长期稳定性保证。
- 这轮检验已有站点利用本地前段化学实测作校准后、在后段用易测指标估计化学指标；不是从未做化学测量的新站直接迁移，也不代表可以取消全部化学抽检。
- 输入固定5项，尚未进行31个非空输入子集的变量预算优化。尚未设置容许误差、计算达标站点比例或检测成本节省比例，不能将750站的结果直接外推至2500站。

## 文件

- 01：750站×4目标的全部MAE、RMSE、R²、偏差、基线误差及样本量。
- 02：最终测试段实际值、预测值、未截断预测和两个基线预测，保留输入值，便于独立复算。
- 03：逐站训练测试样本量和不足标记。
- 04、05：模型原单位参数和训练段内调参记录。
- 06、07：总体及各流域的逐站指标分布汇总。

下一步是结合允许误差和验证证据，逐目标定义达标，再统计每站至少1、2、3、4项达标的数量；分母保留750，并另列证据不足站点。输入组合选择必须在训练阶段完成，不能用最终测试成绩挑组合。
'''
(OUT/'预测基线报告.md').write_text(report,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=True),flush=True)
