from pathlib import Path
import sys,json,hashlib
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REF=ROOT/'预训练模型水质24小时评估_20261103';OLD=ROOT/'XLinear水质24小时条件预测_20261102/results'
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d,n):d.to_csv(HERE/n,index=False,encoding='utf-8-sig',float_format='%.12g')
def score(y,p):
    e=p-y;ss=((y-y.mean())**2).sum();tol=1e-12*max(1.,float((y*y).sum()))
    return {'MAE':np.abs(e).mean(),'RMSE':np.sqrt((e*e).mean()),'偏差':e.mean(),'R2':1-(e*e).sum()/ss if ss>tol else np.nan}
def main():
    idx=pd.read_csv(REF/'测试配对索引.csv');p=pd.read_csv(OLD/'02_精确24小时测试实测与预测.csv');p=p[p['分支']=='conditional_future_easy'].reset_index(drop=True)
    assert len(idx)==len(p)==23136 and idx['站点ID'].equals(p['站点ID'])
    for c in ['起报时间','目标时间','输入历史起点']:assert pd.to_datetime(idx[c],format='mixed').equals(pd.to_datetime(p[c],format='mixed'))
    assert (pd.to_datetime(idx['目标时间'],format='mixed')-pd.to_datetime(idx['起报时间'],format='mixed')==pd.Timedelta(hours=24)).all()
    groups=idx.groupby('站点ID',sort=False).groups;y=p[[c+'_实测' for c in CHEM]].to_numpy(float)
    oldm=pd.read_csv(OLD/'01_逐站点两分支四目标误差.csv').set_index(['站点ID','分支','目标变量'])
    rows=[];frames=[]
    for arm in ['history_only','future_easy']:
        dest=HERE/'results'/arm;done=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert done['completed'] and done['N']==23136 and done['inputs_sha256']==sha(REF/'inputs.npz') and done['cross_learning']==False
        z=np.load(dest/'predictions.npz');pred=z['prediction'];raw=z['raw'];assert np.isfinite(pred).all() and pred.shape==(23136,4)
        pp=idx.copy();pp['输入条件']=arm
        for j,c in enumerate(CHEM):pp[c+'_实测']=y[:,j];pp[c+'_预测']=pred[:,j];pp[c+'_未截断预测']=raw[:,j]
        frames.append(pp)
        for sid,ix in groups.items():
            meta=idx.iloc[ix[0]];oldarm='history_only' if arm=='history_only' else 'conditional_future_easy'
            for j,c in enumerate(CHEM):
                s=score(y[ix,j],pred[ix,j]);oldr=oldm.loc[(sid,oldarm,c)]
                rows.append({'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面'],'模型':'Chronos-2','输入条件':arm,'目标变量':c,'测试配对数':len(ix),**s,'未截断MAE':score(y[ix,j],raw[ix,j])['MAE'],'原始负预测次数':int((raw[ix,j]<0).sum()),'对应XLinearR2':oldr['R2'],'对应XLinearMAE':oldr['MAE'],'较XLinear_R2提高量':s['R2']-oldr['R2'],'较XLinear_MAE降低量':oldr['MAE']-s['MAE']})
    m=pd.DataFrame(rows);assert len(m)==6000;save(m,'01_逐站点两条件四指标误差.csv');save(pd.concat(frames,ignore_index=True),'02_逐配对实测与预测.csv')
    summary=[]
    for (arm,c),g in m.groupby(['输入条件','目标变量'],sort=False):
        summary.append({'输入条件':arm,'目标变量':c,'站点数':len(g),'R2可计算站点数':g.R2.notna().sum(),'R2中位数':g.R2.median(),'MAE中位数':g.MAE.median(),'RMSE中位数':g.RMSE.median(),'R2大于05站点数':int((g.R2>.5).sum()),'R2大于08站点数':int((g.R2>.8).sum()),'R2优于对应XLinear站点数':int((g['较XLinear_R2提高量']>0).sum()),'对应XLinearR2中位数':g['对应XLinearR2'].median()})
    summary=pd.DataFrame(summary);save(summary,'03_总体R2及误差汇总.csv')
    h=m[m['输入条件']=='history_only'].set_index(['站点ID','目标变量']);f=m[m['输入条件']=='future_easy'].copy()
    f['仅历史R2']=[h.loc[(r['站点ID'],r['目标变量']),'R2'] for _,r in f.iterrows()];f['仅历史MAE']=[h.loc[(r['站点ID'],r['目标变量']),'MAE'] for _,r in f.iterrows()]
    f['加入未来易测值_R2提高量']=f.R2-f['仅历史R2'];f['加入未来易测值_MAE降低量']=f['仅历史MAE']-f.MAE;save(f,'04_加入未来易测值的逐站点收益.csv')
    unique=[]
    for arm,g in m.groupby('输入条件',sort=False):
        r={'输入条件':arm,'站点数':750}
        for cut,label in [(.5,'05'),(.8,'08')]:
            cc=g[g.R2>cut].groupby('站点ID').size();r['任一指标R2大于'+label]=len(cc);r['四项均R2大于'+label]=int((cc==4).sum());r['任一指标R2大于'+label+'_且至少20对']=g[(g.R2>cut)&(g['测试配对数']>=20)]['站点ID'].nunique()
        unique.append(r)
    unique=pd.DataFrame(unique);save(unique,'05_高R2站点去重统计.csv');save(f[f.R2>.5].sort_values('R2',ascending=False),'06_条件预测R2大于05的站点指标.csv')
    basin=f.groupby(['流域','目标变量']).agg(站点数=('站点ID','size'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),R2大于05站数=('R2',lambda x:int((x>.5).sum())),R2大于08站数=('R2',lambda x:int((x>.8).sum()))).reset_index();save(basin,'07_分流域条件预测汇总.csv')
    # Independent vectorized metric check using groupby and residual sum of squares.
    for arm in ['history_only','future_easy']:
        pred=np.load(HERE/'results'/arm/'predictions.npz')['prediction'];check=m[m['输入条件']==arm].set_index(['站点ID','目标变量'])
        for j,c in enumerate(CHEM):
            t=pd.DataFrame({'sid':idx['站点ID'],'y':y[:,j],'e':pred[:,j]-y[:,j]});t['ae']=t.e.abs();t['e2']=t.e*t.e;t['yc']=t.y-t.groupby('sid').y.transform('mean');t['yc2']=t.yc*t.yc;t['y2']=t.y*t.y
            stats=t.groupby('sid').agg(mae=('ae','mean'),mse=('e2','mean'),sse=('e2','sum'),sst=('yc2','sum'),sumy2=('y2','sum'))
            rr=1-stats.sse/stats.sst;rr[stats.sst<=1e-12*np.maximum(1.,stats.sumy2)]=np.nan;actual=check.xs(c,level='目标变量').reindex(stats.index)
            assert np.allclose(stats.mae,actual.MAE,rtol=1e-8,atol=1e-8) and np.allclose(np.sqrt(stats.mse),actual.RMSE,rtol=1e-8,atol=1e-8) and np.allclose(rr,actual.R2,equal_nan=True,rtol=1e-8,atol=1e-8)
        pilot=np.load(HERE/'pilot'/arm/'predictions.npz')['raw'];full=np.load(HERE/'results'/arm/'predictions.npz')['raw'];assert np.array_equal(pilot,full[:len(pilot)])
    refaudit=json.loads((REF/'独立核验.json').read_text(encoding='utf-8'));assert refaudit['passed']
    refcfg=json.loads((REF/'实验配置.json').read_text(encoding='utf-8'));source=ROOT/'30天_750站点数据集_20250410-20250510/水质数据_750站点_20250410-20250510.csv';assert sha(source)==refcfg['source_sha256'] and sha(OLD/'02_精确24小时测试实测与预测.csv')==refcfg['old_predictions_sha256']
    download=json.loads((HERE/'下载SHA256核验.json').read_text(encoding='utf-8'));assert download['passed'] and sha(HERE/'weights/model.safetensors')==download['sha256']
    isolation=json.loads((HERE/'输入隔离核验.json').read_text(encoding='utf-8'));assert isolation['passed']
    audit={'passed':True,'stations':750,'exact24h_pairs':23136,'metrics_independently_recomputed':6000,'reference_input_history_audit_passed_and_input_hash_matched':True,'both_arm_first128_replay_exact_match':True,'future_input_and_cross_task_isolation_passed':True,'official_weight_sha256_verified':True,'original_data_and_XLinear_predictions_unchanged':True}
    (HERE/'独立核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    def table(d):return '| '+' | '.join(d.columns)+' |\n|'+ '|'.join(['---']*len(d.columns))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in d.to_numpy())
    report='# Chronos-2 水质24小时条件预测\n\n官方amazon/chronos-2权重，冻结权重、不微调；官方chronos-forecasting 2.3.2安装在项目独立runtime目录，未修改现有环境。权重SHA256已与官方LFS记录核对。\n\n'
    report+='750站，23,136组精确24小时配对，与XLinear和MOIRAI测试配对相同。沿用已核验MOIRAI输入：截至起报的最近6次9项观测，按真实时间置于4小时网格，仅在这6次观测首尾之间做历史前向填充，首条历史之前为NaN；历史32格，未来6格，取第6格。\n\n'
    report+='target是4项化学历史；past_covariates是5项易测历史。条件组future_covariates仅最后一格提供目标时刻5项易测真值，中间5格为NaN；历史组完全不提供future_covariates。未来化学实测不进入模型。\n\n'
    report+='cross_learning=False，每个预测起点独立分组，禁止跨站点或跨起点共享信息。固定FP32、128个预测窗口一批（1152条变量序列），取原生0.5分位数为点预测；负浓度预测截断0，同时保留原始预测。\n\n'
    report+='这是未来易测实测已知、化学历史随每次起报更新的回溯条件实验；不证明可减少化学检测。测试仅10天，阈值为描述性分档，不是业务成功率。模型无本站微调而XLinear有本站监督训练；历史网格填充也与XLinear离散6次编码不同，不是只隔离架构的实验。没有按测试表现调整参数或挑选站点。\n\n'
    view=summary.copy();view[['R2中位数','MAE中位数','RMSE中位数','对应XLinearR2中位数']]=view[['R2中位数','MAE中位数','RMSE中位数','对应XLinearR2中位数']].round(5)
    report+=table(view)+'\n'+table(unique)+'\n## 加入未来信息的变化\n\n'
    gain=pd.DataFrame([{'指标':c,'R2提高站点数':int((g['加入未来易测值_R2提高量']>0).sum()),'全部站点R2差值中位数':round(g['加入未来易测值_R2提高量'].median(),5),'MAE降低站点数':int((g['加入未来易测值_MAE降低量']>0).sum())} for c,g in f.groupby('目标变量',sort=False)])
    report+=table(gain)+'\n独立核验通过：全部6000组指标复算、两组首128预测重放、未来输入和跨样本隔离、原始数据及旧预测哈希、官方下载SHA256。未进行新的统计显著性声明。\n\n参考：[Chronos官方仓库](https://github.com/amazon-science/chronos-forecasting)、[官方权重](https://huggingface.co/amazon/chronos-2)。\n'
    (HERE/'评估报告.md').write_text(report,encoding='utf-8');print(summary.to_string(index=False));print(unique.to_string(index=False));print(gain.to_string(index=False));print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
