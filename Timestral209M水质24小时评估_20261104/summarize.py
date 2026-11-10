from pathlib import Path
import sys,json,hashlib
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REF=ROOT/'预训练模型水质24小时评估_20261103';OLD=ROOT/'XLinear水质24小时条件预测_20261102/results';CHRONOS=ROOT/'Chronos2水质24小时评估_20261103'
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
    oldm=pd.read_csv(OLD/'01_逐站点两分支四目标误差.csv').set_index(['站点ID','分支','目标变量']);cm=pd.read_csv(CHRONOS/'01_逐站点两条件四指标误差.csv').set_index(['站点ID','输入条件','目标变量'])
    rows=[];frames=[]
    for arm in ['history_only','future_easy']:
        dest=HERE/'results'/arm;done=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert done['completed'] and done['N']==23136 and done['inputs_sha256']==sha(REF/'inputs.npz') and done['high_level_wrapper_interpolation_used']==False
        z=np.load(dest/'predictions.npz');pred=z['prediction'];raw=z['raw'];assert np.isfinite(pred).all() and pred.shape==(23136,4)
        pp=idx.copy();pp['输入条件']=arm
        for j,c in enumerate(CHEM):pp[c+'_实测']=y[:,j];pp[c+'_预测']=pred[:,j];pp[c+'_未截断预测']=raw[:,j]
        frames.append(pp)
        for sid,ix in groups.items():
            meta=idx.iloc[ix[0]];oldarm='history_only' if arm=='history_only' else 'conditional_future_easy'
            for j,c in enumerate(CHEM):
                s=score(y[ix,j],pred[ix,j]);oldr=oldm.loc[(sid,oldarm,c)];cr=cm.loc[(sid,arm,c)]
                rows.append({'模型':'Timestral-209M','站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面'],'输入条件':arm,'目标变量':c,'测试配对数':len(ix),**s,'未截断MAE':score(y[ix,j],raw[ix,j])['MAE'],'原始负预测次数':int((raw[ix,j]<0).sum()),'对应XLinearR2':oldr['R2'],'对应XLinearMAE':oldr['MAE'],'较XLinear_R2提高量':s['R2']-oldr['R2'],'较XLinear_MAE降低量':oldr['MAE']-s['MAE'],'对应Chronos2R2':cr['R2'],'对应Chronos2MAE':cr['MAE'],'较Chronos2_R2提高量':s['R2']-cr['R2'],'较Chronos2_MAE降低量':cr['MAE']-s['MAE']})
    m=pd.DataFrame(rows);assert len(m)==6000;save(m,'01_逐站点两条件四指标误差.csv');save(pd.concat(frames,ignore_index=True),'02_逐配对实测与预测.csv')
    summary=[]
    for (arm,c),g in m.groupby(['输入条件','目标变量'],sort=False):
        summary.append({'输入条件':arm,'目标变量':c,'站点数':len(g),'R2可计算站点数':int(g.R2.notna().sum()),'R2中位数':g.R2.median(),'MAE中位数':g.MAE.median(),'RMSE中位数':g.RMSE.median(),'R2大于05站点数':int((g.R2>.5).sum()),'R2大于08站点数':int((g.R2>.8).sum()),'R2优于对应Chronos2站点数':int((g['较Chronos2_R2提高量']>0).sum()),'对应XLinearR2中位数':g['对应XLinearR2'].median(),'对应Chronos2R2中位数':g['对应Chronos2R2'].median()})
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
    save(f.groupby(['流域','目标变量']).agg(站点数=('站点ID','size'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),R2大于05站数=('R2',lambda x:int((x>.5).sum())),R2大于08站数=('R2',lambda x:int((x>.8).sum()))).reset_index(),'07_分流域条件预测汇总.csv')
    # Keep a common descriptive comparison with all previously completed models and persistence.
    x=oldm.reset_index();x=x[x['分支']=='conditional_future_easy'].copy();x['模型']='XLinear';x=x.rename(columns={'精确24小时测试配对数':'测试配对数'})
    cc=cm.reset_index();cc=cc[cc['输入条件']=='future_easy'].copy();cc['模型']='Chronos-2'
    mm=pd.read_csv(REF/'01_逐站点三模型两条件四指标误差.csv');mm=mm[mm['输入条件']=='future_easy'].copy()
    keep=['模型','站点ID','目标变量','测试配对数','R2','MAE','RMSE']
    base=x[keep].copy();base['模型']='起报化学值持久性'
    for metric in ['R2','MAE','RMSE']:base[metric]=x['起报时化学值持久性_'+metric].to_numpy()
    tf=pd.read_csv(ROOT/'TimesFM3水质24小时评估_20261104/01_逐站点两条件四指标误差.csv');tf=tf[tf['输入条件']=='future_easy']
    combined=pd.concat([f[keep],tf[keep],cc[keep],x[keep],mm[keep],base],ignore_index=True)
    comparison=combined.groupby(['模型','目标变量'],sort=False).agg(站点数=('站点ID','size'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median'),R2大于05站点数=('R2',lambda a:int((a>.5).sum())),R2大于08站点数=('R2',lambda a:int((a>.8).sum()))).reset_index();save(comparison,'08_全部模型及持久性逐指标比较.csv')
    coverage=[]
    for name,g in combined.groupby('模型',sort=False):
        r={'模型':name,'站点数':750}
        for cut,label in [(.5,'05'),(.8,'08')]:
            nums=g[g.R2>cut].groupby('站点ID').size();r['任一指标R2大于'+label]=len(nums);r['四项均R2大于'+label]=int((nums==4).sum());r['任一指标R2大于'+label+'_且至少20对']=g[(g.R2>cut)&(g['测试配对数']>=20)]['站点ID'].nunique()
        coverage.append(r)
    coverage=pd.DataFrame(coverage);save(coverage,'09_全部模型高R2站点比较.csv')
    # Independent groupby recomputation of all six thousand metrics.
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
    audit={'passed':True,'stations':750,'exact24h_pairs':23136,'metrics_independently_recomputed':6000,'reference_input_history_audit_passed_and_input_hash_matched':True,'both_arm_first64_replay_exact_match':True,'future_input_and_cross_task_isolation_passed':True,'official_weight_sha256_verified':True,'original_data_and_XLinear_predictions_unchanged':True,'masked_missing_future_covariates_preserved_no_interpolation':True}
    (HERE/'独立核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    def table(d):return '| '+' | '.join(d.columns)+' |\n|'+ '|'.join(['---']*len(d.columns))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in d.to_numpy())
    report='# Timestral-209M 水质24小时条件预测\n\n'
    report+='官方mistralai/Timestral-209M权重与官方源码固定提交，冻结权重不微调；下载SHA256与官方LFS记录一致。模型代码使用项目内源码及现有PyTorch环境，无需修改已有包。\n\n'
    report+='全部750站、23,136组精确24小时配对，与XLinear、MOIRAI、Chronos-2、TimesFM 3.0相同。复用已独立核验输入：截至起报时刻最近6次9项观测，在这6次历史首尾间按4小时网格前向填充；首条历史之前为NaN，32格上下文，预测6格，评估第6格。\n\n'
    report+='使用原生TimestralForecaster.forecast：context为4项化学历史，known_covariates为5项易测历史及未来6格。两组均保持相同known_covariates角色；仅历史组未来6格全为NaN，条件组仅最后一格为目标时刻5项易测实测，中间5格仍为NaN。未传入任何未来化学实测，原生接口保留缺失掩码，不插值未来协变量。\n\n'
    report+='FP32、64窗口一批，每个起报窗口独立分组。使用官方默认对称平均symmetric_averaging=True及quantile_real_cap_k=10000，输出21分位数的0.5分位数；关闭接口非负裁剪以保留原始值，在统一评分前将负浓度截断0，并同时保存未截断预测及误差。此前TimesFM 3.0使用symmetric_averaging=False，Chronos-2无该步骤；此次为各模型已记录固定推理配置的比较，不能仅归因于架构，未根据测试结果调参。\n\n'
    report+='这是未来易测实测已知、化学历史随每次起报更新的回溯条件预测；不证明能够减少化学检测频率或成本。测试只有10天，R²>0.5和>0.8只作描述性分档；缺少业务误差标准，不能称成功率。未进行新统计显著性声明。XLinear有本站监督训练，预训练模型本轮均冻结；历史网格编码方式也有差异。\n\n'
    view=summary.copy();numeric=['R2中位数','MAE中位数','RMSE中位数','对应XLinearR2中位数','对应Chronos2R2中位数'];view[numeric]=view[numeric].round(5)
    report+=table(view)+'\n'+table(unique)+'\n## 加入未来信息的变化\n\n'
    gain=pd.DataFrame([{'指标':c,'R2提高站点数':int((g['加入未来易测值_R2提高量']>0).sum()),'全部站点R2差值中位数':round(g['加入未来易测值_R2提高量'].median(),5),'MAE降低站点数':int((g['加入未来易测值_MAE降低量']>0).sum())} for c,g in f.groupby('目标变量',sort=False)])
    report+=table(gain)+'\n## 所有已完成模型及持久性基线（均为未来易测条件）\n\n'+table(coverage)+'\n'
    cv=comparison.copy();cv[['R2中位数','MAE中位数','RMSE中位数']]=cv[['R2中位数','MAE中位数','RMSE中位数']].round(5);report+=table(cv)+'\n'
    report+='独立核验通过：6000组站点指标复算、首64预测重放、输入隔离、来源及旧预测哈希、官方权重SHA256。\n\n参考：[Timestral官方仓库](https://github.com/mistralai/timestral)、[官方权重和许可](https://huggingface.co/mistralai/Timestral-209M)。\n'
    (HERE/'评估报告.md').write_text(report,encoding='utf-8');print(summary.to_string(index=False));print(unique.to_string(index=False));print(gain.to_string(index=False));print(coverage.to_string(index=False));print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
