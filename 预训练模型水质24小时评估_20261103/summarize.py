"""Station metrics on identical observed +24h pairs, never select a model on test."""
from pathlib import Path
import sys,json,hashlib
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'XLinear水质24小时条件预测_20261102/results'
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
def save(d,name):d.to_csv(HERE/name,index=False,encoding='utf-8-sig',float_format='%.12g')
def score(y,p):
    e=p-y;sst=np.sum((y-y.mean())**2);tol=1e-12*max(1.,float(np.sum(y*y)))
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'偏差':float(e.mean()),'R2':float(1-np.sum(e*e)/sst) if len(y)>=2 and sst>tol else np.nan}
def main():
    idx=pd.read_csv(HERE/'测试配对索引.csv');p=pd.read_csv(OLD/'02_精确24小时测试实测与预测.csv');p=p[p['分支']=='conditional_future_easy'].reset_index(drop=True)
    assert len(p)==len(idx)==23136
    assert p['站点ID'].equals(idx['站点ID'])
    for timecol in ['起报时间','目标时间','输入历史起点']:
        assert pd.to_datetime(p[timecol],format='mixed').equals(pd.to_datetime(idx[timecol],format='mixed'))
    oldm=pd.read_csv(OLD/'01_逐站点两分支四目标误差.csv').set_index(['站点ID','分支','目标变量'])
    y=p[[c+'_实测' for c in CHEM]].to_numpy(float);persist=p[[c+'_起报时化学值持久性' for c in CHEM]].to_numpy(float)
    rows=[];allpred=[];donefiles=[]
    for model in ['MOIRAI-small','MOIRAI-base','MOIRAI-large']:
        for arm in ['history_only','future_easy']:
            dest=HERE/'results'/(model+'__'+arm);done=json.loads((dest/'done.json').read_text(encoding='utf-8'));assert done['completed'] and done['N']==23136
            data=np.load(dest/'predictions.npz');pred=data['prediction'];raw=data['raw'];assert pred.shape==(23136,4) and np.isfinite(pred).all()
            donefiles.append(done)
            pp=idx.copy();pp.insert(0,'模型',model);pp.insert(1,'输入条件',arm)
            for j,c in enumerate(CHEM):pp[c+'_实测']=y[:,j];pp[c+'_预测']=pred[:,j];pp[c+'_未截断预测']=raw[:,j]
            allpred.append(pp)
            for sid,indices in idx.groupby('站点ID',sort=False).groups.items():
                ix=np.asarray(indices);meta=idx.iloc[ix[0]];oldarm='conditional_future_easy' if arm=='future_easy' else 'history_only'
                for j,c in enumerate(CHEM):
                    met=score(y[ix,j],pred[ix,j]);ps=score(y[ix,j],persist[ix,j]);om=oldm.loc[(sid,oldarm,c)]
                    rows.append({'模型':model,'输入条件':arm,'站点ID':sid,'省份':meta['省份'],'流域':meta['流域'],'断面':meta['断面'],'目标变量':c,'测试配对数':len(ix),'测试配对至少20':int(len(ix)>=20),**met,'未截断MAE':score(y[ix,j],raw[ix,j])['MAE'],'持久性R2':ps['R2'],'持久性MAE':ps['MAE'],'对应XLinearR2':float(om['R2']),'对应XLinearMAE':float(om['MAE']),'R2较对应XLinear提高量':met['R2']-float(om['R2']),'MAE较对应XLinear降低量':float(om['MAE'])-met['MAE']})
    m=pd.DataFrame(rows);assert len(m)==18000
    save(m,'01_逐站点三模型两条件四指标误差.csv')
    def aggregate(g,basin):
        model,arm,c=g.iloc[0][['模型','输入条件','目标变量']]
        return {'模型':model,'输入条件':arm,'流域':basin,'目标变量':c,'站点数':len(g),'R2可计算站点数':int(g.R2.notna().sum()),'R2中位数':g.R2.median(),'MAE中位数':g.MAE.median(),'RMSE中位数':g.RMSE.median(),'R2大于0站点数':int((g.R2>0).sum()),'R2大于05站点数':int((g.R2>.5).sum()),'R2大于08站点数':int((g.R2>.8).sum()),'R2较对应XLinear提高站点数':int((g['R2较对应XLinear提高量']>0).sum()),'MAE较对应XLinear降低站点数':int((g['MAE较对应XLinear降低量']>0).sum()),'MAE优于持久性站点数':int((g.MAE<g['持久性MAE']).sum())}
    summary=pd.DataFrame([dict(aggregate(g,'全部'),对应XLinearR2中位数=g['对应XLinearR2'].median(),对应XLinearMAE中位数=g['对应XLinearMAE'].median()) for _,g in m.groupby(['模型','输入条件','目标变量'],sort=False)]);save(summary,'02_总体R2与误差对比.csv')
    basin=pd.DataFrame([aggregate(g,key[-1]) for key,g in m.groupby(['模型','输入条件','目标变量','流域'],sort=False)]);save(basin,'03_分流域R2与误差对比.csv')
    h=m[m['输入条件']=='history_only'].set_index(['模型','站点ID','目标变量']);f=m[m['输入条件']=='future_easy'].copy()
    f['仅历史R2']=[h.loc[(r['模型'],r['站点ID'],r['目标变量']),'R2'] for _,r in f.iterrows()]
    f['仅历史MAE']=[h.loc[(r['模型'],r['站点ID'],r['目标变量']),'MAE'] for _,r in f.iterrows()]
    f['加入未来易测值_R2提高量']=f.R2-f['仅历史R2'];f['加入未来易测值_MAE降低量']=f['仅历史MAE']-f.MAE
    save(f,'04_加入未来易测值的逐站点收益.csv');save(m[m.R2>.5].sort_values(['模型','输入条件','R2'],ascending=[True,True,False]),'05_R2大于05的站点指标明细.csv')
    unique=[]
    for (model,arm),g in m.groupby(['模型','输入条件'],sort=False):
        r={'模型':model,'输入条件':arm,'站点数':750}
        for cutoff,name in [(.5,'05'),(.8,'08')]:
            counts=g[g.R2>cutoff].groupby('站点ID').size();r['至少一项R2大于'+name]=len(counts);r['四项全部R2大于'+name]=int((counts==4).sum());r['至少一项R2大于'+name+'_且配对至少20']=g[(g.R2>cutoff)&(g['测试配对数']>=20)]['站点ID'].nunique()
        unique.append(r)
    unique=pd.DataFrame(unique);save(unique,'06_高R2站点去重统计.csv')
    save(pd.concat(allpred,ignore_index=True),'07_逐配对实测与预测.csv')
    def mdtable(df):
        return '| '+' | '.join(df.columns)+' |\n|'+ '|'.join(['---']*len(df.columns))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in df.to_numpy())
    report='# MOIRAI本地预训练权重：水质24小时条件预测\n\n'
    report+='全部750站、原30天基础数据不变；测试目标为2025-04-30至2025-05-10 00:00（含两端），与XLinear完全相同的23,136组起报—目标配对。每站每次可获得起报时刻之前（含当次）的化学实测。\n\n'
    report+='仅评估用户指定列表中可原生接收未来协变量的MOIRAI-small、base、large。Chronos-tiny为原版T5模型，不是Chronos-2；Chronos-tiny、Sundial和Timer暂不评估。参考官方Chronos流程区分模型输入能力，MOIRAI实际调用官方Uni2TS MoiraiForecast。\n\n'
    report+='冻结本地预训练权重，不微调、不训练适配头，不按测试结果挑参数。固定patch_size=8，20条生成轨迹取样本中位数（偶数样本取中间两项平均），所有负浓度预测截断至0，同时保留未截断值及误差。\n\n'
    report+='## 输入、时间对齐与边界\n\n'
    report+='历史仍仅含起报时刻最近6次9项观测。原数据不等间隔，按真实时间置于4小时网格，仅在这6次观测首尾之间用之前最近值前向填充；此前位置置0并给出缺失和padding掩码。上下文32格，未来6格恰为24小时，取第6格预测评估，不将下一条记录当作24小时。输入历史最早时刻、起报时刻及目标索引均与XLinear对齐。\n\n'
    report+='两条件均输入4项化学历史及5项易测历史；future_easy另外只在目标时刻给5项易测真实值，中间5格的未来易测值保持缺失掩码。history_only对整个未来6格易测值均掩码，不伪造未来易测值。未来化学值不进入模型。输出4项化学指标，均按原单位计分。\n\n'
    report+='MOIRAI的历史前向填充、原生尺度变换与XLinear的6次离散历史编码不同，所以“同测试配对、同原始历史信息”不等于完全相同的数值输入或仅隔离架构的实验。MOIRAI无本站监督训练，XLinear有本站训练；两者是冻结预训练与本站监督方案的对照。\n\n'
    report+='本轮是未来易测实测已知的回溯条件实验，仍允许每次起报获得化学历史，不代表实际提前24小时能获取未来实测，不证明检测成本已下降。测试只有10天，未设业务误差阈值；R²>0.5、>0.8仅为描述性分档，不能称合格率。不可计算的R²留空，比例分母仍为750站。\n\n'
    report+='## 结果判断\n\n三种MOIRAI在加入未来易测值后，四项指标的站点R²中位数均低于对应XLinear。三种MOIRAI中base的四项R²中位数均更好，large未呈现规模增大的稳定收益。small有7个站点至少一项R²>0.8，其中4站测试配对至少20组；少数高表现站点不代表整体改进。以上结论仅适用于本轮冻结权重、短历史、固定配置和既定测试窗口。\n\n'
    report+='## 加入未来易测值后的总体表现\n\n'
    table=summary[summary['输入条件']=='future_easy'][['模型','目标变量','R2中位数','MAE中位数','R2大于05站点数','R2大于08站点数']].copy();table[['R2中位数','MAE中位数']]=table[['R2中位数','MAE中位数']].round(5)
    report+=mdtable(table)+'\n## 高R²站点数（同站多目标去重）\n\n'+mdtable(unique)+'\n'
    report+='## 加入未来信息的变化\n\n'
    gains=[]
    for (model,c),g in f.groupby(['模型','目标变量'],sort=False):
        gains.append({'模型':model,'指标':c,'R2提高站数':int((g['加入未来易测值_R2提高量']>0).sum()),'R2提高量中位数_全部站':round(g['加入未来易测值_R2提高量'].median(),5),'MAE降低站数':int((g['加入未来易测值_MAE降低量']>0).sum())})
    report+=mdtable(pd.DataFrame(gains))+'\n'
    report+='目前只进行描述性比较，未对新的MOIRAI结果复用此前XLinear探索性显著性标记。\n\n官方参考：[Chronos](https://github.com/amazon-science/chronos-forecasting)、[MOIRAI/Uni2TS](https://github.com/SalesforceAIResearch/uni2ts)。\n'
    (HERE/'评估报告.md').write_text(report,encoding='utf-8')
    print(summary.to_string(index=False));print(unique.to_string(index=False))
if __name__=='__main__':main()
