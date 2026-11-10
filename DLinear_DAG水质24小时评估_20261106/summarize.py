"""Compare complete supervised runs and prior fixed models without test-based selection."""
from pathlib import Path
import json,sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OLD=ROOT/'XLinear水质24小时条件预测_20261102/results'
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)'];ARMS=['conditional_future_easy','history_only']
def save(d,name):d.to_csv(HERE/name,index=False,encoding='utf-8-sig',float_format='%.12g')
def table(d):return '| '+' | '.join(d.columns)+' |\n|'+ '|'.join(['---']*len(d.columns))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in d.to_numpy())
def main():
    original=pd.read_csv(OLD/'01_逐站点两分支四目标误差.csv');x=original.copy();x['模型']='XLinear';x=x.rename(columns={'精确24小时测试配对数':'测试配对数'})
    models=[];reference=pd.read_csv(OLD/'02_精确24小时测试实测与预测.csv');reference=reference[reference['分支']=='conditional_future_easy'].sort_values(['站点ID','目标时间']).reset_index(drop=True)
    for model in ['DLinear','DAG']:
        out=HERE/('results_'+model);assert json.loads((out/'独立验证.json').read_text(encoding='utf-8'))['passed']
        m=pd.read_csv(out/'01_逐站点两分支四目标误差.csv');assert len(m)==6000 and m['站点ID'].nunique()==750
        p=pd.read_csv(out/'02_精确24小时测试实测与预测.csv')
        for arm in ARMS:
            pp=p[p['分支']==arm].sort_values(['站点ID','目标时间']).reset_index(drop=True);assert len(pp)==23136 and pp['站点ID'].equals(reference['站点ID'])
            for col in ['起报时间','目标时间','输入历史起点']:assert pd.to_datetime(pp[col],format='mixed').equals(pd.to_datetime(reference[col],format='mixed'))
            for c in CHEM:assert np.allclose(pp[c+'_实测'],reference[c+'_实测'],rtol=0,atol=1e-9)
        m['模型']=model+'（任务适配）';m=m.rename(columns={'精确24小时测试配对数':'测试配对数'});models.append(m)
    new=pd.concat(models,ignore_index=True);assert len(new)==12000
    refidx=x.set_index(['站点ID','分支','目标变量'])
    new['对应XLinearR2']=[refidx.loc[(r['站点ID'],r['分支'],r['目标变量']),'R2'] for _,r in new.iterrows()]
    new['对应XLinearMAE']=[refidx.loc[(r['站点ID'],r['分支'],r['目标变量']),'MAE'] for _,r in new.iterrows()]
    new['较XLinear_R2提高量']=new.R2-new['对应XLinearR2'];new['较XLinear_MAE降低量']=new['对应XLinearMAE']-new.MAE
    save(new,'01_逐站点两模型两条件四指标误差.csv')
    summary=new.groupby(['模型','分支','目标变量'],sort=False).agg(站点数=('站点ID','size'),R2可计算站点数=('R2','count'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median'),R2大于05站点数=('R2',lambda a:int((a>.5).sum())),R2大于08站点数=('R2',lambda a:int((a>.8).sum())),R2较XLinear提高站数=('较XLinear_R2提高量',lambda a:int((a>0).sum())),MAE较XLinear降低站数=('较XLinear_MAE降低量',lambda a:int((a>0).sum()))).reset_index();save(summary,'02_总体误差及R2汇总.csv')
    h=new[new['分支']=='history_only'].set_index(['模型','站点ID','目标变量']);f=new[new['分支']=='conditional_future_easy'].copy()
    f['仅历史R2']=[h.loc[(r['模型'],r['站点ID'],r['目标变量']),'R2'] for _,r in f.iterrows()];f['仅历史MAE']=[h.loc[(r['模型'],r['站点ID'],r['目标变量']),'MAE'] for _,r in f.iterrows()]
    f['加入未来易测值_R2提高量']=f.R2-f['仅历史R2'];f['加入未来易测值_MAE降低量']=f['仅历史MAE']-f.MAE;save(f,'03_加入未来易测值的逐站点收益.csv')
    def coverage(d,groupcols):
        output=[]
        for key,g in d.groupby(groupcols,sort=False):
            key=key if isinstance(key,tuple) else (key,);r=dict(zip(groupcols,key));r['站点数']=750
            for cut,label in [(.5,'05'),(.8,'08')]:
                cc=g[g.R2>cut].groupby('站点ID').size();r['任一指标R2大于'+label]=len(cc);r['四项均R2大于'+label]=int((cc==4).sum());r['任一指标R2大于'+label+'_且至少20对']=g[(g.R2>cut)&(g['测试配对数']>=20)]['站点ID'].nunique()
            output.append(r)
        return pd.DataFrame(output)
    unique=coverage(new,['模型','分支']);save(unique,'04_高R2站点去重统计.csv')
    save(f[f.R2>.5].sort_values(['模型','R2'],ascending=[True,False]),'05_R2大于05的条件预测站点指标.csv')
    basin=f.groupby(['模型','流域','目标变量']).agg(站点数=('站点ID','size'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),R2大于05站数=('R2',lambda a:int((a>.5).sum())),R2大于08站数=('R2',lambda a:int((a>.8).sum()))).reset_index();save(basin,'06_分流域条件预测比较.csv')
    keep=['模型','站点ID','目标变量','测试配对数','R2','MAE','RMSE'];comparisons=[f[keep],x[x['分支']=='conditional_future_easy'][keep]]
    for folder,label in [('Chronos2水质24小时评估_20261103','Chronos-2'),('TimesFM3水质24小时评估_20261104','TimesFM-3.0'),('Timestral209M水质24小时评估_20261104','Timestral-209M')]:
        d=pd.read_csv(ROOT/folder/'01_逐站点两条件四指标误差.csv');d=d[d['输入条件']=='future_easy'].copy();d['模型']=label;comparisons.append(d[keep])
    d=pd.read_csv(ROOT/'预训练模型水质24小时评估_20261103/01_逐站点三模型两条件四指标误差.csv');comparisons.append(d[d['输入条件']=='future_easy'][keep])
    p=x[x['分支']=='conditional_future_easy'].copy();p['模型']='起报化学值持久性'
    for col in ['R2','MAE','RMSE']:p[col]=p['起报时化学值持久性_'+col]
    comparisons.append(p[keep]);combined=pd.concat(comparisons,ignore_index=True)
    allstats=combined.groupby(['模型','目标变量'],sort=False).agg(站点数=('站点ID','size'),R2中位数=('R2','median'),MAE中位数=('MAE','median'),RMSE中位数=('RMSE','median'),R2大于05站点数=('R2',lambda a:int((a>.5).sum())),R2大于08站点数=('R2',lambda a:int((a>.8).sum()))).reset_index();save(allstats,'07_全部模型及持久性逐指标比较.csv')
    allcoverage=coverage(combined,['模型']);save(allcoverage,'08_全部模型高R2站点比较.csv')
    gains=f.groupby(['模型','目标变量'],sort=False).agg(R2提高站点数=('加入未来易测值_R2提高量',lambda a:int((a>0).sum())),R2差值中位数_全部站点=('加入未来易测值_R2提高量','median'),MAE降低站点数=('加入未来易测值_MAE降低量',lambda a:int((a>0).sum()))).reset_index();save(gains,'09_未来易测信息收益汇总.csv')
    report='# DLinear 与 DAG：水质24小时条件预测\n\n'
    report+='使用用户指定的官方DAG论文（arXiv:2509.14933，当前v3题名Dual Correlation Network）和官方DLinear源码。这里的DAG不是有向无环图发现基线，相关性建模也不代表已识别水质变量的因果关系。\n\n'
    report+='全部750站、79,940条基础观测保留。两模型均按站点从随机初始化训练，两条件各750个模型，每模型4输出。测试仍为2025-04-30至2025-05-10 00:00，共23,136组精确24小时配对。\n\n'
    report+='## 时间与训练协议\n\n历史采用原始最近6次9变量观测（截至起报，易测5项＋化学4项），与原XLinear的数值输入和时间配对一致，不插值。6次不等于6小时，采样间隔可能不同。单个输出头通过恰好+24小时的标签训练；pred_len=1代表这个指定24小时目标，绝非将下一条记录当作24小时。目标时刻只读5项易测实测作协变量，4项化学实测仅作标签。\n\n'
    report+='选择轮数：训练标签严格早于4月24日，验证标签4月25日至4月29日之前。验证仅按四化学指标标准化MSE选择轮数。随后重新初始化，使用标签严格早于4月29日的数据重拟合，所有训练/验证标签均早于最早测试起报。每阶段Scaler只拟合相应截止前原始本站数据。Adam lr=0.001，最多100轮，patience=10，全批训练，seed=2025+站点索引，CPU FP32。DAG按站点分4个互不重叠分片并行运行，未混合权重、选择优者或平均预测。\n\n'
    report+='## 必要的模型适配\n\nDLinear保留原生趋势/季节分解及共享时间线性层，官方kernel=25；增加9→4历史通道融合，使5项易测历史也能参与预测，并增加与XLinear相同的5→16→4未来易测修正层，输出层零初始化。无未来组保留同一结构但不读取未来输入。共218参数。此为DLinear任务适配版，不称论文原生模型直接支持未来协变量。\n\n'
    report+='DAG保留官方时间/通道相关模块及原生辅助损失，重排输入为化学4项在前、易测5项在后。固定短历史配置d_model=16、d_ff=32、4头、1层、patch=2、stride=2、alpha=0.5、beta=0.1、dropout=0，共11,280参数；不是原论文长序列配置复现。条件组直接输入目标时刻5项易测值。无未来组在训练及推理都使用时间模块预测出的未来易测值，不读取真实未来易测值，保留历史重构辅助损失。\n\n'
    report+='DAG训练损失为四目标标准化MSE加beta加权原生辅助损失，验证选择只用目标MSE。原始history.csv继承的“训练标准化MSE”列在DAG中实际记录该训练总损失，不能当作纯预测MSE。\n\n'
    report+='官方DAG在多目标展平后将每窗口门控权重repeat(series_dim,...)，顺序与样本优先的展平顺序不同，会将其他预测窗口的门控权重用于当前窗口。vendored副本仅将该处改为repeat_interleave(series_dim,dim=0)，并通过批量与逐样本预测一致性检查。其余支持性改动为重写导入路径、将未使用的Reformer依赖改为可选导入；不替换DAG的FullAttention。原始仓库未修改，逐文件原始及改动后哈希列于source_provenance.json。\n\n'
    report+='## 两模型结果\n\n'
    view=summary.copy();view[['R2中位数','MAE中位数','RMSE中位数']]=view[['R2中位数','MAE中位数','RMSE中位数']].round(5);report+=table(view)+'\n'+table(unique)+'\n'
    report+='## 加入未来易测值的变化\n\n'+table(gains.round({'R2差值中位数_全部站点':5}))+'\n## 与全部既有模型比较（未来易测值条件）\n\n'+table(allcoverage)+'\n'
    report+='原XLinear同为本站监督训练，DLinear/DAG可以更直接对照训练协议。此前预训练模型为冻结权重，且其数值输入经过4小时网格历史填充；比较同一测试样本和原始信息，不是完全相同输入编码的架构实验。\n\n'
    report+='测试仅10天，部分站点配对不足20组；R²阈值仅为描述分档，不是推导成功率。未来易测实测已知、化学历史随每次起报更新，不证明可减少化学检测次数，也未模拟报告延迟。当前数据已多轮探索，未据此重新调参；仍需要更长独立时段验证。不作新的统计显著性声明。\n\n'
    report+='独立核验包括全部12,000组模型指标及两项基线复算、6,000组Scaler核对、精确时间配对/原始实测一致性、全部模型训练与验证时间边界、20次CPU检查点重放、批量隔离及无未来组输入隔离。\n\n参考：[DAG论文](https://arxiv.org/abs/2509.14933)、[DAG官方代码](https://github.com/decisionintelligence/DAG)、[DLinear官方代码](https://github.com/cure-lab/LTSF-Linear)。\n'
    (HERE/'评估报告.md').write_text(report,encoding='utf-8')
    (HERE/'合并分析核验.json').write_text(json.dumps({'passed':True,'station_count_per_model':750,'metrics':12000,'exact24h_pairs_per_arm':23136,'same_keys_and_truth_as_XLinear':True,'independent_model_verifiers':['results_DLinear/独立验证.json','results_DAG/独立验证.json'],'no_test_based_hyperparameter_adjustment':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(unique.to_string(index=False));print(gains.to_string(index=False));print(allcoverage.to_string(index=False))
if __name__=='__main__':main()
