import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
MODES=['不使用邻站','邻站当前值','邻站学习时滞']
def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    z=np.load(HERE/'predictions.npz')
    index=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv')
    test=index[index.test].reset_index(drop=True)
    truth=np.load(ROOT/'冻结预训练模型校正头基线_20261106/inputs.npz')['truth'][index.test]
    np.testing.assert_array_equal(z['truth'],truth)
    rows=[]
    for mode in MODES:
        pred=np.maximum(z[mode],0)
        assert pred.shape==(23136,4) and np.isfinite(pred).all()
        for sid,ii in test.groupby('站点ID').indices.items():
            y=truth[ii];error=pred[ii]-y;sst=np.square(y-y.mean(0)).sum(0)
            r2=1-np.divide(np.square(error).sum(0),sst,out=np.full(4,np.nan),where=sst>1e-12*np.maximum(1,np.square(y).sum(0)))
            for j,c in enumerate(CHEM):rows.append([mode,sid,c,len(ii),np.abs(error[:,j]).mean(),np.sqrt(np.square(error[:,j]).mean()),r2[j]])
    m=pd.DataFrame(rows,columns=['模型','站点ID','指标','测试配对数','MAE','RMSE','R2'])
    m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
    per=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE均值=('MAE','mean'),MAE中位数=('MAE','median'),RMSE均值=('RMSE','mean')).reset_index()
    per.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
    totals=[]
    for mode in MODES:
        g=m[m['模型'].eq(mode)]
        totals.append([mode,per.loc[per['模型'].eq(mode),'平均R2'].mean(),g.loc[g.R2>.5,'站点ID'].nunique(),g.loc[g.R2>.8,'站点ID'].nunique()])
    totals=pd.DataFrame(totals,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数'])
    totals.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
    g=m[m['模型'].eq('邻站学习时滞')].merge(m[m['模型'].eq('邻站当前值')],on=['站点ID','指标'],suffixes=('_时滞','_当前'),validate='one_to_one')
    g['R2变化']=g.R2_时滞-g.R2_当前;g['MAE降低量']=g.MAE_当前-g.MAE_时滞
    g.to_csv(HERE/'时滞相对当前逐站收益.csv',index=False,encoding='utf-8-sig')
    gains=g.groupby('指标',sort=False).agg(平均R2变化=('R2变化','mean'),R2变化中位数=('R2变化','median'),R2提高站数=('R2变化',lambda x:int((x>1e-8).sum())),R2下降站数=('R2变化',lambda x:int((x < -1e-8).sum())),MAE平均降低量=('MAE降低量','mean')).reset_index()
    gains.to_csv(HERE/'分指标时滞收益.csv',index=False,encoding='utf-8-sig')
    matrix=g.pivot(index='站点ID',columns='指标',values='R2变化').to_numpy()
    rng=np.random.default_rng(20261110)
    boot=[np.nanmean(matrix[rng.integers(0,len(matrix),len(matrix))],axis=0).mean() for _ in range(2000)]
    interval=np.quantile(boot,[.025,.975]).tolist()
    edges=pd.read_csv(HERE/'final_时滞选择.csv')
    lag_summary=edges.groupby(['指标','选择滞后_小时']).size().rename('关系数').reset_index()
    lag_summary.to_csv(HERE/'分指标滞后分布.csv',index=False,encoding='utf-8-sig')
    validation=pd.read_csv(HERE/'内部时间验证_关系稳定性.csv')
    v=validation[validation['选择滞后_小时'].gt(0)]['验证r变化'].dropna()
    stability=dict(selected_nonzero_edges=int(validation['选择滞后_小时'].gt(0).sum()),valid_holdout_edges=len(v),improved=int(v.gt(1e-12).sum()),worsened=int(v.lt(-1e-12).sum()),mean_delta_r=v.mean(),median_delta_r=v.median())
    records=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'))
    assert len(set(records[x]['initial_trainable_parameter_sha256'] for x in MODES))==1
    supports=np.load(HERE/'final_时滞图.npz')
    for mode in MODES:
        assert records[mode]['steps']==24 and records[mode]['trainable']==339612
        cp=torch.load(HERE/f'{mode}.pt',map_location='cpu',weights_only=True)['graph']
        expected=supports['lagged'] if mode=='邻站学习时滞' else supports['zero'].copy()
        if mode=='不使用邻站':expected[:]=0;expected[0]=np.eye(750)[None,:,:]
        np.testing.assert_array_equal(cp['temporal_support'].numpy(),expected.astype(np.float32))
    independent=json.loads((HERE/'时滞独立核验.json').read_text(encoding='utf-8'));assert independent['passed']
    audit=dict(passed=True,stations=750,test_pairs=len(test),same_initial_trainable_weights=True,same_declared_trainable_parameters=339612,steps=24,nonzero_lag_edges=int(edges['选择滞后_小时'].gt(0).sum()),receivers_with_nonzero_lag=int(edges.loc[edges['选择滞后_小时'].gt(0),'接收站'].nunique()),lag_minus_current_mean_r2=float(np.nanmean(matrix,axis=0).mean()),station_bootstrap_descriptive_interval95=interval,internal_validation=stability,independent_checks=independent)
    (HERE/'实验核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# 学习邻站时滞的24小时条件预测实验

任务仍为使用本站历史9项、未来24小时本站5项易测实测值，预测24小时后的4项化学值。模型保留Chronos-2末两层LoRA和原空间残差校正框架。为了明确比较邻站当前化学值与过去化学值，空间分支本轮改为读取逐站训练期标准化的化学实测值，通过新增1→64线性投影进入图消息；本站隐藏表示、解码、训练损失及评价口径保持原框架。

## 时滞怎样学习

候选邻站沿用训练期同流域Pearson图，每指标每站最多3条边。对每条边、每项化学指标，比较接收站y(t+24小时)与邻站x(t−滞后)的Pearson相关性。滞后候选为0、4、8、12、24小时；只使用真实、精确时间配对，不按相邻记录替代时间，不插值。要求至少16对，以r*n/(n+16)惩罚小样本；非零滞后得分须比零滞后与0的较大者高0.02，否则保留零滞后。该门槛事先固定，没有用测试结果调整。

这里的“4小时滞后”是读取起报前4小时的邻站值，与预测目标相隔28小时，并非认定水流传播时间为4小时。原候选图来自统计相关性，没有确认物理上下游；报告应称同流域候选邻站。

selection图使用目标时间早于4月24日的数据选择滞后，在4月25日—29日前做内部时间验证；只在零滞后与所选滞后都有数据的共同配对上比较验证相关性，至少6对。这个步骤描述关系稳定性，没有用测试数据调参。最终图以目标时间早于4月29日的数据重新选择，用于训练与测试。最终最晚筛选标签为4月28日20时，最晚可能的邻站源值为4月27日20时。

最终6,484条站点—指标关系中1,384条选择非零滞后，占21.35%；5,100条保持0小时。非零滞后涉及的站点数见实验核验JSON。该比例是候选关系的选择比例，不是预测成功率。

'''+table(lag_summary)+'\n\n## 同结构模型对照\n\n'
    report+='''三组分别为不使用邻站、读取邻站当前值、读取训练期选择的时滞值。零滞后和时滞两组的邻站及边权重完全相同，仅将边分配到不同滞后通道，避免把重新选邻站或调整边权的收益算成时滞收益。无邻站组只有自身连接。

一跳消息为sum_l A_l X(t−l)，二跳为sum_l sum_k A_l A_k X(t−l−k)，最大回看48小时；沿用作者Graph WaveNet的1×1卷积和dropout融合。保留原448维卷积输入；三组均339,612个声明可训练参数，包含新增128个标量投影参数。原节点嵌入和部分原支持缓冲保留以便框架兼容，但本轮不使用自适应跨站传播，节点嵌入不影响输出。这三个新结果应互相比，不能把与旧隐藏表示图实验的差值全归因于时滞。

所有源化学值均为起报时间或以前的真实观测，缺失值在投影后掩码置0，不前向填充、不读取邻站未来化学值。两组读取的时点不同，因此源数据可用性也可能不同；差异反映实际读取策略的净效果，不能完全分离时滞相关性与可用性影响。模型仍使用原允许的未来本站易测协变量。

三组相同初始可训练权重SHA256、相同种子和起报抽样、24次更新，LoRA学习率1e-5，校正模块学习率0.001，统一非负截断。全750站与23,136个测试配对不变。

'''+table(totals)+'\n\n'+table(gains)+'\n\n## 分指标误差\n\n'+table(per)
    report+='''

## 本轮观察

学习时滞相对邻站当前值，平均R²从−0.402498变为−0.400154，增加0.002344；任一项R²超过0.5的站点155→156，超过0.8的站点均为12。相比新增邻站本身的平均收益，时滞带来的额外收益很小。总氮平均增益最大，为0.004427；总磷平均R²略升，但其中位变化为负、下降站数多于提高站数，平均MAE也略恶化。

这支持当前预算下有少量统计时滞预测价值，不能理解为明显或普遍提升。本轮原化学值时滞分支尚未优于此前隐藏表示相关＋自适应图实验的总体平均R²−0.349938；由于输入分支不同，不能用两轮差值单独评判时滞。更长训练、多种子和新的时间测试区间仍有待评估。

## 内部验证和核验

'''
    report+=f"selection阶段选出的{stability['selected_nonzero_edges']}条非零滞后关系中，只有{len(v)}条在内部验证期有至少6个共同有效配对；相关性提高{stability['improved']}条、下降{stability['worsened']}条。平均相关系数变化{stability['mean_delta_r']:.6f}，中位数{stability['median_delta_r']:.6f}，关系稳定性有限。\n\n"
    report+=f"最终时滞模型相对当前值模型的平均R²变化{audit['lag_minus_current_mean_r2']:.6f}。按站点成组重抽样2,000次的描述性95%区间为[{interval[0]:.6f}, {interval[1]:.6f}]，未考虑流域空间依赖及训练种子差异。\n\n"
    report+='''已独立复算100条训练关系的精确时间配对和相关系数；将4月29日及以后全部化学数据替换成随机值后重建图，选出的时滞与图矩阵逐值不变。验证了向量化一跳、二跳传播与Graph WaveNet作者nconv方向和运算相符，并用玩具路径确认8小时＋4小时应读取12小时前的值。已核对训练后保存支持矩阵、初始权重和参数量。

R²为逐站逐指标计算，近常数指标不定义，有限负值保留；总体为四项逐站均值再等权平均。阈值站数按任一项达标去重。当前是单种子、固定24步探索，测试期已被多轮实验查看，不能把相关性、阈值比例或单次收益认定为可推广成功率，也不能把统计时滞称为水文因果关系。

## 文件

- final_时滞选择.csv：逐边逐指标的滞后、训练样本数与相关系数。
- final_时滞图.npz：按时滞、指标分层的750×750矩阵及匹配零时滞矩阵。
- 内部时间验证_关系稳定性.csv：未参与selection选择的后续时段关系比较。
- 时滞相对当前逐站收益.csv：逐站指标误差及R²差值。
- 总体比较.csv、分指标比较.csv、分指标时滞收益.csv：预测结果。
- 时滞独立核验.json、实验核验.json：泄漏、时间配对和传播方向核验。
- build_lags.py、run.py、audit.py、results.py：构图、训练预测、独立检查及报告生成。

原数据及旧实验权重未覆盖。
'''
    (HERE/'实验报告.md').write_text(report,encoding='utf-8')
    print(totals.to_string(index=False));print(gains.to_string(index=False));print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
