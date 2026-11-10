"""Attribute matched-budget changes to adaptation versus cross-station inputs."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.stdout.reconfigure(encoding='utf-8')
CHEM=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    z=np.load(HERE/'predictions.npz')
    geo=np.load(ROOT/'河网三图预测_20261110/predictions.npz')
    dist=np.load(ROOT/'三图联合预测_20261110/predictions.npz')
    original=np.load(ROOT/'GraphWaveNet拓扑对照_20261109/predictions.npz')
    for other in [geo,dist,original]:np.testing.assert_array_equal(z['truth'],other['truth'])
    idx=pd.read_csv(ROOT/'冻结预训练模型校正头基线_20261106/配对索引.csv')
    idx=idx[idx.test].reset_index(drop=True)
    values={'冻结同框架':z['冻结同框架'],'本站LoRA':z['本站LoRA'],'本站LoRA加校正':z['本站LoRA加校正'],'相关加单位加自适应':geo['三图_单位对照'],'相关加距离加自适应':dist['三图联合'],'相关加河网加自适应':geo['三图_河网探索'],'原两图_仅参考':original['GWN固定加自适应']}
    metrics=[]
    for label,p in values.items():
        p=np.maximum(p,0)
        assert p.shape==z['truth'].shape==(23136,4) and np.isfinite(p).all()
        for sid,ii in idx.groupby('站点ID').indices.items():
            y=z['truth'][ii];err=p[ii]-y;sst=np.square(y-y.mean(0)).sum(0)
            r2=1-np.divide(np.square(err).sum(0),sst,out=np.full(4,np.nan),where=sst>1e-12*np.maximum(1,np.square(y).sum(0)))
            for j,c in enumerate(CHEM):metrics.append([label,sid,c,len(ii),np.abs(err[:,j]).mean(),np.sqrt(np.square(err[:,j]).mean()),r2[j]])
    m=pd.DataFrame(metrics,columns=['模型','站点ID','指标','测试配对数','MAE','RMSE','R2'])
    m.to_csv(HERE/'逐站点误差.csv',index=False,encoding='utf-8-sig')
    per=m.groupby(['模型','指标'],sort=False).agg(平均R2=('R2','mean'),R2中位数=('R2','median'),MAE均值=('MAE','mean'),MAE中位数=('MAE','median'),RMSE均值=('RMSE','mean')).reset_index()
    per.to_csv(HERE/'分指标比较.csv',index=False,encoding='utf-8-sig')
    total=[]
    for name,g in m.groupby('模型',sort=False):total.append([name,per.loc[per['模型'].eq(name),'平均R2'].mean(),g.loc[g.R2>.5,'站点ID'].nunique(),g.loc[g.R2>.8,'站点ID'].nunique()])
    total=pd.DataFrame(total,columns=['模型','四指标平均R2','任一项R2超过0.5站数','任一项R2超过0.8站数'])
    total.to_csv(HERE/'总体比较.csv',index=False,encoding='utf-8-sig')
    pairs=[('LoRA本身','本站LoRA','冻结同框架'),('本站校正模块','本站LoRA加校正','本站LoRA'),('邻站信息_相关与自适应','相关加单位加自适应','本站LoRA加校正'),('邻站信息_含距离','相关加距离加自适应','本站LoRA加校正'),('邻站信息_含河网','相关加河网加自适应','本站LoRA加校正')]
    gains=[];detailed=[];boots=[]
    for effect,new,old in pairs:
        g=m[m['模型'].eq(new)].merge(m[m['模型'].eq(old)],on=['站点ID','指标'],suffixes=('_新','_参照'),validate='one_to_one')
        g['比较']=effect;g['R2变化']=g.R2_新-g.R2_参照;g['MAE降低量']=g.MAE_参照-g.MAE_新
        detailed.append(g)
        for indicator,v in g.groupby('指标',sort=False):
            gains.append([effect,indicator,v['R2变化'].mean(),v['R2变化'].median(),int(v['R2变化'].gt(1e-8).sum()),int(v['R2变化'].lt(-1e-8).sum()),v['MAE降低量'].mean(),int(v['MAE降低量'].gt(1e-10).sum())])
        matrix=g.pivot(index='站点ID',columns='指标',values='R2变化').to_numpy()
        rng=np.random.default_rng(20261110)
        estimates=[np.nanmean(matrix[rng.integers(0,len(matrix),len(matrix))],axis=0).mean() for _ in range(2000)]
        ci=np.quantile(estimates,[.025,.975])
        boots.append([effect,np.nanmean(matrix,axis=0).mean(),ci[0],ci[1]])
    gains=pd.DataFrame(gains,columns=['比较','指标','平均R2变化','R2变化中位数','R2提高站数','R2下降站数','MAE平均降低量','MAE降低站数'])
    gains.to_csv(HERE/'分项贡献比较.csv',index=False,encoding='utf-8-sig')
    pd.concat(detailed,ignore_index=True).to_csv(HERE/'逐站点配对收益.csv',index=False,encoding='utf-8-sig')
    boots=pd.DataFrame(boots,columns=['比较','平均R2变化','描述性区间下限','描述性区间上限'])
    boots.to_csv(HERE/'贡献汇总.csv',index=False,encoding='utf-8-sig')
    record=json.loads((HERE/'训练记录.json').read_text(encoding='utf-8'))
    expected=json.loads((ROOT/'河网三图预测_20261110/训练记录.json').read_text(encoding='utf-8'))['三图_单位对照']['initial_graph_hash_without_second_support']
    for name in ['冻结同框架','本站LoRA','本站LoRA加校正']:
        assert record[name]['initial_graph_hash_before_support_replacement']==expected
        assert record[name]['steps']==(0 if name=='冻结同框架' else 24)
        cp=torch.load(HERE/f'{name}.pt',map_location='cpu',weights_only=True)['graph']
        np.testing.assert_array_equal(cp['fixed'].numpy(),np.eye(750,dtype=np.float32))
        np.testing.assert_array_equal(cp['distance'].numpy(),np.eye(750,dtype=np.float32))
        np.testing.assert_array_equal(np.load(HERE/f'{name}_学习矩阵.npy'),np.eye(750,dtype=np.float32))
        assert cp['gcn.mlp.mlp.weight'].shape==(64,448,1,1)
    assert record['本站LoRA加校正']['trainable']==339484
    assert record['本站LoRA']['trainable']==196608
    assert record['冻结同框架']['trainable']==0
    assert record['本站LoRA加校正']['neighbour_perturbation_check']['passed']
    audit=dict(passed=True,test_pairs=len(idx),stations=idx['站点ID'].nunique(),all_three_supports_identity=True,matched_initial_graph_weights=True,neighbour_perturbation_check=record['本站LoRA加校正']['neighbour_perturbation_check'],self_only_declared_trainable=339484,self_only_ineffective_node_embedding_parameters=15000,note='Node embeddings retained for architecture matching; diagonal-only softmax has zero gradients for them. Shared training across stations remains allowed; no other-station inputs at inference.')
    (HERE/'核验.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# 完全不使用邻站的同结构对照

本轮把相关图、距离/河网图、自适应图三条消息传播路径全部限制为自身连接。仍使用全750站联合训练的共享参数，但某站预测只读取本站历史9项和未来24小时本站5项易测值，没有其他站点的输入。模型并非750个单独训练的模型。

冻结同框架、本站LoRA、本站LoRA加校正三个实验使用同一Chronos-2本地权重、相同输入组装和解码路径。冻结组不更新；本站LoRA和本站LoRA加校正均用原24次更新、相同随机种子和相同起报采样。LoRA仍为末两层q/k/v/o、rank 8、学习率1e-5；校正模块学习率0.001。没有使用测试集选轮次。

本站LoRA组冻结校正模块，残差门保持0，只有LoRA影响输出；本站LoRA加校正组允许本站校正模块更新。三图同结构校正模块仍有448维卷积输入，声明可训练参数总数339,484，与已有三图空间模型相同。保留原15,000个节点嵌入参数，但对角约束的自适应图恒为单位阵，其嵌入没有有效梯度，故有效自由度比跨站版少；这是移除跨站关系的结构性结果。

初始化先采用原Pearson图SVD初始化节点嵌入，随后替换消息支持矩阵，避免替换图造成参数初始化变化。图模块初始权重哈希与此前“相关＋单位＋自适应”三图对照完全一致。该空间组为主要同结构比较；此前320维卷积的原两图仅供参考。

数据仍为原750站、23,136个精确24小时测试配对。所有组非负截断一致；逐站逐指标计算R²，近常数目标不定义，负值保留；整体为四指标逐站均值再等权平均。阈值计数按任一项达标站点去重。

## 总体结果

'''+table(total)+'\n\n## 分项贡献\n\n'+table(boots)+'\n\n'+table(gains)+'\n\n## 分指标误差\n\n'+table(per)
    report+='''

## 本轮结论

冻结同框架平均R²为−0.785235；24步本站LoRA为−0.745295，改善约0.039940；进一步加入本站校正模块为−0.396456，改善约0.348839。同结构加入相关和自适应邻站信息后为−0.356348，再改善约0.040108。按本轮嵌套对照，主要平均收益来自本站校正，邻站提供额外且更小的平均收益。这些是重新训练后的净差值，不能当作各组件独立贡献百分比。

邻站信息对高锰酸盐指数、氨氮、总磷、总氮的平均R²变化分别为+0.037878、+0.011047、+0.019578、+0.091929。但氨氮变化中位数为−0.000106、平均MAE也略有恶化，因此不能理解为四项都普遍改善。其余三项平均MAE降低。

无邻站与同结构跨站版本的任一项R²超过0.5站数均为171，超过0.8站数由15变为14。冻结与仅LoRA组高分站数反而更多，表明平均R²的改善与高分站数的优化存在取舍。所有版本总体平均R²仍为负，目前没有达到整体预测表现良好的程度。

## 核验与解释边界

已核对三张图全部为单位阵、卷积形状和参数量与预期一致、初始权重哈希相同、测试真值逐值相同。另在完整编码及解码路径上，将其他749站的历史和未来易测输入全部清空并标记不可用，检查目标站预测不变；具体最大差值见核验JSON。跨站训练共享参数与推理时读取邻站输入是不同概念，本对照去除的是后者。

贡献差值来自分别训练的模型，反映增加相应组件后重新训练的净变化，并非严格可相加的因果分解；邻站信息可能与LoRA及校正模块发生交互。区间采用站点成组重抽样2,000次，仅为描述性不确定性，未覆盖流域空间依赖及训练种子变化。单种子24步且测试期已反复查看，结论只适用于当前实现和训练预算。

所有原空间模型结果复用原预测文件，没有覆盖其权重。冻结组也在本轮完全相同的编码路径下重新评估，没有直接拿不同历史训练预算的官方500步LoRA结果作本轮因果对照。

复现入口：make_runner.py生成沿用原框架的run.py；run.py训练并评估三组；results.py计算指标和核验。详细结果见逐站点误差.csv、逐站点配对收益.csv、分项贡献比较.csv。
'''
    (HERE/'实验报告.md').write_text(report,encoding='utf-8')
    print(total.to_string(index=False));print(boots.to_string(index=False));print(gains.to_string(index=False));print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
