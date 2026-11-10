import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'DLinear_DAG水质24小时评估_20261106'
new=pd.read_csv(HERE/'results_DLinear/01_逐站点两分支四目标误差.csv')
old=pd.read_csv(OLD/'results_DLinear/01_逐站点两分支四目标误差.csv')
dag=pd.read_csv(OLD/'results_DAG/01_逐站点两分支四目标误差.csv')
rows=[];counts=[]
for name,frame in [('DLinear_DAG仓库结构',new),('DLinear_旧适配版',old),('DAG_此前适配版',dag)]:
    for arm,g in frame.groupby('分支'):
        for target,t in g.groupby('目标变量',sort=False):
            rows.append({'模型':name,'分支':arm,'指标':target,'站点数':t['站点ID'].nunique(),'R2有效站点数':t['R2'].notna().sum(),'R2均值':t['R2'].mean(),'R2中位数':t['R2'].median(),'MAE中位数':t['MAE'].median(),'RMSE中位数':t['RMSE'].median(),'R2大于0.5站点数':int(t['R2'].gt(.5).sum()),'R2大于0.8站点数':int(t['R2'].gt(.8).sum())})
        counts.append({'模型':name,'分支':arm,'任一指标R2大于0.5站点数':g.loc[g.R2.gt(.5),'站点ID'].nunique(),'任一指标R2大于0.8站点数':g.loc[g.R2.gt(.8),'站点ID'].nunique()})
A=pd.DataFrame(rows);C=pd.DataFrame(counts)
A.to_csv(HERE/'01_新旧实现逐指标比较.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
C.to_csv(HERE/'02_新旧实现高R2站点比较.csv',index=False,encoding='utf-8-sig')
keys=['站点ID','分支','目标变量']
assert not new.duplicated(keys).any() and not old.duplicated(keys).any()
merged=new[keys+['MAE','R2']].merge(old[keys+['MAE','R2']],on=keys,suffixes=('_新版','_旧版'),validate='one_to_one')
merged['R2提高量']=merged.R2_新版-merged.R2_旧版
merged['MAE降低量']=merged.MAE_旧版-merged.MAE_新版
merged.to_csv(HERE/'03_逐站点新旧实现收益.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
# Verify identical test cases and measured outcomes, independent of model outputs.
P=pd.read_csv(HERE/'results_DLinear/02_精确24小时测试实测与预测.csv')
Q=pd.read_csv(OLD/'results_DLinear/02_精确24小时测试实测与预测.csv')
pk=['站点ID','分支','起报时间','目标时间']
cols=pk+[c for c in P if c.endswith('_实测') or c.endswith('_目标时刻易测值') or c in ['起报站内索引','目标站内索引','输入历史起点']]
pd.testing.assert_frame_equal(P[cols].sort_values(pk).reset_index(drop=True),Q[cols].sort_values(pk).reset_index(drop=True),check_exact=True)
means=A[A['分支'].eq('conditional_future_easy')].groupby('模型')['R2均值'].mean()
text='''# DAG仓库DLinear结构复评

采用固定的本地DAG仓库提交0758990e2c73bb54138ea3e7b11a35cbc5476bcc。
原始数据750站、79,940条；测试23,136个精确+24小时配对，每站分别训练有/无未来易测值两分支。

## 模型实现

- 使用仓库time_series_library/models/DLinear.py，文件仅修改分解模块导入路径。
- moving_avg、series_decomp和融合MLP类逐字提取自仓库源码。
- 窗口6、输出1个精确24小时目标端点；移动平均窗口25；individual=False。
- 季节和趋势时间线性层权重初始化为1/6。
- 输入重排为4项化学指标在前、5项易测指标在后；主干输出截取前4项，不添加通道混合层。
- 条件分支将4项主干化学预测与5项目标时刻易测值拼接，调用仓库MLP(9→32→4, ReLU)，无额外残差相加、无输出置零。隐藏宽度32取自仓库w_future脚本，未按测试集调参。
- 历史分支不添加融合模块，原生通道独立结构意味着历史易测值不会影响化学输出。
- 两分支分别14和466个可训练参数。

## 保留的水质任务适配

本次是仓库模型结构复评，不是仓库基准训练脚本的完整复现。保留此前水质实验训练协议以比较实现差异：逐站训练、训练段标准化、整批Adam(lr=.001)、最多100轮、验证耐心10、选定轮数后重新初始化拟合。仓库脚本采用小批次、学习率调度，且通常输出连续多步，本实验输出按真实时间配对的24小时端点。
选择训练标签早于4月24日；验证目标4月25日至4月29日前；最终拟合标签早于4月29日；测试目标4月30日至5月10日00:00。输入仅使用截至起报时刻6条同站历史，目标时刻5项易测真值为回溯已知条件，化学历史每次使用最新实测值；不能据此推断检测成本降低。

## 验证

仓库原始类与本次适配模型在固定随机输入下前向和参数梯度逐位相同；检查时间线性初始化、批次隔离及历史分支忽略未来值。独立核验所有站点配对、6000行误差、3000份标准化统计及10组保存权重重放。新旧实现的测试时间、真值和易测条件逐项完全相同。详见仓库一致性核验.json和results_DLinear/独立验证.json。

## 条件预测结果

'''
F=A[A['分支'].eq('conditional_future_easy')]
text+=F.to_csv(index=False)
text+='\n四指标先各自对可计算站点求算术平均、再四指标等权平均R²：\n'+means.to_string()+'\n\n'
text+=C.to_csv(index=False)
text+='\nR²缺失来自测试真值接近常量，不将其设为0；保留所有有限负值，不按测试结果截尾。高R²站点数不是推导成功率。旧版及此前DAG结果均保持原样，本轮不重新训练DAG。\n'
(HERE/'复评报告.md').write_text(text,encoding='utf-8')
(HERE/'比较核验.json').write_text(json.dumps({'passed':True,'same_exact_test_pairs_truth_and_future_easy':True,'metric_comparison_rows':len(merged),'conditional_macro_mean_r2':means.to_dict()},ensure_ascii=False,indent=2),encoding='utf-8')
print(F.to_string(index=False));print(C.to_string(index=False));print(means.to_string())
