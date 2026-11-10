"""Reuse the validated common metric calculations; write a Timestral-specific report."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
template=(HERE.parent/'TimesFM3水质24小时评估_20261104/summarize.py').read_text(encoding='utf-8')
prefix=template[:template.index("    report='# TimesFM")]
prefix=prefix.replace("'TimesFM-3.0'","'Timestral-209M'")
needle="    combined=pd.concat([f[keep],cc[keep],x[keep],mm[keep],base],ignore_index=True)"
assert prefix.count(needle)==1
prefix=prefix.replace(needle,"    tf=pd.read_csv(ROOT/'TimesFM3水质24小时评估_20261104/01_逐站点两条件四指标误差.csv');tf=tf[tf['输入条件']=='future_easy']\n    combined=pd.concat([f[keep],tf[keep],cc[keep],x[keep],mm[keep],base],ignore_index=True)")
suffix='''    report='# Timestral-209M 水质24小时条件预测\\n\\n'
    report+='官方mistralai/Timestral-209M权重与官方源码固定提交，冻结权重不微调；下载SHA256与官方LFS记录一致。模型代码使用项目内源码及现有PyTorch环境，无需修改已有包。\\n\\n'
    report+='全部750站、23,136组精确24小时配对，与XLinear、MOIRAI、Chronos-2、TimesFM 3.0相同。复用已独立核验输入：截至起报时刻最近6次9项观测，在这6次历史首尾间按4小时网格前向填充；首条历史之前为NaN，32格上下文，预测6格，评估第6格。\\n\\n'
    report+='使用原生TimestralForecaster.forecast：context为4项化学历史，known_covariates为5项易测历史及未来6格。两组均保持相同known_covariates角色；仅历史组未来6格全为NaN，条件组仅最后一格为目标时刻5项易测实测，中间5格仍为NaN。未传入任何未来化学实测，原生接口保留缺失掩码，不插值未来协变量。\\n\\n'
    report+='FP32、64窗口一批，每个起报窗口独立分组。使用官方默认对称平均symmetric_averaging=True及quantile_real_cap_k=10000，输出21分位数的0.5分位数；关闭接口非负裁剪以保留原始值，在统一评分前将负浓度截断0，并同时保存未截断预测及误差。此前TimesFM 3.0使用symmetric_averaging=False，Chronos-2无该步骤；此次为各模型已记录固定推理配置的比较，不能仅归因于架构，未根据测试结果调参。\\n\\n'
    report+='这是未来易测实测已知、化学历史随每次起报更新的回溯条件预测；不证明能够减少化学检测频率或成本。测试只有10天，R²>0.5和>0.8只作描述性分档；缺少业务误差标准，不能称成功率。未进行新统计显著性声明。XLinear有本站监督训练，预训练模型本轮均冻结；历史网格编码方式也有差异。\\n\\n'
    view=summary.copy();numeric=['R2中位数','MAE中位数','RMSE中位数','对应XLinearR2中位数','对应Chronos2R2中位数'];view[numeric]=view[numeric].round(5)
    report+=table(view)+'\\n'+table(unique)+'\\n## 加入未来信息的变化\\n\\n'
    gain=pd.DataFrame([{'指标':c,'R2提高站点数':int((g['加入未来易测值_R2提高量']>0).sum()),'全部站点R2差值中位数':round(g['加入未来易测值_R2提高量'].median(),5),'MAE降低站点数':int((g['加入未来易测值_MAE降低量']>0).sum())} for c,g in f.groupby('目标变量',sort=False)])
    report+=table(gain)+'\\n## 所有已完成模型及持久性基线（均为未来易测条件）\\n\\n'+table(coverage)+'\\n'
    cv=comparison.copy();cv[['R2中位数','MAE中位数','RMSE中位数']]=cv[['R2中位数','MAE中位数','RMSE中位数']].round(5);report+=table(cv)+'\\n'
    report+='独立核验通过：6000组站点指标复算、首64预测重放、输入隔离、来源及旧预测哈希、官方权重SHA256。\\n\\n参考：[Timestral官方仓库](https://github.com/mistralai/timestral)、[官方权重和许可](https://huggingface.co/mistralai/Timestral-209M)。\\n'
    (HERE/'评估报告.md').write_text(report,encoding='utf-8');print(summary.to_string(index=False));print(unique.to_string(index=False));print(gain.to_string(index=False));print(coverage.to_string(index=False));print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
'''
(HERE/'summarize.py').write_text(prefix+suffix,encoding='utf-8')
import ast
ast.parse(prefix+suffix)
print('ANALYSIS_READY')
