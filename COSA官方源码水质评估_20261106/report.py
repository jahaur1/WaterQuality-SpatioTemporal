from pathlib import Path
import sys,json
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
def table(df):
    lines=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
def main():
    audit=json.loads((HERE/'独立核验.json').read_text(encoding='utf-8'));assert audit['passed'];cfg=json.loads((HERE/'实验配置.json').read_text(encoding='utf-8'))
    A=pd.read_csv(HERE/'02_逐指标汇总.csv');C=pd.read_csv(HERE/'03_总体比较.csv');G=pd.read_csv(HERE/'04_逐站点校正收益.csv');rows=[]
    for (name,target),g in G.groupby(['模型_COSA','目标变量'],sort=False):rows.append({'模型':name,'指标':target,'R2提高站点数_超过1e负6':int(g.R2提高量.gt(1e-6).sum()),'R2降低站点数_超过1e负6':int(g.R2提高量.lt(-1e-6).sum()),'R2提高量中位数':g.R2提高量.median(),'MAE降低站点数_超过1e负8':int(g.MAE降低量.gt(1e-8).sum())})
    gains=pd.DataFrame(rows);gains.to_csv(HERE/'05_逐指标改善站点统计.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    text=f'''# COSA官方源码：冻结水质预测模型的因果在线校正

本次直接执行作者源码中的SimpleOutputAdapter、SimpleAdapter类和get_optimizer函数，未重写校正头、门控、初始化、损失、梯度裁剪、自适应学习率或Adam更新。

官方仓库：https://github.com/bigbases/COSA_ICLR2026
固定提交：{cfg['official_commit']}
论文：COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting（官方README标注ICLR 2026），https://openreview.net/forum?id=L7Z5wBMPrW。
主要源码：source/tta/cosa.py；优化器：source/models/optimizer.py；固定参数参考source/scripts/cosa.sh及source/config.py。
δ-Adapter论文网页未定位到可核实的作者代码链接，因此这轮选用已取得的COSA官方实现，没有把上一轮自写岭回归当成论文复现。

## 实际使用的作者代码

native.py仅通过Python AST选择并编译原始类/函数定义，保留其方法体和原文件行号。SimpleOutputAdapter及其forward、Xavier(gain=.1)初始化和零门控原样执行；继承的_run_adapt_steps原样执行MSE+1e-4×参数平方和、作者梯度裁剪与自适应学习率，并调用作者get_optimizer构建Adam。代码运行位置可从官方核心一致性核验.json核对。
校正结构是每项化学指标独立Linear([本项预测1维，上下文10维]→1维)，再乘该变量的tanh门控并加回原预测；4项共52个参数。上下文为最近10个已到达预测结果的4项标准化化学真值均值，从近到远排列，不足时重复最旧值，最初为空时补零，与作者_get_context_vector数值一致。
上下文不直接拼接未来易测值；5项历史与目标时刻易测信息继续通过冻结骨干的原生协变量接口发挥作用。

## 水质任务适配及边界

- Timestral-209M和Chronos-2沿用此前冻结条件预测缓存，骨干权重不更新；测试站点750、精确24小时配对23,136，标签及原始预测保持相同。
- 每个站点、每个骨干单独初始化一个官方校正头，固定随机种子20261106+站点编号。首次预测零门控，等于原始标准化预测；每站从第一个测试起报时刻开始，未进行测试前离线训练。
- 每当过去预测的目标时间<=当前起报时间，才释放该4项化学标签，保存原预测当时的上下文，调用作者更新函数3步；每个成熟配对只更新一次。以真实时间代替作者规则序列的行号排队，禁止跨站合并上下文或梯度。
- 每次发出预测后保持不变。目标尚未到达的化学值只作为待释放数据，不用于任何更新；日志记录所有释放时刻，独立验证其不晚于预测前已观测时刻。
- 将pred_len设为1表示唯一的+24小时端点，并非未来一步观测；batch_size=1。部分区间标签更新、预测后回改、PAAS周期批次调度均关闭，因为现任务是非等间隔观测的单端点预测。这是官方COSA核心的任务适配，未声称完整复现其规则多步数据基准及全部选项。
- 保留脚本的Linear、buffer10、3步、初始lr=.001、fast/adaptive lr、maxlr=.005、minlr=.0001和逐批lr重置；保留默认逐变量门控、L2=1e-4、Adam weight_decay=.0001。无测试集超参数搜索。
- 用各站4月29日前原始观测固定标准化，不使用测试期统计更新；预测统一截断负值为0，未截断结果另存。

此处是因果在线自适应：测试早期标签在其目标时刻到达后，可用于后续起报的校正。这与上一轮测试期间不更新头的离线岭回归不同，应按更新协议理解比较。用户此前允许每次预测使用已到达的最新化学实测值；化学报告延迟未模拟。

## 总体结果

平均R²是各化学指标先对可计算站点求算术平均，再四指标等权平均；无定义值排除，有限负值全部保留。阈值严格>，站点按任一指标达标去重。

'''
    text+=table(C)+'\n\n## 逐指标结果\n\n'+table(A)+'\n\n## 相对原模型改善\n\n'+table(gains)
    text+='''

## 验证与解释

官方源码检出保持干净，源码、执行模块、骨干权重哈希均核验；作者原生前向及更新方法实际被调用，门控为零时输出恒等，上下文值、排列、补齐与官方方法一致。逐条核验全部标签释放时刻、更新次数与首次可用起报时间；独立复算15,000组指标，并对5个固定站点×2骨干完整重放每次预测和更新，输出及最终权重逐位相同。完整记录见独立核验.json。
部分R²改善不等于达到业务误差要求，也不等于统计显著；改善计数设置微小数值容差，避免标准化和反标准化舍入被误判为收益。未来易测实测条件、历史化学持续可得、在线标签更新均是本实验条件，不能由此推断化学检测成本降低。此前测试段已反复用于模型探索，确认性结论需新的时间段。

01为逐站点5方案误差；02为逐指标汇总；03为总体比较；04为逐站点收益；05为改善站点统计。标签释放与更新记录.csv记录因果更新时序；逐站点在线适配记录.csv记录更新量和门控；两骨干目录保存各750份校正头权重。run.py运行实验，verify.py独立核验，report.py生成本报告。
'''
    (HERE/'实验报告.md').write_text(text,encoding='utf-8');print(C.to_string(index=False));print(gains.to_string(index=False))
if __name__=='__main__':main()
