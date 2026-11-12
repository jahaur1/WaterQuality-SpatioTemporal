# Experimental result index

Snapshot: 2026-11-10. Results are grouped by experimental protocol; this is not a single leaderboard.
Mean R2 retains negative values. Threshold counts mean at least one chemical indicator exceeds the threshold.
Different spatial inputs, LoRA configurations, and training budgets must not be pooled as matched comparisons.

## Matched self-only adaptation versus cross-station inputs

Source: [无邻站同结构对照_20261110/总体比较.csv](../无邻站同结构对照_20261110/总体比较.csv)

| 模型 | 四指标平均R2 | 任一项R2超过0.5站数 | 任一项R2超过0.8站数 |
| --- | --- | --- | --- |
| 冻结同框架 | -0.785235 | 185 | 23 |
| 本站LoRA | -0.745295 | 186 | 22 |
| 本站LoRA加校正 | -0.396456 | 171 | 15 |
| 相关加单位加自适应 | -0.356348 | 171 | 14 |
| 相关加距离加自适应 | -0.353861 | 169 | 12 |
| 相关加河网加自适应 | -0.356687 | 170 | 14 |
| 原两图_仅参考 | -0.349938 | 168 | 10 |

## Graph WaveNet topology comparison, 24 updates

Source: [GraphWaveNet拓扑对照_20261109/总体比较.csv](../GraphWaveNet拓扑对照_20261109/总体比较.csv)

| 模型 | 四指标平均R2 | 任一项R2超过0.5站数 | 任一项R2超过0.8站数 |
| --- | --- | --- | --- |
| GWN固定图 | -0.432648 | 175 | 19 |
| GWN自适应图 | -0.427288 | 177 | 18 |
| GWN固定加自适应 | -0.349938 | 168 | 10 |

## Three fixed/adaptive graph supports

Source: [三图联合预测_20261110/总体比较.csv](../三图联合预测_20261110/总体比较.csv)

| 模型 | 四指标平均R2 | 任一项R2超过0.5站数 | 任一项R2超过0.8站数 |
| --- | --- | --- | --- |
| Pearson加自适应 | -0.349938 | 168 | 10 |
| 距离加自适应 | -0.376422 | 173 | 12 |
| 三图联合 | -0.353861 | 169 | 12 |

## Directed candidate river network and matched identity control

Source: [河网三图预测_20261110/总体比较.csv](../河网三图预测_20261110/总体比较.csv)

| 模型 | 四指标平均R2 | 任一项R2超过0.5站数 | 任一项R2超过0.8站数 |
| --- | --- | --- | --- |
| Pearson加自适应 | -0.349938 | 168 | 10 |
| 三图_直线距离 | -0.353861 | 169 | 12 |
| 三图_单位对照 | -0.356348 | 171 | 14 |
| 三图_河网探索 | -0.356687 | 170 | 14 |

## Same 24-update run on the 4-hour linear interpolation

Source: [插值后三图_单位对照_20261111/总体比较.csv](../插值后三图_单位对照_20261111/总体比较.csv)

History and future easy covariates are read from the filled 4-hour series. Test labels stay the original measurements. This row is not pooled with the unfilled runs above.

| 模型 | 四指标平均R2 | 任一项R2超过0.5站数 | 任一项R2超过0.8站数 |
| --- | --- | --- | --- |
| 冻结 Chronos-2 | -0.7852 | 185 | 23 |
| 三图_单位对照 | -0.3563 | 171 | 14 |
| 插值后三图_单位对照 | -1.6090 | 186 | 20 |

MAE and RMSE are means of the 750 per-station values, in mg/L. The four indicators are not averaged together. Source: [分指标比较.csv](../插值后三图_单位对照_20261111/分指标比较.csv).

| 模型 | 指标 | MAE | RMSE |
| --- | --- | --- | --- |
| 冻结 Chronos-2 | 高锰酸盐指数 | 0.341213 | 0.527691 |
| 冻结 Chronos-2 | 氨氮 | 0.045357 | 0.078812 |
| 冻结 Chronos-2 | 总磷 | 0.010806 | 0.017064 |
| 冻结 Chronos-2 | 总氮 | 0.211625 | 0.315542 |
| 三图_单位对照 | 高锰酸盐指数 | 0.323937 | 0.489064 |
| 三图_单位对照 | 氨氮 | 0.043569 | 0.075219 |
| 三图_单位对照 | 总磷 | 0.010405 | 0.016199 |
| 三图_单位对照 | 总氮 | 0.204112 | 0.300050 |
| 插值后三图_单位对照 | 高锰酸盐指数 | 0.335223 | 0.502138 |
| 插值后三图_单位对照 | 氨氮 | 0.048417 | 0.081003 |
| 插值后三图_单位对照 | 总磷 | 0.010528 | 0.016343 |
| 插值后三图_单位对照 | 总氮 | 0.212065 | 0.309510 |

The total-nitrogen mean R2 is pulled by S1216 (R2 -3510, MAE 0.619 mg/L). The total-nitrogen median R2 is -0.055. Its MAE stays 0.212.

## Exact-time raw chemical lag branch, separate architecture group

Source: [时滞邻站对照_20261110/总体比较.csv](../时滞邻站对照_20261110/总体比较.csv)

| 模型 | 四指标平均R2 | 任一项R2超过0.5站数 | 任一项R2超过0.8站数 |
| --- | --- | --- | --- |
| 不使用邻站 | -0.453913 | 146 | 8 |
| 邻站当前值 | -0.402498 | 155 | 12 |
| 邻站学习时滞 | -0.400154 | 156 | 12 |

## Earlier model families

- XLinear: contemporaneous soft sensing and exact +24h conditional forecasting.
- DLinear / DAG: original task adaptations and a separate author-implementation reassessment.
- Chronos-2, TimesFM and Timestral: pretrained conditional prediction; see individual reports.
- Frozen correction heads / COSA: adaptation baselines with their own update protocols.
- Official single-station Chronos-2 LoRA used 500 updates; joint graph pilot runs mostly used 24.
- Dataset quality, Pearson statistics and candidate station selections are descriptive analyses.

The 30-day measured table, boundary anchors, and 4-hour linear interpolation are in the repository. Monthly raw scrapes, predictions, checkpoints, and some large detailed tables remain local-only. See REPRODUCIBILITY.md.
The test interval has been inspected in multiple rounds; confirmatory results require a new holdout period.
