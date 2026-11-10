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

Original data, predictions, checkpoints and some large detailed tables are local-only. See REPRODUCIBILITY.md.
The test interval has been inspected in multiple rounds; confirmatory results require a new holdout period.
