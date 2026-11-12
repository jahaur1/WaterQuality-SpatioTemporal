"""Score the 24-step graph trained on linearly filled inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHEM = ['高锰酸盐指数(mg/L)', '氨氮(mg/L)', '总磷(mg/L)', '总氮(mg/L)']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    record = json.loads((HERE / '训练记录.json').read_text(encoding='utf-8'))
    assert record['steps'] == 24 and record['loss_name'] == 'smooth_l1'
    current = np.load(HERE / 'predictions.npz')
    original = np.load(ROOT / '河网三图预测_20261110' / 'predictions.npz')
    frozen = np.load(ROOT / '无邻站同结构对照_20261110' / 'predictions.npz')
    np.testing.assert_array_equal(current['truth'], original['truth'])
    np.testing.assert_array_equal(current['truth'], frozen['truth'])
    names = {
        '冻结Chronos-2': frozen['冻结同框架'],
        '三图_单位对照': original['三图_单位对照'],
        '插值后三图_单位对照': current['插值后三图_单位对照'],
    }
    idx = pd.read_csv(ROOT / '冻结预训练模型校正头基线_20261106' / '配对索引.csv')
    test = idx[idx.test].reset_index(drop=True)
    rows = []
    for mode, raw in names.items():
        pred = np.maximum(raw, 0)
        assert pred.shape == (23136, 4) and np.isfinite(pred).all(), mode
        for sid, ii in test.groupby('站点ID').indices.items():
            y = current['truth'][ii]
            error = pred[ii] - y
            sst = np.square(y - y.mean(0)).sum(0)
            r2 = 1 - np.divide(np.square(error).sum(0), sst, out=np.full(4, np.nan), where=sst > 1e-12 * np.maximum(1, np.square(y).sum(0)))
            for j, name in enumerate(CHEM):
                rows.append([mode, sid, name, len(ii), np.abs(error[:, j]).mean(), np.sqrt(np.square(error[:, j]).mean()), r2[j]])
    frame = pd.DataFrame(rows, columns=['模型', '站点ID', '指标', '测试配对数', 'MAE', 'RMSE', 'R2'])
    frame.to_csv(HERE / '逐站点误差.csv', index=False, encoding='utf-8-sig')
    per = frame.groupby(['模型', '指标'], sort=False).agg(平均R2=('R2', 'mean'), MAE均值=('MAE', 'mean'), RMSE均值=('RMSE', 'mean')).reset_index()
    per.to_csv(HERE / '分指标比较.csv', index=False, encoding='utf-8-sig')
    summary = []
    for mode in names:
        part = frame[frame['模型'].eq(mode)]
        best = part.groupby('站点ID').R2.max()
        summary.append([mode, per.loc[per['模型'].eq(mode), '平均R2'].mean(), int(best.gt(0.5).sum()), int(best.gt(0.8).sum())])
    summary = pd.DataFrame(summary, columns=['模型', '四指标平均R2', '任一项R2超过0.5站数', '任一项R2超过0.8站数'])
    summary.to_csv(HERE / '总体比较.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
