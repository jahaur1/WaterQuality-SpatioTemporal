"""Retrain 三图_单位对照 for 24 steps on the linearly filled 4-hour series.

History and future easy covariates are read at exact 4-hour timestamps.
Measured targets and the 24-step optimizer schedule stay the same.
The original checkpoint is not overwritten.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location('river_graph', ROOT / '河网三图预测_20261110' / 'run.py')
river = importlib.util.module_from_spec(spec)
spec.loader.exec_module(river)
CACHE = river.CACHE
CHEM = river.CHEM
EASY = ['水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)', '浊度(NTU)']
FILLED = ROOT / '30天_750站点数据集_20250410-20250510' / '水质数据_750站点_4小时线性插值.csv'
GRID = pd.date_range('2025-04-10 00:00:00', '2025-05-10 00:00:00', freq='4h')


def data():
    idx = pd.read_csv(CACHE / '配对索引.csv')
    raw = pd.read_csv(FILLED, parse_dates=['监测时间'])
    matrix = np.load(ROOT / '全站拓扑联合预测_20261109' / '拓扑与站点.npz')
    stations = matrix['station_ids'].tolist()
    nodes = {station: i for i, station in enumerate(stations)}
    values = np.full((750, len(GRID), 9), np.nan, np.float32)
    station_index = raw['站点ID'].map(nodes)
    time_index = pd.DatetimeIndex(GRID).get_indexer(pd.DatetimeIndex(raw['监测时间']))
    if station_index.isna().any() or np.any(time_index < 0):
        raise RuntimeError('filled series contains a station or time outside the model grid')
    ordered = raw[CHEM + EASY].to_numpy(np.float32)
    values[station_index.to_numpy(int), time_index] = ordered
    if not np.isfinite(values).all():
        raise RuntimeError('filled grid has a missing value')
    times = np.sort(pd.to_datetime(idx['起报时间'], format='mixed').unique())
    h = np.full((len(times), 750, 9, 32), np.nan, np.float32)
    f = np.full((len(times), 750, 9, 16), np.nan, np.float32)
    available = np.ones((len(times), 750), np.float32)
    padded = np.concatenate([np.full((750, 31, 9), np.nan, np.float32), values], axis=1)
    for i, origin in enumerate(times):
        position = int(pd.DatetimeIndex(GRID).get_indexer([pd.Timestamp(origin)])[0])
        if position < 0 or position + 6 >= len(GRID):
            raise RuntimeError(f'origin is outside the filled grid: {origin}')
        h[i] = np.transpose(padded[:, position:position + 32], (0, 2, 1))
        f[i, :, 4:, 5] = values[:, position + 6, 4:]
    time_map = {pd.Timestamp(origin): i for i, origin in enumerate(times)}
    ii = np.array([time_map[pd.Timestamp(origin)] for origin in idx['起报时间']])
    nnodes = np.array([nodes[station] for station in idx['站点ID']])
    truth = np.load(CACHE / 'inputs.npz')['truth']
    y = np.zeros((len(times), 750, 4), np.float32)
    y[ii, nnodes] = truth
    target_position = pd.DatetimeIndex(GRID).get_indexer(pd.DatetimeIndex(pd.to_datetime(idx['目标时间'], format='mixed')))
    if np.any(target_position < 0):
        raise RuntimeError('a measured target is outside the filled grid')
    looked_up = values[nnodes, target_position, :4]
    if not np.allclose(looked_up, truth, equal_nan=False):
        raise RuntimeError('interpolated target differs from a measured pair label')
    masks = {name: np.zeros((len(times), 750), bool) for name in ['refit_train', 'test']}
    for name in masks:
        masks[name][ii, nnodes] = idx[name].to_numpy(bool)
    stats = json.loads((CACHE / '站点训练标准化.json').read_text(encoding='utf-8'))
    scale = np.array([stats[station]['final']['scale'][5:] for station in stations], np.float32)
    finite_history = float(np.isfinite(h).mean())
    return idx, h, f, available, y, masks, matrix['final'], scale, ii, nnodes, finite_history


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    idx, h, f, available, y, masks, adjacency, scale, ii, nodes, finite_history = data()
    train_times = np.flatnonzero(masks['refit_train'].any(1))
    test_times = np.flatnonzero(masks['test'].any(1))
    river.local.set_seed(20261109)
    base = river.backbone('三图_单位对照')
    graph = river.LatentGraph(adjacency, '三图_单位对照').cuda()
    parameters = [p for p in list(base.parameters()) + list(graph.parameters()) if p.requires_grad]
    assert all('lora_' in name for name, p in base.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        [dict(params=graph.parameters(), lr=0.001), dict(params=[p for p in base.parameters() if p.requires_grad], lr=1e-5)],
        weight_decay=0.0001,
    )
    rng = np.random.default_rng(20261109)
    trace = []
    sampled = []
    started = time.time()
    base.eval()
    graph.train()
    for step in range(24):
        t = int(rng.choice(train_times))
        sampled.append(t)
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = river.forward(base, graph, h[t], f[t], available[t])
            mask = torch.tensor(masks['refit_train'][t], device='cuda')
            target = torch.tensor(y[t], device='cuda')
            station_scale = torch.tensor(scale, device='cuda')
            loss = nn.functional.smooth_l1_loss((pred[mask] - target[mask]) / station_scale[mask], torch.zeros_like(target[mask]))
        if not torch.isfinite(loss):
            raise RuntimeError(f'nonfinite loss at step {step + 1}')
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        trace.append(float(loss.detach()))
        if step % 4 == 0:
            print('step', step + 1, 'loss', trace[-1], 'seconds', round(time.time() - started, 1), flush=True)
    graph.eval()
    predictions = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for j, t in enumerate(test_times):
            predictions.append(river.forward(base, graph, h[t], f[t], available[t]).float().cpu().numpy())
            if j % 20 == 0:
                print('test time', j + 1, '/', len(test_times), flush=True)
    positions = {t: i for i, t in enumerate(test_times)}
    selected = idx.test.to_numpy(bool)
    prediction = np.stack(predictions)[np.array([positions[t] for t in ii[selected]]), nodes[selected]]
    truth = np.load(CACHE / 'inputs.npz')['truth'][selected]
    np.savez_compressed(HERE / 'predictions.npz', truth=truth, 插值后三图_单位对照=prediction)
    torch.save(
        dict(graph=graph.state_dict(), lora={k: v.cpu() for k, v in base.state_dict().items() if 'lora_' in k}),
        HERE / '三图_单位对照.pt',
    )
    record = dict(
        steps=24,
        loss=trace,
        sampled_origin_indices=sampled,
        finite_history_fraction=finite_history,
        trainable=sum(p.numel() for p in parameters),
        graph_trainable=sum(p.numel() for p in graph.parameters() if p.requires_grad),
        lora_trainable=sum(p.numel() for p in base.parameters() if p.requires_grad),
        graph_lr=0.001,
        lora_lr=1e-5,
        weight_decay=0.0001,
        gradient_clip_norm=1.0,
        loss_name='smooth_l1',
        source=str(FILLED.relative_to(ROOT)),
        source_sha256=hashlib.sha256(FILLED.read_bytes()).hexdigest(),
        seconds=round(time.time() - started, 1),
    )
    (HERE / '训练记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print('done', record['seconds'], 'first_loss', trace[0], 'finite_history', round(finite_history, 4), flush=True)


if __name__ == '__main__':
    main()
