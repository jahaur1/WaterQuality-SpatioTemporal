"""Fill the 750-station window onto the 4-hour clock. Original file is not modified."""
from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).resolve().parent
SOURCE = FOLDER / '水质数据_750站点_20250410-20250510.csv'
BOUNDARY = FOLDER / '区间边界辅助观测.csv'
DESTINATION = FOLDER / '水质数据_750站点_4小时线性插值.csv'
IDENTITY = ['站点ID', '省份', '流域', '断面名称']
NUMERIC = [
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)', '浊度(NTU)',
    '高锰酸盐指数(mg/L)', '氨氮(mg/L)', '总磷(mg/L)', '总氮(mg/L)',
]
GRID = pd.date_range('2025-04-10 00:00:00', '2025-05-10 00:00:00', freq='4h')


def load(path):
    frame = pd.read_csv(path, usecols=IDENTITY + ['监测时间', *NUMERIC])
    frame['监测时间'] = pd.to_datetime(frame['监测时间'])
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    return frame


def main():
    observed = load(SOURCE)
    anchors = pd.concat([observed, load(BOUNDARY)], ignore_index=True)
    anchors = anchors.drop_duplicates(['站点ID', '监测时间']).sort_values(['站点ID', '监测时间'])
    if anchors.duplicated(['站点ID', '监测时间']).any():
        raise RuntimeError('duplicate station time')
    if observed[NUMERIC].isna().any().any():
        raise RuntimeError('observed numeric field is missing')
    off_grid = observed.loc[~observed['监测时间'].isin(GRID), '监测时间']
    pieces = []
    for sid, station_anchors in anchors.groupby('站点ID', sort=False):
        station_anchors = station_anchors.set_index('监测时间').sort_index()
        filled = station_anchors[NUMERIC].reindex(station_anchors.index.union(GRID))
        filled = filled.interpolate(method='time', limit_area='inside').reindex(GRID)
        station_observed = observed.loc[observed['站点ID'] == sid].set_index('监测时间')
        on_grid = station_observed.index.intersection(GRID)
        filled.loc[on_grid, NUMERIC] = station_observed.loc[on_grid, NUMERIC]
        meta = station_anchors.iloc[0]
        piece = filled.reset_index().rename(columns={'index': '监测时间'})
        for column in IDENTITY:
            piece[column] = meta[column]
        piece['数值来源'] = '线性插值'
        piece.loc[piece['监测时间'].isin(on_grid), '数值来源'] = '实测'
        piece.loc[piece[NUMERIC].isna().any(axis=1), '数值来源'] = '无锚点'
        pieces.append(piece)
    result = pd.concat(pieces, ignore_index=True)
    result = result[IDENTITY + ['监测时间', '数值来源', *NUMERIC]]
    result.to_csv(DESTINATION, index=False, encoding='utf-8-sig')
    counts = result['数值来源'].value_counts().to_dict()
    print('rows', len(result), 'stations', result['站点ID'].nunique(), 'times', result['监测时间'].nunique())
    print('counts', counts)
    print('off_grid_observations', int(len(off_grid)))
    print('grid', len(GRID), GRID.min(), GRID.max())


if __name__ == '__main__':
    main()
