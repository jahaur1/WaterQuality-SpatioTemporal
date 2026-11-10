"""Rebuild the compact result index from committed aggregate CSV files, without ML imports."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GROUPS=[
    ('Matched self-only adaptation versus cross-station inputs','无邻站同结构对照_20261110/总体比较.csv'),
    ('Graph WaveNet topology comparison, 24 updates','GraphWaveNet拓扑对照_20261109/总体比较.csv'),
    ('Three fixed/adaptive graph supports','三图联合预测_20261110/总体比较.csv'),
    ('Directed candidate river network and matched identity control','河网三图预测_20261110/总体比较.csv'),
    ('Exact-time raw chemical lag branch, separate architecture group','时滞邻站对照_20261110/总体比较.csv'),
]
def main():
    parts=['# Experimental result index','',
        'Snapshot: 2026-11-10. Results are grouped by experimental protocol; this is not a single leaderboard.',
        'Mean R2 retains negative values. Threshold counts mean at least one chemical indicator exceeds the threshold.',
        'Different spatial inputs, LoRA configurations, and training budgets must not be pooled as matched comparisons.','']
    for label,relative in GROUPS:
        with (ROOT/relative).open(encoding='utf-8-sig',newline='') as handle:
            rows=list(csv.reader(handle))
        parts+=['## '+label,'','Source: ['+relative+'](../'+relative+')','']
        parts+=['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
        for row in rows[1:]:
            if len(row)>1:
                try:row[1]=f'{float(row[1]):.6f}'
                except ValueError:pass
            parts.append('| '+' | '.join(row)+' |')
        parts.append('')
    parts+=['## Earlier model families','',
        '- XLinear: contemporaneous soft sensing and exact +24h conditional forecasting.',
        '- DLinear / DAG: original task adaptations and a separate author-implementation reassessment.',
        '- Chronos-2, TimesFM and Timestral: pretrained conditional prediction; see individual reports.',
        '- Frozen correction heads / COSA: adaptation baselines with their own update protocols.',
        '- Official single-station Chronos-2 LoRA used 500 updates; joint graph pilot runs mostly used 24.',
        '- Dataset quality, Pearson statistics and candidate station selections are descriptive analyses.',
        '',
        'Original data, predictions, checkpoints and some large detailed tables are local-only. See REPRODUCIBILITY.md.',
        'The test interval has been inspected in multiple rounds; confirmatory results require a new holdout period.','']
    (ROOT/'docs/RESULTS.md').write_text('\n'.join(parts),encoding='utf-8')
    print('Updated docs/RESULTS.md from '+str(len(GROUPS))+' committed result tables.')

if __name__=='__main__':main()
