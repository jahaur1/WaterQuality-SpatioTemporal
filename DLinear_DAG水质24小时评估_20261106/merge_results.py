from pathlib import Path
import json,sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
HERE=Path(__file__).resolve().parent
def save(d,p):d.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.12g')
def main():
    shards=[HERE/f'results_DAG_shard{i}' for i in range(4)]
    cfg=[json.loads((p/'config.json').read_text(encoding='utf-8')) for p in shards];assert len({c['fingerprint'] for c in cfg})==1
    summaries=[json.loads((p/'summary.json').read_text(encoding='utf-8')) for p in shards]
    assert sum(s['基础站点数'] for s in summaries)==750 and sum(s['全部测试精确配对数'] for s in summaries)==23136
    out=HERE/'results_DAG';out.mkdir(exist_ok=True)
    for name,expected,unique in [('01_逐站点两分支四目标误差.csv',6000,['站点ID','分支','目标变量']),('02_精确24小时测试实测与预测.csv',46272,['站点ID','分支','起报时间','目标时间']),('03_站点配对覆盖与不可评估原因.csv',750,['站点ID']),('06_逐站点分支训练与时间记录.csv',1500,['station','arm'])]:
        frames=[pd.read_csv(p/name) for p in shards];d=pd.concat(frames,ignore_index=True)
        assert len(d)==expected and not d.duplicated(unique).any(),name
        save(d.sort_values(unique),out/name)
    (out/'config.json').write_text(json.dumps(cfg[0],ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'基础站点数':750,'站点目标分支指标数':6000,'全部测试精确配对数':23136,'fingerprint':cfg[0]['fingerprint'],'shard_summaries_merged_without_selection':True},ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'合并核验.json').write_text(json.dumps({'passed':True,'shards':4,'stations':750,'test_pairs_per_arm':23136,'metrics':6000,'fingerprint':cfg[0]['fingerprint'],'checkpoint_location':'results_DAG_shard*/stations; station membership disjoint; no weights or predictions averaged','source_summaries':summaries},ensure_ascii=False,indent=2),encoding='utf-8')
    print('MERGED_DAG',750,6000,23136)
if __name__=='__main__':main()
