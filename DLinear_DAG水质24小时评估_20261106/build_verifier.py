from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
s=(HERE.parent/'XLinear水质24小时条件预测_20261102/verify_exact24.py').read_text(encoding='utf-8')
s=s.replace('import json,hashlib,sys,time','import json,hashlib,sys,time,argparse\nsys.stdout.reconfigure(encoding="utf-8")')
s=s.replace('from adapter import XLinearConditional24','from adapter import Conditional24')
s=s.replace("OUT=HERE/'results'",'OUT=None;MODEL=None')
s=s.replace('def main():', '''def stationdir(sid,arm):
    if MODEL=='DLinear':return HERE/'results_DLinear/stations'/sid/arm
    candidates=list(HERE.glob('results_DAG_shard*/stations/'+sid+'/'+arm));assert len(candidates)==1
    return candidates[0]

def main():
    global OUT,MODEL
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['DLinear','DAG'],required=True);args=ap.parse_args();MODEL=args.model;OUT=HERE/('results_'+MODEL)''')
s=s.replace("HERE/'run_forecast24.py'","HERE/'run_train.py'").replace("HERE/'vendor_XLinear.py'","HERE/'vendor_DLinear.py'")
old="    original=ROOT/'XLinear评估_20261102'/'source'/'models'/'XLinear.py';assert sha(original)==sha(HERE/'vendor_DLinear.py')"
assert old in s
s=s.replace(old,"""    prov=readj(HERE/'source_provenance.json');assert sha(HERE/'source_provenance.json')==config['provenance_sha256']
    for record in prov['files']:
        assert sha(HERE/record['source'])==record['source_sha256'] and sha(HERE/record['vendored'])==record['sha256']
    assert sha(HERE/'vendor_DLinear.py')==sha(HERE/'source_DLinear/models/DLinear.py')""")
s=s.replace("OUT/'stations'/sid/arm","stationdir(sid,arm)")
old="model=XLinearConditional24(seq_len=6,d_model=conf['d_model'],t_ff=conf['t_ff'],c_ff=conf['c_ff'],dropout=conf['dropout'],use_future=future).eval()"
assert old in s;s=s.replace(old,"model=Conditional24(MODEL,use_future=future).eval()")
s=s.replace("'original_vendor_byte_exact':True","'official_DLinear_byte_exact_and_DAG_documented_patches_verified':True")
needle="                outz=model(hs,fs if future else None).numpy()"
s=s.replace(needle,needle+"\n                one=model(hs[:1],fs[:1] if future else None).numpy();assert np.allclose(one,outz[:1],rtol=1e-4,atol=1e-5),('batch isolation',MODEL,sid,arm)")
s=s.replace("(OUT/'独立验证.json').write_text", "(OUT/'独立验证.json').write_text")
ast.parse(s);(HERE/'verify.py').write_text(s,encoding='utf-8');print('VERIFIER_READY')
