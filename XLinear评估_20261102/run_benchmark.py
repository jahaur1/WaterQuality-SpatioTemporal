import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, time, subprocess, shlex, re, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
SRC = BASE / 'source'
DATA = Path(r'D:\AI\TATO-main\DATASET\ETT-small')
sys.path.insert(0, str(SRC))
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute

def prepare():
    p = SRC / 'data_provider/data_factory.py'
    s = p.read_text(encoding='utf-8')
    s = s.replace("if flag == 'test':\n        shuffle_flag = False\n        drop_last = True", "if flag == 'test':\n        shuffle_flag = False\n        drop_last = False")
    p.write_text(s, encoding='utf-8')
    p = SRC / 'exp/exp_main.py'
    s = p.read_text(encoding='utf-8')
    s = s.replace('preds = np.array(preds)', 'preds = np.concatenate(preds, axis=0)')
    s = s.replace('trues = np.array(trues)', 'trues = np.concatenate(trues, axis=0)')
    s = s.replace('inputx = np.array(inputx)', 'inputx = np.concatenate(inputx, axis=0)')
    s = s.replace('if i % 20 == 0:', 'if i == 0:')
    p.write_text(s, encoding='utf-8')
    p = SRC / 'utils/tools.py'
    s = p.read_text(encoding='utf-8').replace("plt.savefig(name, bbox_inches='tight')", "plt.savefig(name, bbox_inches='tight')\n    plt.close()")
    p.write_text(s, encoding='utf-8')
    # Limit CPU threads for the small GPU model without changing its calculations.
    p = SRC / 'run_longExp.py'
    s = p.read_text(encoding='utf-8')
    if 'torch.set_num_threads(2)' not in s:
        s = s.replace('import torch', 'import torch\ntorch.set_num_threads(2)', 1)
    p.write_text(s, encoding='utf-8')

def evaluate(name, horizon, run):
    cls = Dataset_ETT_hour if name == 'ETTh1' else Dataset_ETT_minute
    ds = cls(str(DATA), flag='test', size=[96,48,horizon], features='M', data_path=name+'.csv', timeenc=1, freq='h' if name=='ETTh1' else 't')
    result = next((run / 'results').iterdir())
    pred = np.load(result / 'pred.npy', mmap_mode='r')
    assert len(pred) == len(ds), (len(pred), len(ds))
    sums = np.zeros(4, dtype=np.float64)
    count = 0
    for start in range(0, len(ds), 128):
        end = min(start+128,len(ds))
        true = np.stack([ds.data_y[i+96:i+96+horizon] for i in range(start,end)]).astype(np.float32)
        naive = np.stack([ds.data_x[i+95] for i in range(start,end)]).astype(np.float32)[:,None,:]
        err = np.asarray(pred[start:end]) - true
        berr = naive - true
        sums += [np.square(err).sum(dtype=np.float64), np.abs(err).sum(dtype=np.float64), np.square(berr).sum(dtype=np.float64), np.abs(berr).sum(dtype=np.float64)]
        count += true.size
    mse,mae,bmse,bmae = sums/count
    row = dict(dataset=name,seq_len=96,pred_len=horizon,test_windows=len(ds),mse=mse,mae=mae,rmse=float(np.sqrt(mse)),naive_mse=bmse,naive_mae=bmae,mse_improvement_pct=100*(bmse-mse)/bmse)
    (run / 'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    # Save one reproducible forecast window in original units.
    idx = 0
    true = ds.data_y[idx+96:idx+96+horizon]
    original_true = ds.inverse_transform(true)
    original_pred = ds.inverse_transform(np.asarray(pred[idx]))
    pd.DataFrame({'step':np.arange(1,horizon+1),'OT_true':original_true[:,-1],'OT_prediction':original_pred[:,-1]}).to_csv(run/'OT_example.csv',index=False)
    return row

def main():
    prepare()
    metadata = {'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'gpu':torch.cuda.get_device_name(0),'seed':2025,'features':'M','scale':'StandardScaler fitted on training only','test_drop_last':False,'source':r'D:\Project\2023年4月-2025年4月\XLinear-main\XLinear-main','data':{}}
    for name in ['ETTh1','ETTm2']:
        p=DATA/(name+'.csv')
        frame=pd.read_csv(p)
        metadata['data'][name]={'rows':len(frame),'missing':int(frame.isna().sum().sum()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'start':str(frame.iloc[0,0]),'end':str(frame.iloc[-1,0])}
    (BASE/'environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    rows=[]
    for name in ['ETTh1','ETTm2']:
        script=(SRC / 'script/multi_forcatsing' / (name.lower()+'.sh')).read_text(encoding='utf-8')
        blocks=re.split(r'python -u run_longExp.py',script)[1:]
        for block in blocks:
            args=shlex.split(block.replace('\\\n',' '))
            variables={'$random_seed':'2025','$root_path_name':str(DATA),'$data_path_name':name+'.csv','$model_name':'XLinear','$data_name':name}
            args=[variables.get(x,x) for x in args]
            horizon=int(args[args.index('--pred_len')+1])
            run=BASE / f'{name}_{horizon}'
            run.mkdir(exist_ok=True)
            cmd=[sys.executable,'-B','-u',str(SRC/'run_longExp.py'),*args,'--num_workers','0','--freq','h' if name=='ETTh1' else 't']
            (run/'command.json').write_text(json.dumps(cmd,indent=2),encoding='utf-8')
            started=time.time()
            if not (run/'metrics.json').exists():
                print(f'START {name} horizon={horizon}',flush=True)
                with (run/'train.log').open('w',encoding='utf-8') as log:
                    completed=subprocess.run(cmd,cwd=run,stdout=log,stderr=subprocess.STDOUT)
                if completed.returncode:
                    print((run/'train.log').read_text(encoding='utf-8')[-5000:],flush=True)
                    raise RuntimeError(f'{name} {horizon} failed')
                row=evaluate(name,horizon,run)
                row['elapsed_seconds']=time.time()-started
                (run/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            else:
                row=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
            rows.append(row)
            pd.DataFrame(rows).to_csv(BASE/'summary.csv',index=False,encoding='utf-8-sig')
            print('DONE '+json.dumps(row),flush=True)
    print('ALL COMPLETE',flush=True)

if __name__=='__main__':
    main()
