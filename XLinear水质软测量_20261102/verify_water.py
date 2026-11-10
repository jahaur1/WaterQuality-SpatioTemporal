"""Independent, read-only audit of completed XLinear water soft-sensor outputs.

Only writes the requested audit JSON in the existing results directory.
Does not import the training runner, execute optimization, or initialize CUDA.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from adapter import XLinearSoftSensor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASE=ROOT/'30天_750站点数据集_20250410-20250510'
SOURCE=BASE/'水质数据_750站点_20250410-20250510.csv'
RIDGE=BASE/'测5项估4项_时间留出预测基线'/'02_测试集实测与预测值.csv'
XCOL=['水温(℃)','pH(无量纲)','溶解氧(mg/L)','电导率(μS/cm)','浊度(NTU)']
YCOL=['高锰酸盐指数(mg/L)','氨氮(mg/L)','总磷(mg/L)','总氮(mg/L)']
KEY=['站点ID','监测时间']
VAL=pd.Timestamp('2025-04-25')
TEST=pd.Timestamp('2025-04-30')
END=pd.Timestamp('2025-05-10')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loadjson(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(condition,message):
    if not condition:raise AssertionError(message)
def sorted_index(frame):return frame.set_index(KEY).sort_index()
def metrics_from_arrays(y,p):
    error=p-y
    sst=float(np.dot(y-y.mean(),y-y.mean()))
    sse=float(np.dot(error,error))
    tolerance=1e-12*max(1.,float(np.dot(y,y)))
    return {'MAE':float(np.mean(np.abs(error))),
            'RMSE':float(np.sqrt(np.mean(error*error))),
            '偏差':float(np.mean(error)),
            'R2':float(1.-sse/sst) if sst>tolerance else np.nan}

def verify(results):
    started=time.time()
    cfg=loadjson(results/'config.json')
    summary=loadjson(results/'summary.json')
    conf=cfg['config']
    require(conf['inputs']==XCOL and conf['targets']==YCOL,'Input or target whitelist differs')
    require(conf['seq_len']==6,'Expected six observed easy-variable records')
    require(conf['val_start']==str(VAL) and conf['test_start']==str(TEST) and conf['test_end']==str(END),'Date split differs')
    require(conf['refit_pretest'] is True,'Final pre-test refit is required')
    require(summary['站点数']==750 and summary['目标指标评价数量']==3000 and summary['测试记录数']==27326,'Incomplete full-run summary')

    fingerprint=hashlib.sha256(json.dumps({'config':conf,**{k:cfg[k] for k in ['source_dataset_sha256','adapter_sha256','vendor_sha256','runner_sha256']}},sort_keys=True).encode()).hexdigest()
    require(fingerprint==cfg['fingerprint']==summary['fingerprint'],'Experiment fingerprint mismatch')
    digests={
        'source_dataset_sha256':sha(SOURCE),
        'adapter_sha256':sha(HERE/'adapter.py'),
        'vendor_sha256':sha(HERE/'vendor_XLinear.py'),
        'runner_sha256':sha(HERE/'run_water.py'),
    }
    for k,value in digests.items():require(value==cfg[k],f'Changed recorded source: {k}')
    require(digests['source_dataset_sha256']=='9267d65df4c67f934339812b9fdea839696594c8c55af6b8f6fefb445d3cb7eb','Dataset differs from the verified 750-station source')
    provenance=loadjson(HERE/'source_provenance.json')
    original_copy=ROOT/provenance['source_relative_path']
    require(sha(original_copy)==provenance['source_sha256']==provenance['vendor_sha256']==digests['vendor_sha256'],'Vendor differs from recorded original XLinear source')
    original_repository=ROOT.parent/'XLinear-main'/'XLinear-main'/'models'/'XLinear.py'
    repo_check={'path':str(original_repository),'exists':original_repository.exists()}
    if original_repository.exists():
        repo_check['sha256']=sha(original_repository)
        require(repo_check['sha256']==digests['vendor_sha256'],'Vendor differs from user original repository source')

    source=pd.read_csv(SOURCE)
    source['监测时间']=pd.to_datetime(source['监测时间'])
    require(len(source)==79940 and source['站点ID'].nunique()==750,'Source sizes differ')
    require(not source.duplicated(KEY).any(),'Duplicate source keys')
    expected=source[(source['监测时间']>=TEST)&(source['监测时间']<=END)].copy()
    pred=pd.read_csv(results/'02_测试实测与预测.csv')
    ridge=pd.read_csv(RIDGE)
    pred['监测时间']=pd.to_datetime(pred['监测时间'])
    ridge['监测时间']=pd.to_datetime(ridge['监测时间'])
    require(len(pred)==len(ridge)==len(expected)==27326,'Test record totals differ')
    for name,frame in [('XLinear',pred),('ridge',ridge),('source test',expected)]:
        require(not frame.duplicated(KEY).any(),f'Duplicate keys in {name}')
        require(frame['站点ID'].nunique()==750,f'Missing stations in {name}')
    px=sorted_index(pred);rx=sorted_index(ridge);sx=sorted_index(expected)
    require(px.index.equals(rx.index) and px.index.equals(sx.index),'Test keys differ between XLinear, ridge, and source')
    for target in YCOL:
        xtrue=px[target+'_实测'].to_numpy(float)
        require(np.array_equal(xtrue,rx[target+'_实测'].to_numpy(float)),f'XLinear/ridge truth differs: {target}')
        require(np.array_equal(xtrue,sx[target].to_numpy(float)),f'XLinear/source truth differs: {target}')
        require(np.isfinite(px[target+'_预测'].to_numpy(float)).all(),f'Nonfinite prediction: {target}')
        require((px[target+'_预测']>=0).all(),f'Negative postprocessed prediction: {target}')
    require(np.array_equal(px[XCOL].to_numpy(float),sx[XCOL].to_numpy(float)),'Saved current easy inputs differ from source')

    table=pd.read_csv(results/'01_逐站点四目标误差.csv')
    require(len(table)==3000 and not table.duplicated(['站点ID','目标变量']).any(),'Metric table is not 750 unique stations x four targets')
    require(set(table['目标变量'])==set(YCOL),'Metric targets differ')
    require(table.groupby('站点ID').size().eq(4).all(),'Not four targets for each station')
    require((table['RMSE']+1e-9>=table['MAE']).all(),'RMSE below MAE')
    table=table.set_index(['站点ID','目标变量'])

    all_ids=sorted(source['站点ID'].unique())
    done_ids=sorted(p.parent.name for p in (results/'stations').glob('*/done.json'))
    require(done_ids==all_ids,'Station completion directories differ from source')
    selected_cpu=[all_ids[i] for i in [0,187,374,561,749]]
    cpu_checks=[]
    checked_scalers=0;checked_windows=0;checked_metrics=0
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass

    for si,(sid,g) in enumerate(source.groupby('站点ID')):
        g=g.sort_values('监测时间').reset_index(drop=True)
        dest=results/'stations'/sid
        meta=loadjson(dest/'done.json')
        require(meta['station']==sid and meta['fingerprint']==fingerprint,f'{sid}: metadata fingerprint differs')
        times=g['监测时间']
        require((times.diff().dropna()>pd.Timedelta(0)).all(),f'{sid}: nonmonotone times')
        endpoints=np.arange(5,len(g))
        wi=np.stack([np.arange(i-5,i+1) for i in endpoints])
        require((wi[:,-1]==endpoints).all() and (wi.max(1)==endpoints).all(),f'{sid}: bad endpoint indices')
        wtimes=times.iloc[endpoints].reset_index(drop=True)
        n_fit=int((wtimes<VAL).sum());n_val=int(((wtimes>=VAL)&(wtimes<TEST)).sum())
        n_refit=int((wtimes<TEST).sum());testmask=(wtimes>=TEST)&(wtimes<=END)
        for k,value in [('n_raw_observations',len(g)),('n_selection_train_windows',n_fit),('n_validation_windows',n_val),('n_refit_windows',n_refit),('n_test_windows',int(testmask.sum())),('n_window_coldstart_dropped',5)]:
            require(meta[k]==value,f'{sid}: {k} differs')
        require(n_fit>0 and n_val>0 and int(testmask.sum())>0,f'{sid}: empty partition')
        require(wtimes[wtimes<VAL].max()<wtimes[(wtimes>=VAL)&(wtimes<TEST)].min()<wtimes[testmask].min(),f'{sid}: label partitions overlap')

        for prefix,boundary in [('selection',VAL),('final',TEST)]:
            training=g[times<boundary]
            require(meta[prefix+'_scaler_latest_time']==str(training['监测时间'].max()),f'{sid}: {prefix} scaler date metadata differs')
            require(pd.Timestamp(meta[prefix+'_scaler_latest_time'])<boundary,f'{sid}: scaler includes validation/test boundary')
            for label,cols in [('x',XCOL),('y',YCOL)]:
                scaler=StandardScaler().fit(training[cols].to_numpy(float))
                recorded=meta[prefix+'_'+label+'_scaler']
                require(recorded['n_samples']==len(training),f'{sid}: wrong scaler row count')
                np.testing.assert_allclose(recorded['mean'],scaler.mean_,rtol=1e-10,atol=1e-10,err_msg=f'{sid}: scaler mean')
                np.testing.assert_allclose(recorded['scale'],scaler.scale_,rtol=1e-9,atol=1e-10,err_msg=f'{sid}: scaler std')
                checked_scalers+=1

        sg=pred[pred['站点ID']==sid].sort_values('监测时间').reset_index(drop=True)
        testindices=endpoints[testmask]
        source_test=g.iloc[testindices].reset_index(drop=True)
        require(sg['监测时间'].equals(source_test['监测时间']),f'{sid}: station test times differ')
        actual_starts=pd.to_datetime(sg['输入窗口起点'])
        expected_starts=times.iloc[testindices-5].reset_index(drop=True)
        require(actual_starts.equals(expected_starts),f'{sid}: window start is not five preceding observations')
        require((actual_starts<=sg['监测时间']).all(),f'{sid}: future window start')
        expected_span=(sg['监测时间']-expected_starts)/pd.Timedelta(hours=1)
        np.testing.assert_allclose(sg['输入窗口跨度小时'],expected_span,rtol=1e-10,atol=1e-10)
        require(np.array_equal(sg[XCOL].to_numpy(float),source_test[XCOL].to_numpy(float)),f'{sid}: input values differ')
        checked_windows+=len(sg)

        hist=pd.read_csv(dest/'history.csv')
        selection=hist[hist['阶段']=='训练验证选epoch'];refit=hist[hist['阶段']=='前20天重新拟合']
        require(len(selection)==meta['epochs_ran_selection'] and len(refit)==meta['best_epoch']==meta['epochs_ran_refit'],f'{sid}: epoch counts differ')
        bestrow=selection.loc[selection['验证标准化MSE'].idxmin()]
        require(int(bestrow['epoch'])==meta['best_epoch'],f'{sid}: best epoch not minimum validation error')
        np.testing.assert_allclose(bestrow['验证标准化MSE'],meta['validation_best_standardized_mse'],rtol=1e-9,atol=1e-10)

        for target in YCOL:
            actual=metrics_from_arrays(sg[target+'_实测'].to_numpy(float),sg[target+'_预测'].to_numpy(float))
            recorded=table.loc[(sid,target)]
            require(recorded['测试窗口数']==len(sg),f'{sid}: metric test N differs')
            for key,value in actual.items():
                np.testing.assert_allclose(value,recorded[key],rtol=1e-6,atol=1e-7,equal_nan=True,err_msg=f'{sid}/{target}: {key}')
            checked_metrics+=1

        if sid in selected_cpu:
            checkpoint=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
            require(checkpoint['station']==sid and checkpoint['best_epoch']==meta['best_epoch'],f'{sid}: checkpoint metadata differs')
            require(checkpoint['XCOL']==XCOL and checkpoint['YCOL']==YCOL,f'{sid}: checkpoint columns differ')
            require(checkpoint['config']==conf,f'{sid}: checkpoint config differs')
            model=XLinearSoftSensor(seq_len=conf['seq_len'],d_model=conf['d_model'],t_ff=16,c_ff=16,dropout=conf['dropout']).cpu().eval()
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            xs=checkpoint['x_scaler'];ys=checkpoint['y_scaler']
            for label in ['x','y']:
                require(checkpoint[label+'_scaler']==meta['final_'+label+'_scaler'],f'{sid}: checkpoint scaler differs')
            scaled=(g[XCOL].to_numpy(float)-np.array(xs['mean']))/np.array(xs['scale'])
            windows=scaled[wi[testmask]]
            require(windows.shape==(len(sg),6,5),f'{sid}: inference shape differs')
            with torch.inference_mode():replayed_z=model(torch.tensor(windows,dtype=torch.float32,device='cpu')).numpy()
            stored_raw=sg[[t+'_未截断预测' for t in YCOL]].to_numpy(float)
            stored_z=(stored_raw-np.array(ys['mean']))/np.array(ys['scale'])
            # CPU and CUDA float32 accumulation may differ slightly; compare standardized outputs.
            np.testing.assert_allclose(replayed_z,stored_z,rtol=1e-4,atol=1e-4,err_msg=f'{sid}: CPU replay differs from stored CUDA outputs')
            replayed_raw=replayed_z*np.array(ys['scale'])+np.array(ys['mean'])
            saved_post=sg[[t+'_预测' for t in YCOL]].to_numpy(float)
            np.testing.assert_allclose(saved_post,np.maximum(stored_raw,0),rtol=1e-10,atol=1e-10,err_msg=f'{sid}: clipping differs')
            cpu_checks.append({'station':sid,'test_windows':len(sg),'max_abs_standardized_difference':float(np.abs(replayed_z-stored_z).max()),'max_abs_original_unit_difference':float(np.abs(replayed_raw-stored_raw).max()),'comparison_rtol':1e-4,'comparison_atol_standardized':1e-4,'passed':True})
        if si%150==0:print(f'Independent audit: {si+1}/750 stations',flush=True)

    require(checked_scalers==3000 and checked_windows==27326 and checked_metrics==3000,'Audit totals differ')
    require(len(cpu_checks)==5,'CPU replay station count differs')
    return {'passed':True,'audit_scope':'Independent CPU-only checks; no training and no original/result data rewritten',
            'source_rows':len(source),'station_count':750,'metric_rows':len(table),'test_rows':len(pred),
            'same_test_keys_as_ridge':True,'exact_same_four_target_truth_as_ridge_and_source':True,
            'five_current_input_values_match_source':True,'audited_scalers':checked_scalers,
            'scalers_refitted_independently_from_all_station_training_data':True,
            'strict_selection_and_final_scaler_boundaries_passed':True,
            'test_windows_with_same_station_six_observations_ending_at_current_time':checked_windows,
            'no_future_or_cross_station_windows':True,
            'validation_best_epoch_and_fresh_refit_epoch_counts_match_saved_history':True,
            'metric_groups_recomputed_from_saved_predictions':checked_metrics,
            'cpu_checkpoint_replay':cpu_checks,'hashes':digests,
            'vendor_equals_recorded_original_source':True,'original_repository_check':repo_check,
            'elapsed_seconds':round(time.time()-started,3),
            'limitations':['This audit verifies saved data, scaler provenance, window construction, epoch-selection records, and CPU inference; it does not retrain the neural networks.',
                           'Fresh model/optimizer initialization and exclusion of test losses from training choices are established by the separate code review, not by saved checkpoints alone.',
                           'Six observations are not necessarily six hours; tests remain within the same previously observed stations.']}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results')
    args=parser.parse_args()
    results=(HERE/args.results).resolve()
    require(results.is_relative_to(HERE.resolve()) and results.exists(),'Results must be an existing directory under this experiment')
    output=results/'独立验证.json'
    try:
        report=verify(results)
    except Exception as exc:
        output.write_text(json.dumps({'passed':False,'error':repr(exc),'traceback':traceback.format_exc()},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
