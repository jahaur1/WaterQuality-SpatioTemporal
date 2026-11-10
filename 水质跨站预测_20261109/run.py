"""Water-quality LoRA workflow with four historical inter-station messages."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REFERENCE = ROOT/"Chronos2_LoRA水质24小时评估_20261109"
spec = importlib.util.spec_from_file_location("water_lora", REFERENCE/"run.py")
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
CACHE = local.CACHE

def package(history,easy,truth,spatial,rows):
    # Required Chronos order: targets, past-only covariates, known-future covariates.
    packed=torch.full((len(rows),13,38),torch.nan)
    packed[:,:4,:32]=torch.from_numpy(history[rows,:,5:].transpose(0,2,1))
    packed[:,4:8,:32]=torch.from_numpy(spatial[rows].transpose(0,2,1))
    packed[:,8:,:32]=torch.from_numpy(history[rows,:,:5].transpose(0,2,1))
    packed[:,:4,-1]=torch.tensor(truth[rows],dtype=torch.float32)
    packed[:,8:,-1]=torch.from_numpy(easy[rows])
    dummy=torch.full((13,6),torch.nan)
    items=[dict(context=x,future_covariates=dummy,n_targets=4,n_covariates=9,n_future_covariates=5) for x in packed]
    dataset=local.Chronos2Dataset(items,context_length=32,prediction_length=6,batch_size=416,output_patch_size=16,min_past=32,mode="train")
    for k in (0,len(rows)//2,len(rows)-1):
        context,y,f,_=dataset._construct_slice(k)
        torch.testing.assert_close(context,packed[k,:,:32],equal_nan=True)
        torch.testing.assert_close(y[:4,-1],packed[k,:4,-1])
        torch.testing.assert_close(f[8:,-1],packed[k,8:,-1])
        assert torch.isnan(f[:8]).all() and torch.isnan(y[4:]).all()
        assert torch.isnan(y[:,:5]).all() and torch.isnan(f[:,:5]).all()
    return items

def train(stage,items,validation=None,stop_at=None):
    folder=HERE/stage
    folder.mkdir(exist_ok=True)
    local.set_seed(local.SEED)
    pipeline=local.base_model()
    kwargs=dict(eval_steps=100,save_steps=100) if validation is not None else {}
    pipeline.fit(inputs=items,validation_inputs=validation,prediction_length=6,min_past=32,context_length=32,
                 finetune_mode="lora",num_steps=500,batch_size=416,learning_rate=1e-5,output_dir=folder,
                 seed=local.SEED,data_seed=local.SEED,logging_steps=100,disable_tqdm=True,bf16=False,tf32=False,
                 callbacks=[local.Audit(folder,stop_at)],**kwargs)
    assert (folder/"finetuned-ckpt/adapter_model.safetensors").exists()
    del pipeline
    torch.cuda.empty_cache()

def predict(pipeline,history,easy,rows,spatial):
    output=[]
    for start in range(0,len(rows),128):
        rr=rows[start:start+128]
        inputs=local.original.make_inputs(history[rr],easy[rr],True)
        for row,item in zip(rr,inputs):
            if spatial is not None:
                for j in range(4):
                    item["past_covariates"][f"neighbor_{j}"]=spatial[row,:,j].copy()
        cols=13 if spatial is not None else 9
        pred=pipeline.predict(inputs,prediction_length=6,batch_size=128*cols,context_length=32,cross_learning=False)
        output.extend(x[:,pipeline.quantiles.index(.5),5].float().cpu().numpy() for x in pred)
    out=np.stack(output)
    assert np.isfinite(out).all()
    return out

def main():
    torch.set_num_threads(2)
    z=np.load(CACHE/"inputs.npz");history=z["history"];easy=z["future_easy"];truth=z["truth"]
    idx=pd.read_csv(CACHE/"配对索引.csv")
    splits={k:np.flatnonzero(idx[k]) for k in ("selection_train","validation","refit_train","test")}
    previous=np.load(REFERENCE/"predictions.npz")
    np.testing.assert_array_equal(splits["test"],previous["rows"])
    np.testing.assert_array_equal(truth[splits["test"]],previous["truth"])
    assert local.sha(local.BASE/"weights/model.safetensors")==json.loads((REFERENCE/"核验.json").read_text(encoding="utf-8"))["base_sha256"]
    graph=np.load(HERE/"selection_邻站历史.npy",mmap_mode="r")
    train("selection",package(history,easy,truth,graph,splits["selection_train"]),package(history,easy,truth,graph,splits["validation"]))
    log=json.loads((HERE/"selection/training.json").read_text(encoding="utf-8"))
    best=int(Path(log["best_checkpoint"]).name.split("-")[-1])
    print("Selected",best,flush=True)
    graph=np.load(HERE/"final_邻站历史.npy",mmap_mode="r")
    train("final",package(history,easy,truth,graph,splits["refit_train"]),stop_at=best)
    assert json.loads((HERE/"final/training.json").read_text(encoding="utf-8"))["steps"]==best
    base=local.base_model()
    peft_model=local.peft.PeftModel.from_pretrained(base.model,HERE/"final/finetuned-ckpt",is_trainable=False)
    pipeline=local.Chronos2Pipeline(model=peft_model.merge_and_unload().eval())
    test=splits["test"]
    spatial=predict(pipeline,history,easy,test,graph)
    print("Spatial test finished",flush=True)
    ablated=predict(pipeline,history,easy,test,None)
    np.testing.assert_allclose(predict(pipeline,history,easy,test[:128],graph),spatial[:128],rtol=1e-5,atol=1e-6)
    np.savez_compressed(HERE/"predictions.npz",rows=test,truth=truth[test],local=previous["raw"],
                        spatial=spatial,ablated=ablated,persistence=z["last_chemical"][test])
    local.dump(HERE/"训练核验.json",dict(passed=True,best_steps=best,test_pairs=len(test),same_test_rows_and_truth=True,
               trainable=json.loads((HERE/"final/trainable.json").read_text(encoding="utf-8"))["parameters"],
               same_windows_per_batch=32,graph_history_only=True,test_replay_passed=True))

if __name__=="__main__":
    main()
