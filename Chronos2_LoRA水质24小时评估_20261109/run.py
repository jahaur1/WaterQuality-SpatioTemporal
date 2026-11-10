"""Official Chronos-2 LoRA on the existing exact +24h water-quality pairs."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "Chronos2水质24小时评估_20261103"
CACHE = HERE.parent / "冻结预训练模型校正头基线_20261106"
sys.path[:0] = [str(HERE / "deps"), str(BASE / "runtime")]
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import peft
from chronos import Chronos2Pipeline
from chronos.chronos2.dataset import Chronos2Dataset
from transformers import TrainerCallback, set_seed

spec = importlib.util.spec_from_file_location("original_chronos_inference", BASE / "run.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
SEED = 20261109
MAX_STEPS = 500

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def base_model():
    return Chronos2Pipeline.from_pretrained(str(BASE / "weights"), device_map="cuda", torch_dtype=torch.float32, local_files_only=True)

def examples(history, easy, truth, indices):
    """The native sampler has exactly one possible split: 32 history + 6 horizon."""
    packed = torch.full((len(indices), 9, 38), torch.nan)
    packed[:, :4, :32] = torch.tensor(history[indices, :, 5:].transpose(0, 2, 1))
    packed[:, 4:, :32] = torch.tensor(history[indices, :, :5].transpose(0, 2, 1))
    packed[:, :4, -1] = torch.as_tensor(truth[indices], dtype=torch.float32)
    packed[:, 4:, -1] = torch.tensor(easy[indices])
    placeholder = torch.full((9, 6), torch.nan)
    items = [dict(context=row, future_covariates=placeholder, n_targets=4, n_covariates=5, n_future_covariates=5) for row in packed]
    dataset = Chronos2Dataset(items, context_length=32, prediction_length=6, batch_size=288, output_patch_size=16, min_past=32, mode="train")
    for k in [0, len(indices)//2, len(indices)-1]:
        context, label, covariate, _ = dataset._construct_slice(k)
        torch.testing.assert_close(context, packed[k, :, :32], equal_nan=True)
        assert torch.isnan(label[:, :-1]).all() and torch.isnan(label[4:]).all()
        assert torch.isnan(covariate[:, :-1]).all() and torch.isnan(covariate[:4]).all()
        torch.testing.assert_close(label[:4, -1], packed[k, :4, -1])
        torch.testing.assert_close(covariate[4:, -1], packed[k, 4:, -1])
    return items

class Audit(TrainerCallback):
    def __init__(self, folder, stop_at=None):
        self.folder = folder
        self.stop_at = stop_at

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in name for name in trainable)
        dump(self.folder / "trainable.json", {"parameters": sum(trainable.values()), "names": trainable})

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.stop_at:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        dump(self.folder / "training.json", dict(steps=state.global_step, best_checkpoint=state.best_model_checkpoint, history=state.log_history))

def fit(stage, items, validation=None, stop_at=None):
    folder = HERE / stage
    folder.mkdir(exist_ok=True)
    set_seed(SEED)
    base = base_model()
    kwargs = dict(eval_steps=100, save_steps=100) if validation is not None else {}
    base.fit(inputs=items, validation_inputs=validation, prediction_length=6, min_past=32, context_length=32,
             finetune_mode="lora", num_steps=MAX_STEPS, batch_size=288, learning_rate=1e-5, output_dir=folder,
             seed=SEED, data_seed=SEED, logging_steps=50, disable_tqdm=True, bf16=False, tf32=False,
             callbacks=[Audit(folder, stop_at)], **kwargs)
    assert (folder / "finetuned-ckpt/adapter_model.safetensors").is_file()
    del base
    torch.cuda.empty_cache()

def reload_adapter(stage):
    base = base_model()
    adapter = peft.PeftModel.from_pretrained(base.model, HERE / stage / "finetuned-ckpt", is_trainable=False)
    return Chronos2Pipeline(model=adapter.merge_and_unload().eval())

def predict(pipeline, history, easy, rows):
    chunks = []
    for start in range(0, len(rows), 128):
        batch = rows[start:start+128]
        chunks.append(original.forecast(pipeline, history[batch], easy[batch], True))
    return np.concatenate(chunks)

def main():
    torch.set_num_threads(2)
    z = np.load(CACHE / "inputs.npz")
    history, easy, y = z["history"], z["future_easy"], z["truth"]
    idx = pd.read_csv(CACHE / "配对索引.csv")
    splits = {key: np.flatnonzero(idx[key]) for key in ["selection_train", "validation", "refit_train", "test"]}
    assert [len(a) for a in splits.values()] == [27130, 9107, 38171, 23136]
    targets = pd.to_datetime(idx["目标时间"], format="mixed")
    origins = pd.to_datetime(idx["起报时间"], format="mixed")
    assert targets.iloc[splits["selection_train"]].max() < origins.iloc[splits["validation"]].min()
    assert targets.iloc[splits["refit_train"]].max() < origins.iloc[splits["test"]].min()
    cache_cfg = json.loads((CACHE / "训练配置与头权重.json").read_text(encoding="utf-8"))
    for name in ["配对索引.csv", "inputs.npz"]:
        assert sha(CACHE / name) == cache_cfg["fingerprints"][name]
    digest = sha(BASE / "weights/model.safetensors")
    assert digest == "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"
    make = lambda key: examples(history, easy, y, splits[key])
    if not (HERE / "selection/finetuned-ckpt/adapter_model.safetensors").exists():
        fit("selection", make("selection_train"), make("validation"))
    log = json.loads((HERE / "selection/training.json").read_text(encoding="utf-8"))
    assert log["steps"] == MAX_STEPS
    best = int(Path(log["best_checkpoint"]).name.split("-")[-1])
    assert best in range(100, MAX_STEPS+1, 100)
    print("validation-selected steps:", best, flush=True)
    if not (HERE / "final/finetuned-ckpt/adapter_model.safetensors").exists():
        fit("final", make("refit_train"), stop_at=best)
    assert json.loads((HERE / "final/training.json").read_text(encoding="utf-8"))["steps"] == best
    pipeline = reload_adapter("final")
    rows = splits["test"]
    out = predict(pipeline, history, easy, rows)
    assert np.isfinite(out).all()
    replay = predict(pipeline, history, easy, rows[:128])
    np.testing.assert_allclose(replay, out[:128], rtol=1e-5, atol=1e-6)
    frozen = np.load(CACHE / "Chronos2/predictions.npz")["raw"][rows]
    previous = np.load(BASE / "results/future_easy/predictions.npz")["raw"]
    np.testing.assert_array_equal(frozen[np.searchsorted(rows, z["test_new_indices"])], previous[z["test_old_indices"]])
    assert sha(BASE / "weights/model.safetensors") == digest
    np.savez_compressed(HERE / "predictions.npz", raw=out, frozen=frozen, truth=y[rows], rows=rows)
    dump(HERE / "核验.json", dict(passed=True, base_sha256=digest, selection_steps=best,
        trainable=json.loads((HERE / "final/trainable.json").read_text())["parameters"], splits={k:len(v) for k,v in splits.items()},
        peft=peft.__version__, seed=SEED, only_lora_trainable=True, original_test_forecasts_identical=True,
        reloaded_adapter_batch_replay=True, final_adapter_sha256=sha(HERE / "final/finetuned-ckpt/adapter_model.safetensors")))

if __name__ == "__main__":
    main()
