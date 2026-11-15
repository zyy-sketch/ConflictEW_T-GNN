# -*- coding: utf-8 -*-
"""N4: 实测推理时延与运行环境（发布模型 artifacts/xgb_model.json）。

输出 artifacts/runtime_benchmark.json，供正文"实现与环境"段引用。
所有数字均为实测，无估算。
"""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import xgboost as xgb

from conflict_ew.metrics import get_factor_columns
from conflict_ew.paths import get_path

MODEL = ROOT / "artifacts" / "xgb_model.json"
feats = pd.read_csv(get_path("features_file"))
fc = get_factor_columns()
cv_cfg = None
import yaml

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
cv = cfg["model"]["cv"]; TS, TE = cv["test_start"], cv["test_end"]
te = feats[(feats["year"] >= TS) & (feats["year"] <= TE)]
X = te[fc].to_numpy(dtype=np.float32)
assert X.shape[0] == 1650, f"test rows {X.shape[0]} != 1650"

booster = xgb.Booster()
booster.load_model(str(MODEL))
feat_names = booster.feature_names
Xdf = te[feat_names]
X = Xdf.to_numpy(dtype=np.float32)
dmat = xgb.DMatrix(Xdf)

# 预热
booster.predict(dmat)

REP = 50
t = []
for _ in range(REP):
    t0 = time.perf_counter()
    booster.predict(dmat)
    t.append((time.perf_counter() - t0) * 1000)
batch_ms = np.array(t)

# 单行时延（逐行 DMatrix，50 次取中位，单行开销含 DMatrix 构造）
single = []
for i in range(0, 200, 4):
    d1 = xgb.DMatrix(Xdf.iloc[i:i+1])
    booster.predict(d1)  # warm
    t0 = time.perf_counter()
    for _ in range(20):
        booster.predict(d1)
    single.append((time.perf_counter() - t0) * 1000 / 20)
single_ms = np.array(single)

res = {
    "hardware": {"cpu": platform.processor(),
                 "platform": platform.platform(),
                 "python": sys.version.split()[0]},
    "dependencies": {"xgboost": xgb.__version__,
                     "numpy": np.__version__,
                     "pandas": pd.__version__},
    "model_file": {"path": "artifacts/xgb_model.json", "bytes": MODEL.stat().st_size},
    "feature_matrix": {"rows": int(X.shape[0]), "cols": int(X.shape[1]),
                       "bytes": int(X.nbytes)},
    "batch_inference": {"n_repeats": REP, "rows": int(X.shape[0]),
                        "mean_ms": round(float(batch_ms.mean()), 3),
                        "std_ms": round(float(batch_ms.std()), 3),
                        "per_row_us": round(float(batch_ms.mean()) * 1000 / X.shape[0], 2)},
    "single_row_inference": {"n_probes": int(len(single_ms)), "repeats_each": 20,
                             "median_ms": round(float(np.median(single_ms)), 4),
                             "p90_ms": round(float(np.percentile(single_ms, 90)), 4)},
    "measured_at": __import__("datetime").date.today().isoformat(),
}
out = Path(__file__).resolve().parent.parent / "artifacts" / "runtime_benchmark.json"
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, indent=1))
print("saved:", out)
