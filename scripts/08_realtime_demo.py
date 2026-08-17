"""08 — Real-time conflict early-warning prototype (replay + SSE + SHAP).

Demonstrates the streaming architecture from REALTIME_SYSTEM_SPEC.md:
  ingestion(replay) -> feature store -> inference(XGBoost) -> SHAP -> SSE -> dashboard.

Reuses src/conflict_ew: panel build, Kalman features, trained XGBoost.
Zero new dependencies beyond the managed venv (stdlib http.server + shap/xgboost).

Usage:
  python scripts/08_realtime_demo.py            # serves http://127.0.0.1:8765
  python scripts/08_realtime_demo.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.config import load_config  # noqa: E402
from conflict_ew.data import load_raw, build_panel  # noqa: E402
from conflict_ew.features.factors import build_features  # noqa: E402
from conflict_ew.eval import metrics as M  # noqa: E402

import xgboost as xgb  # noqa: E402
import shap  # noqa: E402

ART = Path("artifacts")
MODEL_PATH = ART / "model_xgb.pkl"
FEATURES = ["e_star", "log_level", "growth", "volatility", "slope", "neighbor_e"]


def train_model(cfg, panel_feat):
    """Train XGBoost on the pre-test period; persist for online serving."""
    test_start = int(cfg["model"]["test_start"])
    train = panel_feat[panel_feat["year"] < test_start]
    test = panel_feat[panel_feat["year"] >= test_start]
    Xtr = train[FEATURES].astype(float).values
    ytr = train["label"].astype(int).values
    mp = cfg["model"]["params"]
    pos = max(1, int(ytr.sum()))
    neg = len(ytr) - pos
    spw = (neg / pos) if pos else 1.0
    model = xgb.XGBClassifier(
        max_depth=mp["max_depth"],
        learning_rate=mp["learning_rate"],
        n_estimators=mp["n_estimators"],
        subsample=mp["subsample"],
        colsample_bytree=mp["colsample_bytree"],
        reg_lambda=mp["reg_lambda"],
        min_child_weight=mp["min_child_weight"],
        scale_pos_weight=spw,
        random_state=int(cfg["reproducibility"]["seed"]),
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(Xtr, ytr)
    joblib.dump(model, MODEL_PATH)

    # genuine out-of-sample stats on the held-out test period
    if len(test) and test["label"].nunique() >= 2:
        Xte = test[FEATURES].astype(float).values
        pte = model.predict_proba(Xte)[:, 1]
        auc, ci = M.auc_delong_ci(test["label"].values, pte)
        brier = float(M.brier(test["label"].values, pte))
    else:
        auc, ci, brier = float("nan"), (float("nan"), float("nan")), float("nan")
    return model, {"test_n": int(len(test)), "auc": auc, "auc_ci95": list(ci), "brier": brier}


class Broadcaster:
    """Minimal pub-sub for SSE clients."""

    def __init__(self):
        self._subs = []
        self._lock = threading.Lock()

    def subscribe(self):
        q: deque = deque()
        with self._lock:
            self._subs.append(q)
        return q

    def publish(self, msg: dict):
        with self._lock:
            for q in self._subs:
                q.append(msg)


def build_timeline(panel_feat, cfg):
    """Virtual clock: one tick per (year) in the test period, per country."""
    test_start = int(cfg["model"]["test_start"])
    test = panel_feat[panel_feat["year"] >= test_start].copy()
    ticks = []
    for year in sorted(test["year"].unique()):
        sub = test[test["year"] == year]
        for _, row in sub.iterrows():
            ticks.append(
                {
                    "country": row["country"],
                    "year": int(row["year"]),
                    "features": {f: float(row[f]) for f in FEATURES},
                    "label": int(row["label"]),
                }
            )
    return ticks


def replay_loop(broadcaster, model, explainer, ticks, model_stats, interval=0.4):
    """Push each tick as a streaming event; run in a background thread."""
    broadcaster.publish({"type": "model", **model_stats})
    time.sleep(0.5)
    for i, tick in enumerate(ticks):
        X = pd.DataFrame([tick["features"]])[FEATURES]
        p = float(model.predict_proba(X)[:, 1][0])
        sv = explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        shap_vec = {f: float(sv[0][j]) for j, f in enumerate(FEATURES)}
        top = sorted(shap_vec.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        broadcaster.publish(
            {
                "type": "tick",
                "seq": i + 1,
                "total": len(ticks),
                "country": tick["country"],
                "year": tick["year"],
                "p": round(p, 4),
                "actual_conflict": tick["label"],
                "top_factors": [{"factor": k, "shap": round(v, 4)} for k, v in top],
                "all_shap": {k: round(v, 4) for k, v in shap_vec.items()},
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        time.sleep(interval)
    broadcaster.publish({"type": "done"})


# ---------------------------------------------------------------------------
# HTTP server (stdlib, SSE)
# ---------------------------------------------------------------------------
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    broadcaster: Broadcaster
    model = None
    explainer = None

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (Path(__file__).resolve().parent.parent / "dashboard_rt.html").read_text(encoding="utf-8")
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/stream":
            self._sse()
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        q = Handler.broadcaster.subscribe()
        try:
            self.wfile.write(f"retry: 1000\n\n".encode("utf-8"))
            while True:
                if q:
                    msg = q.popleft()
                    payload = json.dumps(msg, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                else:
                    time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_POST(self):
        if self.path == "/score":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw)
                feats = payload.get("features", {})
                X = pd.DataFrame([{f: float(feats.get(f, 0.0)) for f in FEATURES}])[FEATURES]
                p = float(Handler.model.predict_proba(X)[:, 1][0])
                sv = Handler.explainer.shap_values(X)
                sv = sv[1] if isinstance(sv, list) else sv
                shap_vec = {f: round(float(sv[0][j]), 4) for j, f in enumerate(FEATURES)}
                self._send(200, json.dumps({"p": round(p, 4), "shap": shap_vec}, ensure_ascii=False))
            except Exception as e:  # pragma: no cover
                self._send(400, json.dumps({"error": str(e)}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, *a):  # silence default logging
        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--interval", type=float, default=0.4, help="seconds between stream ticks")
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    print("[rt] loading data + building features ...")
    raw = load_raw(cfg)
    panel = build_panel(raw)
    panel_feat, _, _ = build_features(raw, panel, cfg)
    print(f"[rt] panel: {len(panel_feat)} rows, features={FEATURES}")

    print("[rt] training XGBoost (pre-test period) ...")
    model, stats = train_model(cfg, panel_feat)
    print(f"[rt] OOS test: n={stats['test_n']} AUC={stats['auc']:.3f} "
          f"CI95={stats['auc_ci95']} Brier={stats['brier']:.4f}")

    explainer = shap.TreeExplainer(model)

    broadcaster = Broadcaster()
    Handler.broadcaster = broadcaster
    Handler.model = model
    Handler.explainer = explainer

    ticks = build_timeline(panel_feat, cfg)
    print(f"[rt] timeline: {len(ticks)} streaming events (test period)")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(
        target=replay_loop, args=(broadcaster, model, explainer, ticks, stats, args.interval),
        daemon=True,
    ).start()

    print(f"[rt] serving at http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rt] stopped.")


if __name__ == "__main__":
    main()
