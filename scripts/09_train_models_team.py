from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhl-root", default=os.getenv("NHL_ROOT", "./data/NHL"))
    args = ap.parse_args()

    nhl_root = args.nhl_root
    data_dir = os.path.join(nhl_root, "model_data")
    out_dir  = os.path.join(nhl_root, "models")
    os.makedirs(out_dir, exist_ok=True)

    team_table = os.path.join(data_dir, "team_model_table.parquet")
    df = pd.read_parquet(team_table)
    print("✅ Loaded team_model_table:", df.shape)

    y_cols_all = [c for c in df.columns if c.startswith("y_")]
    feat_cols_all = [c for c in df.columns if c.startswith("team_")]

    numeric_feat_cols = []
    non_numeric_feat_cols = []
    for c in feat_cols_all:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_feat_cols.append(c)
        else:
            non_numeric_feat_cols.append(c)

    print("✅ team_ feature columns:", len(feat_cols_all))
    print("✅ numeric team_ features used:", len(numeric_feat_cols))
    if non_numeric_feat_cols:
        print("⚠️ Dropping non-numeric team_ cols:", non_numeric_feat_cols)

    feat_cols = numeric_feat_cols

    df = df.dropna(subset=["season"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["season"]).copy()

    holdout_season = int(df["season"].max())
    print("Holdout season:", holdout_season)

    candidates = {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=5.0, random_state=42))
        ]),
        "hgb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=400,
                random_state=42
            ))
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]),
    }

    report_rows = []
    for y in y_cols_all:
        d = df.dropna(subset=[y]).copy()
        train_df = d[d["season"] < holdout_season].copy()
        test_df  = d[d["season"] == holdout_season].copy()

        if len(test_df) == 0 or len(train_df) == 0:
            print(f"⚠️ Skipping {y}: train or test empty")
            continue

        X_train = train_df[feat_cols]
        y_train = train_df[y].astype(float)
        X_test  = test_df[feat_cols]
        y_test  = test_df[y].astype(float)

        best = None
        best_mae = 1e18

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae, rmse = eval_metrics(y_test, pred)

            report_rows.append({
                "target": y, "model": name, "holdout_season": holdout_season,
                "MAE": mae, "RMSE": rmse,
                "n_train": len(train_df), "n_test": len(test_df),
                "n_features": len(feat_cols)
            })

            if mae < best_mae:
                best_mae = mae
                best = (name, model, mae, rmse)

        best_name, best_model, best_mae, best_rmse = best
        out_path = os.path.join(out_dir, f"team_{y}.joblib")
        dump(best_model, out_path)
        print(f"✅ Saved best for {y}: {best_name} | MAE={best_mae:.4f} RMSE={best_rmse:.4f} -> {out_path}")

    report = pd.DataFrame(report_rows).sort_values(["target","MAE"]).reset_index(drop=True)
    rep_path = os.path.join(out_dir, "team_model_report.csv")
    report.to_csv(rep_path, index=False)
    print("✅ Saved report:", rep_path)
    print(report.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
