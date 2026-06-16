"""
Train a Ridge regression model to predict game totals.
Also trains XGBoost for comparison.

Walk-forward validation: train on seasons N through Y-1, evaluate on season Y.
Outputs:
  data/processed/predictions.csv  -- game_pk, actual, predicted, close_total
  models are printed with feature importances
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

FEATURE_COLS = [
    # SP stats
    "home_sp_fip", "home_sp_xfip", "home_sp_k_9", "home_sp_bb_9", "home_sp_ip_per_gs",
    "away_sp_fip", "away_sp_xfip", "away_sp_k_9", "away_sp_bb_9", "away_sp_ip_per_gs",
    # Combined SP quality (lower = better pitching matchup = fewer runs)
    "combined_sp_fip",
    # Rolling team offense
    "home_rolling_rpg", "away_rolling_rpg",
    # Rolling run prevention
    "home_rolling_ra9", "away_rolling_ra9",
    # Park + weather
    "park_factor", "temp_f", "wind_mph", "wind_out_num",
    # Rest
    "home_rest_days", "away_rest_days",
    # Prior-season league run environment
    "lg_rpg",
]

TARGET = "total_runs"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combined_sp_fip"] = df["home_sp_fip"] + df["away_sp_fip"]
    df["combined_sp_xfip"] = df.get("home_sp_xfip", np.nan) + df.get("away_sp_xfip", np.nan)
    df["wind_out_num"] = df["wind_out"].astype(float) if "wind_out" in df.columns else 0.0
    return df


def walk_forward_eval(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer(df)
    seasons = sorted(df["season"].dropna().unique().astype(int))

    all_preds = []
    for test_season in seasons[2:]:  # need at least 2 seasons of training data
        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()

        available_feats = [c for c in FEATURE_COLS if c in df.columns]
        X_train = train[available_feats]
        y_train = train[TARGET]
        X_test = test[available_feats]
        y_test = test[TARGET]

        # Drop rows with no target
        mask_train = y_train.notna()
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        mask_test = y_test.notna()
        X_test, y_test = X_test[mask_test], y_test[mask_test]
        test_filtered = test[mask_test]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Ridge pipeline
        ridge_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
        ])
        ridge_pipe.fit(X_train, y_train)
        ridge_preds = ridge_pipe.predict(X_test)

        # XGBoost
        xgb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )),
        ])
        xgb_pipe.fit(X_train, y_train)
        xgb_preds = xgb_pipe.predict(X_test)

        # Bias correction: subtract mean(predicted - line) on training games that have a line
        # Computed on train set only — no leakage into test season
        if "close_total" in train.columns:
            train_lined = train[train["close_total"].notna() & mask_train]
            if len(train_lined) > 50:
                ridge_train_preds = ridge_pipe.predict(train_lined[available_feats])
                ridge_bias = (ridge_train_preds - train_lined["close_total"].values).mean()
                ridge_preds = ridge_preds - ridge_bias

                xgb_train_preds = xgb_pipe.predict(train_lined[available_feats])
                xgb_bias = (xgb_train_preds - train_lined["close_total"].values).mean()
                xgb_preds = xgb_preds - xgb_bias

                print(f"    Bias correction: Ridge={ridge_bias:+.3f}, XGB={xgb_bias:+.3f}")

        mae_ridge = mean_absolute_error(y_test, ridge_preds)
        mae_xgb = mean_absolute_error(y_test, xgb_preds)
        print(f"  Season {test_season}: Ridge MAE={mae_ridge:.3f}, XGB MAE={mae_xgb:.3f}  (n={len(y_test)})")

        preds_df = pd.DataFrame({
            "game_pk": test_filtered["game_pk"].values,
            "date": test_filtered["date"].values,
            "season": test_season,
            "home_team": test_filtered["home_team"].values,
            "away_team": test_filtered["away_team"].values,
            "actual_total": y_test.values,
            "predicted_ridge": ridge_preds,
            "predicted_xgb": xgb_preds,
            "open_total": test_filtered["open_total"].values if "open_total" in test_filtered.columns else np.nan,
            "close_total": test_filtered["close_total"].values if "close_total" in test_filtered.columns else np.nan,
        })
        all_preds.append(preds_df)

    combined = pd.concat(all_preds, ignore_index=True)

    # Overall metrics
    print(f"\nOverall Ridge MAE: {mean_absolute_error(combined['actual_total'], combined['predicted_ridge']):.3f}")
    print(f"Overall XGB MAE:   {mean_absolute_error(combined['actual_total'], combined['predicted_xgb']):.3f}")
    baseline = np.full(len(combined), combined['actual_total'].mean())
    print(f"Baseline (mean):   {mean_absolute_error(combined['actual_total'], baseline):.3f}")

    return combined


def print_ridge_importance(df: pd.DataFrame):
    """Train on all data, print Ridge coefficients as pseudo-importance."""
    df = engineer(df)
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    mask = df[TARGET].notna()
    X = df.loc[mask, available_feats]
    y = df.loc[mask, TARGET]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
    ])
    pipe.fit(X, y)

    coefs = pd.Series(pipe["model"].coef_, index=available_feats).abs().sort_values(ascending=False)
    print("\nRidge feature importance (|coef| on scaled features):")
    print(coefs.to_string())


def train_production_model(df: pd.DataFrame):
    """Train Ridge on ALL available data and save to disk for live prediction."""
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    df = engineer(df)
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    mask = df[TARGET].notna()
    X = df.loc[mask, available_feats]
    y = df.loc[mask, TARGET]

    # Compute bias on games with closing lines
    lined = df[mask & df["close_total"].notna()]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])),
    ])
    pipe.fit(X, y)

    train_preds = pipe.predict(lined[available_feats])
    bias = float((train_preds - lined["close_total"].values).mean())

    model_path = models_dir / "ridge_production.joblib"
    meta_path = models_dir / "ridge_meta.joblib"
    joblib.dump(pipe, model_path)
    joblib.dump({"feature_cols": available_feats, "bias": bias}, meta_path)
    print(f"\nProduction model saved: {model_path.name}  (bias={bias:+.3f}, features={len(available_feats)})")


def main():
    print("Loading features...")
    df = load_features()
    print(f"  {len(df)} games, seasons {df['season'].min():.0f}-{df['season'].max():.0f}")

    print("\nWalk-forward evaluation:")
    preds = walk_forward_eval(df)

    out_path = PROCESSED_DIR / "predictions.csv"
    preds.to_csv(out_path, index=False)
    print(f"\nPredictions saved: {len(preds)} rows -> {out_path.name}")

    print_ridge_importance(df)
    train_production_model(df)


if __name__ == "__main__":
    main()
