"""
Sri Lankan paddy yield prediction: Bagging vs Boosting vs Stacking vs a hybrid RF + ANN model.

AI-GENERATED CODE NOTICE: this file was written with the help of Claude (Anthropic), a generative
AI assistant, and then reviewed and run by the student. The course policy requires this statement.

Run directly for a printed comparison:  python paddy_models.py
"""

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

SEED = 42
# The 25 administrative districts. Special areas (Mahaweli H, Udawalawe) and the national total are excluded.
DISTRICTS = ["AMPARA", "ANURADHAPURA", "BADULLA", "BATTICALOA", "COLOMBO", "GALLE", "GAMPAHA",
             "HAMBANTOTA", "JAFFNA", "KALUTARA", "KANDY", "KEGALLE", "KILINOCHCHI", "KURUNEGALA",
             "MANNAR", "MATALE", "MATARA", "MONARAGALA", "MULLATIVU", "NUWARA_ELIYA", "POLONNARUWA",
             "PUTTALAM", "RATNAPURA", "TRINCOMALEE", "VAVUNIYA"]
# Seasons that were hit by the 2021 chemical-fertiliser import ban and the 2022 fertiliser shortage.
# In this data "Maha 2021" is the 2021/22 Maha season (planted Oct 2021, harvested early 2022).
CRISIS = {(2021, "Maha"), (2022, "Yala")}


# --------------------------------------------------------------------------------------------
# 1. Data cleaning: wide Macro Lab / CBSL / DCS file  ->  one row per district, season and year
# --------------------------------------------------------------------------------------------
def clean_raw(path):
    """Reshape the wide 'Agriculture and fisheries: Paddy' export into a tidy long table."""
    lines = open(path, encoding="utf-8").read().splitlines()
    series_line = next(l for l in lines if l.startswith("# Series:"))
    codes = series_line.split(":", 1)[1].strip().split(", ")
    raw = pd.read_csv(path, comment="#")
    raw.columns = ["year"] + codes
    raw = raw.set_index("year")

    pat = re.compile(r"LK\.AGR\.PADDY\.(SOWN_EXTENT_HA|HARVESTED_EXTENT_HA|AVG_YIELD_KG_PER_HA)\."
                     r"([A-Z_]+)\.(MAHA|YALA)\.(TOTAL|MAJOR|MINOR|RAINFED)\.A")
    recs = []
    for c in codes:
        m = pat.fullmatch(c)
        if not m:
            continue
        var, dist, season, irr = m.groups()
        dist = {"KILLINOCHCHI": "KILINOCHCHI"}.get(dist, dist)   # same district, two spellings
        if dist not in DISTRICTS:
            continue
        s = raw[c].dropna()
        for yr, v in s.items():
            recs.append((dist, season.title(), int(yr), f"{var}_{irr}", float(v)))
    long = pd.DataFrame(recs, columns=["district", "season", "year", "var", "value"])
    tidy = (long.groupby(["district", "season", "year", "var"])["value"].first()
                .unstack("var").reset_index())
    ren = {"SOWN_EXTENT_HA_TOTAL": "sown_ha", "SOWN_EXTENT_HA_MAJOR": "sown_major_ha",
           "SOWN_EXTENT_HA_MINOR": "sown_minor_ha", "SOWN_EXTENT_HA_RAINFED": "sown_rainfed_ha",
           "HARVESTED_EXTENT_HA_TOTAL": "harvested_ha", "AVG_YIELD_KG_PER_HA_TOTAL": "yield_kg_ha",
           "AVG_YIELD_KG_PER_HA_MAJOR": "yield_major", "AVG_YIELD_KG_PER_HA_MINOR": "yield_minor",
           "AVG_YIELD_KG_PER_HA_RAINFED": "yield_rainfed"}
    tidy = tidy.rename(columns=ren)
    keep = ["district", "season", "year", "sown_ha", "sown_major_ha", "sown_minor_ha",
            "sown_rainfed_ha", "harvested_ha", "yield_kg_ha"]
    tidy = tidy[[k for k in keep if k in tidy]]
    tidy = tidy.dropna(subset=["yield_kg_ha", "sown_ha"])
    tidy = tidy[(tidy["yield_kg_ha"] > 0) & (tidy["sown_ha"] > 0)]
    tidy["production_mt"] = tidy["yield_kg_ha"] * tidy["harvested_ha"] / 1000   # derived, for the dashboard
    return tidy.sort_values(["district", "year", "season"]).reset_index(drop=True)


def load(path="paddy_clean.csv"):
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------------
# 2. Feature engineering (only information available when the crop is sown)
# --------------------------------------------------------------------------------------------
def make_features(df, use_harvest_ratio=False):
    d = df.copy()
    # chronological season index: Yala y (Apr-Sep y) comes before Maha y (Oct y - Mar y+1)
    d["t"] = d["year"] * 2 + (d["season"] == "Maha").astype(int)
    d = d.sort_values(["district", "t"]).reset_index(drop=True)

    for c in ["sown_major_ha", "sown_minor_ha", "sown_rainfed_ha"]:
        d[c] = d[c].fillna(0)
    comp = d[["sown_major_ha", "sown_minor_ha", "sown_rainfed_ha"]].sum(axis=1).replace(0, np.nan)
    d["share_major"] = d["sown_major_ha"] / comp
    d["share_minor"] = d["sown_minor_ha"] / comp
    d["share_rainfed"] = d["sown_rainfed_ha"] / comp
    d["log_sown"] = np.log(d["sown_ha"])
    d["is_maha"] = (d["season"] == "Maha").astype(int)

    y = d.set_index(["district", "t"])["yield_kg_ha"]
    sown = d.set_index(["district", "t"])["sown_ha"]
    key = list(zip(d["district"], d["t"]))
    get = lambda s, k: s.reindex(pd.MultiIndex.from_tuples(k)).values
    d["yield_prev_season"] = get(y, [(a, t - 1) for a, t in key])          # previous season, same district
    d["yield_lag1_same"] = get(y, [(a, t - 2) for a, t in key])            # same season last year
    d["yield_lag2_same"] = get(y, [(a, t - 4) for a, t in key])
    d["yield_lag3_same"] = get(y, [(a, t - 6) for a, t in key])
    d["yield_roll3_same"] = d[["yield_lag1_same", "yield_lag2_same", "yield_lag3_same"]].mean(axis=1)
    d["sown_change"] = d["sown_ha"] / get(sown, [(a, t - 2) for a, t in key]) - 1

    # national average yield of the previous season (country-wide shocks such as droughts or policy)
    nat = (d.assign(prod=d["yield_kg_ha"] * d["harvested_ha"].fillna(d["sown_ha"]))
             .groupby("t").apply(lambda g: g["prod"].sum() / g["harvested_ha"].fillna(g["sown_ha"]).sum()))
    d["nat_yield_prev_season"] = d["t"].map(lambda t: nat.get(t - 1, np.nan))

    if use_harvest_ratio:   # known only at harvest time: an 'early-warning' scenario, not a pure forecast
        d["harvest_ratio"] = (d["harvested_ha"] / d["sown_ha"]).clip(0, 1.2)

    d = d.dropna(subset=["yield_prev_season", "yield_lag1_same", "share_major"])
    d["sown_change"] = d["sown_change"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1, 5)
    d["nat_yield_prev_season"] = d["nat_yield_prev_season"].fillna(d["yield_prev_season"])
    if use_harvest_ratio:
        d["harvest_ratio"] = d["harvest_ratio"].fillna(1.0)

    dummies = pd.get_dummies(d["district"], prefix="dist", dtype=int)
    d = pd.concat([d, dummies], axis=1)
    d["target"] = d["yield_kg_ha"]
    return d.reset_index(drop=True)


def feature_columns(X):
    base = ["year", "is_maha", "log_sown", "sown_change", "share_major", "share_minor", "share_rainfed",
            "yield_prev_season", "yield_lag1_same", "yield_lag2_same", "yield_roll3_same",
            "nat_yield_prev_season"]
    if "harvest_ratio" in X:
        base.append("harvest_ratio")
    return base + [c for c in X.columns if c.startswith("dist_")]


# --------------------------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------------------------
def _mlp():
    """Small feed-forward neural network (ANN). Inputs and target are standardised."""
    net = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                       max_iter=2000, early_stopping=True, random_state=SEED)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), net),
                                      transformer=StandardScaler())


def _lag_filled(X):
    """Linear / neural models cannot take NaN: fill the deeper lags with the 1-year lag."""
    X = X.copy()
    for c in ["yield_lag2_same", "yield_roll3_same"]:
        if c in X:
            X[c] = X[c].fillna(X["yield_lag1_same"])
    return X


def build_models(rf_params=None, gb_params=None, use=None):
    rf_params = {"n_estimators": 400, "max_depth": 12, "min_samples_leaf": 3, "max_features": 0.5,
                 "n_jobs": -1, "random_state": SEED, **(rf_params or {})}
    gb_params = {"n_estimators": 600, "learning_rate": 0.03, "max_depth": 4, **(gb_params or {})}
    m, notes = {}, []
    m["Bagging (decision trees)"] = BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3),
                                                     n_estimators=200, max_samples=0.8,
                                                     n_jobs=-1, random_state=SEED)
    m["Random Forest"] = RandomForestRegressor(**rf_params)
    m["Gradient Boosting (sklearn)"] = GradientBoostingRegressor(
        n_estimators=gb_params["n_estimators"], learning_rate=gb_params["learning_rate"],
        max_depth=gb_params["max_depth"], subsample=0.8, random_state=SEED)
    if HAS_XGB:
        m["XGBoost"] = XGBRegressor(n_estimators=gb_params["n_estimators"],
                                    learning_rate=gb_params["learning_rate"],
                                    max_depth=gb_params["max_depth"], subsample=0.8,
                                    colsample_bytree=0.8, random_state=SEED, n_jobs=4)
    else:
        notes.append("xgboost not installed: XGBoost skipped.")
    if HAS_LGBM:
        m["LightGBM"] = LGBMRegressor(n_estimators=gb_params["n_estimators"],
                                      learning_rate=gb_params["learning_rate"],
                                      num_leaves=2 ** gb_params["max_depth"] - 1, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
                                      random_state=SEED, verbose=-1)
    else:
        notes.append("lightgbm not installed: LightGBM skipped.")
    if HAS_CAT:
        m["CatBoost"] = CatBoostRegressor(iterations=gb_params["n_estimators"],
                                          learning_rate=gb_params["learning_rate"],
                                          depth=gb_params["max_depth"] + 2, random_seed=SEED,
                                          verbose=0, thread_count=4)
    else:
        notes.append("catboost not installed: CatBoost skipped.")
    m["ANN (MLP)"] = _mlp()

    # Stacking: diverse level-0 learners, a ridge regression meta-learner trained on out-of-fold predictions
    boost = m.get("LightGBM", m.get("XGBoost", m["Gradient Boosting (sklearn)"]))
    m["Stacking (RF + Boosting + ANN -> Ridge)"] = StackingRegressor(
        estimators=[("rf", RandomForestRegressor(**rf_params)), ("boost", boost),
                    ("ann", _mlp()), ("ridge", make_pipeline(StandardScaler(), RidgeCV()))],
        final_estimator=RidgeCV(), cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=1)
    if use:
        m = {k: v for k, v in m.items() if any(u.lower() in k.lower() for u in use)}
    return m, notes, rf_params


class RFANNHybrid:
    """Hybrid model: a Random Forest learns the main signal, then an ANN learns what the forest got wrong.
    The ANN is trained on the forest's OUT-OF-FOLD residuals, so it does not just memorise training noise.
    Final prediction = RF prediction + ANN residual correction."""

    def __init__(self, rf_params):
        self.rf_params = rf_params

    def fit(self, X, y):
        self.rf = RandomForestRegressor(**self.rf_params).fit(X, y)
        oof = cross_val_predict(RandomForestRegressor(**self.rf_params), X, y,
                                cv=KFold(5, shuffle=True, random_state=SEED))
        Z = X.assign(rf_pred=oof)
        self.ann = _mlp().fit(Z, y - oof)
        return self

    def predict(self, X):
        p = self.rf.predict(X)
        return p + self.ann.predict(X.assign(rf_pred=p))


def _metrics(name, y, p):
    err = y - p
    return {"model": name,
            "RMSE (kg/ha)": float(np.sqrt(np.mean(err ** 2))),
            "MAE (kg/ha)": float(np.mean(np.abs(err))),
            "MAPE (%)": float(np.mean(np.abs(err) / y) * 100),
            "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}


def run_experiment(X, train_end=2017, rf_params=None, gb_params=None, use=None, include_hybrid=True):
    feats = feature_columns(X)
    tr = X["year"] <= train_end
    Xtr, Xte = _lag_filled(X.loc[tr, feats]), _lag_filled(X.loc[~tr, feats])
    ytr, yte = X.loc[tr, "target"].values, X.loc[~tr, "target"].values

    preds = {"Baseline: same season last year": X.loc[~tr, "yield_lag1_same"].values,
             "Baseline: 3-year average": X.loc[~tr, "yield_roll3_same"].values}
    models, notes, rfp = build_models(rf_params, gb_params, use)
    fitted, imps = {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        preds[name] = mdl.predict(Xte)
        fitted[name] = mdl
        if hasattr(mdl, "feature_importances_"):
            imps[name] = pd.Series(mdl.feature_importances_ / np.sum(mdl.feature_importances_), feats)
    if include_hybrid:
        hyb = RFANNHybrid(rfp).fit(Xtr, ytr)
        preds["Hybrid (RF + ANN residual)"] = hyb.predict(Xte)
        fitted["Hybrid (RF + ANN residual)"] = hyb

    test = X.loc[~tr, ["district", "season", "year", "sown_ha", "harvested_ha", "target"]].copy()
    for k, v in preds.items():
        test[k] = v
    crisis = np.array([(a, b) in CRISIS for a, b in zip(test["year"], test["season"])])
    table = pd.DataFrame([_metrics(k, yte, v) for k, v in preds.items()]).set_index("model")
    table_nc = pd.DataFrame([_metrics(k, yte[~crisis], v[~crisis]) for k, v in preds.items()]).set_index("model")
    table["RMSE excl. 2021-22 crisis"] = table_nc["RMSE (kg/ha)"]
    return {"table": table, "test": test, "importances": pd.DataFrame(imps), "notes": notes,
            "models": fitted, "features": feats, "n_train": int(tr.sum()), "n_test": int((~tr).sum())}


def group_errors(test, model):
    """Mean absolute error of one model by district and by season, for interpretation."""
    e = (test["target"] - test[model]).abs()
    return (test.assign(abs_err=e).groupby("district")["abs_err"].mean().sort_values(),
            test.assign(abs_err=e).groupby(["year", "season"])["abs_err"].mean())


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists("paddy_clean.csv"):
        clean_raw("paddy-a (1).csv").to_csv("paddy_clean.csv", index=False)
    X = make_features(load("paddy_clean.csv"))
    res = run_experiment(X)
    pd.set_option("display.width", 200)
    print(f"train rows: {res['n_train']}  test rows: {res['n_test']}")
    print(res["table"].round(3).sort_values("RMSE (kg/ha)"))
    for n in res["notes"]:
        print("NOTE:", n)
    res["table"].to_csv("results.csv")
    res["importances"].to_csv("feature_importances.csv")
    t = res["test"]
    nat = t.groupby(["year", "season"]).mean(numeric_only=True).sort_index(level=[0, 1], ascending=[True, False])
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    lbl = [f"{s[:1]}{y % 100:02d}" for y, s in nat.index]
    for c in ["target", "Random Forest", "LightGBM", "Stacking (RF + Boosting + ANN -> Ridge)",
              "Hybrid (RF + ANN residual)"]:
        if c in nat:
            ax[0].plot(lbl, nat[c], marker="o", label="actual" if c == "target" else c)
    ax[0].set_title("Average district yield (kg/ha), test seasons (Y = Yala, M = Maha)"); ax[0].legend(fontsize=8)
    imp = res["importances"].mean(axis=1).sort_values().tail(15)
    imp.plot.barh(ax=ax[1], title="Mean feature importance across tree ensembles (top 15)")
    plt.tight_layout(); plt.savefig("paddy_results.png", dpi=130)
