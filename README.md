# 🌾 Sri Lankan Paddy Yield Prediction: Bagging vs Boosting vs Stacking vs Hybrid

Predicts paddy yield (kg/ha) for each Sri Lankan district and cultivation season (Maha / Yala) with ensemble learning, and links it to the rural economy through CBSL Samba paddy prices. Includes an interactive Streamlit dashboard.

> **AI-generated code statement:** the code in this repository (`paddy_models.py`, `app.py` and the notebook) was generated with the help of **Claude (Anthropic)**, a generative AI assistant, and was reviewed, run and interpreted by the student, as the module's GenAI policy requires.

## Files

| File | Purpose |
|---|---|
| `paddy_yield_ensemble.ipynb` | Main notebook: data processing, EDA, models, comparison, stability checks, interpretation (already executed) |
| `paddy_models.py` | Data cleaning, feature engineering, all models and evaluation. Run it directly for a printed comparison |
| `app.py` | Streamlit dashboard (5 tabs: overview, model comparison, district explorer, what-if predictor, rural economy) |
| `paddy-a (1).csv` | Raw download: "Agriculture and fisheries: Paddy" (DCS / CBSL via Macro Lab), annual, 1951 to 2026 |
| `paddy_clean.csv` | Cleaned table: one row per district, season and year (2,240 rows, 25 districts, 1978 to 2025) |
| `samba_paddy_price.csv` | CBSL monthly Samba paddy price, Rs/kg, Jan 2021 to Sep 2026 |
| `results.csv`, `feature_importances.csv`, `paddy_results.png` | Saved outputs of `python paddy_models.py` |
| `requirements.txt` | Python dependencies |

## Setup and run

```bash
pip install -r requirements.txt
python paddy_models.py          # prints the comparison table, saves results.csv and paddy_results.png
jupyter notebook paddy_yield_ensemble.ipynb
streamlit run app.py            # opens the dashboard in your browser
```

Keep all files in the same folder. If `xgboost`, `lightgbm` or `catboost` is missing, that model is skipped and a note is printed. Full run time is about 20 seconds for the script and about 1.5 minutes for the notebook.

**Live dashboard (optional):** push this folder to a public GitHub repository, sign in at share.streamlit.io, choose "New app", select the repo and `app.py`, and deploy. Put the resulting link in your submission.

## Data

- **Source:** Department of Census and Statistics (DCS) paddy statistics and the Central Bank of Sri Lanka (CBSL), compiled by Macro Lab ("Agriculture and fisheries: Paddy", annual). Samba paddy price: CBSL via Macro Lab (`LK.AGR.PADDY.PRICE_SAMBA_PADDY.M`).
- **Variables per district, season and year:** sown extent (total, major irrigation, minor irrigation, rainfed), harvested extent, average yield (kg/ha).
- **Season labels:** "Maha 2021" is the 2021/22 Maha season (planted Oct 2021, harvested early 2022). Yala 2021 is Apr to Sep 2021. You can see this in the data: the fertiliser-ban crash appears in Maha 2021.
- **Cleaning:** reshaped 850 wide columns into a long table, merged the two spellings of Kilinochchi, kept the 25 administrative districts (dropped the national total and the Mahaweli H / Udawalawe special areas), and removed rows without yield or sown area. Northern districts have gaps during the civil-war years.

## Method

**Target:** average paddy yield, kg per harvested hectare.

**Features** (all known at sowing time): district (one-hot), season, year, log sown area and its change on last year, irrigation shares (major / minor / rainfed), previous season's yield, same-season yield 1 and 2 years ago, 3-year same-season average, national yield of the previous season.

**Split:** time-ordered. Train 1978 to 2017 (1,515 rows), test 2018 to 2025 (368 rows). No shuffling. The test set includes the 2021 fertiliser ban.

**Models**

| Family | Models |
|---|---|
| Baselines | Same season last year; 3-year average |
| Bagging | Bagged decision trees; Random Forest |
| Boosting | sklearn Gradient Boosting; XGBoost; LightGBM; CatBoost |
| Neural network | MLP (64-32), standardised inputs and target |
| Stacking | RF + LightGBM + ANN + Ridge, with a Ridge meta-learner on 5-fold out-of-fold predictions |
| Hybrid | RF + ANN: the ANN is trained on the Random Forest's out-of-fold residuals and adds a correction |

## Results (test 2018 to 2025)

| Model | RMSE (kg/ha) | MAE | MAPE | R² | RMSE excl. crisis |
|---|---|---|---|---|---|
| **LightGBM** | **633** | 469 | 12.8% | 0.55 | 509 |
| CatBoost | 637 | 471 | 12.8% | 0.55 | 515 |
| Stacking | 640 | **469** | 13.0% | 0.54 | **488** |
| XGBoost | 644 | 479 | 12.9% | 0.54 | 534 |
| Gradient Boosting | 646 | 484 | 13.1% | 0.53 | 530 |
| ANN (MLP) | 653 | 485 | 13.6% | 0.52 | 496 |
| Hybrid RF + ANN | 656 | 488 | 13.5% | 0.52 | 504 |
| Random Forest | 661 | 494 | 13.5% | 0.51 | 513 |
| Bagging (trees) | 671 | 500 | 13.7% | 0.49 | 518 |
| 3-year average | 746 | 551 | 15.1% | 0.38 | 551 |
| Same season last year | 774 | 576 | 15.8% | 0.33 | 625 |

"Excl. crisis" removes Maha 2021/22 and Yala 2022. Across four split years (2011, 2014, 2017, 2020) Stacking and CatBoost have the best average rank.

## Interpretation (short version, full version in the notebook)

- All ensembles beat the naive baseline by 13 to 18% RMSE.
- **Boosting beats bagging** because the problem is mainly bias, not variance: yield is a persistent district level plus small additive corrections, and boosting fits those corrections step by step. Random Forest averaging pulls predictions towards the training mean, so it under-predicts the 2018 to 2020 record yields. The gap is larger than seed noise (RF 662 ± 2 vs LightGBM 639 ± 3).
- **Stacking is the most robust**: it combines trees and an ANN that make different errors, and it is the best model in normal (non-crisis) seasons.
- **The hybrid improves Random Forest** (661 → 656), but one residual-correction step is weaker than full boosting.
- **The 2021 fertiliser ban** (−37% national Maha yield) is missed by every model. Policy shocks are not in the historical features.
- **Most important features:** the 3-year same-season average and last year's yield. Irrigation mix and sown area add the rest.
- **Economy:** dry-zone irrigated districts (Hambantota, Polonnaruwa, Ampara, Anuradhapura) lead. Wet-zone districts and Jaffna trail. Maha 2025 production is worth about Rs 458 bn at Samba prices.

## Limitations

- No rainfall, temperature or fertiliser-supply features; adding them is the obvious next step.
- District-level averages only.
- Production is derived as yield × harvested area.
- The Samba price series only starts in 2021, so it is used for economic context, not as a model feature.

## Possible extensions

- Add CHIRPS rainfall or Department of Meteorology data per district
- Walk-forward validation (retrain every season)
- SHAP values for per-prediction explanations
- An ARIMA + RF hybrid on the national series
