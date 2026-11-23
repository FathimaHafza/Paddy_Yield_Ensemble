import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
import plotly.io as pio
 
import paddy_models as pm
 
# ---------------------------------------------------------------- palette
# Hand-picked paddy-field palette, used the same way in every chart.
INK = "#4A4A48"       # actual values, text
MAHA = "#8CC084"      # soft paddy green: Maha season
YALA = "#F2CC6B"      # pale harvest gold: Yala season
CLAY = "#EFA27A"      # peach clay: highlights, money
RIVER = "#8FC1D4"     # sky / tank-water blue
PLUM = "#C4A7CF"      # lavender dusk
STONE = "#BFBAB0"     # baselines, secondary
GRID = "#EEEBE5"      # gridlines
PAPER = "#FFFFFF"     # clean white background
 
SEASON_COLORS = {"Maha": MAHA, "Yala": YALA}
FAMILY_COLORS = {"Baseline": STONE, "Bagging": RIVER, "Boosting": MAHA, "Neural net": YALA,
                 "Stacking": CLAY, "Hybrid": PLUM, "": STONE}
MODEL_COLORS = {"Bagging (decision trees)": "#BBDCE8", "Random Forest": RIVER,
                "Gradient Boosting (sklearn)": "#C6E2BE", "XGBoost": "#A9D3A0", "LightGBM": MAHA,
                "CatBoost": "#6FA96A", "ANN (MLP)": YALA,
                "Stacking (RF + Boosting + ANN -> Ridge)": CLAY, "Hybrid (RF + ANN residual)": PLUM,
                "Baseline: same season last year": STONE, "Baseline: 3-year average": "#DAD6CE"}
 
pio.templates["paddy"] = go.layout.Template(layout=dict(
    font=dict(family="'Source Sans Pro', 'Segoe UI', sans-serif", color=INK, size=13),
    title=dict(font=dict(size=17, color=INK), x=0.01),
    paper_bgcolor=PAPER, plot_bgcolor=PAPER,
    colorway=[MAHA, YALA, CLAY, RIVER, PLUM, STONE, "#C6E2BE", "#6FA96A"],
    xaxis=dict(gridcolor=GRID, linecolor=STONE, zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor=STONE, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", title_font_color=STONE),
    hoverlabel=dict(bgcolor="white", font_color=INK)))
pio.templates.default = "plotly_white+paddy"
CHART = dict(width="stretch", theme=None)   # use our light template, not Streamlit's dark/light chart theme
 
st.set_page_config(page_title="Paddy Yield Ensemble Lab", page_icon="🌾", layout="wide")
# ---------------------------------------------------------------- light theme (kept inside app.py)
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #FFFFFF !important; color: #3A3A38 !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background-color: #F4F8F1 !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp label, .stApp span, .stApp li,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"] { color: #3A3A38 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #7A7A75 !important; }
.stApp input, .stApp textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-testid="stNumberInputContainer"], [data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important; color: #3A3A38 !important; border-color: #DCE3D6 !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background-color: #FFFFFF !important; }
[data-baseweb="tag"] { background-color: #8CC084 !important; }
[data-baseweb="tab-highlight"] { background-color: #6FA96A !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #6FA96A !important; }
[data-testid="stAlert"] { background-color: transparent !important; }
[data-testid="stAlertContainer"] { background-color: #EEF6EA !important; border: 1px solid #CFE3C8 !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background-color: #EAF4F8 !important; border-color: #C9E1EA !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { background-color: #FFF6DE !important; border-color: #F2DFA6 !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { background-color: #FDECE6 !important; border-color: #F3C9B9 !important; }
[data-testid="stAlertContainer"] *, [data-testid="stAlertContainer"] p { color: #3A3A38 !important; }
[data-testid="stCheckbox"] label > div:not([data-testid]) { background-color: #FFFFFF !important; border: 1px solid #9DBE95 !important; }
[data-testid="stCheckbox"] label:has(input:checked) > div:not([data-testid]) { background-color: #6FA96A !important; border-color: #6FA96A !important; }
[data-testid="stRadioOption"] > div > div:first-child { background-color: #FFFFFF !important; border: 1px solid #9DBE95 !important; }
[data-testid="stRadioOption"] > div > div:first-child > div { background-color: #FFFFFF !important; }
[data-testid="stRadioOption"][data-selected="true"] > div > div:first-child { background-color: #6FA96A !important; border-color: #6FA96A !important; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"] {
    background-color: #FFFFFF !important; border: 1px solid #DCE3D6 !important; color: #3A3A38 !important; }
[data-testid="stMultiSelectTagsContainer"] span[class] { background-color: #8CC084 !important; color: #FFFFFF !important; }
[data-testid="stMultiSelectTagsContainer"] span[class] * { color: #FFFFFF !important; }
.react-aria-SelectionIndicator { background-color: #6FA96A !important; }
[data-testid="stTab"][aria-selected="true"] p { color: #3E7A3A !important; }
[role="listbox"], [role="option"] { background-color: #FFFFFF !important; color: #3A3A38 !important; }
[role="option"]:hover { background-color: #F4F8F1 !important; }
[data-testid="stMetric"] { background-color: #F7FAF5 !important; border: 1px solid #E3ECDD; border-radius: 10px; padding: 10px 14px; }
[data-testid="stSidebar"] * { color: #3A3A38; }
[data-testid="stSkipToContent"] { color: #3A3A38 !important; }
button[kind="secondary"] { background-color: #FFFFFF !important; color: #3A3A38 !important; }
[data-baseweb="select"] svg, [data-testid="stNumberInputContainer"] button, [data-testid="stNumberInputContainer"] button svg {
    background-color: #FFFFFF !important; color: #3A3A38 !important; fill: #3A3A38 !important; }
[data-baseweb="checkbox"] > div:first-child, [data-baseweb="radio"] > div:first-child {
    background-color: #FFFFFF !important; border-color: #9DBE95 !important; }
[data-testid="stTable"] table { background-color: #FFFFFF !important; border-collapse: collapse; width: 100%; }
[data-testid="stTable"] th { background-color: #F4F8F1 !important; color: #3A3A38 !important; font-weight: 600; }
[data-testid="stTable"] td, [data-testid="stTable"] th { border-bottom: 1px solid #EEEBE5 !important; color: #3A3A38 !important; }
</style>
""", unsafe_allow_html=True)
 
st.title("🌾 Sri Lankan Paddy Yield: Ensemble Learning Dashboard")
st.caption("District by season paddy statistics, 1978 to 2025 (Department of Census and Statistics / CBSL). "
           "Bagging, Boosting, Stacking and a hybrid RF + ANN model compared on unseen seasons. ")
 
 
# ---------------------------------------------------------------- data
@st.cache_data
def load_data(file=None):
    return pd.read_csv(file) if file is not None else pm.load("paddy_clean.csv")
 
 
@st.cache_data
def load_price():
    p = pd.read_csv("samba_paddy_price.csv", comment="#")
    p["date"] = pd.to_datetime(p["period"])
    return p[["date", "value"]].rename(columns={"value": "samba_rs_kg"})
 
 
up = st.sidebar.file_uploader("Cleaned CSV (optional, same columns as paddy_clean.csv)", type="csv")
df = load_data(up)
df["district_name"] = df["district"].str.replace("_", " ").str.title()
 
st.sidebar.header("Model settings")
train_end = st.sidebar.slider("Train on seasons up to (year)", 2008, 2022, 2017,
                              help="Everything after this year is the unseen test set (time-ordered split).")
use_hr = st.sidebar.checkbox("Add harvest ratio feature (harvested / sown)", False,
                             help="Only known at harvest time. Turns the task into an early-warning estimate.")
st.sidebar.subheader("Bagging: Random Forest")
rf_p = {"n_estimators": st.sidebar.slider("Trees", 100, 800, 400, 50),
        "max_depth": st.sidebar.slider("Max depth", 3, 30, 12),
        "min_samples_leaf": st.sidebar.slider("Min samples per leaf", 1, 30, 3)}
st.sidebar.subheader("Boosting: XGBoost / LightGBM / CatBoost")
gb_p = {"n_estimators": st.sidebar.slider("Boosting rounds", 100, 1500, 600, 50),
        "max_depth": st.sidebar.slider("Tree depth", 2, 8, 4),
        "learning_rate": st.sidebar.select_slider("Learning rate", [0.01, 0.02, 0.03, 0.05, 0.1, 0.2], 0.03)}
ALL = ["Bagging", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "ANN", "Stacking"]
use = st.sidebar.multiselect("Models", ALL, default=ALL)
hybrid = st.sidebar.checkbox("Hybrid (RF + ANN residual)", True)
 
 
@st.cache_resource(show_spinner="Training ensemble models (about 20 seconds)...")
def train(df, train_end, use_hr, rf_p, gb_p, use, hybrid):
    X = pm.make_features(df, use_harvest_ratio=use_hr)
    return X, pm.run_experiment(X, train_end, rf_p, gb_p, list(use), hybrid)
 
 
X, res = train(df, train_end, use_hr, rf_p, gb_p, tuple(use), hybrid)
for n in res["notes"]:
    st.warning(n)
table = res["table"].sort_values("RMSE (kg/ha)")
test = res["test"]
models = [c for c in test.columns if c not in ("district", "season", "year", "sown_ha", "harvested_ha", "target")]
ml_models = [m for m in models if not m.startswith("Baseline")]
best = table.drop([i for i in table.index if i.startswith("Baseline")]).index[0]
base_rmse = table.loc["Baseline: same season last year", "RMSE (kg/ha)"]
 
tabs = st.tabs([" Overview", " Model comparison", "District explorer", " What-if predictor",
                " Rural economy"])
 
# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("District-season records", f"{len(df):,}")
    c[1].metric("Years covered", f"{df['year'].min()} to {df['year'].max()}")
    c[2].metric(f"Best model (RMSE {table.loc[best, 'RMSE (kg/ha)']:.0f} kg/ha)", best)
    c[3].metric("Error cut vs naive baseline",
                f"{(1 - table.loc[best, 'RMSE (kg/ha)'] / base_rmse) * 100:.1f}%")
 
    nat = (df.assign(prod=df["yield_kg_ha"] * df["harvested_ha"])
             .groupby(["year", "season"]).apply(lambda g: g["prod"].sum() / g["harvested_ha"].sum())
             .rename("yield").reset_index())
    fig = px.line(nat, x="year", y="yield", color="season", markers=True,
                  color_discrete_map=SEASON_COLORS,
                  labels={"yield": "National average yield (kg/ha)"},
                  title="Fifty years of Sri Lankan paddy yield by season")
    fig.add_vrect(x0=2020.5, x1=2022.5, fillcolor=CLAY, opacity=0.18, line_width=0,
                  annotation_text="2021 fertiliser ban / 2022 crisis", annotation_position="top left")
    st.plotly_chart(fig, **CHART)
 
    recent = df[df["year"] >= df["year"].max() - 9]
    rank = recent.groupby(["district_name", "season"])["yield_kg_ha"].mean().reset_index()
    order = rank.groupby("district_name")["yield_kg_ha"].mean().sort_values().index
    fig = px.bar(rank, x="yield_kg_ha", y="district_name", color="season", barmode="group",
                 orientation="h", category_orders={"district_name": list(order)},
                 color_discrete_map=SEASON_COLORS, height=700,
                 labels={"yield_kg_ha": "Average yield, last 10 years (kg/ha)", "district_name": ""},
                 title="Which districts and seasons achieve better yields?")
    st.plotly_chart(fig, **CHART)
    st.info("Dry-zone districts served by major irrigation schemes (Hambantota, Polonnaruwa, Ampara, "
            "Anuradhapura) lead, while the wet-zone districts (Colombo, Galle, Kalutara, Gampaha) and Jaffna "
            "trail. Yala is on average slightly MORE productive than Maha: it is the smaller, drier season, so it "
            "is cultivated mostly where irrigation water is guaranteed.")
 
# ---------------------------------------------------------------- comparison
with tabs[1]:
    st.subheader(f"Test-set performance: {res['n_test']} district-seasons after {train_end} "
                 f"(trained on {res['n_train']})")
    fam = {"Stacking": "Stacking", "Hybrid": "Hybrid", "Bagging": "Bagging", "Random Forest": "Bagging", "Boost": "Boosting", "XGBoost": "Boosting",
           "LightGBM": "Boosting", "CatBoost": "Boosting",
           "ANN": "Neural net", "Baseline": "Baseline"}
    t2 = table.copy()
    t2.insert(0, "Family", [next((v for k, v in fam.items() if k in i), "") for i in t2.index])
    t2["Better than naive (%)"] = (1 - t2["RMSE (kg/ha)"] / base_rmse) * 100
    st.table(t2.style.format({c: "{:.1f}" for c in t2.columns if c != "Family"} | {"R2": "{:.3f}"})
                   .highlight_min(subset=["RMSE (kg/ha)", "MAE (kg/ha)", "RMSE excl. 2021-22 crisis"],
                                  color="#E5F3E1"))
    st.caption("Lower RMSE/MAE is better; R2 closer to 1 is better. The last column drops Maha 2021/22 and "
               "Yala 2022, the fertiliser-ban seasons that no model could foresee from history.")
    fig = px.bar(t2.reset_index(), x="RMSE (kg/ha)", y="model", color="Family", orientation="h",
                 color_discrete_map=FAMILY_COLORS, title="RMSE by model (lower is better)", height=450)
    fig.update_layout(yaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, **CHART)
 
    pick = st.multiselect("Show on chart", models, default=[m for m in [best, "Random Forest",
                          "Hybrid (RF + ANN residual)", "Baseline: same season last year"] if m in models])
    nat_t = test.groupby(["year", "season"])[["target"] + models].mean().reset_index()
    nat_t["label"] = nat_t["season"] + " " + nat_t["year"].astype(str)
    nat_t = nat_t.sort_values(["year", "season"], ascending=[True, False])
    fig = go.Figure()
    fig.add_scatter(x=nat_t["label"], y=nat_t["target"], name="Actual", line=dict(color=INK, width=3))
    for m in pick:
        fig.add_scatter(x=nat_t["label"], y=nat_t[m], name=m, mode="lines+markers",
                        line=dict(color=MODEL_COLORS.get(m, STONE), width=2,
                                  dash="dot" if m.startswith("Baseline") else "solid"))
    fig.update_layout(title="Average district yield per test season: actual vs predicted",
                      yaxis_title="kg/ha")
    st.plotly_chart(fig, **CHART)
 
    imp = res["importances"]
    if not imp.empty:
        imp = imp.copy()
        imp["mean"] = imp.mean(axis=1)
        top = imp.sort_values("mean").tail(15).drop(columns="mean")
        top.index = top.index.str.replace("dist_", "district: ")
        fig = px.bar(top.reset_index().melt(id_vars="index"), x="value", y="index", color="variable",
                     orientation="h", barmode="group", height=650, color_discrete_map=MODEL_COLORS,
                     labels={"value": "Normalised importance", "index": "", "variable": "Model"},
                     title="What drives the predictions? (top 15 features)")
        st.plotly_chart(fig, **CHART)
        st.caption("Past yield of the same district dominates: soil, water and farming practice change slowly. "
                   "Irrigation mix and sown area add the rest.")
 
# ---------------------------------------------------------------- district explorer
with tabs[2]:
    dists = sorted(df["district_name"].unique())
    dn = st.selectbox("District", dists, index=dists.index("Polonnaruwa") if "Polonnaruwa" in dists else 0)
    code = dn.upper().replace(" ", "_")
    sea = st.radio("Season", ["Maha", "Yala"], horizontal=True)
    hist = df[(df["district"] == code) & (df["season"] == sea)]
    tt = test[(test["district"] == code) & (test["season"] == sea)]
    fig = go.Figure()
    fig.add_scatter(x=hist["year"], y=hist["yield_kg_ha"], name="Actual yield", line=dict(color=INK, width=2.5))
    for m in [best, "Hybrid (RF + ANN residual)", "Stacking (RF + Boosting + ANN -> Ridge)"]:
        if m in tt:
            fig.add_scatter(x=tt["year"], y=tt[m], name=m, mode="lines+markers",
                            line=dict(color=MODEL_COLORS.get(m, STONE), width=2))
    fig.add_vline(x=train_end + 0.5, line_dash="dash", line_color=STONE, annotation_text="train | test")
    fig.update_layout(title=f"{dn}, {sea} season", yaxis_title="kg/ha")
    st.plotly_chart(fig, **CHART)
    c = st.columns(3)
    last = hist.sort_values("year").iloc[-1]
    c[0].metric(f"Latest {sea} yield ({int(last['year'])})", f"{last['yield_kg_ha']:,.0f} kg/ha")
    c[1].metric("Sown area", f"{last['sown_ha']:,.0f} ha")
    if len(tt):
        c[2].metric(f"{best} mean abs. error here", f"{(tt['target'] - tt[best]).abs().mean():,.0f} kg/ha")
    err = (test.assign(e=(test["target"] - test[best]).abs()).groupby("district")["e"].mean()
               .sort_values().reset_index())
    err["district"] = err["district"].str.replace("_", " ").str.title()
    st.plotly_chart(px.bar(err, x="district", y="e", labels={"e": "Mean absolute error (kg/ha)", "district": ""},
                           title=f"Where is {best} most and least accurate?",
                           color="e", color_continuous_scale=[MAHA, YALA, CLAY])
                    .update_layout(coloraxis_showscale=False), **CHART)
 
# ---------------------------------------------------------------- what-if
with tabs[3]:
    st.write("Pick a district and season, adjust the inputs, and every trained model predicts the yield.")
    c = st.columns(3)
    dn2 = c[0].selectbox("District ", dists, index=dists.index("Anuradhapura") if "Anuradhapura" in dists else 0)
    code2 = dn2.upper().replace(" ", "_")
    sea2 = c[1].radio("Season ", ["Maha", "Yala"], horizontal=True)
    yr2 = c[2].number_input("Year", 1980, 2030, int(df["year"].max()) + 1)
    hist2 = X[(X["district"] == code2) & (X["season"] == sea2)].sort_values("year")
    ref = hist2.iloc[-1] if len(hist2) else X.iloc[-1]
    c = st.columns(3)
    sown = c[0].number_input("Sown area (ha)", 100, 200000, int(ref["sown_ha"]), 500)
    sm = c[1].slider("Major irrigation share", 0.0, 1.0, float(ref["share_major"]), 0.05)
    smi = c[2].slider("Minor irrigation share", 0.0, 1.0 - sm, min(float(ref["share_minor"]), 1.0 - sm), 0.05)
    c = st.columns(3)
    l1 = c[0].number_input("Same season last year (kg/ha)", 500, 8000, int(ref["target"]), 50)
    lp = c[1].number_input("Previous season (kg/ha)", 500, 8000, int(ref["yield_prev_season"]), 50)
    r3 = c[2].number_input("3-year average, same season (kg/ha)", 500, 8000, int(ref["yield_roll3_same"]), 50)
    row = {f: 0 for f in res["features"]}
    row.update({"year": yr2, "is_maha": int(sea2 == "Maha"), "log_sown": np.log(sown),
                "sown_change": sown / ref["sown_ha"] - 1, "share_major": sm, "share_minor": smi,
                "share_rainfed": max(0.0, 1 - sm - smi), "yield_prev_season": lp, "yield_lag1_same": l1,
                "yield_lag2_same": r3, "yield_roll3_same": r3, "nat_yield_prev_season": ref["nat_yield_prev_season"]})
    if "harvest_ratio" in row:
        row["harvest_ratio"] = st.slider("Harvest ratio", 0.3, 1.0, 0.95, 0.01)
    if f"dist_{code2}" in row:
        row[f"dist_{code2}"] = 1
    xr = pd.DataFrame([row])[res["features"]]
    out = pd.Series({n: float(m.predict(xr)[0]) for n, m in res["models"].items()}).sort_values()
    fig = px.bar(out.reset_index(), x=0, y="index", orientation="h", text_auto=".0f",
                 labels={"0": "Predicted yield (kg/ha)", "index": ""}, title=f"{dn2}, {sea2} {yr2}",
                 color="index", color_discrete_map=MODEL_COLORS)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, **CHART)
    est = out.median() * sown * 0.95 / 1000
    st.success(f"Median prediction {out.median():,.0f} kg/ha. With about 95% of the sown area harvested, that is "
               f"roughly {est:,.0f} tonnes of paddy.")
 
# ---------------------------------------------------------------- economy
with tabs[4]:
    price = load_price()
    st.write("Monthly Samba paddy price from the Central Bank of Sri Lanka (Rs. per kg), 2021 onward. "
             "Combining it with production shows what each season's harvest is worth to farmers.")
    st.plotly_chart(px.line(price, x="date", y="samba_rs_kg", markers=True,
                            labels={"samba_rs_kg": "Samba paddy, Rs/kg", "date": ""},
                            title="Samba paddy price (CBSL)")
                    .update_traces(line_color=CLAY, marker_color=CLAY), **CHART)
    # harvest months: Yala y -> Aug-Oct y ; Maha y (2021/22 style) -> Feb-Apr y+1
    def season_price(y, s):
        m = pd.date_range(f"{y}-08-01", periods=3, freq="MS") if s == "Yala" else \
            pd.date_range(f"{y + 1}-02-01", periods=3, freq="MS")
        v = price.set_index("date").reindex(m)["samba_rs_kg"]
        return v.mean() if v.notna().any() else np.nan
    prod = df.groupby(["year", "season"])["production_mt"].sum().reset_index()
    prod["price_rs_kg"] = [season_price(y, s) for y, s in zip(prod["year"], prod["season"])]
    prod = prod.dropna(subset=["price_rs_kg"])
    prod["value_rs_bn"] = prod["production_mt"] * 1000 * prod["price_rs_kg"] / 1e9
    prod["label"] = prod["season"] + " " + prod["year"].astype(str)
    if len(prod):
        prod = prod.sort_values(["year", "season"], ascending=[True, False])
        fig = go.Figure()
        fig.add_bar(x=prod["label"], y=prod["production_mt"] / 1e6, name="Production (million t)", marker_color=MAHA)
        fig.add_scatter(x=prod["label"], y=prod["value_rs_bn"], name="Farm-gate value (Rs bn)", yaxis="y2",
                        mode="lines+markers", line=dict(color=YALA, width=3), marker=dict(color=YALA, size=8))
        fig.update_layout(title="Paddy production and estimated farm-gate value per season",
                          yaxis=dict(title="million tonnes"),
                          yaxis2=dict(title="Rs billion", overlaying="y", side="right"))
        st.plotly_chart(fig, **CHART)
        st.table(prod[["label", "production_mt", "price_rs_kg", "value_rs_bn"]]
                     .rename(columns={"label": "Season", "production_mt": "Production (t)",
                                      "price_rs_kg": "Avg harvest-month price (Rs/kg)",
                                      "value_rs_bn": "Value (Rs bn)"})
                     .style.format({"Production (t)": "{:,.0f}", "Avg harvest-month price (Rs/kg)": "{:.1f}",
                                    "Value (Rs bn)": "{:.1f}"}).hide(axis="index"))
        st.caption("Value = district production summed nationally x average Samba price in the harvest months "
                   "(Yala: Aug to Oct, Maha: Feb to Apr of the following year). Samba is one variety, so this is "
                   "an indicative estimate, not official farm income.")
    loss = st.slider("What-if: yield change across the country (%)", -40, 20, -10)
    lastp = prod.iloc[-1] if len(prod) else None
    if lastp is not None:
        st.metric(f"Change in farm-gate value, {lastp['label']} basis",
                  f"Rs {lastp['value_rs_bn'] * loss / 100:+.1f} bn")
 