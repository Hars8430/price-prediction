import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
NEEDS_SCALE = {"Linear Regression", "Neural Network"}

DALLAS_NEIGHBORHOODS = ["Uptown", "Highland Park", "Plano", "Frisco", "Oak Cliff", "Garland"]
GURGAON_SECTORS      = ["Golf Course Road", "DLF Phase 1-3", "Sohna Road",
                         "New Gurgaon", "Palam Vihar", "Manesar",
                         "Sector 29", "Sector 56-57"]  # extra sectors from real CSV
FURNISHING_MAP = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Fully Furnished"}

# ── Data generation ────────────────────────────────────────────────────────────
def generate_dallas_data(n=1200, seed=42):
    np.random.seed(seed)
    neighborhoods = {
        "Uptown":        550_000,
        "Highland Park": 900_000,
        "Plano":         420_000,
        "Frisco":        480_000,
        "Oak Cliff":     280_000,
        "Garland":       250_000,
    }
    records = []
    per_hood = n // len(neighborhoods)
    for hood, base in neighborhoods.items():
        for _ in range(per_hood):
            sqft          = max(600, np.random.normal(2000, 500))
            bedrooms      = int(np.clip(np.random.normal(3, 1), 1, 7))
            bathrooms     = int(np.clip(np.random.normal(2, 1), 1, 5))
            age           = max(0, int(np.random.exponential(10)))
            school_rating = round(np.clip(np.random.normal(5, 2), 1, 10), 1)
            dist_downtown = round(max(0.5, np.random.normal(12, 6)), 1)
            has_pool      = int(np.random.random() < 0.2)
            garage        = int(np.clip(np.random.poisson(1.5), 0, 4))
            price = (base
                     + sqft          * 150
                     + bedrooms      * 10_000
                     + bathrooms     *  7_000
                     - age           *  2_000
                     + school_rating * 10_000
                     - dist_downtown *  3_000
                     + has_pool      * 30_000
                     + garage        *  6_000
                     + np.random.normal(0, 25_000))
            records.append({
                "neighborhood" : hood,
                "sqft"         : round(sqft),
                "bedrooms"     : bedrooms,
                "bathrooms"    : bathrooms,
                "age_years"    : age,
                "school_rating": school_rating,
                "dist_downtown": dist_downtown,
                "has_pool"     : has_pool,
                "garage_spaces": garage,
                "price_usd"    : max(50_000, round(price, -2)),
            })
    return pd.DataFrame(records)


def generate_gurgaon_data(n=1200, seed=99):
    np.random.seed(seed)
    sectors = {
        "Golf Course Road" : 14_000,
        "DLF Phase 1-3"    : 12_000,
        "Sohna Road"       :  7_500,
        "New Gurgaon"      :  6_500,
        "Palam Vihar"      :  5_500,
        "Manesar"          :  4_500,
    }
    records = []
    per_sector = n // len(sectors)
    for sector, base_psf in sectors.items():
        for _ in range(per_sector):
            sqft       = max(400, np.random.normal(1300, 350))
            bhk        = int(np.clip(np.random.normal(3, 1), 1, 5))
            floor      = int(np.random.randint(0, 31))
            age        = max(0, int(np.random.exponential(5)))
            amenities  = round(np.clip(np.random.normal(6, 2), 1, 10), 1)
            dist_metro = round(max(0.2, np.random.exponential(2.5)), 1)
            parking    = int(np.clip(np.random.poisson(1), 0, 3))
            furnishing = int(np.random.choice([0, 1, 2], p=[0.35, 0.40, 0.25]))
            psf = max(2000, base_psf
                      + amenities  * 200
                      - dist_metro * 300
                      + floor      *  20
                      - age        * 100
                      + furnishing * 400
                      + np.random.normal(0, base_psf * 0.08))
            price = sqft * psf
            records.append({
                "sector"    : sector,
                "sqft"      : round(sqft),
                "bhk"       : bhk,
                "floor"     : floor,
                "age_years" : age,
                "amenities" : amenities,
                "dist_metro": dist_metro,
                "parking"   : parking,
                "furnishing": furnishing,
                "price_inr" : max(1_500_000, round(price, -3)),
            })
    return pd.DataFrame(records)


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df, target_col, cat_cols):
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return (X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc,
            scaler, encoders, feature_names)


def get_models():
    return {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting" : GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42),
        "Neural Network"    : MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu", solver="adam",
            alpha=0.01, max_iter=1000,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=25, random_state=42),
    }


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "R²"  : round(r2_score(y_test, y_pred), 4),
        "MAE" : round(mean_absolute_error(y_test, y_pred), 0),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 0),
        "MAPE": round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2),
    }


def train_all(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc):
    results = {}
    for name, model in get_models().items():
        Xtr = X_train_sc if name in NEEDS_SCALE else X_train
        Xte = X_test_sc  if name in NEEDS_SCALE else X_test
        model.fit(Xtr, y_train)
        m = compute_metrics(model, Xte, y_test)
        results[name] = {**m, "model": model}
    return results


def predict_price(model_name, results, scaler, feature_names, values):
    row = pd.DataFrame([dict(zip(feature_names, values))])
    model = results[model_name]["model"]
    if model_name in NEEDS_SCALE:
        row = scaler.transform(row)
    return model.predict(row)[0]


# ── Cached training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Dallas models…")
def train_dallas():
    df = generate_dallas_data(1200)
    (X_tr, X_te, y_tr, y_te,
     X_tr_sc, X_te_sc,
     scaler, enc, feat) = preprocess(df, "price_usd", ["neighborhood"])
    results = train_all(X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc)
    return results, scaler, enc, feat, df


@st.cache_resource(show_spinner="Training Gurgaon models…")
def train_gurgaon():
    df = generate_gurgaon_data(1200)
    (X_tr, X_te, y_tr, y_te,
     X_tr_sc, X_te_sc,
     scaler, enc, feat) = preprocess(df, "price_inr", ["sector"])
    results = train_all(X_tr, X_te, y_tr, y_te, X_tr_sc, X_te_sc)
    return results, scaler, enc, feat, df


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🏠 Real Estate Price Predictor")
st.caption("Dallas, TX (USD)  ·  Gurgaon, Haryana (INR)")

city_tab, model_tab = st.tabs(["🔮 Predict Price", "📊 Model Performance"])

# ── Predict tab ────────────────────────────────────────────────────────────────
with city_tab:
    city = st.radio("Select City", ["Dallas, TX", "Gurgaon, Haryana"], horizontal=True)
    st.divider()

    if city == "Dallas, TX":
        res_d, scaler_d, enc_d, feat_d, df_d = train_dallas()

        col1, col2, col3 = st.columns(3)
        with col1:
            hood = st.selectbox("Neighborhood", DALLAS_NEIGHBORHOODS)
            sqft = st.slider("Area (sq ft)", 600, 5000, 2200, step=50)
            bedrooms = st.slider("Bedrooms", 1, 7, 3)
        with col2:
            bathrooms = st.slider("Bathrooms", 1, 5, 2)
            age = st.slider("Property Age (years)", 0, 50, 8)
            school = st.slider("School Rating", 1.0, 10.0, 7.0, step=0.5)
        with col3:
            dist = st.slider("Distance from Downtown (mi)", 0.5, 40.0, 10.0, step=0.5)
            pool = st.checkbox("Has Pool")
            garage = st.slider("Garage Spaces", 0, 4, 2)

        model_choice = st.selectbox("Model", list(res_d.keys()), index=2)

        if st.button("Predict Price 💰", use_container_width=True):
            # FIX: encode neighborhood safely — handle unseen labels gracefully
            known = list(enc_d["neighborhood"].classes_)
            if hood not in known:
                st.error(f"Neighborhood '{hood}' was not seen during training.")
            else:
                hood_enc = enc_d["neighborhood"].transform([hood])[0]
                values   = [hood_enc, sqft, bedrooms, bathrooms, age,
                            school, dist, int(pool), garage]
                price    = predict_price(model_choice, res_d, scaler_d, feat_d, values)
                st.success(f"### Predicted Price: **${price:,.0f}**")
                st.caption(f"Model: {model_choice}  |  R² = {res_d[model_choice]['R²']}")

    else:  # Gurgaon
        res_g, scaler_g, enc_g, feat_g, df_g = train_gurgaon()

        col1, col2, col3 = st.columns(3)
        with col1:
            sector = st.selectbox("Sector", sorted(enc_g["sector"].classes_))
            sqft   = st.slider("Area (sq ft)", 400, 5000, 1400, step=50)
            bhk    = st.slider("BHK", 1, 5, 3)
        with col2:
            floor    = st.slider("Floor Number", 0, 40, 8)
            age      = st.slider("Property Age (years)", 0, 30, 4)
            amenities = st.slider("Amenities Score", 1.0, 10.0, 7.0, step=0.5)
        with col3:
            dist_metro = st.slider("Distance from Metro (km)", 0.2, 15.0, 1.5, step=0.1)
            parking    = st.slider("Parking Spaces", 0, 3, 1)
            furnishing = st.selectbox("Furnishing", list(FURNISHING_MAP.values()))

        furnishing_enc = {v: k for k, v in FURNISHING_MAP.items()}[furnishing]
        model_choice   = st.selectbox("Model", list(res_g.keys()), index=2)

        if st.button("Predict Price 💰", use_container_width=True):
            known = list(enc_g["sector"].classes_)
            if sector not in known:
                st.error(f"Sector '{sector}' was not seen during training.")
            else:
                sec_enc = enc_g["sector"].transform([sector])[0]
                values  = [sec_enc, sqft, bhk, floor, age,
                           amenities, dist_metro, parking, furnishing_enc]
                price   = predict_price(model_choice, res_g, scaler_g, feat_g, values)
                st.success(f"### Predicted Price: **₹{price/1e7:.2f} Cr** (₹{price:,.0f})")
                st.caption(f"Model: {model_choice}  |  R² = {res_g[model_choice]['R²']}")

# ── Model performance tab ──────────────────────────────────────────────────────
with model_tab:
    city2 = st.radio("City", ["Dallas, TX", "Gurgaon, Haryana"],
                     horizontal=True, key="model_city")

    if city2 == "Dallas, TX":
        res, _, _, _, _ = train_dallas()
    else:
        res, _, _, _, _ = train_gurgaon()

    rows = []
    for name, info in res.items():
        rows.append({
            "Model"  : name,
            "R²"     : info["R²"],
            "MAE"    : f"{info['MAE']:,.0f}",
            "RMSE"   : f"{info['RMSE']:,.0f}",
            "MAPE %": info["MAPE"],
        })
    df_metrics = pd.DataFrame(rows).set_index("Model")
    best = max(res, key=lambda k: res[k]["R²"])
    st.dataframe(df_metrics, use_container_width=True)
    st.info(f"🏆 Best model: **{best}** — R² = {res[best]['R²']}")
