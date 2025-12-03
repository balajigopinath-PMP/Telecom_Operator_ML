from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

DATA_FILE = "2025_06_16_MobileOperators.csv"
MODEL_FILE = "operator_cluster_model.pkl"

# ---------- LOAD DATA ----------
df_raw = pd.read_csv(
    DATA_FILE,
    sep=';',
    encoding='latin1',
    engine='python'
)

df = df_raw.rename(columns={
    'Full name:': 'full_name',
    'Short name': 'short_name',
    'Headquarters': 'hq',
    'Description': 'description',
    'Cellular Networks Installed': 'networks',
    'Supported Cellular Data Links': 'data_links',
    'Cellular Network Operator': 'is_operator',
    'Covered Countries': 'covered_countries',
    'Covered Regions': 'covered_regions',
    'Founded': 'founded'
})

def extract_country(val):
    if isinstance(val, str):
        return val.replace('covered', '').strip()
    return None

df['country'] = df['covered_countries'].apply(extract_country)

def has_term(text, term):
    if not isinstance(text, str):
        return 0
    return 1 if term.lower() in text.lower() else 0

def count_bands(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0
    return len([x for x in text.split(',') if x.strip() != ""])

df['has_2g'] = df['networks'].apply(lambda x: has_term(x, 'GSM'))
df['has_3g'] = df['networks'].apply(lambda x: has_term(x, 'UMTS'))
df['has_4g'] = df['networks'].apply(lambda x: has_term(x, 'LTE'))
df['has_5g'] = df['networks'].apply(lambda x: has_term(x, '5G') or has_term(x, 'NR'))
df['band_count'] = df['networks'].apply(count_bands)

df['founded'] = pd.to_numeric(df['founded'], errors='coerce')
df['founded'] = df['founded'].fillna(df['founded'].median())

countries = sorted([c for c in df['country'].dropna().unique() if c])

# ---------- LOAD ML MODEL ----------
model_bundle = joblib.load(MODEL_FILE)
pipeline = model_bundle['pipeline']
feature_cols = model_bundle['feature_cols']
cluster_score_map = model_bundle['cluster_score_map']

# ---------- ITINERARY SCORE ----------
def compute_itinerary_score(itinerary_raw, row):
    """
    itinerary_raw: user text "Delhi, Agra, Jaipur"
    row: operator row (with 'covered_regions' and 'description')
    """
    if not itinerary_raw or not isinstance(itinerary_raw, str):
        return 0.0

    text_blob = ""
    for col in ['covered_regions', 'description', 'country']:
        val = row.get(col)
        if isinstance(val, str):
            text_blob += " " + val.lower()

    if not text_blob.strip():
        return 0.0

    # Split cities, normalize
    cities = [c.strip().lower() for c in itinerary_raw.split(',') if c.strip()]
    if not cities:
        return 0.0

    matches = 0
    for city in cities:
        if city in text_blob:
            matches += 1

    return matches / len(cities)


# ---------- RECOMMENDATION ----------
def recommend_operators(country, priority, itinerary_raw):
    subset = df[df['country'] == country].copy()
    if subset.empty:
        return []

    # Coverage score from band_count (normalized within country)
    max_bands = subset['band_count'].max() or 1
    subset['coverage_score'] = subset['band_count'] / max_bands

    # ML tech-maturity cluster
    X_sub = subset[feature_cols]
    clusters = pipeline.predict(X_sub)
    subset['cluster'] = clusters
    subset['cluster_score'] = subset['cluster'].map(cluster_score_map)

    # Itinerary match score
    subset['itinerary_score'] = subset.apply(
        lambda row: compute_itinerary_score(itinerary_raw, row), axis=1
    )

    # Combine scores based on priority (weights)
    # All scores are approx 0–1
    if priority == 'coverage':
        w_cov, w_tech, w_it = 0.6, 0.25, 0.15
    elif priority == '4g':
        w_cov, w_tech, w_it = 0.3, 0.55, 0.15
    elif priority == '5g':
        w_cov, w_tech, w_it = 0.25, 0.6, 0.15
    else:  # 'overall'
        w_cov, w_tech, w_it = 0.4, 0.4, 0.2

    subset['final_score'] = (
        w_cov * subset['coverage_score'] +
        w_tech * subset['cluster_score'] +
        w_it * subset['itinerary_score']
    )

    subset = subset.sort_values('final_score', ascending=False)

    results = []
    for _, row in subset.head(3).iterrows():
        results.append({
            "full_name": row.get('full_name'),
            "short_name": row.get('short_name'),
            "hq": row.get('hq'),
            "description": row.get('description'),
            "networks": row.get('networks'),
            "data_links": row.get('data_links'),
            "coverage_score": round(row.get('coverage_score', 0), 2),
            "cluster_score": round(row.get('cluster_score', 0), 2),
            "itinerary_score": round(row.get('itinerary_score', 0), 2),
            "final_score": round(row.get('final_score', 0), 2),
        })
    return results


# ---------- ROUTES ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    selected_country = None
    selected_priority = "overall"
    itinerary = ""
    recommendations = []

    if request.method == 'POST':
        selected_country = request.form.get('country')
        selected_priority = request.form.get('priority', 'overall')
        itinerary = request.form.get('itinerary', "")

        if selected_country:
            recommendations = recommend_operators(selected_country, selected_priority, itinerary)

    return render_template(
        'index.html',
        countries=countries,
        selected_country=selected_country,
        selected_priority=selected_priority,
        itinerary=itinerary,
        recommendations=recommendations
    )


if __name__ == '__main__':
    app.run(debug=True)
