
import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from feature_extractor import extract_features_from_list


st.set_page_config(
    page_title="PhishGuard: Phishing Detection Decision Support System",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "models/best_url_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
HISTORY_PATH = "data/analysis_history.csv"

CREDENTIAL_KEYWORDS = [
    "login", "verify", "account", "secure", "update", "bank", "payment",
    "signin", "wallet", "confirm", "password", "unlock", "billing"
]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def ensure_history_store():
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
        pd.DataFrame().to_csv(HISTORY_PATH, index=False)


def safe_probability(value):
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def assign_risk_level(prob):
    if prob is None or pd.isna(prob):
        return "Unknown"
    if prob >= 0.80:
        return "High"
    if prob >= 0.50:
        return "Medium"
    return "Low"


def label_name(value):
    return "Phishing" if int(value) == 1 else "Legitimate"


def predict_urls(urls, model, feature_columns):
    feature_df = extract_features_from_list(urls)
    X = feature_df[feature_columns].copy()

    preds = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = [None] * len(X)

    result_df = feature_df.copy()
    result_df["prediction"] = preds
    result_df["phishing_probability"] = probs
    result_df["risk_level"] = result_df["phishing_probability"].apply(assign_risk_level)
    result_df["prediction_label"] = result_df["prediction"].apply(label_name)
    result_df["analyzed_at"] = datetime.now().isoformat()
    return result_df


def append_history(df):
    ensure_history_store()
    history_df = load_history()
    combined = pd.concat([history_df, df], ignore_index=True)
    combined.to_csv(HISTORY_PATH, index=False)


def load_history():
    ensure_history_store()
    try:
        history_df = pd.read_csv(HISTORY_PATH)
    except pd.errors.EmptyDataError:
        history_df = pd.DataFrame()

    if "analyzed_at" in history_df.columns:
        history_df["analyzed_at"] = pd.to_datetime(history_df["analyzed_at"], errors="coerce")
    return history_df


def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #eef4fb 0%, #f8fbff 100%);
    }
    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3 {
        color: #163a63;
    }
    .hero-box {
        background: linear-gradient(135deg, #dfeefe 0%, #f8fbff 100%);
        border: 1px solid #d7e6f5;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #163a63;
        margin-bottom: 12px;
    }
    .result-box {
        background: white;
        border: 1px solid #dbe7f3;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
    .factor-item {
        background: white;
        border-left: 5px solid #f59e0b;
        padding: 12px 14px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        font-weight: 600;
        color: #334155;
    }
    .metric-card {
        background: white;
        border: 1px solid #dbe7f3;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        height: 100%;
    }
    .metric-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #163a63;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.1;
    }
    .small-note {
        color: #64748b;
        font-size: 13px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=safe_probability(prob) * 100,
        number={"suffix": "%"},
        title={"text": "Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.30},
            "steps": [
                {"range": [0, 50], "color": "#15803d"},
                {"range": [50, 80], "color": "#f59e0b"},
                {"range": [80, 100], "color": "#dc2626"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def get_key_risk_factors(row):
    factors = []
    if row.get("url_entropy", 0) >= 4.5:
        factors.append("High URL Entropy")
    if int(row.get("has_hyphen_in_domain", 0)) == 1:
        factors.append("Contains Hyphen in Domain")
    if int(row.get("suspicious_file_extension", 0)) == 1:
        factors.append("Suspicious File Extension")
    if int(row.get("has_ip_address", 0)) == 1:
        factors.append("Uses IP Address Instead of Domain")
    if row.get("subdomain_count", 0) >= 3:
        factors.append("Too Many Subdomains")
    if row.get("number_of_digits", 0) >= 8:
        factors.append("High Number of Digits")
    if row.get("url_length", 0) >= 75:
        factors.append("Unusually Long URL")
    if row.get("query_param_count", 0) >= 3:
        factors.append("Many Query Parameters")
    if int(row.get("https_flag", 1)) == 0:
        factors.append("No HTTPS Detected")

    if not factors:
        factors.append("No Strong Suspicious Pattern Detected")
    return factors[:5]


def plot_feature_importance(model, feature_columns, title="Feature Importance", top_n=8):
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is available for tree-based models only.")
        return

    pretty_names = {
        "url_length": "URL Length",
        "url_entropy": "URL Entropy",
        "subdomain_count": "Subdomain Count",
        "has_ip_address": "Has IP Address",
        "query_param_count": "Query Param Count",
        "number_of_digits": "Number of Digits",
        "domain_name_length": "Domain Name Length",
        "https_flag": "HTTPS Flag",
        "dot_count": "Dot Count",
        "tld_length": "TLD Length",
        "path_length": "Path Length",
        "has_hyphen_in_domain": "Hyphen in Domain",
        "suspicious_file_extension": "Suspicious Extension",
        "token_count": "Token Count",
        "percentage_numeric_chars": "Numeric Char Ratio"
    }

    imp_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(top_n)

    imp_df["Feature"] = imp_df["Feature"].map(lambda x: pretty_names.get(x, x))

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_distribution(df, title="Risk Score Distribution"):
    if df.empty:
        st.info("No data available.")
        return

    fig = px.histogram(
        df,
        x="phishing_probability",
        nbins=20,
        title=title
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Phishing Probability",
        yaxis_title="URL Count"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_prediction_pie(df):
    chart_df = (
        df["prediction_label"]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    fig = px.pie(
        chart_df,
        names="Class",
        values="Count",
        title="Phishing vs Legitimate URLs",
        hole=0.35
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def plot_financial_sector_targeting(df):
    target_keywords = {
        "Bank": ["bank", "banking"],
        "Login": ["login", "signin", "sign-in"],
        "Secure": ["secure", "security", "safe"],
        "Account": ["account", "profile"],
        "Verify": ["verify", "verification", "confirm"],
        "Payment": ["payment", "pay", "billing", "invoice"],
        "Wallet": ["wallet", "card", "credit", "debit"],
    }

    rows = []
    url_series = df["URL"].fillna("").astype(str).str.lower()

    for label, keywords in target_keywords.items():
        count = sum(any(keyword in url for keyword in keywords) for url in url_series)
        rows.append({"Category": label, "Count": count})

    chart_df = pd.DataFrame(rows).sort_values("Count", ascending=False)

    if chart_df["Count"].sum() == 0:
        st.info("No strong finance-related targeting keywords detected in the current URLs.")
        return

    fig = px.bar(
        chart_df,
        x="Category",
        y="Count",
        title="Financial Sector Targeting",
        text="Count",
        color="Category",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def build_feature_details_table(row):
    feature_rows = [
        ("Length of URL (characters)", int(row.get("url_length", 0))),
        ("Domain", row.get("root_domain", "-") or row.get("host", "-") or "-"),
        ("Host", row.get("host", "-") or "-"),
        ("TLD", row.get("tld", "-") or "-"),
        ("HTTPS", "Yes" if int(row.get("https_flag", 0)) == 1 else "No"),
        ("Subdomain Count", int(row.get("subdomain_count", 0))),
        ("Digits in URL", int(row.get("number_of_digits", 0))),
        ("URL Entropy", round(float(row.get("url_entropy", 0.0)), 3)),
        ("Has IP Address", "Yes" if int(row.get("has_ip_address", 0)) == 1 else "No"),
        ("Suspicious Extension", "Yes" if int(row.get("suspicious_file_extension", 0)) == 1 else "No"),
    ]
    return pd.DataFrame(feature_rows, columns=["Feature", "Analyze URL details"])


def show_domain_cards(row):
    def domain_card(title, value):
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{title}</div>
                <div style="font-size: 24px; font-weight: 800; color: #1e293b; word-break: break-word;">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        domain_card("Host", row.get("host", "-") or "-")
    with c2:
        domain_card("Root Domain", row.get("root_domain", "-") or "-")
    with c3:
        domain_card("Subdomain", row.get("subdomain", "-") or "-")
    with c4:
        domain_card("TLD", row.get("tld", "-") or "-")


def show_url_indicator_cards(row):
    def render_card(title, value, subtitle="", color="#163a63"):
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{title}</div>
                <div style="font-size: 28px; font-weight: 800; color:{color}; line-height:1.1;">{value}</div>
                <div class="small-note">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        https_value = "Yes" if int(row.get("https_flag", 0)) == 1 else "No"
        https_color = "#16a34a" if https_value == "Yes" else "#dc2626"
        render_card("HTTPS", https_value, "Secure protocol detected" if https_value == "Yes" else "No HTTPS detected", https_color)

    with c2:
        ip_value = "Yes" if int(row.get("has_ip_address", 0)) == 1 else "No"
        ip_color = "#dc2626" if ip_value == "Yes" else "#16a34a"
        render_card("IP Address", ip_value, "Uses numeric IP instead of domain" if ip_value == "Yes" else "Domain name used", ip_color)

    with c3:
        sub_count = int(row.get("subdomain_count", 0))
        sub_color = "#f59e0b" if sub_count >= 3 else "#163a63"
        render_card("Subdomains", str(sub_count), "High count can be suspicious" if sub_count >= 3 else "Normal range", sub_color)

    with c4:
        digit_count = int(row.get("number_of_digits", 0))
        digit_color = "#f59e0b" if digit_count >= 8 else "#163a63"
        render_card("Digits in URL", str(digit_count), "Many digits may be suspicious" if digit_count >= 8 else "Normal range", digit_color)


def show_result_summary(row):
    risk_level = row["risk_level"]
    prediction_label = row["prediction_label"]
    source_domain = row.get("root_domain", "") or row.get("host", "") or "unknown domain"

    risk_styles = {
        "High": {"bg": "#fee2e2", "border": "#fca5a5", "text": "#b91c1c", "icon": "⚠️"},
        "Medium": {"bg": "#fef3c7", "border": "#fcd34d", "text": "#b45309", "icon": "⚡"},
        "Low": {"bg": "#dcfce7", "border": "#86efac", "text": "#15803d", "icon": "✅"},
        "Unknown": {"bg": "#e2e8f0", "border": "#cbd5e1", "text": "#334155", "icon": "ℹ️"},
    }
    style = risk_styles.get(risk_level, risk_styles["Unknown"])

    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            gap:10px;
            padding:12px 22px;
            border-radius:999px;
            background:{style['bg']};
            color:{style['text']};
            border:2px solid {style['border']};
            font-weight:800;
            font-size:20px;
            margin:0 0 18px 0;
            box-shadow: 0 6px 16px rgba(0,0,0,0.05);
        ">
            <span style="font-size:22px;">{style['icon']}</span>
            <span>{risk_level} Risk</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if row["prediction"] == 0:
        box = f"""
        <div style="
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border: 1.5px solid #86efac;
            border-left: 8px solid #22c55e;
            border-radius: 18px;
            padding: 22px 24px;
            margin: 6px 0 18px 0;
            box-shadow: 0 8px 20px rgba(34,197,94,0.08);">
            <div style="font-size:16px;font-weight:700;color:#166534;margin-bottom:8px;">Legitimate URL Detected</div>
            <div style="font-size:18px;line-height:1.6;color:#166534;">
                This URL is predicted as <b>{prediction_label}</b> and appears to come from <b>{source_domain}</b>.
            </div>
        </div>
        """
    else:
        box = f"""
        <div style="
            background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
            border: 1.5px solid #fca5a5;
            border-left: 8px solid #ef4444;
            border-radius: 18px;
            padding: 22px 24px;
            margin: 6px 0 18px 0;
            box-shadow: 0 8px 20px rgba(239,68,68,0.08);">
            <div style="font-size:16px;font-weight:700;color:#991b1b;margin-bottom:8px;">Phishing Warning</div>
            <div style="font-size:18px;line-height:1.6;color:#991b1b;">
                This URL is predicted as <b>{prediction_label}</b>. Inspect the source domain <b>{source_domain}</b> carefully before visiting it.
            </div>
        </div>
        """
    st.markdown(box, unsafe_allow_html=True)


def render_single_page(model, feature_columns):
    st.markdown("## Single URL Analysis")
    url = st.text_input("Enter URL for analysis", placeholder="https://example.com/login", key="single_url_input")

    if st.button("Analyze URL", type="primary", key="single_url_button"):
        if not url.strip():
            st.warning("Please enter a URL.")
            return

        result_df = predict_urls([url], model, feature_columns)
        append_history(result_df)
        st.session_state["single_result_df"] = result_df

    result_df = st.session_state.get("single_result_df")
    if result_df is None or result_df.empty:
        return

    row = result_df.iloc[0]

    st.markdown("### Analysis Result")
    show_result_summary(row)

    c1, c2, c3 = st.columns([1.3, 1.1, 1.1])

    with c1:
        st.markdown('<div class="card-title">URL Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="result-box">
                <div><b>Prediction:</b> {row['prediction_label']}</div>
                <div style="margin-top:10px;"><b>Phishing Probability:</b> {safe_probability(row['phishing_probability']):.2%}</div>
                <div style="margin-top:10px;"><b>Source Domain:</b> {row.get('root_domain', '-') or row.get('host', '-')}</div>
                <div style="margin-top:10px;"><b>Host:</b> {row.get('host', '-')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        gauge_chart(row["phishing_probability"])

    with c3:
        st.markdown('<div class="card-title">Key Risk Factors</div>', unsafe_allow_html=True)
        for factor in get_key_risk_factors(row):
            st.markdown(f'<div class="factor-item">⚠️ {factor}</div>', unsafe_allow_html=True)

    st.markdown("### Feature Importance")
    plot_feature_importance(model, feature_columns, title="Model Feature Importance")

    st.markdown("### Important Feature Details")
    feature_details = build_feature_details_table(row)
    st.dataframe(feature_details, use_container_width=True, hide_index=True)

    st.markdown("### Domain Information")
    show_domain_cards(row)

    st.markdown("### URL Indicators")
    show_url_indicator_cards(row)


def show_batch_kpis(df):
    total = len(df)
    phishing_count = int((df["prediction"] == 1).sum())
    legitimate_count = int((df["prediction"] == 0).sum())
    avg_risk = df["phishing_probability"].fillna(0).mean()
    high_risk = int((df["risk_level"] == "High").sum())
    phishing_pct = (phishing_count / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total URLs", total, "URLs in current batch")
    with c2:
        render_metric_card("Phishing URLs", phishing_count, f"{phishing_pct:.1f}% of current batch")
    with c3:
        render_metric_card("Legitimate URLs", legitimate_count, "Predicted safe URLs")
    with c4:
        render_metric_card("High Risk URLs", high_risk, "URLs with probability ≥ 80%")

    st.markdown(
        f"""
        <div class="result-box">
            <b>Average Phishing Probability:</b> {avg_risk:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )


def results_table(df):
    st.markdown("### Prediction Results")
    table_df = df.copy()
    table_df["phishing_probability"] = table_df["phishing_probability"].fillna(0.0)
    table_df = table_df.sort_values("phishing_probability", ascending=False)

    preferred_cols = [
        "URL",
        "prediction_label",
        "risk_level",
        "phishing_probability",
        "host",
        "root_domain",
        "subdomain",
        "tld",
        "url_length",
        "https_flag",
        "subdomain_count",
        "number_of_digits",
        "url_entropy",
        "analyzed_at",
    ]
    available_cols = [col for col in preferred_cols if col in table_df.columns]
    renamed = {
        "URL": "URL",
        "prediction_label": "Predicted Result",
        "risk_level": "Risk Level",
        "phishing_probability": "Risk Score",
        "host": "Host",
        "root_domain": "Root Domain",
        "subdomain": "Subdomain",
        "tld": "TLD",
        "url_length": "URL Length",
        "https_flag": "HTTPS",
        "subdomain_count": "Subdomain Count",
        "number_of_digits": "Digits in URL",
        "url_entropy": "URL Entropy",
        "analyzed_at": "Analyzed At",
    }
    display_df = table_df[available_cols].rename(columns=renamed)

    st.dataframe(display_df, use_container_width=True, height=360, hide_index=True)


def make_download_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def render_batch_page(model, feature_columns):
    st.markdown("## Batch URL Analysis")

    input_tab, upload_tab = st.tabs(["Paste URLs", "Upload CSV"])
    pasted = ""
    uploaded_file = None
    selected_url_column = None

    with input_tab:
        pasted = st.text_area(
            "Paste one URL per line",
            height=180,
            placeholder="https://example.com\nhttp://suspicious-site.biz/login\nwww.sample.org",
            key="batch_paste_box"
        )

    with upload_tab:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload_csv")
        if uploaded_file is not None:
            uploaded_preview = pd.read_csv(uploaded_file)
            st.dataframe(uploaded_preview.head(), use_container_width=True)
            selected_url_column = st.selectbox("Select URL column", uploaded_preview.columns.tolist(), key="batch_url_column")
            uploaded_file.seek(0)

    if st.button("Analyze Batch", type="primary", key="batch_button"):
        urls = []
        if uploaded_file is not None:
            upload_df = pd.read_csv(uploaded_file)
            urls = upload_df[selected_url_column].dropna().astype(str).tolist()
        else:
            urls = [line.strip() for line in pasted.splitlines() if line.strip()]

        if not urls:
            st.warning("Please provide at least one URL.")
        else:
            result_df = predict_urls(urls, model, feature_columns)
            append_history(result_df)
            st.session_state["batch_result_df"] = result_df

    result_df = st.session_state.get("batch_result_df")
    if result_df is None or result_df.empty:
        return

    st.markdown("### Batch Intelligence Overview")
    show_batch_kpis(result_df)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        plot_risk_distribution(result_df, title="Risk Score Distribution")
    with row2_col2:
        plot_feature_importance(model, feature_columns, title="Feature Importance")

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        plot_prediction_pie(result_df)
    with row3_col2:
        plot_financial_sector_targeting(result_df)

    results_table(result_df)

    st.download_button(
        label="Download Results CSV",
        data=make_download_csv(result_df),
        file_name="url_prediction_results.csv",
        mime="text/csv"
    )


def filter_dashboard_data(df, period_label, tld_filter, custom_start, custom_end):
    filtered = df.copy()

    if filtered.empty:
        return filtered

    now = pd.Timestamp.now()

    if period_label == "Last 24 hours":
        filtered = filtered[filtered["analyzed_at"] >= now - pd.Timedelta(hours=24)]
    elif period_label == "Last 7 days":
        filtered = filtered[filtered["analyzed_at"] >= now - pd.Timedelta(days=7)]
    elif period_label == "Last 30 days":
        filtered = filtered[filtered["analyzed_at"] >= now - pd.Timedelta(days=30)]
    elif period_label == "Custom range" and custom_start and custom_end:
        start = pd.to_datetime(custom_start)
        end = pd.to_datetime(custom_end) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["analyzed_at"] >= start) & (filtered["analyzed_at"] < end)]

    if tld_filter != "All":
        normalized_tld = tld_filter.lstrip(".")
        filtered = filtered[filtered["tld"].fillna("").astype(str).str.lower() == normalized_tld.lower()]

    return filtered


def render_dashboard():
    st.markdown("## Dashboard")
    history_df = load_history()

    if history_df.empty:
        st.info("No analysis history found yet. Analyze some URLs first to populate the dashboard.")
        return

    if "prediction_label" not in history_df.columns and "prediction" in history_df.columns:
        history_df["prediction_label"] = history_df["prediction"].apply(label_name)
    if "risk_level" not in history_df.columns and "phishing_probability" in history_df.columns:
        history_df["risk_level"] = history_df["phishing_probability"].apply(assign_risk_level)

    filter_col1, filter_col2, filter_col3 = st.columns([1.4, 1.2, 1.6])

    with filter_col1:
        period_label = st.selectbox(
            "Time filter",
            ["Last 24 hours", "Last 7 days", "Last 30 days", "Custom range"],
            index=1,
            key="dashboard_period"
        )

    with filter_col2:
        available_tlds = sorted([
            f".{tld}" for tld in history_df["tld"].dropna().astype(str).unique() if str(tld).strip()
        ])
        tld_filter = st.selectbox("Filter by TLD", ["All"] + available_tlds, key="dashboard_tld")

    custom_start = None
    custom_end = None
    with filter_col3:
        if period_label == "Custom range":
            cstart, cend = st.columns(2)
            with cstart:
                custom_start = st.date_input("Start date", key="dashboard_start")
            with cend:
                custom_end = st.date_input("End date", key="dashboard_end")

    filtered = filter_dashboard_data(history_df, period_label, tld_filter, custom_start, custom_end)

    total_urls = len(filtered)
    phishing_urls = int((filtered["prediction"] == 1).sum()) if total_urls else 0
    phishing_pct = (phishing_urls / total_urls * 100) if total_urls else 0.0
    avg_risk = filtered["phishing_probability"].fillna(0).mean() if total_urls else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        render_metric_card("Total URLs analyzed", total_urls, "Filtered time period")
    with k2:
        render_metric_card("% Phishing URLs", f"{phishing_pct:.1f}%", "Filtered time period")
    with k3:
        render_metric_card("Avg risk score", f"{avg_risk:.2%}", "Filtered time period")

    st.markdown("### Risk Distribution")
    plot_risk_distribution(filtered, title="Risk Distribution")

    st.markdown("### Lifetime Metrics")
    lifetime_total = len(history_df)
    lifetime_phishing = int((history_df["prediction"] == 1).sum()) if lifetime_total else 0

    l1, l2 = st.columns(2)
    with l1:
        render_metric_card("Total URLs ever analyzed", lifetime_total, "All recorded analyses")
    with l2:
        render_metric_card("Total phishing detected", lifetime_phishing, "All recorded analyses")

    plot_risk_distribution(history_df, title="Lifetime Risk Distribution")


def main():
    apply_custom_css()

    st.markdown(
        '<div class="hero-box"><h1>🛡️ PhishGuard: Phishing Detection Decision Support System</h1></div>',
        unsafe_allow_html=True
    )

    try:
        model, feature_columns = load_model()
    except Exception as e:
        st.error(f"Failed to load model files: {e}")
        st.stop()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to page", ["Dashboard", "Single URL Analysis", "Batch URL Analysis"])

    if page == "Dashboard":
        render_dashboard()
    elif page == "Single URL Analysis":
        render_single_page(model, feature_columns)
    else:
        render_batch_page(model, feature_columns)


if __name__ == "__main__":
    main()
