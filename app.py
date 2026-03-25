import math
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from feature_extractor import extract_features_from_list, FEATURE_COLUMNS


st.set_page_config(
    page_title="PhisGuard: Phishing Detection Decision Support System",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "models/best_url_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

THREAT_TLDS = {".xyz", ".tk", ".zip", ".cf", ".ga", ".ml", ".top", ".work", ".support", ".click"}
CREDENTIAL_KEYWORDS = [
    "login", "verify", "account", "secure", "update", "bank", "payment",
    "signin", "wallet", "confirm", "password", "unlock", "billing"
]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def assign_risk_level(prob):
    if prob is None:
        return "Unknown"
    if prob >= 0.80:
        return "High"
    if prob >= 0.50:
        return "Medium"
    return "Low"


def label_name(value):
    return "Phishing" if int(value) == 1 else "Legitimate"


def risk_badge_html(risk_level):
    color_map = {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#22c55e",
        "Unknown": "#64748b"
    }
    color = color_map.get(risk_level, "#64748b")
    return f"""
    <div style="
        display:inline-block;
        padding:8px 16px;
        border-radius:999px;
        background:{color}22;
        color:{color};
        font-weight:700;
        font-size:18px;
        border:1px solid {color}55;
    ">
        {risk_level} Risk
    </div>
    """


def add_readable_labels(df):
    out = df.copy()
    out["prediction_label"] = out["prediction"].apply(label_name)
    return out


def predict_urls(urls, model, feature_columns):
    feature_df = extract_features_from_list(urls)
    # st.write(feature_df.columns.tolist())
    # st.write(feature_df.head())
    # st.write("Extractor columns:", feature_df.columns.tolist())
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
    return result_df


def gauge_chart(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
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
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="Risk Score chart")


def get_key_risk_factors(row):
    factors = []

    if row["url_entropy"] >= 4.5:
        factors.append("High URL Entropy")
    if row["has_hyphen_in_domain"] == 1:
        factors.append("Contains Hyphen in Domain")
    if row["suspicious_file_extension"] == 1:
        factors.append("Suspicious File Extension")
    if row["has_ip_address"] == 1:
        factors.append("Uses IP Address Instead of Domain")
    if row["subdomain_count"] >= 3:
        factors.append("Too Many Subdomains")
    if row["number_of_digits"] >= 8:
        factors.append("High Number of Digits")
    if row["url_length"] >= 75:
        factors.append("Unusually Long URL")
    if row["query_param_count"] >= 3:
        factors.append("Many Query Parameters")
    if row["https_flag"] == 0:
        factors.append("No HTTPS Detected")

    if not factors:
        factors.append("No Strong Suspicious Pattern Detected")

    return factors[:5]


def show_single_result(row):
    st.markdown("### Analysis Result")

    c1, c2, c3 = st.columns([1.4, 1.2, 1.2])

    with c1:
        st.markdown('<div class="card-title">URL Analysis</div>', unsafe_allow_html=True)
        st.markdown(
        f"""
        <div class="result-box">
            <div><b>Prediction:</b> {row['prediction_label']}</div>
            <div style="margin-top:10px;"><b>Phishing Probability:</b> {row['phishing_probability']:.2%}</div>
            <div style="margin-top:10px;"><b>Source Domain:</b> {row.get('root_domain', '-') or row.get('host', '-')}</div>
            <div style="margin-top:10px;"><b>Host:</b> {row.get('host', '-')}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    risk_level = row["risk_level"]
    prediction_label = row["prediction_label"]
    source_domain = row.get("root_domain", "") or row.get("host", "") or "unknown domain"

    risk_styles = {
        "High": {
            "bg": "#fee2e2",
            "border": "#fca5a5",
            "text": "#b91c1c",
            "icon": "⚠️"
        },
        "Medium": {
            "bg": "#fef3c7",
            "border": "#fcd34d",
            "text": "#b45309",
            "icon": "⚡"
        },
        "Low": {
            "bg": "#dcfce7",
            "border": "#86efac",
            "text": "#15803d",
            "icon": "✅"
        }
    }

    style = risk_styles.get(risk_level, risk_styles["Low"])

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
            margin:8px 0 18px 0;
            box-shadow: 0 6px 16px rgba(0,0,0,0.05);
        ">
            <span style="font-size:22px;">{style['icon']}</span>
            <span>{risk_level} Risk</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if row["prediction"] == 0:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
                border: 1.5px solid #86efac;
                border-left: 8px solid #22c55e;
                border-radius: 18px;
                padding: 22px 24px;
                margin: 6px 0 18px 0;
                box-shadow: 0 8px 20px rgba(34,197,94,0.08);
            ">
                <div style="
                    font-size:16px;
                    font-weight:700;
                    color:#166534;
                    margin-bottom:8px;
                    letter-spacing:0.2px;
                ">
                    Legitimate URL Detected
                </div>
                <div style="
                    font-size:18px;
                    line-height:1.6;
                    color:#166534;
                ">
                    This URL is predicted as <b>{prediction_label}</b> and appears to come from
                    <b>{source_domain}</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
                border: 1.5px solid #fca5a5;
                border-left: 8px solid #ef4444;
                border-radius: 18px;
                padding: 22px 24px;
                margin: 6px 0 18px 0;
                box-shadow: 0 8px 20px rgba(239,68,68,0.08);
            ">
                <div style="
                    font-size:16px;
                    font-weight:700;
                    color:#991b1b;
                    margin-bottom:8px;
                    letter-spacing:0.2px;
                ">
                    Phishing Warning
                </div>
                <div style="
                    font-size:18px;
                    line-height:1.6;
                    color:#991b1b;
                ">
                    This URL is predicted as <b>{prediction_label}</b>. Inspect the source domain
                    <b>{source_domain}</b> carefully before visiting it.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        gauge_chart(float(row["phishing_probability"]))

    with c3:
        st.markdown('<div class="card-title">Key Risk Factors</div>', unsafe_allow_html=True)
        factors = get_key_risk_factors(row)
        for factor in factors:
            st.markdown(
                f"""
                <div class="factor-item">
                    ⚠️ {factor}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Domain Information")
    show_domain_cards(row)
    
    st.markdown("### URL Indicators")
    show_url_indicator_cards(row)


def show_kpis(df):
    total = len(df)
    phishing_count = int((df["prediction"] == 1).sum())
    legitimate_count = int((df["prediction"] == 0).sum())
    avg_risk = float(df["phishing_probability"].mean()) if len(df) else 0.0
    high_risk = int((df["risk_level"] == "High").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total URLs", total)
    c2.metric("Phishing URLs", phishing_count)
    c3.metric("Legitimate URLs", legitimate_count)
    c4.metric("High Risk URLs", high_risk)

    st.markdown(
        f"""
        <div class="avg-risk-box">
            <b>Average Phishing Probability:</b> {avg_risk:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )


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
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key="Phishing vs Legitimate URLs chart")


def plot_feature_importance(model, feature_columns):
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance is available for Random Forest only.")
        return

    imp_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(8)

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
    imp_df["Feature"] = imp_df["Feature"].map(lambda x: pretty_names.get(x, x))

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True, key="Feature Importance chart")

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
        count = 0
        for url in url_series:
            if any(keyword in url for keyword in keywords):
                count += 1
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
        text="Count"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig, width="stretch", key="financial_sector_targeting_chart")


def plot_top_threat_tlds(df, top_n=5):
    if "tld" not in df.columns:
        st.info("TLD data not available.")
        return

    tld_df = df[df["prediction"] == 1].copy()
    tld_df["tld_display"] = "." + tld_df["tld"].fillna("").astype(str)

    chart_df = (
        tld_df[tld_df["tld_display"] != "."]
        ["tld_display"]
        .value_counts()
        .head(top_n)
        .rename_axis("TLD")
        .reset_index(name="Count")
    )

    if chart_df.empty:
        st.info("No phishing TLD patterns found yet.")
        return

    fig = px.bar(
        chart_df,
        x="Count",
        y="TLD",
        orientation="h",
        title="Top Threat TLDs"
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True, key="Top Threat TLDs chart")


def extract_keyword_counts(urls):
    counts = {kw: 0 for kw in CREDENTIAL_KEYWORDS}
    for url in urls:
        u = str(url).lower()
        for kw in CREDENTIAL_KEYWORDS:
            if kw in u:
                counts[kw] += 1

    rows = [{"Keyword": k.title(), "Count": v} for k, v in counts.items() if v > 0]
    return pd.DataFrame(rows)


def plot_keyword_chart(df):
    keyword_df = extract_keyword_counts(df["URL"].tolist())
    if keyword_df.empty:
        st.info("No credential-related keywords detected in current URLs.")
        return

    keyword_df = keyword_df.sort_values("Count", ascending=False).head(8)
    fig = px.bar(
        keyword_df,
        x="Keyword",
        y="Count",
        title="Credential Harvesting Keywords"
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key="Credential Harvesting Keywords chart")


def plot_suspicious_patterns(df):
    rows = [
        {"Pattern": "IP Address", "Count": int(df["has_ip_address"].sum())},
        {"Pattern": "Hyphens in Domain", "Count": int(df["has_hyphen_in_domain"].sum())},
        {"Pattern": "Suspicious Extensions", "Count": int(df["suspicious_file_extension"].sum())},
        {"Pattern": "No HTTPS", "Count": int((df["https_flag"] == 0).sum())},
    ]
    chart_df = pd.DataFrame(rows)

    fig = px.bar(
        chart_df,
        x="Pattern",
        y="Count",
        title="Suspicious URL Patterns"
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key="Suspicious URL Patterns chart")


def plot_probability_distribution(df):
    if len(df) < 8:
        st.info("Probability distribution becomes more useful when analyzing a larger batch of URLs.")
        return

    fig = px.histogram(
        df,
        x="phishing_probability",
        nbins=20,
        title="Phishing Probability Distribution"
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key="Phishing Probability Distribution chart")


def plot_top_domains(df, top_n=8):
    chart_df = (
        df["root_domain"]
        .fillna("unknown")
        .replace("", "unknown")
        .value_counts()
        .head(top_n)
        .rename_axis("Root Domain")
        .reset_index(name="Count")
    )

    fig = px.bar(
        chart_df,
        x="Root Domain",
        y="Count",
        title="Top Source Domains"
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key="Top Source Domains chart")

def plot_phishing_url_patterns(df):
    phishing_df = df[df["prediction"] == 1].copy()

    if phishing_df.empty:
        st.info("No phishing URLs detected, so no phishing-specific URL patterns are available.")
        return

    rows = [
        {"Pattern": "IP Address", "Count": int(phishing_df["has_ip_address"].sum())},
        {"Pattern": "Hyphens in Domain", "Count": int(phishing_df["has_hyphen_in_domain"].sum())},
        {"Pattern": "Suspicious Extensions", "Count": int(phishing_df["suspicious_file_extension"].sum())},
        {"Pattern": "No HTTPS", "Count": int((phishing_df["https_flag"] == 0).sum())},
        {"Pattern": "Long URL", "Count": int((phishing_df["url_length"] >= 75).sum())},
        {"Pattern": "Many Subdomains", "Count": int((phishing_df["subdomain_count"] >= 3).sum())},
        {"Pattern": "Many Digits", "Count": int((phishing_df["number_of_digits"] >= 8).sum())},
    ]

    chart_df = pd.DataFrame(rows)

    fig = px.bar(
        chart_df,
        x="Pattern",
        y="Count",
        title="Phishing URL Patterns",
        text="Count"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="",
        yaxis_title="Count"
    )
    st.plotly_chart(fig, use_container_width=True, key="Phishing URL Patterns chart")

def plot_suspicious_keyword_frequency(df):
    phishing_df = df[df["prediction"] == 1].copy()

    if phishing_df.empty:
        st.info("No phishing URLs detected, so no suspicious keyword frequency chart is available.")
        return

    suspicious_keywords = {
        "Login": ["login", "signin", "sign-in"],
        "Verify": ["verify", "verification", "confirm"],
        "Account": ["account", "profile"],
        "Secure": ["secure", "security", "safe"],
        "Update": ["update", "updated"],
        "Bank": ["bank", "banking"],
        "Payment": ["payment", "pay", "billing", "invoice"],
        "Password": ["password", "passwd"],
        "Wallet": ["wallet", "card", "credit", "debit"],
        "Alert": ["alert", "warning", "notice"],
    }

    url_series = phishing_df["URL"].fillna("").astype(str).str.lower()

    rows = []
    for label, keywords in suspicious_keywords.items():
        count = 0
        for url in url_series:
            if any(keyword in url for keyword in keywords):
                count += 1
        rows.append({"Keyword": label, "Count": count})

    chart_df = pd.DataFrame(rows)
    chart_df = chart_df[chart_df["Count"] > 0].sort_values("Count", ascending=False)

    if chart_df.empty:
        st.info("No strong phishing-related keywords were found in the current phishing URLs.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=chart_df["Keyword"],
        y=chart_df["Count"],
        text=chart_df["Count"],
        textposition="outside",
        marker=dict(
            color="#e74c3c",
            line=dict(color="#c0392b", width=1.5)
        ),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))

    fig.update_layout(
        title={
            "text": "Credential Harvesting Keywords",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 18, "color": "#163a63"}
        },
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(
            title="",
            tickfont=dict(size=13, color="#163a63"),
            showgrid=False,
            showline=True,
            linecolor="#6b7f99",
            linewidth=2
        ),
        yaxis=dict(
            title="Count",
            tickfont=dict(size=12, color="#163a63"),
            showgrid=True,
            gridcolor="#d9e2ec",
            griddash="dot",
            zeroline=False,
            showline=True,
            linecolor="#6b7f99",
            linewidth=2
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="suspicious_keyword_frequency_chart")

def results_table(df):
    st.markdown("### Prediction Results")

    preferred_cols = [
        "URL", "host", "root_domain", "subdomain", "tld",
        "prediction_label", "phishing_probability", "risk_level",
        "url_length", "https_flag", "subdomain_count",
        "number_of_digits", "url_entropy"
    ]

    available_cols = [col for col in preferred_cols if col in df.columns]

    if not available_cols:
        st.warning("No display columns are available in the prediction results.")
        return

    st.dataframe(df[available_cols], use_container_width=True, height=320)

def make_download_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(180deg, #eef4fb 0%, #f8fbff 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #163a63;
    }
    .card-title {
        font-size: 20px;
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
    .avg-risk-box {
        margin-top: 10px;
        padding: 12px 16px;
        border-radius: 12px;
        background: #eaf3ff;
        border: 1px solid #cfe0f5;
        color: #163a63;
        font-size: 16px;
    }
    .hero-box {
        background: linear-gradient(135deg, #dfeefe 0%, #f8fbff 100%);
        border: 1px solid #d7e6f5;
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)


def show_url_indicator_cards(row):
    def render_card(title, value, subtitle="", color="#163a63"):
        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #dbe7f3;
                border-radius:16px;
                padding:16px 18px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.05);
                min-height:120px;
            ">
                <div style="font-size:14px; color:#64748b; font-weight:600; margin-bottom:10px;">
                    {title}
                </div>
                <div style="font-size:28px; font-weight:800; color:{color}; line-height:1.1;">
                    {value}
                </div>
                <div style="font-size:13px; color:#64748b; margin-top:8px;">
                    {subtitle}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        https_value = "Yes" if int(row["https_flag"]) == 1 else "No"
        https_color = "#16a34a" if int(row["https_flag"]) == 1 else "#dc2626"
        render_card("HTTPS", https_value, "Secure protocol detected" if https_value == "Yes" else "No HTTPS detected", https_color)

    with c2:
        ip_value = "Yes" if int(row["has_ip_address"]) == 1 else "No"
        ip_color = "#dc2626" if int(row["has_ip_address"]) == 1 else "#16a34a"
        render_card("IP Address", ip_value, "Uses numeric IP instead of domain" if ip_value == "Yes" else "Domain name used", ip_color)

    with c3:
        sub_count = int(row["subdomain_count"])
        sub_color = "#f59e0b" if sub_count >= 3 else "#163a63"
        render_card("Subdomains", str(sub_count), "High count can be suspicious" if sub_count >= 3 else "Normal range", sub_color)

    with c4:
        digit_count = int(row["number_of_digits"])
        digit_color = "#f59e0b" if digit_count >= 8 else "#163a63"
        render_card("Digits in URL", str(digit_count), "Many digits may be suspicious" if digit_count >= 8 else "Normal range", digit_color)

def show_domain_cards(row):
    def domain_card(title, value):
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 1px solid #dbe7f3;
                border-radius: 16px;
                padding: 18px 20px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.05);
                min-height: 120px;
            ">
                <div style="
                    font-size: 14px;
                    color: #64748b;
                    font-weight: 600;
                    margin-bottom: 12px;
                ">
                    {title}
                </div>
                <div style="
                    font-size: 34px;
                    font-weight: 800;
                    color: #1e293b;
                    line-height: 1.1;
                    word-break: break-word;
                ">
                    {value}
                </div>
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





def main():
    apply_custom_css()
    st.markdown('<div class="hero-box"><h1>🛡️ PhishGuard: Phishing Detection Decision Support System</h1></div>', unsafe_allow_html=True)

    try:
        model, feature_columns = load_model()
    except Exception as e:
        st.error(f"Failed to load model files: {e}")
        st.stop()

    st.markdown("## URL Analysis")

    col_input, col_mode = st.columns([3, 1])
    with col_input:
        analysis_mode = st.radio("Choose input mode", ["Single URL", "Batch URLs"], horizontal=True)

    result_df = None
    single_row = None

    if analysis_mode == "Single URL":
        url = st.text_input("Enter URL for analysis", placeholder="https://example.com/login")
        if st.button("Analyze", type="primary"):
            if not url.strip():
                st.warning("Please enter a URL.")
            else:
                result_df = predict_urls([url], model, feature_columns)
                single_row = result_df.iloc[0]

    else:
        input_tab, upload_tab = st.tabs(["Paste URLs", "Upload CSV"])

        with input_tab:
            pasted = st.text_area(
                "Paste one URL per line",
                height=180,
                placeholder="https://example.com\nhttp://suspicious-site.biz/login\nwww.sample.org"
            )

        with upload_tab:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if st.button("Analyze Batch", type="primary"):
            urls = []

            if uploaded_file is not None:
                df_up = pd.read_csv(uploaded_file)
                st.write("Uploaded file preview:")
                st.dataframe(df_up.head(), use_container_width=True)
                url_column = st.selectbox("Select URL column", df_up.columns.tolist())
                urls = df_up[url_column].dropna().astype(str).tolist()
            else:
                urls = [line.strip() for line in pasted.splitlines() if line.strip()]

            if not urls:
                st.warning("Please provide at least one URL.")
            else:
                result_df = predict_urls(urls, model, feature_columns)

    if single_row is not None:
        show_single_result(single_row)
        
        

    if result_df is not None:
        result_df = add_readable_labels(result_df)

        if len(result_df) > 1:
            st.markdown("## Batch Intelligence Overview")
            show_kpis(result_df)

            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                plot_prediction_pie(result_df)
            with r1c2:
                plot_feature_importance(model, feature_columns)
            with r1c3:
                plot_financial_sector_targeting(result_df)

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                plot_suspicious_keyword_frequency(result_df)
            with r2c2:
                plot_phishing_url_patterns(result_df)
          
            
            results_table(result_df)
            
            
                
            st.download_button(
                label="Download Results CSV",
                data=make_download_csv(result_df),
                file_name="url_prediction_results.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()  