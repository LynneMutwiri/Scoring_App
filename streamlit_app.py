import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SACCO Credit Scoring Dashboard",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA
# =========================================================
model_package = joblib.load("credit_default_best_model.joblib")
features = model_package["selected_features"]
model_name = model_package.get("model_name", "Model")
metrics = model_package.get("validation_metrics", {})
target_col = model_package.get("target_col", "Target")
df = pd.read_csv("snapshot_training_set.csv")

possible_id_cols = ["MemberNumber", "member_id", "Mem No", "Member No", "CustomerID"]
id_col = next((col for col in possible_id_cols if col in df.columns), None)

pipeline = model_package["pipeline"]
final_model = pipeline.named_steps.get("model", None)

# =========================================================
# CUSTOM THEME
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #050505 0%, #0b0f0b 55%, #111811 100%);
    color: #f5f5f5;
}

/* reduce top whitespace */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.8rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* headings */
h1, h2, h3, h4 {
    color: #d9ff57 !important;
}
p, label, div, span {
    color: #f3f3f3;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0b0b 0%, #111611 100%);
    border-right: 1px solid rgba(217,255,87,0.15);
}

/* cards */
.metric-card {
    background: linear-gradient(135deg, rgba(18,18,18,0.95), rgba(25,30,20,0.92));
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(217,255,87,0.18);
    box-shadow: 0 0 18px rgba(201,255,72,0.08);
    margin-bottom: 14px;
}
.glow-card {
    background: linear-gradient(135deg, rgba(217,255,87,0.12), rgba(255,214,10,0.10));
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(217,255,87,0.25);
    box-shadow: 0 0 16px rgba(217,255,87,0.16);
    margin-bottom: 14px;
}

.small-label {
    font-size: 13px;
    color: #d6d6d6 !important;
    margin-bottom: 6px;
}
.big-value {
    font-size: 26px;
    font-weight: 800;
    color: #d9ff57 !important;
}
.section-label {
    font-size: 20px;
    font-weight: 800;
    color: #ffe44d !important;
    margin-bottom: 10px;
}

/* risk boxes */
.risk-high {
    background: linear-gradient(135deg, #2d1200, #4d1f00);
    color: #ffd2a6;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #ff9b42;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
}
.risk-medium {
    background: linear-gradient(135deg, #3c3500, #5f5100);
    color: #fff0a8;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #ffd84d;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
}
.risk-low {
    background: linear-gradient(135deg, #0a2d12, #13451d);
    color: #c9ff72;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #7eff9e;
    text-align: center;
    font-size: 22px;
    font-weight: 800;
}

/* buttons */
div.stButton > button {
    background: linear-gradient(90deg, #d9ff57 0%, #ffe44d 100%);
    color: black !important;
    border: none;
    border-radius: 14px;
    font-weight: 800;
    padding: 0.65rem 1.1rem;
}
div.stButton > button:hover {
    box-shadow: 0 0 18px rgba(217,255,87,0.28);
}

/* input fields */
input {
    color: black !important;
    background-color: #f5f5f5 !important;
}
label {
    color: #d9ff57 !important;
}

/* select boxes: selected value */
div[data-baseweb="select"] > div {
    background-color: #d9ff57 !important;
    color: black !important;
}
div[data-baseweb="select"] * {
    color: black !important;
}

/* dropdown menu options */
ul[role="listbox"] {
    background-color: #f5f5f5 !important;
}
ul[role="listbox"] li {
    color: black !important;
    background-color: #f5f5f5 !important;
}
ul[role="listbox"] li:hover {
    background-color: #d9ff57 !important;
    color: black !important;
}
div[role="option"] {
    color: black !important;
    background-color: #f5f5f5 !important;
}
div[role="option"]:hover {
    background-color: #d9ff57 !important;
    color: black !important;
}

/* radio labels */
div[role="radiogroup"] label {
    color: #f5f5f5 !important;
}

/* dataframes */
[data-testid="stDataFrame"] {
    background-color: rgba(15,15,15,0.9) !important;
    border-radius: 14px;
    padding: 6px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def safe_value(val):
    if pd.isna(val):
        return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0

def nice_num(x):
    try:
        if pd.isna(x):
            return "N/A"
        x = float(x)
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.2f}"
    except Exception:
        return str(x)

def risk_label(prob):
    if prob >= 0.70:
        return "High Risk", "risk-high", "Strong likelihood of default. Tight review is recommended before approval."
    elif prob >= 0.40:
        return "Moderate Risk", "risk-medium", "Moderate risk detected. Consider additional checks, collateral, or revised terms."
    else:
        return "Low Risk", "risk-low", "Borrower appears relatively safer based on the available model inputs."

def borrower_summary_cols(row):
    preferred = [
        "LoanAmount", "Principal", "Days In Arrears", "InstallmentAmount",
        "Balance", "OutstandingBalance", "MonthlyIncome", "Net Salary"
    ]
    return [c for c in preferred if c in row.index][:4]

def draw_target_chart_small(data, target_name):
    counts = data[target_name].value_counts(dropna=False).sort_index()
    fig = plt.figure(figsize=(4.2, 2.4))
    ax = fig.add_subplot(111)
    ax.bar(counts.index.astype(str), counts.values, color=["#d9ff57", "#ffe44d", "#7eff9e"][:len(counts)])
    ax.set_title("Target Class Distribution", color="white", fontsize=10)
    ax.set_xlabel(target_name, color="white", fontsize=9)
    ax.set_ylabel("Count", color="white", fontsize=9)
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    plt.tight_layout()
    return fig

def draw_metrics_bar(metrics_dict):
    if not metrics_dict:
        return None
    fig = plt.figure(figsize=(4.8, 2.8))
    ax = fig.add_subplot(111)
    names = list(metrics_dict.keys())
    vals = list(metrics_dict.values())
    ax.barh(names, vals, color=["#d9ff57", "#ffe44d", "#7eff9e"][:len(vals)])
    ax.set_title("Validation Metrics", color="white", fontsize=10)
    ax.set_facecolor("#0f0f0f")
    fig.patch.set_facecolor("#0f0f0f")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#666666")
    plt.tight_layout()
    return fig

def get_feature_importance_df():
    if final_model is not None and hasattr(final_model, "feature_importances_"):
        return pd.DataFrame({
            "Feature": features,
            "Importance": final_model.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return None

def build_local_driver_table(user_input_dict, reference_df, importance_df):
    if importance_df is None:
        return None

    rows = []
    for feature in features:
        if feature not in reference_df.columns:
            continue

        ref_series = pd.to_numeric(reference_df[feature], errors="coerce")
        borrower_val = user_input_dict.get(feature, 0)

        try:
            borrower_val = float(borrower_val)
        except Exception:
            borrower_val = 0.0

        median_val = ref_series.median()
        std_val = ref_series.std()

        if pd.isna(std_val) or std_val == 0:
            scaled_diff = 0.0
        else:
            scaled_diff = (borrower_val - median_val) / std_val

        importance_val = float(
            importance_df.loc[importance_df["Feature"] == feature, "Importance"].iloc[0]
        )

        driver_score = scaled_diff * importance_val

        rows.append({
            "Feature": feature,
            "Borrower Value": borrower_val,
            "Portfolio Median": median_val,
            "Importance": importance_val,
            "Driver Score": driver_score
        })

    if not rows:
        return None

    local_df = pd.DataFrame(rows)
    local_df["Abs Score"] = local_df["Driver Score"].abs()
    local_df = local_df.sort_values("Abs Score", ascending=False).reset_index(drop=True)
    return local_df


def style_driver_direction(score):
    if score > 0:
        return "↑ pushes risk higher"
    elif score < 0:
        return "↓ pushes risk lower"
    return "• neutral"

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🟢 Dashboard Menu")
page = st.sidebar.radio(
    "Go to section",
    ["Overview", "Borrower Scoring", "Portfolio Insights", "Explainability", "Model Insights", "Decision Guide"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
st.sidebar.success(model_name)

st.sidebar.markdown("### Quick Stats")
st.sidebar.write(f"Features: **{len(features)}**")
st.sidebar.write(f"Target: **{target_col}**")
st.sidebar.write(f"Records: **{len(df):,}**")

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="glow-card" style="margin-top:0 !important;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
            <div style="font-size:32px; font-weight:900; color:#d9ff57;">🟢 SACCO Credit Scoring Dashboard</div>
            <div style="font-size:16px; color:#f4f4f4; margin-top:6px;">
                Real-time borrower risk evaluation, borrower lookup, and portfolio-level monitoring
            </div>
        </div>
        <div style="text-align:right;">
            <div class="small-label">Deployment Stack</div>
            <div class="big-value">FastAPI + Streamlit</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="small-label">Test AUC</div>
            <div class="big-value">{metrics.get('Test AUC', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="small-label">Test KS</div>
            <div class="big-value">{metrics.get('Test KS', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="small-label">Number of Features</div>
            <div class="big-value">{len(features)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="small-label">Prediction Target</div>
            <div class="big-value">{target_col}</div>
        </div>
        """, unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="section-label">System Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
        This dashboard supports <b>credit risk decision-making</b> in a SACCO environment by:
        <ul>
            <li>Scoring existing borrowers using stored records</li>
            <li>Scoring new applicants using manual data entry</li>
            <li>Providing a risk category and default probability</li>
            <li>Displaying portfolio-level insights for monitoring</li>
            <li>Showing explainability through feature importance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Performance Snapshot</div>', unsafe_allow_html=True)
        fig = draw_metrics_bar(metrics)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

# =========================================================
# PAGE: BORROWER SCORING
# =========================================================
elif page == "Borrower Scoring":
    mode = st.radio("Choose input mode", ["Select Existing Borrower", "Manual Entry"], horizontal=True)
    user_input = {}
    borrower_row = None
    borrower_id = None
    predict_clicked = False

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="section-label">Borrower Inputs</div>', unsafe_allow_html=True)

        if mode == "Select Existing Borrower" and id_col is not None:
            borrower_id = st.selectbox("Select Borrower Number", df[id_col].astype(str).unique())
            borrower_row = df[df[id_col].astype(str) == borrower_id].iloc[0]

            st.markdown("""
            <div class="glow-card">
                Existing borrower record loaded. You can still adjust the values before scoring.
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(2)
            for i, feature in enumerate(features):
                with cols[i % 2]:
                    user_input[feature] = st.number_input(
                        feature,
                        value=safe_value(borrower_row[feature]) if feature in borrower_row.index else 0.0,
                        step=0.01
                    )
        else:
            st.markdown("""
            <div class="glow-card">
                Manual entry mode is active. Use this for a new applicant or scenario testing.
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(2)
            for i, feature in enumerate(features):
                with cols[i % 2]:
                    user_input[feature] = st.number_input(feature, value=0.0, step=0.01)

        b1, b2 = st.columns(2)
        with b1:
            predict_clicked = st.button("🚀 Score Borrower", use_container_width=True)
        with b2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

    with right:
        st.markdown('<div class="section-label">Borrower Snapshot</div>', unsafe_allow_html=True)

        if borrower_row is not None:
            if id_col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="small-label">Borrower ID</div>
                    <div class="big-value">{borrower_id}</div>
                </div>
                """, unsafe_allow_html=True)

            summary_cols = borrower_summary_cols(borrower_row)
            c1, c2 = st.columns(2)
            for i, col in enumerate(summary_cols):
                html = f"""
                <div class="metric-card">
                    <div class="small-label">{col}</div>
                    <div class="big-value">{nice_num(borrower_row[col])}</div>
                </div>
                """
                if i % 2 == 0:
                    c1.markdown(html, unsafe_allow_html=True)
                else:
                    c2.markdown(html, unsafe_allow_html=True)

            with st.expander("View full selected-feature profile"):
                feature_view = pd.DataFrame({
                    "Feature": features,
                    "Value": [borrower_row[f] if f in borrower_row.index else np.nan for f in features]
                })
                st.dataframe(feature_view, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div class="metric-card">
                No borrower selected yet. In manual mode, fill the values on the left and score the applicant.
            </div>
            """, unsafe_allow_html=True)

    if predict_clicked:
        st.markdown("---")
        st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=user_input,
                timeout=30
            )
            result = response.json()

            if "error" in result:
                st.error(result["error"])
            else:
                prob = float(result["default_probability"])
                label, css_class, interpretation = risk_label(prob)

                importance_df = get_feature_importance_df()
                local_driver_df = build_local_driver_table(user_input, df, importance_df)

                r1, r2 = st.columns([1, 1])

                with r1:
                    st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-card">
                        <b>Interpretation</b><br><br>
                        {interpretation}
                    </div>
                    """, unsafe_allow_html=True)

                with r2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="small-label">Default Probability</div>
                        <div class="big-value">{prob:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(max(prob, 0.0), 1.0))
                    st.caption("The higher the gauge, the greater the predicted likelihood of default.")

                x1, x2, x3 = st.columns(3)
                with x1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="small-label">Risk Category</div>
                        <div class="big-value">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with x2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="small-label">Scoring Mode</div>
                        <div class="big-value">{mode}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with x3:
                    ref = borrower_id if borrower_id is not None else "Manual"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="small-label">Borrower Reference</div>
                        <div class="big-value">{ref}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("Show submitted values"):
                    submitted = pd.DataFrame({
                        "Feature": list(user_input.keys()),
                        "Value": list(user_input.values())
                    })
                    st.dataframe(submitted, use_container_width=True, hide_index=True)

                st.markdown("### Borrower-Specific Risk Drivers")

                if local_driver_df is not None:
                    top_drivers = local_driver_df.head(5)

                    dcol1, dcol2 = st.columns([1.1, 1])

                    with dcol1:
                        plot_df = top_drivers.sort_values("Driver Score", ascending=True)

                        fig = plt.figure(figsize=(6, 3.2))
                        ax = fig.add_subplot(111)

                        colors = ["#ffb347" if x > 0 else "#7eff9e" for x in plot_df["Driver Score"]]
                        ax.barh(plot_df["Feature"], plot_df["Driver Score"], color=colors)

                        ax.set_title("Top Borrower-Specific Risk Drivers", color="white", fontsize=11)
                        ax.set_xlabel("Directional Driver Score", color="white", fontsize=9)
                        ax.set_ylabel("Feature", color="white", fontsize=9)
                        ax.set_facecolor("#0f0f0f")
                        fig.patch.set_facecolor("#0f0f0f")
                        ax.tick_params(colors="white", labelsize=8)
                        for spine in ax.spines.values():
                            spine.set_color("#666666")
                        plt.tight_layout()

                        st.pyplot(fig)

                    with dcol2:
                        st.markdown("""
                        <div class="metric-card">
                            <b>Interpretation Guide</b><br><br>
                            Orange bars suggest features that are pushing the borrower toward higher predicted risk relative to the portfolio profile.<br><br>
                            Green bars suggest features that are comparatively lowering predicted risk.
                        </div>
                        """, unsafe_allow_html=True)

                        explain_table = top_drivers.copy()
                        explain_table["Direction"] = explain_table["Driver Score"].apply(style_driver_direction)
                        explain_table = explain_table[[
                            "Feature", "Borrower Value", "Portfolio Median", "Direction"
                        ]]
                        st.dataframe(explain_table, use_container_width=True, hide_index=True)

                else:
                    st.info("Borrower-specific risk drivers are not available for the current model.")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Ensure the API is still running.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# =========================================================
# PAGE: PORTFOLIO INSIGHTS
# =========================================================
elif page == "Portfolio Insights":
    st.markdown('<div class="section-label">Portfolio-Level Insights</div>', unsafe_allow_html=True)

    top_left, top_right = st.columns([0.9, 1.1])

    with top_left:
        if target_col in df.columns:
            fig = draw_target_chart_small(df, target_col)
            st.pyplot(fig)

    with top_right:
        numeric_cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            selected_num = st.selectbox("Choose numeric feature to visualize", numeric_cols)

            fig = plt.figure(figsize=(5.2, 2.6))
            ax = fig.add_subplot(111)
            ax.hist(df[selected_num].dropna(), bins=20, color="#d9ff57", edgecolor="#111111")
            ax.set_title(f"Distribution of {selected_num}", color="white", fontsize=10)
            ax.set_xlabel(selected_num, color="white", fontsize=9)
            ax.set_ylabel("Frequency", color="white", fontsize=9)
            ax.set_facecolor("#0f0f0f")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#666666")
            plt.tight_layout()
            st.pyplot(fig)

    if target_col in df.columns and numeric_cols:
        grouped = df.groupby(target_col)[selected_num].mean()

        fig = plt.figure(figsize=(5.0, 2.8))
        ax = fig.add_subplot(111)
        ax.bar(grouped.index.astype(str), grouped.values, color="#ffe44d")
        ax.set_title(f"Average {selected_num} by {target_col}", color="white", fontsize=10)
        ax.set_xlabel(target_col, color="white", fontsize=9)
        ax.set_ylabel(f"Mean", color="white", fontsize=9)
        ax.set_facecolor("#0f0f0f")
        fig.patch.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#666666")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Portfolio Snapshot")
    st.dataframe(df[features].head(15), use_container_width=True)

# =========================================================
# PAGE: EXPLAINABILITY
# =========================================================
elif page == "Explainability":
    st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)

    importance_df = get_feature_importance_df()

    if importance_df is not None:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("""
            <div class="metric-card">
                This section shows the relative importance of the deployed features in the final Gradient Boosting model.
                Higher values indicate stronger contribution to the model’s prediction behavior.
            </div>
            """, unsafe_allow_html=True)

            top_n = st.slider("Select number of top features", min_value=5, max_value=min(14, len(importance_df)), value=min(10, len(importance_df)))
            plot_df = importance_df.head(top_n).sort_values("Importance", ascending=True)

            fig = plt.figure(figsize=(6.5, 3.8))
            ax = fig.add_subplot(111)
            ax.barh(plot_df["Feature"], plot_df["Importance"], color="#d9ff57")
            ax.set_title("Top Feature Importances", color="white", fontsize=11)
            ax.set_xlabel("Importance", color="white", fontsize=9)
            ax.set_ylabel("Feature", color="white", fontsize=9)
            ax.set_facecolor("#0f0f0f")
            fig.patch.set_facecolor("#0f0f0f")
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#666666")
            plt.tight_layout()
            st.pyplot(fig)

        with right:
            st.markdown("""
            <div class="glow-card">
                <b>Why this matters</b><br><br>
                Explainability improves trust in model-assisted credit decisions by showing which variables have stronger influence in the scoring process.
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(importance_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Feature importance is not available for the current model artifact.")

# =========================================================
# PAGE: MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":
    st.markdown('<div class="section-label">Model Insights</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("""
        <div class="metric-card">
            <b>Selected Model</b><br><br>
            Gradient Boosting was selected as the final production model because it achieved the strongest balance between predictive performance and generalization.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <b>Selected Features</b><br><br>
            The deployed model uses a reduced set of variables selected during the modeling workflow to support efficient and focused prediction.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        fig = draw_metrics_bar(metrics)
        if fig is not None:
            st.pyplot(fig)

    st.markdown("### Final Feature List")
    feature_df = pd.DataFrame({"Selected Features": features})
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

# =========================================================
# PAGE: DECISION GUIDE
# =========================================================
elif page == "Decision Guide":
    st.markdown('<div class="section-label">Decision Guide</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("""<div class="risk-low">Low Risk</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            Typical action:
            <ul>
                <li>Proceed with normal review</li>
                <li>Maintain standard loan terms</li>
                <li>Monitor as part of routine operations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown("""<div class="risk-medium">Moderate Risk</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            Typical action:
            <ul>
                <li>Request more review</li>
                <li>Consider added guarantor/collateral checks</li>
                <li>Review borrower affordability carefully</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown("""<div class="risk-high">High Risk</div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            Typical action:
            <ul>
                <li>Escalate for stricter review</li>
                <li>Consider revised terms or rejection</li>
                <li>Use as an early warning signal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glow-card">
        <b>Important note:</b> The model is intended to support decision-making, not replace professional judgment. Final credit decisions should still consider institutional policy, compliance requirements, and contextual borrower information.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built as a SACCO decision-support dashboard using Gradient Boosting, FastAPI, and Streamlit.")