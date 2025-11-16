import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# SciPy for chi-square test (p-values). If unavailable, we still compute chi-square statistic.
try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# DOCX viewer
try:
    from docx import Document
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_CSV = "employee_attrition_dataset.csv"
DEFAULT_DOCX = "Velorium_Technologies_Case study.docx"

ATTRITION_FIELD = "Attrition"
ID_FIELD = "Employee_ID"

# Known categorical fields from your dataset header (used as a hint)
KNOWN_CATEGORICAL = {
    'Gender', 'Marital_Status', 'Department', 'Job_Role', 'Work_Life_Balance',
    'Job_Satisfaction', 'Overtime', 'Work_Environment_Satisfaction',
    'Relationship_with_Manager', 'Job_Involvement'
}

# -----------------------------
# Utilities
# -----------------------------
def normalize_yes_no(val):
    s = str(val).strip().lower() if val is not None else ""
    if s == "yes":
        return 1
    if s == "no":
        return 0
    return np.nan

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Strip whitespace from string values
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def detect_numeric_columns(df: pd.DataFrame, exclude_cols=None, sample_size=1000, threshold=0.9):
    if exclude_cols is None:
        exclude_cols = set()
    numeric_cols, categorical_cols = [], []
    cols = [c for c in df.columns if c not in exclude_cols]
    n = len(df)
    sample = df if n <= sample_size else df.sample(sample_size, random_state=42)

    for c in cols:
        # Try converting to numeric
        coerced = pd.to_numeric(sample[c], errors='coerce')
        valid_ratio = coerced.notna().mean()
        if valid_ratio >= threshold:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def point_biserial(x: pd.Series, y: pd.Series) -> float:
    # y must be binary 0/1. x numeric. Handle NaNs by aligning.
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y}).dropna()
    if df.empty:
        return np.nan
    # Check binary
    unique_y = sorted(df["y"].dropna().unique())
    if not (len(unique_y) == 2 and unique_y[0] == 0 and unique_y[1] == 1):
        return np.nan

    x_all = df["x"].values
    y_all = df["y"].values
    n = len(x_all)
    if n == 0:
        return np.nan

    mx = np.nanmean(x_all)
    sx = np.nanstd(x_all)
    if sx == 0 or np.isnan(sx):
        return np.nan

    x1 = x_all[y_all == 1]
    x0 = x_all[y_all == 0]
    if len(x1) == 0 or len(x0) == 0:
        return np.nan

    m1 = np.nanmean(x1)
    m0 = np.nanmean(x0)
    p = len(x1) / n
    q = 1 - p
    if p == 0 or q == 0:
        return np.nan
    return ((m1 - m0) * math.sqrt(p * q)) / sx

def compute_overview(df: pd.DataFrame):
    total = len(df)
    attrition_count = int(df[ATTRITION_FIELD].fillna(0).sum())
    rate = (attrition_count / total) if total else 0.0
    return total, attrition_count, rate

def categorical_breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.groupby(col, dropna=False)[ATTRITION_FIELD].agg(["count", "sum"]).reset_index()
    d = d.rename(columns={"count": "total", "sum": "attrition"})
    d["attritionRate"] = (d["attrition"] / d["total"]).fillna(0.0)
    # Replace NaN category with "Unknown"
    d[col] = d[col].fillna("Unknown")
    return d

def chi_square_attrition(df: pd.DataFrame, col: str):
    # Build contingency table: rows=category, columns=[No, Yes]
    temp = df[[col, ATTRITION_FIELD]].copy()
    temp[col] = temp[col].fillna("Unknown")
    temp[ATTRITION_FIELD] = temp[ATTRITION_FIELD].fillna(0)
    contingency = pd.crosstab(temp[col], temp[ATTRITION_FIELD])

    if contingency.empty or contingency.shape[1] < 2:
        return {"chi2": np.nan, "p": np.nan, "df": 0, "contingency": contingency}

    if SCIPY_AVAILABLE:
        chi2, p, df, _ = chi2_contingency(contingency.values)
        return {"chi2": chi2, "p": p, "df": df, "contingency": contingency}
    else:
        # Manual chi-square statistic (no p-value)
        # E = row_total * col_total / grand_total
        obs = contingency.values.astype(float)
        row_totals = obs.sum(axis=1)[:, None]
        col_totals = obs.sum(axis=0)[None, :]
        grand_total = obs.sum()
        expected = (row_totals @ col_totals) / grand_total
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = np.nansum(((obs - expected) ** 2) / expected)
        df_val = (obs.shape[0] - 1) * (obs.shape[1] - 1)
        return {"chi2": chi2, "p": np.nan, "df": df_val, "contingency": contingency}

def compute_numeric_correlations(df: pd.DataFrame, numeric_cols):
    corrs = {}
    y = df[ATTRITION_FIELD]
    for c in numeric_cols:
        if c == ID_FIELD or c == ATTRITION_FIELD:
            continue
        r = point_biserial(df[c], y)
        corrs[c] = r
    # Drop NaNs
    corrs = {k: v for k, v in corrs.items() if pd.notna(v)}
    return corrs

def compute_categorical_chis(df: pd.DataFrame, categorical_cols):
    stats = {}
    for c in categorical_cols:
        if c == ID_FIELD or c == ATTRITION_FIELD:
            continue
        stats[c] = chi_square_attrition(df, c)
    return stats

def rank_top_drivers(numeric_corrs: dict, categorical_stats: dict, topn=15):
    scores = []
    for f, r in numeric_corrs.items():
        scores.append({"feature": f, "type": "numeric", "score": abs(r), "detail": {"corr": r}})
    for f, s in categorical_stats.items():
        chi = s.get("chi2", np.nan)
        if pd.notna(chi):
            scores.append({"feature": f, "type": "categorical", "score": float(chi), "detail": {"chi2": chi, "p": s.get("p", np.nan)}})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:topn]

def load_docx_text(path: str):
    if not DOCX_AVAILABLE:
        return "python-docx is not installed. Install with: pip install python-docx"
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        doc = Document(path)
        # Concatenate paragraphs, preserve simple headings if possible
        text_parts = []
        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                text_parts.append(t)
        return "\n\n".join(text_parts) if text_parts else "(Document contains no visible paragraphs)"
    except Exception as e:
        return f"Failed to read DOCX: {e}"

# -----------------------------
# Sidebar: Inputs
# -----------------------------
st.sidebar.title("Configuration")

# Dataset source
st.sidebar.subheader("Dataset")
use_default_csv = st.sidebar.checkbox("Use default dataset", value=True)
uploaded_csv = None
csv_path = DEFAULT_CSV
if not use_default_csv:
    uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv is not None:
        csv_path = uploaded_csv
    else:
        st.sidebar.info(f"If not uploading, default path will be used: {DEFAULT_CSV}")

# Case study docx
st.sidebar.subheader("Case Study DOCX")
use_default_docx = st.sidebar.checkbox("Use default case study file", value=True)
uploaded_docx = None
docx_path = DEFAULT_DOCX
if not use_default_docx:
    uploaded_docx = st.sidebar.file_uploader("Upload DOCX", type=["docx"])
    if uploaded_docx is not None:
        docx_path = uploaded_docx
    else:
        st.sidebar.info(f"If not uploading, default path will be used: {DEFAULT_DOCX}")

st.sidebar.markdown("---")

# -----------------------------
# Load and preprocess data
# -----------------------------
@st.cache_data(show_spinner=True)
def load_data(source):
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV not found at path: {source}")
        df = pd.read_csv(source)
    else:
        # file-like from uploader
        df = pd.read_csv(source)

    df = strip_columns(df)

    # Normalize Attrition to 0/1
    if ATTRITION_FIELD in df.columns:
        df[ATTRITION_FIELD] = df[ATTRITION_FIELD].apply(normalize_yes_no)
    else:
        raise ValueError(f"Attrition column '{ATTRITION_FIELD}' not present in dataset")

    # Ensure Employee_ID numeric if exists
    if ID_FIELD in df.columns:
        df[ID_FIELD] = pd.to_numeric(df[ID_FIELD], errors="coerce")

    # Detect numeric/categorical dynamically
    exclude = {ATTRITION_FIELD}
    num_cols, cat_cols = detect_numeric_columns(df, exclude_cols=exclude)

    # Force known categorical hints, if present
    for c in KNOWN_CATEGORICAL:
        if c in df.columns and c not in cat_cols:
            if c in num_cols:
                num_cols.remove(c)
            cat_cols.append(c)

    # Convert detected numeric columns to numeric dtype
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df, num_cols, cat_cols

# Load
load_error = None
try:
    df, numeric_cols, categorical_cols = load_data(csv_path)
except Exception as e:
    df, numeric_cols, categorical_cols = None, [], []
    load_error = str(e)

# -----------------------------
# Layout
# -----------------------------
st.title("Employee Attrition Analytics Dashboard")

if load_error:
    st.error(f"Failed to load dataset: {load_error}")
    st.stop()

# KPIs
total, attr_count, rate = compute_overview(df)
kpi_cols = st.columns(3)
kpi_cols[0].metric("Total Employees", f"{total}")
kpi_cols[1].metric("Attrition Count", f"{attr_count}")
kpi_cols[2].metric("Attrition Rate", f"{rate*100:.1f}%")

st.markdown("---")

# -----------------------------
# Top Drivers of Attrition
# -----------------------------
with st.spinner("Computing feature importance..."):
    numeric_corrs = compute_numeric_correlations(df, numeric_cols)
    categorical_stats = compute_categorical_chis(df, categorical_cols)
    top_drivers = rank_top_drivers(numeric_corrs, categorical_stats, topn=15)

st.subheader("Top Drivers of Attrition")
if top_drivers:
    td_df = pd.DataFrame(top_drivers)
    td_df["score"] = td_df["score"].round(3)
    st.dataframe(td_df, use_container_width=True)
else:
    st.info("No drivers computed (check data and feature types).")

st.markdown("---")

# -----------------------------
# Overview Charts
# -----------------------------
overview_cols = st.columns(2)

# Attrition by Department
if "Department" in df.columns:
    dept_breakdown = categorical_breakdown(df, "Department")
    chart_dept = alt.Chart(dept_breakdown).mark_bar().encode(
        x=alt.X('Department:N', sort='-y', title='Department'),
        y=alt.Y('attritionRate:Q', title='Attrition Rate', axis=alt.Axis(format='%')),
        tooltip=['Department', 'total', 'attrition', alt.Tooltip('attritionRate:Q', format=".1%")]
    ).properties(title="Attrition Rate by Department", height=350)
    overview_cols[0].altair_chart(chart_dept, use_container_width=True)
else:
    overview_cols[0].info("Department column not found.")

# Attrition by Job Role
if "Job_Role" in df.columns:
    role_breakdown = categorical_breakdown(df, "Job_Role")
    chart_role = alt.Chart(role_breakdown).mark_bar(color="#e15759").encode(
        x=alt.X('Job_Role:N', sort='-y', title='Job Role'),
        y=alt.Y('attritionRate:Q', title='Attrition Rate', axis=alt.Axis(format='%')),
        tooltip=['Job_Role', 'total', 'attrition', alt.Tooltip('attritionRate:Q', format=".1%")]
    ).properties(title="Attrition Rate by Job Role", height=350)
    overview_cols[1].altair_chart(chart_role, use_container_width=True)
else:
    overview_cols[1].info("Job_Role column not found.")

st.markdown("---")

# -----------------------------
# Deep Dive: Interactive Exploration
# -----------------------------
st.subheader("Deep Dive Exploration")

deep_cols = st.columns(2)

# Categorical exploration
if categorical_cols:
    selected_cat = deep_cols[0].selectbox("Explore Categorical Feature", options=sorted(categorical_cols))
    cat_data = categorical_breakdown(df, selected_cat)
    chart_cat = alt.Chart(cat_data).mark_bar().encode(
        x=alt.X(f'{selected_cat}:N', sort='-y', title=selected_cat),
        y=alt.Y('attritionRate:Q', title='Attrition Rate', axis=alt.Axis(format='%')),
        color=alt.Color('attritionRate:Q', scale=alt.Scale(scheme='teals')),
        tooltip=[selected_cat, 'total', 'attrition', alt.Tooltip('attritionRate:Q', format=".1%")]
    ).properties(title=f"Attrition Rate by {selected_cat}", height=350)
    deep_cols[0].altair_chart(chart_cat, use_container_width=True)

    # Chi-square
    chi_stats = categorical_stats.get(selected_cat, {})
    chi2 = chi_stats.get("chi2", np.nan)
    pval = chi_stats.get("p", np.nan)
    df_val = chi_stats.get("df", 0)
    deep_cols[0].markdown(
        f"- Chi-square statistic: **{(0.0 if pd.isna(chi2) else chi2):.3f}**  "
        f"- Degrees of freedom: **{df_val}**  "
        f"- p-value: **{'N/A' if pd.isna(pval) else f'{pval:.5f}'}**"
    )
else:
    deep_cols[0].info("No categorical features detected.")

# Numeric exploration
if numeric_cols:
    selected_num = deep_cols[1].selectbox("Explore Numeric Feature", options=sorted(numeric_cols))
    # Point-biserial correlation
    r_pb = numeric_corrs.get(selected_num, np.nan)
    deep_cols[1].markdown(f"- Point-biserial correlation with Attrition: **{(0.0 if pd.isna(r_pb) else r_pb):.3f}**")

    # Density plots for y=0 vs y=1
    plot_df = df[[selected_num, ATTRITION_FIELD]].dropna()
    plot_df = plot_df.rename(columns={ATTRITION_FIELD: "AttritionFlag"})
    # Cast to categorical labels
    plot_df["AttritionFlag"] = plot_df["AttritionFlag"].map({0: "No", 1: "Yes"})
    # Create overlapping density chart
    density = alt.Chart(plot_df).transform_density(
        selected_num, groupby=['AttritionFlag'], as_=[selected_num, 'density']
    ).mark_area(opacity=0.5).encode(
        x=alt.X(f'{selected_num}:Q', title=selected_num),
        y='density:Q',
        color='AttritionFlag:N'
    ).properties(title=f"Distribution of {selected_num} by Attrition", height=350)
    deep_cols[1].altair_chart(density, use_container_width=True)
else:
    deep_cols[1].info("No numeric features detected.")

st.markdown("---")

# -----------------------------
# Case Study Viewer
# -----------------------------
st.subheader("Velorium Technologies Case Study")
if isinstance(docx_path, str):
    docx_status = f"Using file: {docx_path}"
else:
    docx_status = "Using uploaded DOCX file"
st.caption(docx_status)

if DOCX_AVAILABLE:
    try:
        if isinstance(docx_path, str) and os.path.exists(docx_path):
            case_text = load_docx_text(docx_path)
        elif uploaded_docx is not None:
            # Save to temp and load
            case_text = load_docx_text(uploaded_docx)
        else:
            case_text = f"Could not find DOCX at path: {DEFAULT_DOCX}. Upload one from the sidebar."
    except Exception as e:
        case_text = f"Failed to load DOCX: {e}"
else:
    case_text = "DOCX viewing requires python-docx. Please install it: pip install python-docx"

st.text_area("Case Study Content", case_text, height=300)

st.markdown("---")

# -----------------------------
# Data Table and Export
# -----------------------------
st.subheader("Raw Data Preview")
st.dataframe(df.head(100), use_container_width=True)

st.download_button(
    label="Download Cleaned Data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="employee_attrition_cleaned.csv",
    mime="text/csv"
)

# -----------------------------
# Notes
# -----------------------------
st.markdown("""
**Notes and assumptions**
- Attrition is normalized from Yes/No to 1/0 (binary).
- Numeric feature strength is evaluated via point-biserial correlation (linear association) with Attrition.
- Categorical feature strength is evaluated via chi-square test of independence (and p-value if SciPy is available).
- Feature detection is dynamic: columns are classified as numeric if ≥90% of sampled values can be coerced to numbers; otherwise categorical.
- Use the sidebar to upload alternate datasets and case study files.
""")