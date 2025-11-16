import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# SciPy for chi-square test
try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="Velorium Retention Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    .action-box {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================
DEFAULT_CSV = "employee_attrition_dataset.csv"

ATTRITION_FIELD = "Attrition"
ID_FIELD = "Employee_ID"

KNOWN_CATEGORICAL = {
    'Gender', 'Marital_Status', 'Department', 'Job_Role', 'Overtime',
    'Work_Life_Balance', 'Job_Satisfaction', 'Work_Environment_Satisfaction',
    'Relationship_with_Manager', 'Job_Involvement'
}

# Risk factors and interventions
INTERVENTION_MAPPING = {
    'Work_Life_Balance': {
        'low': ['Explore flexible work arrangements', 'Reduce concurrent project assignments', 'Encourage time off'],
        'medium': ['Review workload distribution', 'Implement wellness programs'],
        'high': ['Immediate workload audit', 'Assign mentorship for stress management']
    },
    'Years_Since_Last_Promotion': {
        'high': ['Create visible growth path', 'Offer leadership training', 'Consider role expansion'],
        'medium': ['Schedule career development discussion', 'Define next steps for advancement'],
        'low': ['Continue current development plan']
    },
    'Job_Satisfaction': {
        'low': ['Conduct stay interview', 'Clarify role expectations', 'Enhance role autonomy'],
        'medium': ['Increase recognition programs', 'Improve feedback frequency'],
        'high': ['Maintain engagement', 'Leverage as internal advocate']
    },
    'Relationship_with_Manager': {
        'low': ['Management coaching session', 'Consider role reassignment', 'Mediated 1:1 meetings'],
        'medium': ['Strengthen 1:1 cadence', 'Focus on feedback quality'],
        'high': ['Leverage strong relationship for retention']
    },
    'Performance_Rating': {
        'low': ['Performance improvement plan', 'Identify skill gaps', 'Provide targeted coaching'],
        'medium': ['Set clear improvement goals', 'Regular check-ins'],
        'high': ['High-potential development track', 'Leadership opportunities']
    }
}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def normalize_yes_no(val):
    """Convert Yes/No to 1/0"""
    s = str(val).strip().lower() if val is not None else ""
    if s == "yes":
        return 1
    if s == "no":
        return 0
    return np.nan

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and values"""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def detect_numeric_columns(df: pd.DataFrame, exclude_cols=None, sample_size=1000, threshold=0.9):
    """Detect numeric vs categorical columns"""
    if exclude_cols is None:
        exclude_cols = set()
    numeric_cols, categorical_cols = [], []
    cols = [c for c in df.columns if c not in exclude_cols]
    n = len(df)
    sample = df if n <= sample_size else df.sample(sample_size, random_state=42)

    for c in cols:
        coerced = pd.to_numeric(sample[c], errors='coerce')
        valid_ratio = coerced.notna().mean()
        if valid_ratio >= threshold:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def compute_overview(df: pd.DataFrame):
    """Compute key metrics"""
    total = len(df)
    attrition_count = int(df[ATTRITION_FIELD].fillna(0).sum())
    rate = (attrition_count / total) if total else 0.0
    return total, attrition_count, rate

def categorical_breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Break down attrition by category"""
    d = df.groupby(col, dropna=False)[ATTRITION_FIELD].agg(["count", "sum"]).reset_index()
    d = d.rename(columns={"count": "total", "sum": "attrition"})
    d["attritionRate"] = (d["attrition"] / d["total"]).fillna(0.0)
    d[col] = d[col].fillna("Unknown")
    return d

# ============================================
# MACHINE LEARNING MODELS
# ============================================

@st.cache_data(show_spinner=True)
def load_data(source):
    """Load and preprocess data"""
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV not found at path: {source}")
        df = pd.read_csv(source)
    else:
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

    # Force known categorical hints
    for c in KNOWN_CATEGORICAL:
        if c in df.columns and c not in cat_cols:
            if c in num_cols:
                num_cols.remove(c)
            cat_cols.append(c)

    # Convert detected numeric columns
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df, num_cols, cat_cols

@st.cache_resource
def train_attrition_model(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """Train Random Forest model for attrition prediction"""
    df_model = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_cols:
        if col in df_model.columns and col != ATTRITION_FIELD:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].fillna('Unknown'))
            le_dict[col] = le
    
    # Select features for model
    feature_cols = [c for c in numeric_cols if c != ID_FIELD and c != ATTRITION_FIELD] + [c for c in categorical_cols if c != ATTRITION_FIELD]
    
    X = df_model[feature_cols].fillna(df_model[feature_cols].mean())
    y = df_model[ATTRITION_FIELD].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, le_dict, feature_cols, feature_importance

def predict_attrition_risk(employee_id: int, df: pd.DataFrame, numeric_cols: list, 
                           categorical_cols: list, model, scaler, le_dict, feature_cols):
    """Predict attrition risk for individual employee"""
    employee = df[df[ID_FIELD] == employee_id].copy()
    
    if employee.empty:
        return None
    
    df_model = employee.copy()
    
    # Encode categorical
    for col in categorical_cols:
        if col in df_model.columns and col != ATTRITION_FIELD:
            if col in le_dict:
                df_model[col] = le_dict[col].transform(df_model[col].fillna('Unknown'))
    
    X = df_model[feature_cols].fillna(df_model[feature_cols].mean())
    X_scaled = scaler.transform(X)
    
    risk_prob = model.predict_proba(X_scaled)[0][1]
    return risk_prob

def get_risk_category(prob):
    """Categorize risk level"""
    if prob >= 0.7:
        return "🔴 High Risk", "risk-high"
    elif prob >= 0.4:
        return "🟡 Medium Risk", "risk-medium"
    else:
        return "🟢 Low Risk", "risk-low"

def generate_retention_insights(employee_data: pd.Series, feature_importance: pd.DataFrame, 
                                numeric_cols: list, categorical_cols: list) -> dict:
    """Generate personalized retention insights"""
    insights = {
        'risk_drivers': [],
        'interventions': [],
        'email_draft': '',
        'manager_notes': ''
    }
    
    # Analyze key risk factors
    risk_factors = []
    
    # Work-Life Balance
    if 'Work_Life_Balance' in employee_data.index:
        wlb = employee_data['Work_Life_Balance']
        if wlb <= 2:
            risk_factors.append(('Work-Life Balance', wlb, 'low'))
    
    # Years Since Promotion
    if 'Years_Since_Last_Promotion' in employee_data.index:
        yslp = employee_data['Years_Since_Last_Promotion']
        if yslp >= 5:
            risk_factors.append(('Career Stagnation', yslp, 'high'))
    
    # Job Satisfaction
    if 'Job_Satisfaction' in employee_data.index:
        js = employee_data['Job_Satisfaction']
        if js <= 2:
            risk_factors.append(('Job Satisfaction', js, 'low'))
    
    # Manager Relationship
    if 'Relationship_with_Manager' in employee_data.index:
        rwm = employee_data['Relationship_with_Manager']
        if rwm <= 2:
            risk_factors.append(('Manager Relationship', rwm, 'low'))
    
    # Overtime
    if 'Overtime' in employee_data.index:
        ot = employee_data['Overtime']
        if ot == 1:  # Yes
            risk_factors.append(('Excessive Overtime', 1, 'high'))
    
    insights['risk_drivers'] = risk_factors
    
    # Generate interventions
    for factor, value, severity in risk_factors:
        for key, mapping in INTERVENTION_MAPPING.items():
            if key in factor or factor in key:
                interventions = mapping.get(severity, [])
                insights['interventions'].extend(interventions)
    
    # Generate email draft
    emp_name = employee_data.get('Employee_ID', 'Team Member')
    dept = employee_data.get('Department', 'Your')
    
    insights['email_draft'] = f"""Subject: Let's Talk About Your Growth at Velorium

Hi {emp_name},

I wanted to reach out for a quick conversation. As your manager, I value your contributions to our {dept} team, and I've noticed some things I want to address directly.

I see you've been putting in solid effort, and I want to make sure we're creating an environment where you can thrive—not just survive. 

Could we find 30 minutes this week for a 1:1? I'd love to understand:
- How you're feeling about your current role
- What's working and what's not
- What would make your work here more rewarding

This isn't a corrective conversation. It's me trying to listen better.

Let me know your availability.

Best,
Your Manager"""

    insights['manager_notes'] = f"""
**Immediate Actions:**
1. Schedule empathetic 1:1 conversation this week
2. Acknowledge recent contributions and performance
3. Explore specific pain points without judgment
4. Co-create a personalized development roadmap

**Key Discussion Points:**
- Career aspirations and growth timeline
- Work-life balance concerns
- Manager support and feedback quality
- Learning and development opportunities
- Compensation and benefits alignment

**Follow-up:**
- Document agreed actions
- Set review date (2-4 weeks)
- Ensure transparency in growth opportunities
"""
    
    return insights

# ============================================
# PAGE LAYOUT
# ============================================

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

use_default_csv = st.sidebar.checkbox("Use default dataset", value=True)
csv_path = DEFAULT_CSV
if not use_default_csv:
    uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv is not None:
        csv_path = uploaded_csv

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dashboard Features")
st.sidebar.markdown("""
- **Overview**: Key metrics and organizational trends
- **Risk Profiles**: Individual employee risk assessment
- **Deep Dive**: Feature-level analysis
- **Retention Actions**: Personalized interventions
- **Manager Copilot**: AI-guided retention strategies
""")

# Load data
load_error = None
try:
    df, numeric_cols, categorical_cols = load_data(csv_path)
except Exception as e:
    df, numeric_cols, categorical_cols = None, [], []
    load_error = str(e)

if load_error:
    st.error(f"❌ Failed to load dataset: {load_error}")
    st.stop()

# Train model
model, scaler, le_dict, feature_cols, feature_importance = train_attrition_model(df, numeric_cols, categorical_cols)

# ============================================
# MAIN DASHBOARD
# ============================================

st.title("🎯 Velorium Technologies - Retention Copilot")
st.markdown("*Data-driven insights to prevent attrition and enable earlier empathy*")
st.markdown("---")

# ============================================
# SECTION 1: EXECUTIVE OVERVIEW
# ============================================

st.header("📈 Executive Overview")

col1, col2, col3, col4 = st.columns(4)

total, attr_count, rate = compute_overview(df)
high_risk_count = sum(1 for emp_id in df[ID_FIELD].unique() 
                      if predict_attrition_risk(emp_id, df, numeric_cols, categorical_cols, 
                                               model, scaler, le_dict, feature_cols) >= 0.7)

with col1:
    st.metric("Total Employees", f"{total:,}")

with col2:
    st.metric("Historical Attrition", f"{attr_count}")

with col3:
    st.metric("Attrition Rate", f"{rate*100:.1f}%")

with col4:
    st.metric("High Risk Employees", f"{high_risk_count}")

st.markdown("---")

# ============================================
# SECTION 2: DEPARTMENT INSIGHTS
# ============================================

col1, col2 = st.columns(2)

# Attrition by Department
if "Department" in df.columns:
    with col1:
        st.subheader("📍 Attrition Rate by Department")
        dept_breakdown = categorical_breakdown(df, "Department")
        dept_breakdown = dept_breakdown.sort_values('attritionRate', ascending=False)
        
        chart_dept = alt.Chart(dept_breakdown).mark_bar().encode(
            x=alt.X('Department:N', sort='-y', title='Department'),
            y=alt.Y('attritionRate:Q', title='Attrition Rate', axis=alt.Axis(format='%')),
            color=alt.condition(
                alt.datum.attritionRate > 0.3,
                alt.value('#d32f2f'),
                alt.condition(
                    alt.datum.attritionRate > 0.15,
                    alt.value('#f57c00'),
                    alt.value('#388e3c')
                )
            ),
            tooltip=['Department:N', 'total:Q', 'attrition:Q', alt.Tooltip('attritionRate:Q', format=".1%")]
        ).properties(height=300)
        
        st.altair_chart(chart_dept, use_container_width=True)

# Attrition by Job Level
if "Job_Level" in df.columns:
    with col2:
        st.subheader("📊 Attrition Rate by Job Level")
        level_breakdown = categorical_breakdown(df, "Job_Level")
        level_breakdown = level_breakdown.sort_values('attritionRate', ascending=False)
        
        chart_level = alt.Chart(level_breakdown).mark_bar(color="#667eea").encode(
            x=alt.X('Job_Level:N', sort='-y', title='Job Level'),
            y=alt.Y('attritionRate:Q', title='Attrition Rate', axis=alt.Axis(format='%')),
            tooltip=['Job_Level:N', 'total:Q', 'attrition:Q', alt.Tooltip('attritionRate:Q', format=".1%")]
        ).properties(height=300)
        
        st.altair_chart(chart_level, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 3: TOP RISK DRIVERS
# ============================================

st.header("⚠️ Top Risk Drivers")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Feature Importance in Churn Prediction")
    top_features = feature_importance.head(10)
    
    chart_features = alt.Chart(top_features).mark_barh().encode(
        y=alt.Y('feature:N', sort='-x'),
        x=alt.X('importance:Q', title='Importance Score'),
        color=alt.value('#667eea')
    ).properties(height=350)
    
    st.altair_chart(chart_features, use_container_width=True)

with col2:
    st.subheader("📋 Risk Factor Analysis")
    
    # Analyze individual risk factors
    if 'Work_Life_Balance' in df.columns:
        wlb_dist = df['Work_Life_Balance'].value_counts().sort_index()
        st.markdown("**Work-Life Balance Distribution:**")
        st.bar_chart(wlb_dist)
    
    if 'Job_Satisfaction' in df.columns:
        st.markdown("**Job Satisfaction Distribution:**")
        js_dist = df['Job_Satisfaction'].value_counts().sort_index()
        st.bar_chart(js_dist)

st.markdown("---")

# ============================================
# SECTION 4: INDIVIDUAL RISK ASSESSMENT
# ============================================

st.header("👤 Employee Risk Profiles")

risk_tab1, risk_tab2, risk_tab3 = st.tabs(["High Risk Employees", "At-Risk Analysis", "Individual Profile"])

with risk_tab1:
    st.subheader("🔴 High-Risk Employees Requiring Immediate Action")
    
    high_risk_employees = []
    for emp_id in df[ID_FIELD].unique():
        risk_prob = predict_attrition_risk(emp_id, df, numeric_cols, categorical_cols, 
                                           model, scaler, le_dict, feature_cols)
        if risk_prob and risk_prob >= 0.6:
            emp_data = df[df[ID_FIELD] == emp_id].iloc[0]
            high_risk_employees.append({
                'Employee_ID': emp_id,
                'Department': emp_data.get('Department', 'N/A'),
                'Job_Role': emp_data.get('Job_Role', 'N/A'),
                'Risk_Score': risk_prob,
                'Work_Life_Balance': emp_data.get('Work_Life_Balance', 'N/A'),
                'Job_Satisfaction': emp_data.get('Job_Satisfaction', 'N/A'),
                'Tenure_Years': emp_data.get('Years_at_Company', 'N/A')
            })
    
    if high_risk_employees:
        risk_df = pd.DataFrame(high_risk_employees).sort_values('Risk_Score', ascending=False)
        
        st.dataframe(
            risk_df.style.format({
                'Risk_Score': '{:.1%}',
                'Work_Life_Balance': '{:.0f}',
                'Job_Satisfaction': '{:.0f}'
            }),
            use_container_width=True
        )
        
        st.markdown(f"**Total High-Risk Employees:** {len(high_risk_employees)}")
        st.markdown(f"**Recommended Action:** Priority conversations with managers for immediate interventions")
    else:
        st.info("✅ No employees in high-risk category currently")

with risk_tab2:
    st.subheader("🟡 Medium Risk Analysis (40-60% Risk)")
    
    medium_risk_employees = []
    for emp_id in df[ID_FIELD].unique():
        risk_prob = predict_attrition_risk(emp_id, df, numeric_cols, categorical_cols, 
                                           model, scaler, le_dict, feature_cols)
        if risk_prob and 0.4 <= risk_prob < 0.6:
            emp_data = df[df[ID_FIELD] == emp_id].iloc[0]
            medium_risk_employees.append({
                'Employee_ID': emp_id,
                'Department': emp_data.get('Department', 'N/A'),
                'Risk_Score': risk_prob,
                'Work_Life_Balance': emp_data.get('Work_Life_Balance', 'N/A'),
                'Years_Since_Promotion': emp_data.get('Years_Since_Last_Promotion', 'N/A')
            })
    
    if medium_risk_employees:
        medium_df = pd.DataFrame(medium_risk_employees).sort_values('Risk_Score', ascending=False)
        st.dataframe(
            medium_df.style.format({'Risk_Score': '{:.1%}'}),
            use_container_width=True
        )
        st.markdown(f"**Total Medium-Risk Employees:** {len(medium_risk_employees)}")
        st.markdown("**Recommended Action:** Regular check-ins and proactive career development planning")
    else:
        st.info("✅ No employees in medium-risk category")

with risk_tab3:
    st.subheader("🔍 Individual Employee Deep Dive")
    
    selected_emp = st.selectbox(
        "Select Employee",
        options=sorted(df[ID_FIELD].unique()),
        format_func=lambda x: f"Employee {int(x)}"
    )
    
    if selected_emp:
        emp_data = df[df[ID_FIELD] == selected_emp].iloc[0]
        risk_prob = predict_attrition_risk(selected_emp, df, numeric_cols, categorical_cols, 
                                          model, scaler, le_dict, feature_cols)
        
        # Display risk assessment
        risk_label, risk_class = get_risk_category(risk_prob)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div style='text-align: center;'><h3>{risk_label}</h3><h2>{risk_prob:.1%}</h2></div>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Department", emp_data.get('Department', 'N/A'))
            st.metric("Job Role", emp_data.get('Job_Role', 'N/A'))
        
        with col3:
            st.metric("Tenure (Years)", f"{emp_data.get('Years_at_Company', 0):.0f}")
            st.metric("Monthly Income", f"₹{emp_data.get('Monthly_Income', 0):,.0f}")
        
        st.markdown("---")
        
        # Generate retention insights
        insights = generate_retention_insights(emp_data, feature_importance, numeric_cols, categorical_cols)
        
        # Display risk drivers
        st.subheader("⚠️ Risk Drivers")
        if insights['risk_drivers']:
            for factor, value, severity in insights['risk_drivers']:
                if severity == 'high':
                    st.markdown(f"<div class='recommendation-box'><strong>🔴 {factor}:</strong> {value:.1f}</div>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='recommendation-box'><strong>🟡 {factor}:</strong> {value:.1f}</div>", 
                               unsafe_allow_html=True)
        else:
            st.info("No significant risk drivers detected")
        
        # Display interventions
        st.subheader("💡 Recommended Interventions")
        for intervention in insights['interventions']:
            st.markdown(f"<div class='action-box'>✓ {intervention}</div>", unsafe_allow_html=True)
        
        # Display email draft
        st.subheader("📧 Manager Email Draft")
        st.text_area("Copy and customize this message:", value=insights['email_draft'], height=200, disabled=True)
        
        # Display manager notes
        st.subheader("📝 Manager Action Plan")
        st.markdown(insights['manager_notes'])

st.markdown("---")

# ============================================
# SECTION 5: RETENTION STRATEGY
# ============================================

st.header("🎯 Strategic Retention Actions")

strategy_col1, strategy_col2 = st.columns(2)

with strategy_col1:
    st.subheader("🏢 Department-Level Actions")
    
    dept_risk = []
    for dept in df['Department'].unique():
        dept_data = df[df['Department'] == dept]
        dept_attrition = dept_data[ATTRITION_FIELD].mean()
        dept_risk.append((dept, dept_attrition))
    
    dept_risk.sort(key=lambda x: x[1], reverse=True)
    
    for dept, attrition in dept_risk:
        if attrition > 0.2:
            st.markdown(f"""
            <div class='recommendation-box'>
            <strong>{dept}</strong> (Attrition: {attrition:.1%})
            <ul>
            <li>Conduct focus group discussions</li>
            <li>Review compensation competitiveness</li>
            <li>Enhance manager training in retention</li>
            <li>Create career path clarity</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

with strategy_col2:
    st.subheader("🎓 Organization-Wide Initiatives")
    
    st.markdown("""
    <div class='recommendation-box'>
    <h4>1️⃣ Early Warning System</h4>
    <p>Deploy this copilot in weekly manager 1:1s to identify flight risks early and intervene proactively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='recommendation-box'>
    <h4>2️⃣ Career Pathways</h4>
    <p>Create transparent, multi-track career progression (technical, managerial, specialist) with clear milestones.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='recommendation-box'>
    <h4>3️⃣ Work-Life Balance Programs</h4>
    <p>Implement flexible schedules, project rotation, and wellness initiatives to reduce burnout.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='recommendation-box'>
    <h4>4️⃣ Manager Development</h4>
    <p>Train managers in empathetic leadership, career coaching, and retention conversations.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SECTION 6: MODEL PERFORMANCE
# ============================================

st.header("📊 Model Performance & Insights")

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.subheader("🎯 Model Metrics")
    
    # Calculate metrics
    y_true = df[ATTRITION_FIELD].fillna(0)
    y_pred_probs = []
    for emp_id in df[ID_FIELD].unique():
        risk_prob = predict_attrition_risk(emp_id, df, numeric_cols, categorical_cols, 
                                           model, scaler, le_dict, feature_cols)
        y_pred_probs.append(risk_prob if risk_prob else 0.5)
    
    if len(y_pred_probs) == len(y_true):
        auc_score = roc_auc_score(y_true, y_pred_probs)
        st.metric("AUC-ROC Score", f"{auc_score:.3f}")
        st.metric("Model Accuracy", f"{model.score(scaler.transform(df[feature_cols].fillna(df[feature_cols].mean())), y_true):.1%}")

with perf_col2:
    st.subheader("📈 Key Insights")
    st.markdown("""
    - **Model Approach:** Random Forest with feature importance ranking
    - **Key Predictors:** Work-Life Balance, Job Satisfaction, Tenure, Manager Relationship
    - **Intervention Focus:** Proactive conversations, growth clarity, workload management
    - **Success Metric:** Reduced time-to-intervention and improved retention
    """)

st.markdown("---")

# ============================================
# FOOTER
# ============================================

st.markdown("""
---
### 📌 How to Use This Copilot:

1. **Review Risk Profiles:** Start with "High Risk Employees" tab to identify immediate action items
2. **Understand Drivers:** Review feature importance and individual risk factors
3. **Personalize Actions:** Use the email draft and manager notes as starting points for meaningful conversations
4. **Track Impact:** Monitor risk scores and attrition rates monthly

**Philosophy:** Data didn't make us colder. It made us listen. — Raghav Sethi, CEO

**Questions or Feedback?** Contact the HR Analytics team for dashboard enhancements and model refinements.
""")
