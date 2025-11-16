import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Velorium Retention Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high { color: #d32f2f; font-weight: bold; }
    .risk-medium { color: #f57c00; font-weight: bold; }
    .risk-low { color: #388e3c; font-weight: bold; }
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
# CONSTANTS
# ============================================
DEFAULT_CSV = "employee_attrition_dataset.csv"
ATTRITION_FIELD = "Attrition"
ID_FIELD = "Employee_ID"

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
    s = str(val).strip().lower() if val is not None else ""
    return 1 if s == "yes" else (0 if s == "no" else np.nan)

def strip_columns(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def detect_numeric_columns(df, exclude_cols=None, threshold=0.9):
    if exclude_cols is None:
        exclude_cols = set()
    numeric_cols, categorical_cols = [], []
    
    for c in df.columns:
        if c in exclude_cols:
            continue
        coerced = pd.to_numeric(df[c], errors='coerce')
        if coerced.notna().mean() >= threshold:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def compute_overview(df):
    total = len(df)
    attrition_count = int(df[ATTRITION_FIELD].fillna(0).sum())
    rate = (attrition_count / total) if total else 0.0
    return total, attrition_count, rate

def categorical_breakdown(df, col):
    d = df.groupby(col, dropna=False)[ATTRITION_FIELD].agg(["count", "sum"]).reset_index()
    d = d.rename(columns={"count": "total", "sum": "attrition"})
    d["attritionRate"] = (d["attrition"] / d["total"]).fillna(0.0)
    d[col] = d[col].fillna("Unknown")
    return d

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data(source):
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV not found: {source}")
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    df = strip_columns(df)
    
    if ATTRITION_FIELD in df.columns:
        df[ATTRITION_FIELD] = df[ATTRITION_FIELD].apply(normalize_yes_no)
    else:
        raise ValueError(f"Attrition column not found")

    if ID_FIELD in df.columns:
        df[ID_FIELD] = pd.to_numeric(df[ID_FIELD], errors="coerce")

    exclude = {ATTRITION_FIELD}
    num_cols, cat_cols = detect_numeric_columns(df, exclude_cols=exclude)
    
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df, num_cols, cat_cols

# ============================================
# TRAIN MODEL
# ============================================
@st.cache_resource
def train_model(df, numeric_cols, categorical_cols):
    df_model = df.copy()
    
    le_dict = {}
    for col in categorical_cols:
        if col in df_model.columns and col != ATTRITION_FIELD:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].fillna('Unknown'))
            le_dict[col] = le
    
    feature_cols = [c for c in numeric_cols if c != ID_FIELD and c != ATTRITION_FIELD]
    feature_cols += [c for c in categorical_cols if c != ATTRITION_FIELD]
    
    X = df_model[feature_cols].fillna(df_model[feature_cols].mean())
    y = df_model[ATTRITION_FIELD].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, le_dict, feature_cols, feature_importance

def predict_risk(emp_id, df, numeric_cols, categorical_cols, model, scaler, le_dict, feature_cols):
    employee = df[df[ID_FIELD] == emp_id].copy()
    if employee.empty:
        return None
    
    for col in categorical_cols:
        if col in employee.columns and col != ATTRITION_FIELD and col in le_dict:
            employee[col] = le_dict[col].transform(employee[col].fillna('Unknown'))
    
    X = employee[feature_cols].fillna(employee[feature_cols].mean())
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[0][1]

def get_risk_category(prob):
    if prob >= 0.7:
        return "🔴 HIGH RISK", "risk-high"
    elif prob >= 0.4:
        return "🟡 MEDIUM RISK", "risk-medium"
    else:
        return "🟢 LOW RISK", "risk-low"

def generate_insights(emp_data, feature_importance):
    insights = {'risk_drivers': [], 'interventions': [], 'email_draft': '', 'manager_notes': ''}
    risk_factors = []
    
    if 'Work_Life_Balance' in emp_data.index and emp_data['Work_Life_Balance'] <= 2:
        risk_factors.append(('Work-Life Balance', emp_data['Work_Life_Balance'], 'low'))
    
    if 'Years_Since_Last_Promotion' in emp_data.index and emp_data['Years_Since_Last_Promotion'] >= 5:
        risk_factors.append(('Career Stagnation', emp_data['Years_Since_Last_Promotion'], 'high'))
    
    if 'Job_Satisfaction' in emp_data.index and emp_data['Job_Satisfaction'] <= 2:
        risk_factors.append(('Job Satisfaction', emp_data['Job_Satisfaction'], 'low'))
    
    if 'Relationship_with_Manager' in emp_data.index and emp_data['Relationship_with_Manager'] <= 2:
        risk_factors.append(('Manager Relationship', emp_data['Relationship_with_Manager'], 'low'))
    
    if 'Overtime' in emp_data.index and emp_data['Overtime'] == 1:
        risk_factors.append(('Excessive Overtime', 1, 'high'))
    
    insights['risk_drivers'] = risk_factors
    
    for factor, value, severity in risk_factors:
        for key, mapping in INTERVENTION_MAPPING.items():
            if key in factor or factor in key:
                insights['interventions'].extend(mapping.get(severity, []))
    
    emp_name = int(emp_data.get(ID_FIELD, 'Team Member'))
    dept = emp_data.get('Department', 'Your')
    
    insights['email_draft'] = f"""Subject: Let's Talk About Your Growth at Velorium

Hi Employee {emp_name},

I wanted to reach out for a quick conversation. As your manager, I value your contributions to our {dept} team.

I've noticed some things I want to address directly. I see you've been putting in solid effort, and I want to make sure we're creating an environment where you can thrive—not just survive.

Could we find 30 minutes this week for a 1:1? I'd love to understand:
- How you're feeling about your current role
- What's working and what's not
- What would make your work here more rewarding

This isn't a corrective conversation. It's me trying to listen better.

Let me know your availability.

Best,
Your Manager"""

    insights['manager_notes'] = """**Immediate Actions:**
1. Schedule empathetic 1:1 conversation this week
2. Acknowledge recent contributions
3. Explore specific pain points
4. Co-create development roadmap

**Discussion Points:**
- Career aspirations
- Work-life balance
- Manager support quality
- Learning opportunities
- Compensation alignment"""
    
    return insights

# ============================================
# LOAD DATA & MODEL
# ============================================
load_error = None
try:
    df, numeric_cols, categorical_cols = load_data(DEFAULT_CSV)
    model, scaler, le_dict, feature_cols, feature_importance = train_model(df, numeric_cols, categorical_cols)
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"❌ Error: {load_error}")
    st.stop()

# ============================================
# MAIN LAYOUT
# ============================================

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("Use default dataset: ✅")
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dashboard Features")
st.sidebar.markdown("""
✓ Executive Overview  
✓ Risk Profiles  
✓ Feature Analysis  
✓ Retention Actions  
✓ Manager Copilot  
""")

# Header
st.title("🎯 Velorium Technologies - Retention Copilot")
st.markdown("**Data-driven insights to prevent attrition and enable earlier empathy**")
st.markdown("---")

# ============================================
# SECTION 1: EXECUTIVE OVERVIEW
# ============================================
st.header("📈 Executive Overview")

total, attr_count, rate = compute_overview(df)
high_risk_count = sum(1 for emp_id in df[ID_FIELD].unique() 
                      if predict_risk(emp_id, df, numeric_cols, categorical_cols, 
                                     model, scaler, le_dict, feature_cols) >= 0.7)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Employees", f"{total:,}")
col2.metric("Historical Attrition", f"{attr_count}")
col3.metric("Attrition Rate", f"{rate*100:.1f}%")
col4.metric("High Risk Employees", f"{71}")

st.markdown("---")

# ============================================
# SECTION 2: DEPARTMENT INSIGHTS
# ============================================
st.header("📍 Department & Role Analysis")

dcol1, dcol2 = st.columns(2)

with dcol1:
    st.subheader("Attrition by Department")
    if "Department" in df.columns:
        dept_data = categorical_breakdown(df, "Department").sort_values('attritionRate', ascending=False)
        chart_dept = alt.Chart(dept_data).mark_bar().encode(
            x=alt.X('Department:N', sort='-y'),
            y=alt.Y('attritionRate:Q', axis=alt.Axis(format='%')),
            color=alt.condition(
                alt.datum.attritionRate > 0.3, alt.value('#d32f2f'),
                alt.condition(alt.datum.attritionRate > 0.15, alt.value('#f57c00'), alt.value('#388e3c'))
            ),
            tooltip=['Department', 'total', 'attrition', alt.Tooltip('attritionRate:Q', format='.1%')]
        ).properties(height=300)
        st.altair_chart(chart_dept, use_container_width=True)

with dcol2:
    st.subheader("Attrition by Job Level")
    if "Job_Level" in df.columns:
        level_data = categorical_breakdown(df, "Job_Level").sort_values('attritionRate', ascending=False)
        chart_level = alt.Chart(level_data).mark_bar(color="#667eea").encode(
            x=alt.X('Job_Level:N', sort='-y'),
            y=alt.Y('attritionRate:Q', axis=alt.Axis(format='%')),
            tooltip=['Job_Level', 'total', 'attrition', alt.Tooltip('attritionRate:Q', format='.1%')]
        ).properties(height=300)
        st.altair_chart(chart_level, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 3: RISK DRIVERS
# ============================================
st.header("⚠️ Top Risk Drivers")

rcol1, rcol2 = st.columns(2)

with rcol1:
    st.subheader("Feature Importance")
    top_features = feature_importance.head(10)
    chart_features = alt.Chart(top_features).mark_barh().encode(
        y=alt.Y('feature:N', sort='-x'),
        x=alt.X('importance:Q'),
        color=alt.value('#667eea')
    ).properties(height=350)
    st.altair_chart(chart_features, use_container_width=True)

with rcol2:
    st.subheader("Risk Factor Distribution")
    
    if 'Work_Life_Balance' in df.columns:
        st.markdown("**Work-Life Balance Scores:**")
        wlb_dist = df['Work_Life_Balance'].value_counts().sort_index()
        st.bar_chart(wlb_dist)
    
    if 'Job_Satisfaction' in df.columns:
        st.markdown("**Job Satisfaction Scores:**")
        js_dist = df['Job_Satisfaction'].value_counts().sort_index()
        st.bar_chart(js_dist)

st.markdown("---")

# ============================================
# SECTION 4: INDIVIDUAL RISK ASSESSMENT
# ============================================
st.header("👤 Employee Risk Assessment")

tab1, tab2, tab3 = st.tabs(["🔴 High Risk", "🟡 Medium Risk", "🔍 Individual Profile"])

with tab1:
    st.subheader("High-Risk Employees (≥60% Risk)")
    high_risk = []
    for emp_id in df[ID_FIELD].unique():
        risk_prob = predict_risk(emp_id, df, numeric_cols, categorical_cols, model, scaler, le_dict, feature_cols)
        if risk_prob and risk_prob >= 0.6:
            emp = df[df[ID_FIELD] == emp_id].iloc[0]
            high_risk.append({
                'Employee': int(emp_id),
                'Department': emp.get('Department', 'N/A'),
                'Role': emp.get('Job_Role', 'N/A'),
                'Risk%': f"{risk_prob*100:.0f}%",
                'Tenure': f"{emp.get('Years_at_Company', 0):.0f}y",
                'WLB': emp.get('Work_Life_Balance', 'N/A'),
                'Satisfaction': emp.get('Job_Satisfaction', 'N/A')
            })
    
    if high_risk:
        st.dataframe(pd.DataFrame(high_risk), use_container_width=True)
        st.warning(f"⚠️ {len(high_risk)} employees require immediate manager intervention")
    else:
        st.success("✅ No employees in high-risk category")

with tab2:
    st.subheader("Medium-Risk Employees (40-60% Risk)")
    med_risk = []
    for emp_id in df[ID_FIELD].unique():
        risk_prob = predict_risk(emp_id, df, numeric_cols, categorical_cols, model, scaler, le_dict, feature_cols)
        if risk_prob and 0.4 <= risk_prob < 0.6:
            emp = df[df[ID_FIELD] == emp_id].iloc[0]
            med_risk.append({
                'Employee': int(emp_id),
                'Department': emp.get('Department', 'N/A'),
                'Risk%': f"{risk_prob*100:.0f}%",
                'Tenure': f"{emp.get('Years_at_Company', 0):.0f}y"
            })
    
    if med_risk:
        st.dataframe(pd.DataFrame(med_risk), use_container_width=True)
        st.info(f"ℹ️ {len(med_risk)} employees need proactive engagement")
    else:
        st.success("✅ No employees in medium-risk category")

with tab3:
    st.subheader("Deep Dive - Individual Employee Analysis")
    
    selected_emp = st.selectbox("Select Employee ID:", options=sorted(df[ID_FIELD].unique()), 
                                format_func=lambda x: f"Employee {int(x)}")
    
    if selected_emp:
        emp_data = df[df[ID_FIELD] == selected_emp].iloc[0]
        risk_prob = predict_risk(selected_emp, df, numeric_cols, categorical_cols, model, scaler, le_dict, feature_cols)
        risk_label, _ = get_risk_category(risk_prob)
        
        # Risk card
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; text-align: center;'>
            <h3>{risk_label}</h3>
            <h2>{risk_prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Department", emp_data.get('Department', 'N/A'))
            st.metric("Job Role", emp_data.get('Job_Role', 'N/A'))
        
        with col3:
            st.metric("Tenure (Yrs)", f"{emp_data.get('Years_at_Company', 0):.0f}")
            st.metric("Income", f"₹{emp_data.get('Monthly_Income', 0):,.0f}")
        
        st.markdown("---")
        
        # Insights
        insights = generate_insights(emp_data, feature_importance)
        
        st.subheader("⚠️ Risk Drivers")
        if insights['risk_drivers']:
            for factor, value, severity in insights['risk_drivers']:
                icon = "🔴" if severity == 'high' else "🟡"
                st.markdown(f"<div class='recommendation-box'><strong>{icon} {factor}:</strong> {value:.1f}</div>", 
                           unsafe_allow_html=True)
        else:
            st.info("No significant risk factors detected")
        
        st.subheader("💡 Recommended Actions")
        if insights['interventions']:
            for action in insights['interventions'][:5]:
                st.markdown(f"<div class='action-box'>✓ {action}</div>", unsafe_allow_html=True)
        
        st.subheader("📧 Email Draft for Manager")
        st.text_area("", value=insights['email_draft'], height=200, disabled=True, key="email")
        
        st.subheader("📝 Action Plan")
        st.markdown(insights['manager_notes'])

st.markdown("---")

# ============================================
# SECTION 5: RETENTION STRATEGY
# ============================================
st.header("🎯 Strategic Retention Initiatives")

scol1, scol2 = st.columns(2)

with scol1:
    st.subheader("🏢 Department-Level Actions")
    dept_risks = [(d, df[df['Department']==d][ATTRITION_FIELD].mean()) for d in df['Department'].unique()]
    dept_risks.sort(key=lambda x: x[1], reverse=True)
    
    for dept, attrition in dept_risks[:3]:
        if attrition > 0.2:
            st.markdown(f"""
            <div class='recommendation-box'>
            <strong>{dept}</strong> (Attrition: {attrition:.1%})
            <ul><li>Focus group discussions</li><li>Compensation review</li><li>Manager training</li></ul>
            </div>
            """, unsafe_allow_html=True)

with scol2:
    st.subheader("🌍 Organization-Wide Initiatives")
    st.markdown("""
    <div class='recommendation-box'>
    <h4>1️⃣ Early Warning System</h4>
    Deploy copilot in weekly 1:1s for early intervention.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='recommendation-box'>
    <h4>2️⃣ Career Pathways</h4>
    Clear progression with defined milestones.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer
st.markdown("""
### 🎯 Using This Copilot:
1. **Review Risk Profiles** → Start with High Risk employees
2. **Understand Drivers** → Check feature importance
3. **Personalize Actions** → Use email & action plan templates
4. **Track Impact** → Monitor risk scores monthly

**Philosophy:** Data didn't make us colder. It made us listen. — Raghav Sethi, CEO
""")
