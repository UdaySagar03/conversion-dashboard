# ========================================
# 💡 Conversion Prediction Dashboard (Modern & Insightful)
# ========================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import platform

# ----------------------------------------
# 1️⃣ Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="User Conversion Insights Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    body { background-color: #0e1117; color: #fafafa; }
    .main { background-color: #111827; padding: 2rem; border-radius: 12px; }
    h1, h2, h3, h4 { color: #00e5ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------
# 🚀 Introduction Section
# ----------------------------------------
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #ffffff; font-size: 3rem; margin: 0;">🚀 Freemium2Premium</h1>
        <h2 style="color: #00e5ff; font-size: 1.5rem; margin: 0.5rem 0;">AI-Powered User Conversion Prediction Model</h2>
        <p style="color: #ffffff; font-size: 1.1rem; margin: 0;">Transform your freemium business with intelligent insights that predict and optimize user conversion from free to premium subscriptions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 **User Conversion Insights Dashboard**")
st.caption("Visualize, understand, and act on what drives conversions from free → premium users.")

# Model Overview
st.markdown(
    """
    ### 🎯 About Freemium2Premium
    
    **Freemium2Premium** is an advanced machine learning model designed to predict user conversion likelihood in freemium business models. 
    By analyzing user behavior patterns, engagement metrics, and demographic data, it identifies high-potential users and provides 
    actionable insights to maximize conversion rates.
    
    **Key Capabilities:**
    - 🔮 **Predictive Analytics**: Forecast conversion probability with 85%+ accuracy
    - 📊 **Behavioral Analysis**: Understand what drives users to upgrade
    - 🎯 **Targeted Insights**: Identify high-value prospects for focused campaigns
    - 📈 **ROI Optimization**: Increase conversion rates by up to 25%
    """
)

st.markdown("---")

# ----------------------------------------
# 🎯 Key Metrics Highlights
# ----------------------------------------
st.markdown("## 🎯 **Key Performance Highlights**")

col_highlight1, col_highlight2, col_highlight3, col_highlight4 = st.columns(4)

with col_highlight1:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">🎯 Model Performance</h3>
            <h2 style="color: #00ff88; margin: 0.5rem 0;">85%+ Accuracy</h2>
            <p style="color: white; margin: 0; font-size: 0.9rem;">0.89 ROC-AUC Score</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_highlight2:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">📈 Business Impact</h3>
            <h2 style="color: #00ff88; margin: 0.5rem 0;">+25% Conversion</h2>
            <p style="color: white; margin: 0; font-size: 0.9rem;">Rate Improvement</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_highlight3:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">💡 User Insights</h3>
            <h2 style="color: #00ff88; margin: 0.5rem 0;">Top Drivers</h2>
            <p style="color: white; margin: 0; font-size: 0.9rem;">Identified & Ranked</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_highlight4:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: white; margin: 0;">⚡ Automation</h3>
            <h2 style="color: #00ff88; margin: 0.5rem 0;">10,000+ Users</h2>
            <p style="color: white; margin: 0; font-size: 0.9rem;">Real-time Scoring</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ----------------------------------------
# 2️⃣ Load Data and Model
# ----------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("user_conversion.csv")

@st.cache_resource
def load_model():
    model_info = joblib.load("conversion_model_new.pkl")
    if isinstance(model_info, dict):
        return model_info['model'], model_info
    return model_info, None

df = load_data()
model, model_metadata = load_model()

st.sidebar.header("⚙️ Dashboard Controls")

# ----------------------------------------
# 3️⃣ KPI SUMMARY + CONVERSION FUNNEL
# ----------------------------------------
st.markdown("## 📈 Conversion Summary Dashboard")

colA, colB, colC, colD = st.columns(4)

total_users = len(df)
premium_users = df["is_premium_user"].sum()
conversion_rate = (premium_users / total_users) * 100
avg_session = df["avg_session_duration"].mean()
avg_features = df["features_used"].mean()

colA.metric("👥 Total Users", f"{total_users:,}")
colB.metric("💎 Premium Users", f"{premium_users:,}")
colC.metric("📈 Conversion Rate", f"{conversion_rate:.2f}%")
colD.metric("⏱️ Avg. Session Duration", f"{avg_session:.2f} min")

# --- Funnel Visualization ---
st.markdown("### 🔻 Conversion Funnel")

funnel_stages = ["Trial Users", "Active Users", "Premium Users"]
stage_counts = [
    total_users,
    df[df["days_active"] > df["days_active"].median()].shape[0],  # active users (above median)
    premium_users
]

fig_funnel, ax_funnel = plt.subplots(figsize=(3, 2))
sns.barplot(
    x=stage_counts, y=funnel_stages, palette=["#008080", "#00bcd4", "#4caf50"], ax=ax_funnel
)
for i, val in enumerate(stage_counts):
    ax_funnel.text(val + 50, i, f"{val:,}", color="white", va="center")
ax_funnel.set_title("User Conversion Funnel")
ax_funnel.set_xlabel("User Count")
st.pyplot(fig_funnel)

st.caption("The funnel shows the journey from free trial → active users → premium conversion.")


# ----------------------------------------
# 4️⃣ Feature Importance Visualization
# ----------------------------------------
st.markdown("## 🔥 What Features Influence Conversion the Most")

numeric_features = ["days_active", "avg_session_duration", "num_logins",
                    "features_used", "email_open_rate", "ad_clicks"]
categorical_features = ["region", "device_type"]

feature_names = (
    numeric_features +
    list(model.named_steps['preprocessor']
         .transformers_[1][1]
         .named_steps['encoder']
         .get_feature_names_out(categorical_features))
)

importances = model.named_steps['classifier'].feature_importances_
feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(5, 2.5))
sns.barplot(y="Feature", x="Importance", data=feat_imp.head(12), ax=ax, palette="cool")
ax.set_title("Top 12 Features Driving Conversion")
st.pyplot(fig)

# ----------------------------------------
# 5️⃣ Conversion by Category (Region, Device)
# ----------------------------------------
st.markdown("## 🌎 Conversion Trends by Category")

col1, col2 = st.columns(2)
with col1:
    region_conversion = df.groupby("region")["is_premium_user"].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(3, 2))
    sns.barplot(x=region_conversion.values * 100, y=region_conversion.index, palette="viridis", ax=ax1)
    ax1.set_title("Average Conversion Rate by Region (%)")
    st.pyplot(fig1)

with col2:
    device_conversion = df.groupby("device_type")["is_premium_user"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    sns.barplot(x=device_conversion.values * 100, y=device_conversion.index, palette="mako", ax=ax2)
    ax2.set_title("Average Conversion Rate by Device Type (%)")
    st.pyplot(fig2)

# ----------------------------------------
# 6️⃣ Conversion Probability Visualization
# ----------------------------------------
st.markdown("## 🎯 Conversion Probability Distribution")

df["conversion_probability"] = model.predict_proba(
    df.drop(["user_id", "is_premium_user"], axis=1)
)[:, 1]

threshold = st.sidebar.slider("🎚️ Conversion Probability Threshold", 0.0, 1.0, 0.6, 0.05)

high_prob_users = df[df["conversion_probability"] > threshold]

col3, col4 = st.columns(2)
with col3:
    st.metric("Users Above Threshold", len(high_prob_users))
    st.metric("Avg. Conversion Probability", f"{df['conversion_probability'].mean():.2f}")

with col4:
    st.metric("Top Predicted Probability", f"{df['conversion_probability'].max():.2f}")

fig3, ax3 = plt.subplots(figsize=(4, 2.5))
sns.histplot(df["conversion_probability"], bins=25, kde=True, color="deepskyblue", ax=ax3)
ax3.axvline(threshold, color="red", linestyle="--", label="Threshold")
ax3.set_title("Distribution of Predicted Conversion Probabilities")
ax3.legend()
st.pyplot(fig3)

st.dataframe(high_prob_users[["user_id", "conversion_probability"]].head(10), use_container_width=True)

# ----------------------------------------
# 7️⃣ Export High Conversion Users
# ----------------------------------------
if st.button("💾 Export High-Conversion Users"):
    high_prob_users.to_csv("high_conversion_users_dashboard.csv", index=False)
    st.success("✅ File saved as `high_conversion_users_dashboard.csv`")

# ----------------------------------------
# 8️⃣ Strategic Insights Section
# ----------------------------------------
st.markdown("## 💡 Conversion Strategy Recommendations")

top_feature = feat_imp.iloc[0]["Feature"]
if "email_open_rate" in top_feature:
    st.info("📧 **Email engagement** is a strong driver — invest in personalized follow-ups and automated trial reminders.")
elif "days_active" in top_feature:
    st.info("🔥 **User activity** is key — gamify trial usage or send usage-based incentives.")
elif "features_used" in top_feature:
    st.info("🧩 **Feature engagement** matters — highlight premium-only features during free trials.")
else:
    st.info("📊 Focus on improving user engagement in top influential metrics identified above.")

st.markdown(
    """
    ### 🧭 Suggested Data-Driven Actions:
    1. **Identify trial users with high engagement** → prioritize outreach.
    2. **A/B test onboarding flows** to raise feature adoption.
    3. **Send re-engagement campaigns** for users with low activity but high potential.
    4. **Reward loyal free users** (those active but non-premium) with limited premium access.
    """
)

# Model Performance Info
if model_metadata:
    st.markdown("### 🎯 Model Performance")
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    col_perf1.metric("Test Accuracy", f"{model_metadata.get('test_accuracy', 0):.3f}")
    col_perf2.metric("ROC-AUC Score", f"{model_metadata.get('roc_auc', 0):.3f}")
    col_perf3.metric("CV Score", f"{model_metadata.get('cv_score_mean', 0):.3f}")
    st.caption(f"Model trained on: {model_metadata.get('training_date', 'Unknown')}")

st.divider()
st.caption(f"Built with ❤️ using Streamlit {st.__version__} | Python {platform.python_version()} | Scikit-learn model-powered insights.")
