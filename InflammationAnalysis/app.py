import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from groq import Groq
import os

st.set_page_config(page_title="Anemia Risk Assessment", layout="wide")
st.title("Agentic AI System for Anemia Risk Assessment")
st.markdown("*Built on multibiomarker correlation *")

@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("training_medians.pkl", "rb") as f:
        training_medians = pickle.load(f)
    return model, scaler, feature_names, training_medians

model, scaler, feature_names, training_medians = load_model()

try:
    api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=api_key)
except:
    api_key = st.sidebar.text_input("Groq API Key (optional):", type="password")
    client = Groq(api_key=api_key) if api_key else None

biomarker_info = {
    "albumin": ("Albumin", "g/dL", 3.5, 5.5),
    "wbc": ("WBC", "K/μL", 4.0, 11.0),
    "hematocrit": ("Hematocrit", "%", 36.0, 46.0),
    "crp": ("CRP", "mg/L", 0.0, 10.0),
    "platelet": ("Platelets", "K/μL", 150, 400),
    "neutrophil": ("Neutrophils", "K/μL", 2.0, 7.5),
    "lymphocyte": ("Lymphocytes", "K/μL", 1.0, 4.0),
    "ferritin": ("Ferritin", "ng/mL", 30, 300)
}

st.sidebar.header("Enter Biomarker Values")
inputs = {}
provided = []

for feature in feature_names:
    label, unit, min_val, max_val = biomarker_info[feature]
    use = st.sidebar.checkbox(f"{label}", key=f"use_{feature}")
    if use:
        val = st.sidebar.number_input(f"{label} ({unit})", value=float((min_val+max_val)/2), step=0.1)
        inputs[feature] = val
        provided.append(feature)
    else:
        inputs[feature] = training_medians[feature]

if st.sidebar.button("Assess Risk"):
    X = pd.DataFrame([inputs])
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Anemia Risk", f"{prob*100:.1f}%")
    risk = "🔴 HIGH" if prob > 0.7 else "🟡 MODERATE" if prob > 0.4 else "🟢 LOW"
    col2.metric("Risk Level", risk)
    col3.metric("Classification", "ANEMIA" if pred == 1 else "NO ANEMIA")
    
    st.markdown("---")
    st.subheader("Provided Biomarkers")
    for bio in provided:
        st.write(f"• {biomarker_info[bio][0]}: {inputs[bio]:.1f} {biomarker_info[bio][1]}")
    
    st.markdown("---")
    st.subheader("SHAP Explainability")
    
    shap_data = []
    for idx, feature in enumerate(feature_names):
        if feature in provided:
            shap_data.append({
                "Feature": biomarker_info[feature][0],
                "SHAP Value": shap_values[0][idx],
                "Impact": "Increases Risk" if shap_values[0][idx] > 0 else "Decreases Risk"
            })
    
    shap_df = pd.DataFrame(shap_data).sort_values("SHAP Value", key=abs, ascending=False)
    for _, row in shap_df.iterrows():
        color = "🔴" if row["SHAP Value"] > 0 else "🟢"
        st.write(f"{color} **{row['Feature']}**: {row['SHAP Value']:+.3f} ({row['Impact']})")
    
    if client:
        st.markdown("---")
        st.subheader(" AI Clinical Reasoning")
        st.caption(" Please consult a doctor for accurate medical diagnosis")
        
        biomarker_text = "\n".join([f"- {biomarker_info[b][0]}: {inputs[b]:.1f} {biomarker_info[b][1]}" for b in provided])
        shap_text = "\n".join([f"- {row['Feature']}: {row['SHAP Value']:+.3f}" for _, row in shap_df.iterrows()])
        
        prompt = f"""You are a clinical AI assistant. A patient has been assessed for anemia risk.

BIOMARKERS:
{biomarker_text}

PREDICTION:
- Anemia Risk: {prob*100:.1f}%
- Classification: {"Anemia Detected" if pred == 1 else "No Anemia"}

SHAP ANALYSIS (Feature Contributions):
{shap_text}

CONTEXT:
- Model built on discovering albumin-hemoglobin correlation (r=0.446, p<0.001) and relationships between inflammation biomarkers
- Uses XGBoost integrating multiple biomarkers for comprehensive anemia risk assessment

Provide:
1. Clinical interpretation of the anemia risk
2. Which biomarkers are most concerning and why
3. Recommended next steps
4. Relevant pathophysiology

Keep response clear and actionable."""

        with st.spinner("Generating clinical reasoning"):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq API error: {str(e)}")
    else:
        st.info("Enter Groq API key in sidebar for AI clinical reasoning")
