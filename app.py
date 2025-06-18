
import joblib
import pandas as pd
import gradio as gr

# Load trained model
clf = joblib.load("sepsis_model.pkl")

# Define feature names (must match training data)
feature_names = [
    "IL6", "CRP", "PCT", "IL8", "TNFa", "SAA",
    "Lactate", "WBC", "IL1b", "VEGF", "IL10"
]

# Define prediction logic
def predict_risk(values):
    input_df = pd.DataFrame([values], columns=feature_names)
    prediction = clf.predict(input_df)[0]
    return f"🧠 Predicted Sepsis Risk: {prediction.upper()}"

# Gradio interface
def predict_sepsis_risk(IL6, CRP, PCT, IL8, TNFa, SAA, Lactate, WBC, IL1b, VEGF, IL10):
    values = [IL6, CRP, PCT, IL8, TNFa, SAA, Lactate, WBC, IL1b, VEGF, IL10]
    return predict_risk(values)

gr.Interface(
    fn=predict_sepsis_risk,
    inputs=[
        gr.Number(label="IL-6 (pg/mL)"),
        gr.Number(label="CRP (mg/L)"),
        gr.Number(label="PCT (ng/mL)"),
        gr.Number(label="IL-8 (pg/mL)"),
        gr.Number(label="TNF-α (pg/mL)"),
        gr.Number(label="SAA (mg/L)"),
        gr.Number(label="Lactate (mmol/L)"),
        gr.Number(label="WBC Count (/μL)"),
        gr.Number(label="IL-1β (pg/mL)"),
        gr.Number(label="VEGF (pg/mL)"),
        gr.Number(label="IL-10 (pg/mL)")
    ],
    outputs="text",
    title="🧬 Sepsis Risk Detector",
    description="Enter biomarker values to predict sepsis risk using AI."
).launch(server_port=10000)
