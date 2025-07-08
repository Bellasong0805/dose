
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Dose Adjustment Simulator", layout="centered")
st.title("💊 AI-Powered Dose Adjustment Simulator")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500

    data = pd.DataFrame({
        "age": np.random.randint(18, 80, size=n),
        "weight": np.random.normal(70, 12, size=n),
        "height": np.random.normal(170, 10, size=n),
        "sex": np.random.randint(0, 2, size=n),
    })

    data["starting_dose"] = data["weight"] * np.random.normal(10, 1, size=n)
    data["Cmax"] = data["starting_dose"] / data["weight"] * np.random.normal(1.0, 0.1, size=n)
    data["Tmax"] = np.random.normal(1.5, 0.2, size=n)
    data["BP_pre"] = np.random.normal(125, 10, size=n)
    data["BP_post_1h"] = data["BP_pre"] - data["Cmax"] * np.random.normal(4, 1, size=n)
    data["symptom_score"] = np.clip(data["Cmax"] * np.random.normal(2.5, 0.5, size=n), 0, 10)
    data["dose_adjusted"] = np.where(
        (data["BP_post_1h"] < 100) | (data["symptom_score"] > 7),
        data["starting_dose"] * 0.8,
        np.where(data["BP_post_1h"] > 135, data["starting_dose"] * 1.2, data["starting_dose"])
    )

    X = data[["age", "weight", "height", "sex", "starting_dose", "BP_pre", "Cmax", "Tmax", "symptom_score"]]
    y = data["dose_adjusted"]

    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model()

def adjust_dose(predicted_dose, bp_post, symptom_score):
    dose = predicted_dose
    messages = []

    if bp_post < 100:
        dose *= 0.8
        messages.append("Low blood pressure: dose decreased.")
    elif bp_post > 140:
        dose *= 1.2
        messages.append("High blood pressure: dose increased.")

    if symptom_score > 7:
        dose *= 0.7
        messages.append("⚠️ High symptom score: dose reduced.")

    if dose > 1000:
        dose = 1000
        messages.append("Dose capped at 1000mg.")

    return dose, messages

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        weight = st.number_input("Weight (kg)", 30, 120, 60)
        sex = st.selectbox("Sex", ["Male", "Female"])
    with col2:
        height = st.number_input("Height (cm)", 130, 200, 170)
        starting_dose = st.number_input("Previous dose (mg)", 100, 1000, 500, step=50)
        bp_pre = st.number_input("Blood pressure before dose", 80, 180, 125)

    symptom_score = st.slider("Symptom score (0 = none, 10 = severe)", 0.0, 10.0, 5.0)
    submitted = st.form_submit_button("Predict & Add to Chart")

if "dose_history" not in st.session_state:
    st.session_state.dose_history = []
    st.session_state.time_point = []

if submitted:
    sex_val = 0 if sex == "Male" else 1
    Cmax = starting_dose / weight * np.random.normal(1.0, 0.1)
    Tmax = np.random.normal(1.5, 0.2)
    bp_post = bp_pre - Cmax * np.random.normal(4, 1)

    new_input = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "sex": sex_val,
        "starting_dose": starting_dose,
        "BP_pre": bp_pre,
        "Cmax": Cmax,
        "Tmax": Tmax,
        "symptom_score": symptom_score
    }])

    predicted = model.predict(new_input)[0]
    final_dose, messages = adjust_dose(predicted, bp_post, symptom_score)

    st.session_state.dose_history.append(final_dose)
    st.session_state.time_point.append(len(st.session_state.dose_history))

    st.success(f"Predicted dose: {predicted:.1f} mg → Adjusted: {final_dose:.1f} mg")
    for m in messages:
        st.warning(m)

    fig, ax = plt.subplots()
    ax.plot(st.session_state.time_point, st.session_state.dose_history, marker='o')
    ax.set_xlabel("Dose number")
    ax.set_ylabel("Dose (mg)")
    ax.set_title("Dose Adjustment Over Time")
    ax.grid(True)
    st.pyplot(fig)
