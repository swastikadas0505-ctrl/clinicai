"""ClinicAI - Smart Clinic Appointment Assistant (Streamlit)."""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from train_model import (DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET,
                         main as train_main)

st.set_page_config(page_title="ClinicAI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(180deg,#f4faf6 0%,#eef6fc 100%);}
.header {background: linear-gradient(90deg,#1b8a5a,#3a9bd5); padding:26px 30px;
         border-radius:16px; color:white; margin-bottom:18px;}
.header h1 {margin:0; color:white; font-size:2.2rem;}
.header p {margin:4px 0 0 0; font-size:1.05rem; opacity:.95;}
.card {background:white; border-radius:14px; padding:18px; text-align:center;
       box-shadow:0 2px 8px rgba(0,0,0,.07); border-top:4px solid #1b8a5a;}
.card .v {font-size:1.9rem; font-weight:700; color:#14633f;}
.card .l {font-size:.85rem; color:#5b6b75;}
.good {background:#e6f6ec; border-left:6px solid #1b8a5a; padding:16px; border-radius:10px;}
.risk {background:#fff4e0; border-left:6px solid #e89a1c; padding:16px; border-radius:10px;}
.note {background:#eaf4fb; border-left:6px solid #3a9bd5; padding:12px; border-radius:10px; font-size:.9rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load_everything():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH)
            and os.path.exists(DATA_PATH)):
        train_main()  # auto-train if the user skipped the training step
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    data = pd.read_csv(DATA_PATH)
    return model, metrics, data


try:
    model, metrics, df = load_everything()
except Exception as e:
    st.error(f"Could not load the model or data: {e}. Run `python train_model.py` first.")
    st.stop()

# ---------------- Header ----------------
st.markdown("""<div class="header"><h1>🩺 ClinicAI</h1>
<p>Smarter Appointments, Better Patient Care</p></div>""", unsafe_allow_html=True)
st.markdown("""<div class="note"><b>Demo notice:</b> This prototype uses a
<b>synthetic (computer-generated) dataset</b>, not real patient data. Results
show how the system works and are <b>not</b> real-world or clinically validated predictions.</div>""",
            unsafe_allow_html=True)
st.write("")

# ---------------- Summary cards ----------------
total = len(df)
miss_rate = df[TARGET].mean() * 100
c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{total:,}", "Appointments (synthetic dataset)"),
    (c2, f"{100 - miss_rate:.1f}%", "Attendance rate"),
    (c3, f"{miss_rate:.1f}%", "Missed appointment rate"),
    (c4, f"{metrics['accuracy'] * 100:.1f}%", "Test-set accuracy (synthetic data)"),
]:
    col.markdown(f'<div class="card"><div class="v">{val}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)

with st.expander("Full test-set evaluation (synthetic data)"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision (missed class)", f"{metrics['precision']:.2f}")
    m2.metric("Recall (missed class)", f"{metrics['recall']:.2f}")
    m3.metric("F1-score (missed class)", f"{metrics['f1']:.2f}")
    cm = pd.DataFrame(metrics["confusion_matrix"],
                      index=["Actually attended", "Actually missed"],
                      columns=["Predicted attend", "Predicted miss"])
    st.write("Confusion matrix (test set):")
    st.dataframe(cm)

st.write("")
left, right = st.columns([1, 1], gap="large")

# ---------------- Prediction form ----------------
with left:
    st.subheader("📋 Appointment Prediction")
    with st.form("predict_form"):
        age = st.number_input("Patient age (years)", 0, 120, 35)
        prev = st.number_input("Previous appointments", 0, 200, 4)
        missed = st.number_input("Previous missed appointments", 0, 200, 0)
        wait = st.number_input("Waiting time (days until appointment)", 0, 365, 7)
        reminder = st.selectbox("Reminder sent?", ["Yes", "No"])
        go = st.form_submit_button("Predict Attendance", type="primary",
                                   use_container_width=True)

    if go:
        if missed > prev:
            st.error("Previous missed appointments cannot be more than previous appointments.")
        else:
            x = pd.DataFrame([[age, prev, missed, wait, 1 if reminder == "Yes" else 0]],
                             columns=FEATURES)
            p_miss = float(model.predict_proba(x)[0][1])
            p_attend = 1 - p_miss
            if p_miss >= 0.5:
                st.markdown(f'<div class="risk"><h3>⚠️ At Risk of Missing Appointment</h3>'
                            f'Model-estimated chance of missing: <b>{p_miss * 100:.0f}%</b></div>',
                            unsafe_allow_html=True)
                st.write("**Why?** Patterns like a longer wait, past missed visits, or no "
                         "reminder were linked with missed visits in the training data.")
                st.info("💡 Suggested support: send a friendly reminder (SMS/call), confirm the "
                        "appointment, and offer easy rescheduling. Never deny care based on this score.")
            else:
                st.markdown(f'<div class="good"><h3>✅ Likely to Attend</h3>'
                            f'Model-estimated chance of attending: <b>{p_attend * 100:.0f}%</b></div>',
                            unsafe_allow_html=True)
                st.write("**Why?** The entered details resemble appointments that were "
                         "usually attended in the training data.")
                st.info("💡 Suggested support: a standard confirmation message is enough.")
            st.caption("Probability comes from the trained Random Forest on synthetic data; it is a "
                       "demonstration estimate, not a clinical prediction.")

# ---------------- Charts ----------------
with right:
    st.subheader("📊 Dataset Insights")
    counts = df[TARGET].map({0: "Attended", 1: "Missed"}).value_counts()
    st.write("Attended vs Missed appointments")
    st.bar_chart(counts, color="#1b8a5a")

    d = df.copy()
    d["Waiting time"] = pd.cut(d["waiting_days"], [-1, 7, 14, 30, 90],
                               labels=["0-7 days", "8-14 days", "15-30 days", "31+ days"])
    rate = d.groupby("Waiting time", observed=True)[TARGET].mean() * 100
    st.write("Missed appointment rate (%) by waiting time")
    st.bar_chart(rate, color="#3a9bd5")

# ---------------- Project info ----------------
st.divider()
st.subheader("ℹ️ About This Project")
a, b, c = st.columns(3)
a.markdown("**The problem**\n\nPatients sometimes miss clinic appointments, wasting doctors' "
           "time and delaying care for others.")
b.markdown("**How ML helps**\n\nA Random Forest learns patterns from past appointments and "
           "flags those that may need an extra reminder, so staff can offer support early.")
c.markdown("**Limitations**\n\nTrained on synthetic data only; not validated on real patients; "
           "ignores many real factors (transport, illness, etc.). Use only as a demo, "
           "and never to deny care.")
