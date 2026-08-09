import os
import streamlit as st

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from io import BytesIO
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="PreventAI", layout="wide")

st.markdown("""
<style>

body {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

/* Shared hover effect */
.hover-effect {
    transition: all 0.3s ease-in-out;
    cursor: pointer;
}
.hover-effect:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,255,255,0.35);
    background: rgba(0,255,255,0.08);
}
/* Hero Card */
.hero-card {
    padding:2rem;
    border-radius:22px;
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(18px);
    text-align:center;
    margin-bottom:2rem;
    transition:0.3s;
}
.hero-card:hover {
    transform:translateY(-6px);
    box-shadow:0 25px 50px rgba(0,255,255,0.4);
    background:rgba(0,255,255,0.12);
}

/* Metric Cards */
.metric-card {
    padding:1rem;
    border-radius:16px;
    background:rgba(255,255,255,0.06);
    backdrop-filter:blur(12px);
    text-align:center;
    margin-bottom:12px;
    transition:0.3s;
    cursor:pointer;
}
.metric-card:hover {
    transform:scale(1.05);
    background:rgba(0,255,255,0.1);
    box-shadow:0 12px 30px rgba(0,255,255,0.3);
}

/* Recommendation Cards */
.reco-card {
    padding:1rem;
    border-radius:14px;
    background:rgba(0,255,150,0.08);
    margin-bottom:10px;
    transition:0.3s;
    cursor:pointer;
}
.reco-card:hover {
    background:rgba(0,255,150,0.18);
    box-shadow:0 10px 25px rgba(0,255,150,0.3);
}

/* Doctor Cards */
.doctor-card {
    padding:1rem;
    border-radius:14px;
    background:rgba(255,255,255,0.06);
    margin-bottom:10px;
    transition:0.3s;
    cursor:pointer;
}
.doctor-card:hover {
    background:rgba(0,200,255,0.12);
    box-shadow:0 15px 30px rgba(0,200,255,0.4);
}

/* Small Info Box */
.small-box {
    padding:0.6rem;
    border-radius:10px;
    background:rgba(255,255,255,0.05);
    font-size:0.85rem;
    transition:0.3s;
}
.small-box:hover {
    background:rgba(255,255,255,0.1);
    box-shadow:0 8px 20px rgba(255,255,255,0.2);
}

/* Section Headers */
.live-title {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    color: #00e5ff;
    letter-spacing: 0.5px;
}

/* Premium Card for Tab 2 */
.live-card {
    padding: 1.4rem;
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        rgba(0, 255, 255, 0.08),
        rgba(255, 255, 255, 0.05)
    );
    backdrop-filter: blur(14px);
    margin-bottom: 20px;
    transition: all 0.3s ease-in-out;
    border: 1px solid rgba(0,255,255,0.15);
}

.live-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0,255,255,0.35);
    border: 1px solid rgba(0,255,255,0.4);
}

.med-box {
    padding: 0.7rem 1rem;
    border-radius: 12px;
    background: rgba(0,200,255,0.08);
    margin-bottom: 8px;
    transition: 0.3s;
}

.med-box:hover {
    background: rgba(0,200,255,0.18);
    box-shadow: 0 8px 20px rgba(0,200,255,0.25);
}

.status-low {
    color: #ff4b5c;
    font-weight: 600;
}

.status-mid {
    color: #ffb703;
    font-weight: 600;
}

.status-good {
    color: #06d6a0;
    font-weight: 600;
}

.live-strip {
    padding: 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 1rem;
    transition: 0.3s;
}

.live-strip:hover {
    background: rgba(0,255,255,0.08);
    box-shadow: 0 12px 25px rgba(0,255,255,0.25);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero-card">
<h1 style="font-size:3rem;">🏥 PreventAI</h1>
<h4>Clinical Preventive Intelligence System</h4>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SESSION
# =====================================================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "results" not in st.session_state:
    st.session_state.results = {}
if "med_list" not in st.session_state:
    st.session_state.med_list = []

# =====================================================
# LOAD MODEL
# =====================================================
# FIX: the original code did `pickle.load(open("model.pkl","rb"))` with a
# relative path and no error handling. If Streamlit is launched from a
# different working directory (very common in deployment), or if model.pkl
# is missing/corrupt, this throws an unhandled exception and the ENTIRE app
# fails to render -- which is why the dashboard appeared "not visible" even
# though the values were entered correctly. We now resolve the path
# relative to this script and fail gracefully with a clear message instead
# of crashing the whole page.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = None
model_load_error = None
try:
    model = load_model()
except Exception as e:
    model_load_error = str(e)

if model_load_error:
    st.error(
        "⚠️ Could not load the prediction model (model.pkl). "
        "The clinical risk dashboard cannot run until this is fixed.\n\n"
        f"Details: {model_load_error}\n\n"
        "Make sure `model.pkl` is placed in the same folder as this script "
        "and is not tracked/ignored by .gitignore if you're deploying from GitHub."
    )

# =====================================================
# LOCATION FUNCTIONS
# =====================================================
def get_city_coordinates(city):
    try:
        geolocator = Nominatim(user_agent="preventai")
        loc = geolocator.geocode(city, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return None, None

def find_nearby_healthcare(lat, lon):
    overpass_url = "https://overpass-api.de/api/interpreter"
    for radius in [7000, 15000, 30000]:
        query = f"""
        [out:json];
        (
          node["amenity"~"hospital|clinic"](around:{radius},{lat},{lon});
        );
        out;
        """
        try:
            response = requests.post(overpass_url, data={"data": query}, timeout=20)
            data = response.json()
        except Exception:
            continue

        facilities = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "Healthcare Facility")

            if any(x in name.lower() for x in ["dental", "eye", "physio", "veterinary"]):
                continue

            distance = round(geodesic((lat, lon), (el["lat"], el["lon"])).km, 2)

            facilities.append({
                "name": name,
                "lat": el["lat"],
                "lon": el["lon"],
                "distance": distance,
                "url": f"https://www.openstreetmap.org/node/{el['id']}"
            })

        if facilities:
            return sorted(facilities, key=lambda x: x["distance"])[:6]

    return []

# =====================================================
# MEAL / CALORIE PARSER
# =====================================================
# FIX: the original parser only matched patterns like "2 idli" (a digit
# immediately followed by a food word). "two idli", "idli" (no quantity),
# or "idli and dosa" all silently produced 0 calories. This version
# understands number words and defaults quantity to 1 when a food word
# appears with no explicit quantity.
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def parse_meal_calories(meal_input, calorie_db):
    words = meal_input.lower().replace(",", " ").split()
    total = 0
    items = []
    pending_qty = None

    for word in words:
        word = word.strip(".")
        if word in ("and", "with", "plus"):
            continue

        qty = None
        if word.isdigit():
            qty = int(word)
        elif word in NUMBER_WORDS:
            qty = NUMBER_WORDS[word]

        if qty is not None:
            pending_qty = qty
            continue

        # strip a trailing 's' for simple plurals (e.g. "eggs" -> "egg")
        food_key = word if word in calorie_db else word.rstrip("s")

        if food_key in calorie_db:
            used_qty = pending_qty if pending_qty is not None else 1
            total += used_qty * calorie_db[food_key]
            items.append((used_qty, food_key))
            pending_qty = None

    return total, items

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("Patient Information")

patient_name = st.sidebar.text_input("Patient Name", key="patient_name")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
report_date = datetime.now().strftime("%d %B %Y")

st.sidebar.markdown("---")
st.sidebar.header("Clinical Inputs")

glucose = st.sidebar.number_input("Glucose (mg/dL)", 0.0, 400.0, 100.0, key="glucose")
bp = st.sidebar.number_input("Blood Pressure (mmHg)", 0.0, 200.0, 80.0, key="bp")
insulin = st.sidebar.number_input("Insulin (µU/mL)", 0.0, 500.0, 80.0, key="insulin")
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0, key="bmi")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.0, 0.5, key="dpf")
age = st.sidebar.number_input("Age", 1, 100, 30, key="age")

physical_activity = st.sidebar.selectbox("Physical Activity", ["low", "medium", "high"], key="physical_activity")
smoking = st.sidebar.selectbox("Smoking", ["no", "yes"], key="smoking")
family_history = st.sidebar.selectbox("Family Heart History", ["no", "yes"], key="family_history")
city = st.sidebar.text_input("City", key="city")
confirm = st.sidebar.checkbox("Confirm data accuracy", key="confirm")

# =====================================================
# ANALYZE
# =====================================================
if st.sidebar.button("Generate Clinical Report", disabled=(model is None)):

    if not confirm:
        st.warning("Please confirm data before generating the report.")
    else:
        # FIX: the prediction logic ran with no error handling. If the
        # feature dataframe didn't match what the model expects, or the
        # model object was invalid, this raised an exception mid-script and
        # the dashboard never rendered. Now we catch that and show a clear
        # error instead of a blank page.
        try:
            features = pd.DataFrame(
                [[glucose, bp, insulin, bmi, dpf, age]],
                columns=["Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            )

            diabetes = model.predict_proba(features)[0][1] * 100

            heart = 0
            if age > 45: heart += 15
            if bmi > 27: heart += 20
            if bp > 130: heart += 25
            if physical_activity == "low": heart += 20
            if smoking == "yes": heart += 10
            if family_history == "yes": heart += 10

            heart = min(heart, 100)
            index = (0.5 * diabetes) + (0.4 * heart)

            st.session_state.results = {"diabetes": diabetes, "heart": heart, "index": index}
            st.session_state.analyzed = True
            st.success("Clinical report generated.")

        except Exception as e:
            st.session_state.analyzed = False
            st.error(f"Could not generate the report: {e}")

# =====================================================
# DASHBOARD
# =====================================================

tab1, tab2 = st.tabs(["📊 Risk Dashboard", "🧠 Live Monitoring"])

with tab1:

    if st.session_state.analyzed:
        r = st.session_state.results

        # Gauge + Metrics
        col1, col2 = st.columns([2, 1])

        with col1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=r["index"],
                title={'text': "PreventAI Health Index"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "#00ff99"},
                        {'range': [40, 70], 'color': "#ffcc00"},
                        {'range': [70, 100], 'color': "#ff0066"}
                    ],
                    'bar': {'color': "#00e5ff"}
                }
            ))
            gauge.update_layout(height=450)
            st.plotly_chart(gauge, use_container_width=True)

        with col2:
            st.markdown(
                f'<div class="metric-card"><h3>Diabetes Risk</h3><h2>{r["diabetes"]:.1f}%</h2></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="metric-card"><h3>Heart Risk</h3><h2>{r["heart"]:.1f}%</h2></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # =====================================================
        # PERSONALIZED RECOMMENDATIONS
        # =====================================================
        st.markdown("## 🧠 Personalized Preventive Plan")

        recommendations = []
        lab_tests = []
        follow_up = "6 months"

        if glucose > 140:
            recommendations.append("Elevated glucose detected. Reduce refined carbohydrates and schedule HbA1c test.")
            lab_tests.append("HbA1c Test")
            follow_up = "3 months"

        if bmi > 27:
            recommendations.append("BMI above recommended range. Target 5–7% weight reduction.")
            follow_up = "3 months"

        if bp > 130:
            recommendations.append("Elevated blood pressure noted. Reduce sodium intake and monitor weekly.")
            lab_tests.append("Lipid Profile")

        if smoking == "yes":
            recommendations.append("Smoking cessation strongly recommended.")

        if physical_activity == "low":
            recommendations.append("Increase physical activity to minimum 150 minutes/week.")

        if r["index"] > 70:
            recommendations.append("High composite risk detected. Immediate physician consultation advised.")
            follow_up = "1 month"

        if not recommendations:
            recommendations.append("Current parameters within acceptable range. Maintain healthy lifestyle.")

        for rec in recommendations:
            st.markdown(f'<div class="reco-card">✔ {rec}</div>', unsafe_allow_html=True)

        st.markdown(f"**Suggested Follow-Up:** {follow_up}")

        if lab_tests:
            st.markdown("**Suggested Lab Tests:**")
            for test in lab_tests:
                st.markdown(f"- {test}")

        # =====================================================
        # 5-YEAR RISK FORECASTING
        # =====================================================
        st.markdown("## 🔮 5-Year Risk Forecast")

        base_index_forecast = r["index"]
        years = np.arange(1, 6)

        no_change = [base_index_forecast for _ in range(5)]

        moderate_drop = 3 if base_index_forecast > 60 else 2
        active_drop = 6 if base_index_forecast > 60 else 4

        moderate = [max(base_index_forecast - (i * moderate_drop), 0) for i in range(5)]
        active = [max(base_index_forecast - (i * active_drop), 0) for i in range(5)]

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(
            x=years, y=no_change, mode="lines+markers",
            name="No Lifestyle Change", line=dict(width=4, color="#ff4b5c")
        ))
        forecast_fig.add_trace(go.Scatter(
            x=years, y=moderate, mode="lines+markers",
            name="Moderate Improvement", line=dict(width=4, color="#ffb703")
        ))
        forecast_fig.add_trace(go.Scatter(
            x=years, y=active, mode="lines+markers",
            name="Active Lifestyle Change", line=dict(width=4, color="#06d6a0"),
            fill="tozeroy", fillcolor="rgba(6,214,160,0.15)"
        ))
        forecast_fig.update_layout(
            height=420, xaxis_title="Years", yaxis_title="Projected PreventAI Index",
            yaxis=dict(range=[0, 100]), template="plotly_dark"
        )
        st.plotly_chart(forecast_fig, use_container_width=True)

        st.markdown("""
        <div class="small-box">
        🔴 No Change: Risk remains stable at current level.<br>
        🟡 Moderate Improvement: Gradual reduction through basic lifestyle adjustments.<br>
        🟢 Active Lifestyle Change: Significant long-term risk reduction trajectory.
        </div>
        """, unsafe_allow_html=True)

        # =====================================================
        # SMART CARE (nearby providers)
        # =====================================================
        # FIX: previously this whole block, INCLUDING the PDF report
        # download further below, was nested inside `if city:`. That meant
        # if a user left City blank, they never got a PDF download button
        # at all. Nearby-provider lookup still requires a city (it needs
        # coordinates), but the PDF report generation has been moved out
        # so it always appears regardless of whether a city was entered.
        st.markdown("## 🏥 Nearby Healthcare Providers")

        if city:
            lat, lon = get_city_coordinates(city)
            if lat and lon:
                facilities = find_nearby_healthcare(lat, lon)
                if facilities:
                    df = pd.DataFrame(facilities)
                    fig_map = px.scatter_mapbox(
                        df, lat="lat", lon="lon", hover_name="name",
                        hover_data=["distance"], zoom=11,
                        color="distance", color_continuous_scale="Turbo"
                    )
                    fig_map.update_layout(mapbox_style="open-street-map", height=450)
                    st.plotly_chart(fig_map, use_container_width=True)

                    for f in facilities:
                        st.markdown(f"""
            <div class="doctor-card">
            <b>{f['name']}</b><br>
            Distance: {f['distance']} km<br>
            <a href="{f['url']}" target="_blank">View Location</a>
            </div>
            """, unsafe_allow_html=True)
                else:
                    st.warning("No facilities found within 30km.")
            else:
                st.error("City not found. Check the spelling and try again.")
        else:
            st.info("Enter a city in the sidebar to see nearby healthcare providers.")

        # =====================================================
        # PROFESSIONAL PDF (now always available once analyzed)
        # =====================================================
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("<b>PreventAI Clinical Health Report</b>", styles["Title"]))
            elements.append(Spacer(1, 0.3 * inch))

            elements.append(Paragraph(f"Patient Name: {patient_name or 'N/A'}", styles["Normal"]))
            elements.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
            elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
            elements.append(Paragraph(f"Report Date: {report_date}", styles["Normal"]))
            elements.append(Spacer(1, 0.3 * inch))

            elements.append(Paragraph(f"PreventAI Index: {r['index']:.2f}", styles["Heading2"]))
            elements.append(Paragraph(f"Diabetes Risk: {r['diabetes']:.2f}%", styles["Normal"]))
            elements.append(Paragraph(f"Heart Risk: {r['heart']:.2f}%", styles["Normal"]))

            elements.append(Spacer(1, 0.3 * inch))
            elements.append(Paragraph("Detailed Preventive Strategy:", styles["Heading2"]))
            elements.append(ListFlowable([ListItem(Paragraph(rec, styles["Normal"])) for rec in recommendations]))

            elements.append(Spacer(1, 0.3 * inch))
            elements.append(Paragraph(f"Suggested Follow-Up: {follow_up}", styles["Normal"]))

            if lab_tests:
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph("Suggested Lab Tests:", styles["Heading3"]))
                elements.append(ListFlowable([ListItem(Paragraph(test, styles["Normal"])) for test in lab_tests]))

            elements.append(Spacer(1, 0.5 * inch))
            elements.append(Paragraph("Physician Signature: ____________________", styles["Normal"]))

            doc.build(elements)
            buffer.seek(0)

            st.download_button(
                "Download Clinical PDF Report",
                buffer,
                "PreventAI_Clinical_Report.pdf",
                "application/pdf"
            )
        except Exception as e:
            st.error(f"Could not generate the PDF report: {e}")

    else:
        st.info("Enter patient clinical values in the sidebar and click Generate Clinical Report to view risk analysis.")


# =====================================================
# TAB 2 — LIVE MONITORING
# =====================================================
with tab2:

    st.markdown("## 🧠 Live Preventive Monitoring")

    base_index = st.session_state.results.get("index", 0) if "results" in st.session_state else 0

    # =====================================================
    # SMART MEAL ANALYZER
    # =====================================================
    st.markdown("### 🍽 Smart Meal Analyzer")
    st.markdown('<div class="doctor-card">', unsafe_allow_html=True)

    calorie_db = {
        "idli": 60,
        "dosa": 120,
        "rice": 200,
        "chapati": 110,
        "egg": 70,
        "banana": 100,
        "milk": 150,
        "bread": 80,
        "chicken": 250
    }

    meal_input = st.text_input("Enter meal (e.g., 2 idli and 1 egg, or just 'banana')", key="meal_input")
    total_calories = 0

    if meal_input:
        total_calories, parsed_items = parse_meal_calories(meal_input, calorie_db)
        if parsed_items:
            item_text = ", ".join(f"{q} {f}" for q, f in parsed_items)
            st.success(f"Detected: {item_text} — Estimated Calories: {total_calories} kcal")
        else:
            st.warning("Couldn't recognize any known food items in that entry. Try items like: " + ", ".join(calorie_db.keys()))

    recommended_calories = 2200 if gender == "Male" else 1800

    if total_calories > 0:
        st.write(f"Recommended Daily Calories: {recommended_calories}")
        if total_calories > recommended_calories:
            st.warning("High calorie intake detected.")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # ACTIVITY + SLEEP
    # =====================================================
    col1, col2 = st.columns(2)

    step_penalty = 0
    sleep_penalty = 0

    with col1:
        st.markdown("### 🚶 Activity Tracker")
        st.markdown('<div class="doctor-card">', unsafe_allow_html=True)

        steps = st.number_input("Today's Steps", 0, 50000, 4000, key="steps")

        if steps < 5000:
            st.error("Low Activity Level")
            step_penalty = 5
        elif steps < 8000:
            st.warning("Moderate Activity")
            step_penalty = 2
        else:
            st.success("Active Lifestyle")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 😴 Sleep Monitor")
        st.markdown('<div class="doctor-card">', unsafe_allow_html=True)

        sleep_hours = st.slider("Hours Slept", 0, 12, 6, key="sleep")

        if sleep_hours < 6:
            st.error("Poor Sleep Quality")
            sleep_penalty = 4
        elif sleep_hours <= 8:
            st.success("Optimal Sleep")
        else:
            st.warning("Excess Sleep Duration")

        st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # MEDICINE TRACKER
    # =====================================================
    st.markdown("### 💊 Medicine Scheduler")
    st.markdown('<div class="doctor-card">', unsafe_allow_html=True)

    med_name = st.text_input("Medicine Name", key="med_name")
    med_time = st.time_input("Select Time", key="med_time")
    med_type = st.selectbox("Before/After Food", ["Before Food", "After Food"], key="med_type")

    if st.button("Add Medicine", key="add_med"):
        if med_name.strip() != "":
            st.session_state.med_list.append({
                "name": med_name,
                "time": med_time,
                "type": med_type
            })
            st.success(f"Added {med_name} to schedule.")
        else:
            st.warning("Enter a medicine name first.")

    if st.session_state.med_list:
        st.markdown("#### Scheduled Medicines")
        for med in st.session_state.med_list:
            st.info(f"💊 {med['name']} • {med['time']} • {med['type']}")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # LIVE LIFESTYLE IMPACT
    # =====================================================
    st.markdown("## 🔥 Live Lifestyle Impact on Risk")

    lifestyle_penalty = step_penalty + sleep_penalty
    adjusted_index = base_index + lifestyle_penalty

    if total_calories > recommended_calories:
        adjusted_index += 3

    adjusted_index = min(adjusted_index, 100)

    colA, colB = st.columns(2)

    with colA:
        st.metric("Base PreventAI Index", f"{base_index:.2f}")

    with colB:
        st.metric(
            "Adjusted Live Index",
            f"{adjusted_index:.2f}",
            delta=f"{adjusted_index - base_index:.2f}"
        )

    if adjusted_index > base_index:
        st.warning("Current lifestyle behavior is increasing projected risk.")
    else:
        st.success("Current lifestyle supports long-term risk reduction.")
