import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import statsmodels.formula.api as smf

st.set_page_config(layout="wide")
st.title("Ship Price Dashboard")

# ---- Load Data ----
FILE_PATH = "RegressionData.csv"
df = pd.read_csv(FILE_PATH)

required = ["Price", "Age", "DWT", "Capesize"]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

for c in required:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=required)

# ---- Sidebar Controls ----
st.sidebar.header("Input Variables")

age = st.sidebar.slider(
    "Age",
    int(df.Age.min()),
    int(df.Age.max()),
    int(df.Age.mean())
)

dwt = st.sidebar.slider(
    "DWT",
    float(df.DWT.min()),
    float(df.DWT.max()),
    float(df.DWT.mean())
)

cape = st.sidebar.selectbox(
    "Capesize",
    [0, 1]
)

# ---- Regression Model ----
model = smf.ols("Price ~ Age + DWT + Capesize", data=df).fit()

new_data = pd.DataFrame({
    "Age": [age],
    "DWT": [dwt],
    "Capesize": [cape]
})

prediction = model.predict(new_data)[0]

st.metric("Predicted Ship Price", f"${prediction:,.2f}")

# ---- Scatter Plot ----
chart = alt.Chart(df).mark_circle(size=60).encode(
    x="Age",
    y="Price",
    tooltip=["Age", "Price"]
).interactive()

st.subheader("Ship Prices vs Age")
st.altair_chart(chart, use_container_width=True)

# ---- Regression Summary ----
st.subheader("Regression Summary")

st.text(model.summary())
