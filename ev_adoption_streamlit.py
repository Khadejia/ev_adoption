import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="U.S. Electric Vehicle Overview", layout="centered")

CSV_PATH = "EV_Data.csv"
df = pd.read_csv(CSV_PATH)

st.title("U.S. Electric Vehicle Overview")
st.markdown(
    "Use the options on the left to explore EV adoption details by state and year.")


required_cols = ["state", "year", "EV Registrations", "EV Share (%)", "Stations",
                 "gasoline_price_per_gallon", "Per_Cap_Income", "Incentives"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()


st.sidebar.header("Options")
state_choice = st.sidebar.selectbox(
    "Choose State", sorted(df["state"].dropna().unique()))
year_choice = st.sidebar.selectbox("Choose Year", sorted(
    df["year"].dropna().unique().astype(int).tolist(), reverse=True))


selected = df[(df["state"] == state_choice) & (df["year"] == int(year_choice))]

if selected.empty:
    st.warning(
        f"No data for {state_choice} in {year_choice}. Try a different selection.")
else:
    row = selected.iloc[0]

    st.subheader(f"{state_choice} — {year_choice}")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("EV Registrations", f"{row['EV Registrations']:,}")
        st.metric("EV Share (%)", f"{row['EV Share (%)']}")
        st.metric("Charging Stations", f"{row['Stations']:,}")

    with c2:
        st.metric("Gasoline Price ($/gal)", row["gasoline_price_per_gallon"])
        st.metric("Per Cap Income", f"${row['Per_Cap_Income']:,}")
        st.metric("Incentives", row["Incentives"])

    st.markdown("### Adoption History for Selected State")
    trend = df[df["state"] == state_choice].sort_values("year")

    if not trend.empty:
        trend2 = trend[["year", "EV Share (%)", "Stations"]].set_index("year")
        st.line_chart(trend2)
    else:
        st.info("No trend data available for this state.")
