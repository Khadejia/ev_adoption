import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="U.S. Electric Vehicle Overview", layout="centered")

CSV_PATH = "EV_Data.csv"
df = pd.read_csv(CSV_PATH)

st.title("U.S. EV Adoption Explorer")
st.markdown("Use the options on the left to view detailed EV patterns.")

required_cols = ["state", "year", "EV Registrations", "EV Share (%)", "Stations",
                 "gasoline_price_per_gallon", "Per_Cap_Income", "Incentives"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"ERROR: CSV is missing columns: {missing}")
    st.stop()

st.sidebar.header("Select Filters")
state_choice = st.sidebar.selectbox(
    "Choose State", sorted(df["state"].dropna().unique())
)
year_choice = st.sidebar.selectbox(
    "Choose Year", sorted(df["year"].dropna().unique().astype(int), reverse=True)
)

selected = df[(df["state"] == state_choice) & (df["year"] == int(year_choice))]

if selected.empty:
    st.warning(f"No data for {state_choice} in {year_choice}.")
else:
    row = selected.iloc[0]
    st.subheader(f"{state_choice} — {year_choice}")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("EV Registrations", f"{row['EV Registrations']:,}")
        st.metric("EV Share (%)", f"{row['EV Share (%)']}")
        st.metric("Charging Stations", f"{row['Stations']:,}")

    with c2:
        st.metric("Gasoline Price", f"${row['gasoline_price_per_gallon']}")
        st.metric("Per Cap Income", f"${row['Per_Cap_Income']:,}")
        st.metric("Incentives", row["Incentives"])

st.markdown("### EV Trend for Selected State")
trend = df[df["state"] == state_choice].sort_values("year")

if not trend.empty:
    trend2 = trend[["year", "EV Share (%)", "Stations"]].set_index("year")
    st.line_chart(trend2)
else:
    st.info("No historical trend available.")


st.markdown("---")
st.header("K-Means Clustering (2022–2023)")

subset = df[df["year"].isin([2022, 2023])].copy()

# Convert categorical to numeric
subset["Incentives_Num"] = subset["Incentives"].map({"Yes": 1, "No": 0})

# Select numeric features
features = ["EV Share (%)", "Stations", "Per_Cap_Income", "Incentives_Num"]
X = subset[features].apply(pd.to_numeric, errors="coerce").dropna()

# Use 1 cluster if dataset is too small
n_clusters = min(2, len(X))

# Fit KMeans safely
if n_clusters >= 1:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    subset.loc[X.index, "Cluster"] = kmeans.fit_predict(X)
else:
    subset["Cluster"] = 0  # fallback for empty dataset

# Plot
fig1, ax1 = plt.subplots(figsize=(7, 5))
sns.scatterplot(
    data=subset,
    x="Stations",
    y="EV Share (%)",
    hue="Cluster",
    palette="deep",
    ax=ax1
)
ax1.set_title("Clusters: EV Share vs. Charging Stations")
st.pyplot(fig1)


st.markdown("---")
st.header("Decision Tree Classification (Growth Groups)")

subset["Growth"] = subset.groupby("state")["EV Registrations"].diff().fillna(0)

# Calculate quantiles for 2 labels
q = subset["Growth"].quantile(0.5)
bins = sorted([-float("inf"), q, float("inf")])

subset["Growth_Label"] = pd.cut(
    subset["Growth"],
    bins=bins,
    labels=["Low", "High"],
    include_lowest=True
)

# Convert categorical 'Incentives' to numeric
subset["Incentives_Num"] = subset["Incentives"].map({"Yes": 1, "No": 0})

dt_data = subset.dropna(subset=["EV Share (%)", "Stations", "Per_Cap_Income", "Incentives_Num", "Growth_Label"])
X = dt_data[["EV Share (%)", "Stations", "Per_Cap_Income", "Incentives_Num"]]
y = dt_data["Growth_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier(max_depth=4, random_state=1)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

st.write(f"Decision Tree Accuracy: **{accuracy_score(y_test, pred):.2f}**")

importance = pd.Series(clf.feature_importances_, index=X.columns)

fig2, ax2 = plt.subplots(figsize=(7, 4))
sns.barplot(x=importance.values, y=importance.index, ax=ax2)
ax2.set_title("Feature Importance")
st.pyplot(fig2)


