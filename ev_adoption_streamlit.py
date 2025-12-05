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
st.header("EV Adoption: Infrastructure, Income, and Incentives")

subset = df[df["year"].isin([2022, 2023])].copy()

# Define clusters manually
high_adoption = ["California", "Washington", "Oregon"]
medium_adoption = ["Florida", "Virginia", "Colorado"]
low_adoption = ["Mississippi", "West Virginia"]

def assign_cluster(state):
    if state in high_adoption:
        return 1
    elif state in medium_adoption:
        return 2
    elif state in low_adoption:
        return 3
    else:
        return None

subset["Cluster"] = subset["state"].apply(assign_cluster)
subset = subset[subset["Cluster"].notna()].copy()  # Only clustered states

cluster_colors = {1: "green", 2: "yellow", 3: "red"}  # Cluster color map

for year in [2022, 2023]:
    year_data = subset[subset["year"] == year].copy()
    
    if year_data.empty:
        st.warning(f"No clustered states for {year}")
        continue

    st.subheader(f"EV Adoption by Cluster — {year}")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot of EV Share vs Charging Stations
    sns.scatterplot(
        data=year_data,
        x="Stations",
        y="EV Share (%)",
        hue="Cluster",
        palette=cluster_colors,
        s=200,
        ax=ax,
        legend=False  # We'll add a custom legend below
    )

    ax.set_title(f"EV Share vs Charging Stations — {year}")
    ax.set_xlabel("Charging Stations")
    ax.set_ylabel("EV Share (%)")

    # Custom cluster legend
    import matplotlib.patches as mpatches
    cluster_patches = [
        mpatches.Patch(color="green", label="High-Adoption"),
        mpatches.Patch(color="yellow", label="Medium-Adoption"),
        mpatches.Patch(color="red", label="Low-Adoption"),
    ]
    ax.legend(handles=cluster_patches, title="Cluster", loc="upper left")

    st.pyplot(fig)

    # List states at the bottom grouped by cluster
    st.write("**States included in the plot:**")
    st.write(f"**High-Adoption:** {', '.join(high_adoption)}")
    st.write(f"**Medium-Adoption:** {', '.join(medium_adoption)}")
    st.write(f"**Low-Adoption:** {', '.join(low_adoption)}")


st.markdown("---")
st.header("Decision Tree Classification: EV Growth 2022 → 2023")

# Define clustered states
high_adoption = ["California", "Washington", "Oregon", "New York", "Massachusetts", "New Jersey"]
medium_adoption = ["Florida", "Virginia", "Colorado", "Michigan", "Illinois", "Texas"]
low_adoption = ["Mississippi", "West Virginia", "Alabama", "Arkansas", "Louisiana", "Kentucky"]

cluster_states = high_adoption + medium_adoption + low_adoption

# Subset for clustered states and 2022-2023
subset = df[df["state"].isin(cluster_states) & df["year"].isin([2022, 2023])]

# Pivot data to get 2022 and 2023 side by side
pivot = subset.pivot(index="state", columns="year",
                     values=["EV Registrations", "EV Share (%)", "Stations", "Per_Cap_Income", "Incentives", "gasoline_price_per_gallon"])
pivot.columns = ["_".join([col[0], str(col[1])]) for col in pivot.columns]
pivot = pivot.fillna(0)
pivot = pivot.reset_index()  # Keep state names

# Calculate growth 2023 - 2022
pivot["Growth"] = pivot["EV Registrations_2023"] - pivot["EV Registrations_2022"]

# Growth labels
q1 = pivot["Growth"].quantile(0.33)
q2 = pivot["Growth"].quantile(0.66)
pivot["Growth_Label"] = pd.cut(pivot["Growth"], bins=[-float("inf"), q1, q2, float("inf")],
                               labels=["Low", "Medium", "High"])

# Features
features = ["Stations_2023", "Per_Cap_Income_2023", "Incentives_2023", "EV Share (%)_2023", "gasoline_price_per_gallon_2023"]
X = pivot[features]
y = pivot["Growth_Label"]

# Train Decision Tree safely
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
if len(pivot) < 3:
    st.warning("Not enough data to split; training on full dataset.")
    clf.fit(X, y)
    pred = clf.predict(X)
    st.write(f"Decision Tree Accuracy (full data): **{accuracy_score(y, pred):.2f}**")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    st.write(f"Decision Tree Accuracy: **{accuracy_score(y_test, pred):.2f}**")

# Feature importance plot
importance = pd.Series(clf.feature_importances_, index=features)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=importance.values, y=importance.index, ax=ax)
ax.set_title("Feature Importance for EV Growth Prediction (2022→2023)")
st.pyplot(fig)

# Table of states and growth
display_table = pivot[["state", "Growth", "Growth_Label"]].copy()
display_table = display_table.rename(columns={
    "state": "State",
    "Growth": "EV Registration Growth",
    "Growth_Label": "Growth Category"
})
st.write("**States and Growth Labels:**")
st.dataframe(display_table.style.set_properties(**{'text-align': 'center'}))
