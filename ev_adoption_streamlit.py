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
high_adoption = ["California", "Washington", "Oregon", "New York", "Massachusetts", "New Jersey"]
medium_adoption = ["Florida", "Virginia", "Colorado", "Michigan", "Illinois", "Texas"]
low_adoption = ["Mississippi", "West Virginia", "Alabama", "Arkansas", "Louisiana", "Kentucky"]

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
st.header("Decision Tree Classification: EV Growth 2022 vs 2023")

# --- Define clustered states ---
high_adoption = ["California", "Washington", "Oregon", "New York", "Massachusetts", "New Jersey"]
medium_adoption = ["Florida", "Virginia", "Colorado", "Michigan", "Illinois", "Texas"]
low_adoption = ["Mississippi", "West Virginia", "Alabama", "Arkansas", "Louisiana", "Kentucky"]
cluster_states = high_adoption + medium_adoption + low_adoption

# --- Function to create Decision Tree per year ---
def decision_tree_growth(prev_year, curr_year, cluster_states):
    # Ensure all clustered states are included
    all_states = pd.DataFrame({"state": cluster_states})
    
    prev_data = df[(df["year"] == prev_year) & df["state"].isin(cluster_states)].merge(
        all_states, on="state", how="right"
    )
    curr_data = df[(df["year"] == curr_year) & df["state"].isin(cluster_states)].merge(
        all_states, on="state", how="right"
    )
    
    # Fill missing numeric values with mean; if mean is NaN (all missing), fill with 0
    numeric_cols = ["EV Registrations", "EV Share (%)", "Stations", 
                    "Per_Cap_Income", "Incentives", "gasoline_price_per_gallon"]
    for col in numeric_cols:
        prev_mean = prev_data[col].mean()
        prev_data[col] = prev_data[col].fillna(prev_mean if not pd.isna(prev_mean) else 0)
        
        curr_mean = curr_data[col].mean()
        curr_data[col] = curr_data[col].fillna(curr_mean if not pd.isna(curr_mean) else 0)

    # Calculate growth
    growth_df = curr_data[["state", "EV Registrations", "EV Share (%)", "Stations",
                           "Per_Cap_Income", "Incentives", "gasoline_price_per_gallon"]].copy()
    growth_df["Growth"] = growth_df["EV Registrations"] - prev_data["EV Registrations"]
    
    # Growth labels
    q1 = growth_df["Growth"].quantile(0.33)
    q2 = growth_df["Growth"].quantile(0.66)
    growth_df["Growth_Label"] = pd.cut(growth_df["Growth"], bins=[-float("inf"), q1, q2, float("inf")],
                                       labels=["Low", "Medium", "High"])
    
    # Features for model (current year)
    all_features = ["Stations", "Per_Cap_Income", "Incentives", "EV Share (%)", "gasoline_price_per_gallon"]
    
    # Only keep features with variance > 0
    features = [f for f in all_features if growth_df[f].var() > 0]
    
    if not features:
        st.warning(f"No valid features with variance for {curr_year}. Skipping Decision Tree.")
        return
    
    X = growth_df[features]
    y = growth_df["Growth_Label"]
    
    # Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    
    # Feature importance plot
    importance = pd.Series(clf.feature_importances_, index=features)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=importance.values, y=importance.index, ax=ax)
    ax.set_title(f"Feature Importance for EV Growth Prediction ({curr_year})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
    
    # Display table with titles
    display_table = growth_df[["state", "Growth", "Growth_Label"]].rename(columns={
        "state": "State",
        "Growth": "EV Registration Growth",
        "Growth_Label": "Growth Category"
    })
    st.write(f"**States and Growth Labels ({curr_year}):**")
    st.dataframe(display_table.style.set_properties(**{'text-align': 'center'}))


decision_tree_growth(prev_year=2021, curr_year=2022, cluster_states=cluster_states)
st.markdown("---")
decision_tree_growth(prev_year=2022, curr_year=2023, cluster_states=cluster_states)

