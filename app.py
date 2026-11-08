import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load/Generate customer transaction dataset with pandas
# (Creating a robust simulated RFM customer dataset for clustering)
print("Loading customer transaction data...")
np.random.seed(42)
n_customers = 300

data = {
    "CustomerID": range(1001, 1001 + n_customers),
    "Recency": np.random.randint(1, 365, size=n_customers),  # Days since last purchase
    "Frequency": np.random.randint(
        1, 25, size=n_customers
    ),  # Number of purchases
    "Monetary": np.random.uniform(
        50.0, 5000.0, size=n_customers
    ),  # Total spend amount
}
df = pd.DataFrame(data)

# Step 2: Select numerical behavior metrics (Frequency, Monetary, Recency)
X = df[["Recency", "Frequency", "Monetary"]]

# Step 3: Standardize features with sklearn.preprocessing.StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply KMeans with k=4 (within the 3-5 range) and assign cluster labels
print("Applying KMeans clustering (k=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Step 5: Create a summary report of each segment's characteristics
print("\n" + "=" * 50)
print("CUSTOMER SEGMENT SUMMARY REPORT")
print("=" * 50)
summary_report = (
    df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
    .mean()
    .reset_index()
)
summary_report["Customer_Count"] = df["Cluster"].value_counts().sort_index()
print(summary_report.to_string(index=False))
print("=" * 50)

# Optional: Visualize clusters (Monetary vs Frequency)
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Frequency"],
    df["Monetary"],
    c=df["Cluster"],
    cmap="viridis",
    edgecolors="k",
    alpha=0.7,
)
plt.title("Customer Segments (Frequency vs Monetary)")
plt.xlabel("Purchase Frequency")
plt.ylabel("Monetary Spend ($)")
plt.grid(True)
plt.show()