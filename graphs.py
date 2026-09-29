import pandas as pd
import matplotlib.pyplot as plt


# ============================================
# 1. LOAD DATASET
# ============================================

data = pd.read_csv(
    "data/OLX_Car_Data_CSV.csv",
    encoding="latin1"
)

print("Dataset loaded successfully!")


# ============================================
# 2. REMOVE DUPLICATES
# ============================================

data = data.drop_duplicates()


# ============================================
# 3. HANDLE MISSING VALUES
# ============================================

categorical_columns = [
    "Brand",
    "Condition",
    "Fuel",
    "Model",
    "Registered City",
    "Transaction Type"
]

for column in categorical_columns:
    data[column] = data[column].fillna("Unknown")

data["KMs Driven"] = data["KMs Driven"].fillna(
    data["KMs Driven"].median()
)

data["Year"] = data["Year"].fillna(
    data["Year"].median()
)


# ============================================
# 4. REMOVE EXTREME PRICE VALUES
# ============================================

Q1 = data["Price"].quantile(0.01)
Q99 = data["Price"].quantile(0.99)

data = data[
    (data["Price"] >= Q1) &
    (data["Price"] <= Q99)
]


# ============================================
# GRAPH 1: PRICE DISTRIBUTION
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(data["Price"], bins=50)

plt.title("Car Price Distribution")
plt.xlabel("Price (PKR)")
plt.ylabel("Number of Cars")

plt.tight_layout()
plt.savefig("graphs/price_distribution.png")
plt.close()

print("Graph 1 saved!")


# ============================================
# GRAPH 2: PRICE VS YEAR
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Year"],
    data["Price"],
    alpha=0.4
)

plt.title("Car Price vs Year")
plt.xlabel("Year")
plt.ylabel("Price (PKR)")

plt.tight_layout()
plt.savefig("graphs/price_vs_year.png")
plt.close()

print("Graph 2 saved!")


# ============================================
# GRAPH 3: PRICE VS KMs DRIVEN
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["KMs Driven"],
    data["Price"],
    alpha=0.4
)

plt.title("Car Price vs KMs Driven")
plt.xlabel("KMs Driven")
plt.ylabel("Price (PKR)")

plt.tight_layout()
plt.savefig("graphs/price_vs_kms.png")
plt.close()

print("Graph 3 saved!")


# ============================================
# FINISHED
# ============================================

print("\n====================================")
print("ALL 3 GRAPHS CREATED SUCCESSFULLY!")
print("====================================")