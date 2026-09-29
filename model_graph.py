import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


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
# 5. SEPARATE FEATURES AND TARGET
# ============================================

X = data.drop("Price", axis=1)
y = data["Price"]


# ============================================
# 6. CONVERT CATEGORICAL DATA
# ============================================

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# ============================================
# 7. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================
# 8. CREATE MODEL
# ============================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


# ============================================
# 9. TRAIN MODEL
# ============================================

print("Training model...")

model.fit(X_train, y_train)

print("Model trained successfully!")


# ============================================
# 10. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)


# ============================================
# 11. ACTUAL VS PREDICTED GRAPH
# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price (PKR)")
plt.ylabel("Predicted Price (PKR)")

plt.tight_layout()

plt.savefig(
    "graphs/actual_vs_predicted.png"
)

plt.close()


# ============================================
# 12. FINISHED
# ============================================

print("\n====================================")
print("ACTUAL VS PREDICTED GRAPH CREATED!")
print("====================================")