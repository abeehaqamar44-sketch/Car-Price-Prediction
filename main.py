import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "data/OLX_Car_Data_CSV.csv",
    encoding="latin1"
)

print("Original Dataset Shape:", data.shape)


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

print("Duplicate Rows:", data.duplicated().sum())

data = data.drop_duplicates()

print("Shape after removing duplicates:", data.shape)


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

categorical_columns = [
    "Brand",
    "Condition",
    "Fuel",
    "Model",
    "Registered City",
    "Transaction Type"
]

# Fill categorical missing values
for column in categorical_columns:
    data[column] = data[column].fillna("Unknown")

# Fill numerical missing values
data["KMs Driven"] = data["KMs Driven"].fillna(
    data["KMs Driven"].median()
)

data["Year"] = data["Year"].fillna(
    data["Year"].median()
)


# ============================================================
# 4. REMOVE EXTREME PRICE VALUES
# ============================================================

Q1 = data["Price"].quantile(0.01)
Q99 = data["Price"].quantile(0.99)

data = data[
    (data["Price"] >= Q1) &
    (data["Price"] <= Q99)
]

print("Shape after removing extreme prices:", data.shape)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop("Price", axis=1)
y = data["Price"]


# ============================================================
# 6. CONVERT CATEGORICAL DATA INTO NUMBERS
# ============================================================

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)


# ============================================================
# 7. SAVE FEATURE COLUMNS
# ============================================================

feature_columns = X.columns.tolist()

joblib.dump(
    feature_columns,
    "feature_columns.pkl"
)

print("Feature columns saved.")


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 9. CREATE RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 12. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n========================================")
print("        MODEL RESULTS")
print("========================================")

print(f"MAE:       {mae:,.2f} PKR")
print(f"RMSE:      {rmse:,.2f} PKR")
print(f"R² Score:  {r2:.4f}")


# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "car_price_model.pkl"
)

print("\nModel saved as: car_price_model.pkl")


# ============================================================
# 14. USER CAR PRICE PREDICTION
# ============================================================

print("\n========================================")
print("       CAR PRICE PREDICTION")
print("========================================")

print("\nEnter the following car information:\n")


# User inputs

brand = input("Brand: ")

condition = input("Condition (Used/New): ")

fuel = input("Fuel (Petrol/Diesel/CNG/Hybrid): ")

kms = float(input("KMs Driven: "))

model_name = input("Model: ")

registered_city = input("Registered City: ")

transaction_type = input("Transaction Type (Cash/Installment): ")

year = float(input("Year: "))


# ============================================================
# 15. CREATE INPUT DATAFRAME
# ============================================================

user_data = pd.DataFrame({
    "Brand": [brand],
    "Condition": [condition],
    "Fuel": [fuel],
    "KMs Driven": [kms],
    "Model": [model_name],
    "Registered City": [registered_city],
    "Transaction Type": [transaction_type],
    "Year": [year]
})


# ============================================================
# 16. ENCODE USER INPUT
# ============================================================

user_data = pd.get_dummies(
    user_data,
    columns=categorical_columns,
    drop_first=True
)


# ============================================================
# 17. MATCH TRAINING FEATURES
# ============================================================

user_data = user_data.reindex(
    columns=feature_columns,
    fill_value=False
)


# ============================================================
# 18. PREDICT PRICE
# ============================================================

predicted_price = model.predict(user_data)[0]


# ============================================================
# 19. DISPLAY RESULT
# ============================================================

print("\n========================================")
print("        PREDICTED CAR PRICE")
print("========================================")

print(
    f"Estimated Price: Rs. {predicted_price:,.0f}"
)

print("========================================")