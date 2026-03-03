import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/bengaluru_ola.csv")

# Create binary target
df["is_cancelled"] = df["Booking Status"].apply(
    lambda x: 0 if x == "Success" else 1
)

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Extract useful time features
df["booking_day"] = df["Date"].dt.dayofweek   # 0 = Monday
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
df["booking_hour"] = df["Time"].dt.hour

# Select ONLY booking-time features
features = [
    "Vehicle Type",
    "Pickup Location",
    "Drop Location",
    "Payment Method",
    "booking_day",
    "booking_hour"
]

df_model = df[features + ["is_cancelled"]]

# Fill missing payment method
df_model["Payment Method"] = df_model["Payment Method"].fillna("Unknown")

# Separate X and y
X = df_model.drop("is_cancelled", axis=1)
y = df_model["is_cancelled"]

# Categorical columns
categorical_cols = [
    "Vehicle Type",
    "Pickup Location",
    "Drop Location",
    "Payment Method"
]

# Encoder
encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encode
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

print("Training shape:", X_train_encoded.shape)
print("Testing shape:", X_test_encoded.shape)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_encoded, y_train)

# Predict
y_pred = model.predict(X_test_encoded)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nPickup Location vs Cancellation:\n")
print(pd.crosstab(df["Pickup Location"], df["is_cancelled"]))

print("\nDrop Location vs Cancellation:\n")
print(pd.crosstab(df["Drop Location"], df["is_cancelled"]))
print("\nCheck cancellation by hour:\n")
print(pd.crosstab(df["booking_hour"], df["is_cancelled"]))

import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save model and encoder
joblib.dump(model, "models/cancel_model.pkl")
joblib.dump(encoder, "models/encoder.pkl")

print("\nModel and encoder saved successfully!")