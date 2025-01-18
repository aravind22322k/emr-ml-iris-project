import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load processed data
processed_data_path = "/tmp/processed_iris"
df = pd.read_parquet(processed_data_path)

# Split data into features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model locally
model_path = "/tmp/model.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")
