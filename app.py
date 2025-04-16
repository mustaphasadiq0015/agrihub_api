from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Render!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
            
# Load model and encoders
model = xgb.XGBRegressor()
model.load_model("commodity_price_model.json")
label_encoders = joblib.load("label_encoders.pkl")

# Define features
features = ["admin1", "admin2", "market", "category", "commodity",
            "unit", "pricetype", "currency", "year", "month", "day"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    # Encode categorical variables
    for col in label_encoders:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = input_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    prediction = model.predict(input_df[features])[0]
    return jsonify({'predicted_price': round(prediction, 2)})
