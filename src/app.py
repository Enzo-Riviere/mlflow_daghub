from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.datasets import load_digits

app = FastAPI()

# Charger le modèle avec joblib
model = joblib.load("rf_model.pkl")

digits = load_digits()
# Les noms des features sont simplement "pixel_0", "pixel_1", ..., car load_digits n'a pas feature_names
feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
df = pd.DataFrame(digits.data, columns=feature_names)
df['target'] = digits.target

@app.get("/predict")
def predict():
    random_line = df.sample(n=1)

    x = random_line.drop(columns=["target"]).to_dict(orient="records")[0]
    X_input = random_line.drop(columns=["target"]).values
    y_pred = model.predict(X_input)

    return {
        "input": x,
        "prediction": int(y_pred[0]),
        "actual": int(random_line["target"].iloc[0])
    }
