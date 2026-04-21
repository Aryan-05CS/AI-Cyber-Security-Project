import time
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

def simulate_stream():
    df = pd.read_csv("data/cicids.csv").sample(20)

    for i, row in df.iterrows():
        sample = row.drop("Label").values.reshape(1, -1)

        prediction = model.predict(sample)[0]

        print("\n🔍 Incoming Traffic...")

        if prediction != 0:
            print(f"🚨 ALERT: Attack detected! Type: {prediction}")
        else:
            print("✅ Normal traffic")

        time.sleep(2)


if __name__ == "__main__":
    simulate_stream()