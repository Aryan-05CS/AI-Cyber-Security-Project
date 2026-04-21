from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = "secret123"

# 🔥 Load ML components
model = joblib.load("models/model.pkl")
selector = joblib.load("models/selector.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
columns = joblib.load("models/columns.pkl")

# Load dataset
df = pd.read_csv("data/cicids.csv")

# Stats
attack_count = 0
normal_count = 0


# 🔐 LOGIN
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == "admin" and request.form['password'] == "1234":
            session['logged_in'] = True
            return redirect(url_for('home'))
    return render_template('login.html')


# 🏠 HOME
@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template(
        'index.html',
        attack_count=attack_count,
        normal_count=normal_count
    )


# 🔍 PREDICT BUTTON
@app.route('/predict')
def predict():
    global attack_count, normal_count

    sample = df.sample(1)

    sample.columns = sample.columns.str.strip()
    sample.rename(columns={"Label": "label"}, inplace=True)
    sample = sample.drop("label", axis=1, errors='ignore')

    sample = sample[columns]
    sample = selector.transform(sample)

    pred = model.predict(sample)[0]
    label = label_encoder.inverse_transform([pred])[0]

    if label != "BENIGN":
        result = f"🚨 {label} attack detected!"
        attack_count += 1
    else:
        result = "✅ Normal Traffic"
        normal_count += 1

    return render_template(
        'index.html',
        prediction_text=result,
        attack_count=attack_count,
        normal_count=normal_count
    )


# 📡 LIVE STREAM
@app.route('/stream')
def stream():
    sample = df.sample(1)

    sample.columns = sample.columns.str.strip()
    sample.rename(columns={"Label": "label"}, inplace=True)
    sample = sample.drop("label", axis=1, errors='ignore')

    sample = sample[columns]
    sample = selector.transform(sample)

    pred = model.predict(sample)[0]
    label = label_encoder.inverse_transform([pred])[0]

    return jsonify({"prediction": label})


if __name__ == "__main__":
    app.run(debug=True)