import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)

app.secret_key = "secret123"  # Used for session handling

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# Load the trained scaler
scaler = pickle.load(open("scaler.pkl", "rb"))  # Make sure this file exists

# Dummy user database (Replace with a real database)
users = {"testuser": "password123"}

# ==== SIMPLE IN-MEMORY STORAGE FOR DASHBOARD ====
# Each item will be a dict with keys:
# id, user, date, age, sex, chol, trestbps, result
predictions_log = []
# 
# Home Route (Redirects to Login)
@app.route("/")
def home():
    return redirect(url_for("login"))


# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if users.get(username) == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials! Try again.")
    return render_template("login.html")


# Registration Page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users:
            return render_template("register.html", error="Username already exists!")
        users[username] = password  # Add user (Replace with a real database)
        return redirect(url_for("login"))
    return render_template("register.html")


# Dashboard Page (Only if logged in)
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    # ---- Build stats from predictions_log ----
    total_predictions = len(predictions_log)
    high_risk = sum(1 for p in predictions_log if p["result"] == 1)
    low_risk = total_predictions - high_risk

    # Age buckets: <40, 40–50, 50–60, 60+
    age_buckets = [0, 0, 0, 0]
    for p in predictions_log:
        age = p["age"]
        if age < 40:
            age_buckets[0] += 1
        elif 40 <= age < 50:
            age_buckets[1] += 1
        elif 50 <= age < 60:
            age_buckets[2] += 1
        else:
            age_buckets[3] += 1

    stats = {
        "total_predictions": total_predictions,
        "high_risk": high_risk,
        "low_risk": low_risk,
        "age_buckets": age_buckets,
    }

    # Show last 10 predictions (most recent first)
    recent_predictions = list(reversed(predictions_log))[:10]

    return render_template(
        "dashboard.html",
        user=session.get("user"),
        stats=stats,
        recent_predictions=recent_predictions,
    )


# Prediction Page
@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            # Extract form values individually so we can store them
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = float(request.form["fbs"])
            restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = float(request.form["thal"])

            features = [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]

            # Scale the input
            scaled_features = scaler.transform(np.array(features).reshape(1, -1))

            # Make prediction (0 or 1)
            prediction = int(model.predict(scaled_features)[0])

            # Result message (what result.html displays)
            if prediction == 1:
                result_text = "💔 Heart Disease Detected! Consult a doctor."
            else:
                result_text = "❤️ No Heart Disease Detected. Stay healthy!"

            # ---- Save this prediction to the in-memory log ----
            predictions_log.append(
                {
                    "id": len(predictions_log) + 1,
                    "user": session.get("user"),
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "age": int(age),
                    "sex": int(sex),
                    "chol": int(chol),
                    "trestbps": int(trestbps),
                    "result": prediction,
                    # Optional: you could store more fields if you want
                }
            )

            return render_template("result.html", result=result_text)

        except Exception as e:
            print("Prediction error:", e)
            return render_template(
                "predictor.html", error="Invalid input! Please check your values."
            )

    return render_template("predictor.html")


# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
