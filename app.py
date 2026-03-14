from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import joblib
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask import jsonify
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = "supersecretkey"

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Per-user chat memory
chat_history = {}

# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect("disaster.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disaster_type TEXT,
            probability REAL,
            result TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(disaster_type, probability, result):
    conn = sqlite3.connect("disaster.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (disaster_type, probability, result)
        VALUES (?, ?, ?)
    """, (disaster_type, probability, result))
    conn.commit()
    conn.close()

# =========================
# LOAD MODELS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

flood_model = joblib.load(os.path.join(BASE_DIR, "flood_model.pkl"))

landslide_model = joblib.load(os.path.join(BASE_DIR, "landslide_model.pkl"))
landslide_imputer = joblib.load(os.path.join(BASE_DIR, "landslide_imputer.pkl"))
landslide_scaler = joblib.load(os.path.join(BASE_DIR, "landslide_scaler.pkl"))
landslide_columns = joblib.load(os.path.join(BASE_DIR, "landslide_columns.pkl"))
landslide_threshold = joblib.load(os.path.join(BASE_DIR, "landslide_threshold.pkl"))

# =========================
# REGISTER
# =========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        conn = sqlite3.connect("disaster.db")
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                           (username, email, password))
            conn.commit()
            flash("Account created successfully!")
            return redirect(url_for("login"))
        except:
            flash("User already exists!")
        conn.close()

    return render_template("register.html")

# =========================
# LOGIN
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("disaster.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[1]
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid Credentials")

    return render_template("login.html")

# =========================
# FORGOT PASSWORD
# =========================
@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form["email"]
        new_password = generate_password_hash(request.form["new_password"])

        conn = sqlite3.connect("disaster.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password=? WHERE email=?",
                       (new_password, email))
        conn.commit()
        conn.close()

        flash("Password updated successfully!")
        return redirect(url_for("login"))

    return render_template("forgot.html")

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("disaster.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE 'High%'")
    high_risk = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(DISTINCT disaster_type) FROM predictions")
    disaster_types = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT disaster_type, probability, result, date
        FROM predictions
        ORDER BY id DESC LIMIT 5
    """)
    history = cursor.fetchall()

    cursor.execute("""
        SELECT disaster_type, COUNT(*)
        FROM predictions
        GROUP BY disaster_type
    """)
    data = cursor.fetchall()

    conn.close()

    labels = [row[0] for row in data] if data else []
    values = [row[1] for row in data] if data else []

    if total_predictions > 0:
        risk_percent = round((high_risk / total_predictions) * 100, 2)
    else:
        risk_percent = 0

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        high_risk=high_risk,
        disaster_types=disaster_types,
        history=history,
        labels=labels,
        values=values,
        risk_percent=risk_percent
    )

# =========================
# FLOOD PAGE (FIX ADDED)
# =========================
@app.route("/flood")
def flood():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("flood.html")

# =========================
# LANDSLIDE PAGE (FIX ADDED)
# =========================
@app.route("/landslide")
def landslide():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("landslide.html")

# =========================
# FLOOD PREDICTION
# =========================
@app.route('/predict_flood', methods=['POST'])
def predict_flood():

    if "user" not in session:
        return redirect(url_for("login"))

    Jan = float(request.form['Jan'])
    Feb = float(request.form['Feb'])
    Mar = float(request.form['Mar'])
    Apr = float(request.form['Apr'])
    May = float(request.form['May'])

    PreMonsoon = Jan + Feb + Mar + Apr + May
    Rainfall_Variability = np.std([Jan, Feb, Mar, Apr, May])
    Dry_Spell = sum([x < 10 for x in [Jan, Feb, Mar, Apr, May]])

    flood_input = np.array([[Jan, Feb, Mar, Apr, May,
                             PreMonsoon, Rainfall_Variability, Dry_Spell]])

    flood_prob = flood_model.predict_proba(flood_input)[0][1]
    flood_pred = 1 if flood_prob > 0.40 else 0
    flood_result = "High Flood Risk" if flood_pred == 1 else "Low Flood Risk"

    flood_causes = []
    if PreMonsoon > 500:
        flood_causes.append("Extremely high pre-monsoon rainfall detected.")
    if Rainfall_Variability > 50:
        flood_causes.append("High rainfall variability increases sudden flood chances.")
    if Dry_Spell >= 3:
        flood_causes.append("Irregular rainfall pattern observed.")
    if not flood_causes:
        flood_causes.append("Rainfall levels are within safe limits.")

    if flood_pred == 1:
        flood_safety = [
            "Avoid low-lying areas immediately.",
            "Keep emergency supplies ready.",
            "Switch off electricity if water enters home.",
            "Follow official evacuation alerts."
        ]
    else:
        flood_safety = [
            "Maintain drainage system.",
            "Stay updated with weather forecasts."
        ]

    save_prediction("Flood", float(flood_prob), flood_result)

    return render_template("flood.html",
                           flood_result=flood_result,
                           flood_prob=round(flood_prob, 3),
                           flood_causes=flood_causes,
                           flood_safety=flood_safety)

# =========================
# LANDSLIDE PREDICTION
# =========================
@app.route('/predict_landslide', methods=['POST'])
def predict_landslide():

    if "user" not in session:
        return redirect(url_for("login"))

    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    country = request.form['country']
    category = request.form['category']
    setting = request.form['setting']

    input_dict = {
        'latitude': latitude,
        'longitude': longitude,
        'year': year,
        'month': month,
        'day': day,
        'country_name': country,
        'landslide_category': category,
        'landslide_setting': setting
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)

    for col in landslide_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[landslide_columns]
    input_imputed = landslide_imputer.transform(input_df)

    landslide_prob = landslide_model.predict_proba(input_imputed)[0][1]
    landslide_pred = 1 if landslide_prob > landslide_threshold else 0
    landslide_result = "High Landslide Risk" if landslide_pred == 1 else "Low Landslide Risk"

    if landslide_pred == 1:
        landslide_causes = [
            "Unstable terrain conditions detected.",
            "Rainfall and slope combination increases risk."
        ]
        landslide_safety = [
            "Avoid steep slopes.",
            "Monitor cracks in soil.",
            "Follow evacuation alerts.",
            "Improve hillside drainage."
        ]
    else:
        landslide_causes = ["Current environmental conditions are stable."]
        landslide_safety = [
            "Maintain vegetation cover.",
            "Regularly monitor terrain stability."
        ]

    save_prediction("Landslide", float(landslide_prob), landslide_result)

    return render_template("landslide.html",
                           landslide_result=landslide_result,
                           landslide_prob=round(landslide_prob, 3),
                           landslide_causes=landslide_causes,
                           landslide_safety=landslide_safety)


# =========================
# CHATBOT ROUTE
# =========================
@app.route("/chat", methods=["POST"])
def chat():

    if "user" not in session:
        return jsonify({"reply": "Please login first."})

    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"reply": "Message is empty!"})

    username = session["user"]

    # Create memory per user
    if username not in chat_history:
        chat_history[username] = []

    chat_history[username].append({
        "role": "user",
        "content": user_message
    })

    completion = client.chat.completions.create(
        model="stepfun/step-3.5-flash:free",
        messages=chat_history[username],
    )

    bot_reply = completion.choices[0].message.content

    chat_history[username].append({
        "role": "assistant",
        "content": bot_reply
    })

    return jsonify({"reply": bot_reply})

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)