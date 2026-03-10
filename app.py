from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import pickle
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# =========================
# LOGIN CREDENTIALS
# =========================
USERNAME = "admin"
PASSWORD = "1234"

# =========================
# DATABASE INIT
# =========================
def init_db():
    conn = sqlite3.connect("disaster.db")
    cursor = conn.cursor()

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

flood_model = pickle.load(open(os.path.join(BASE_DIR, "flood_model.pkl"), "rb"))

landslide_model = pickle.load(open(os.path.join(BASE_DIR, "landslide_model.pkl"), "rb"))
landslide_imputer = pickle.load(open(os.path.join(BASE_DIR, "landslide_imputer.pkl"), "rb"))
landslide_scaler = pickle.load(open(os.path.join(BASE_DIR, "landslide_scaler.pkl"), "rb"))
landslide_columns = pickle.load(open(os.path.join(BASE_DIR, "landslide_columns.pkl"), "rb"))
landslide_threshold = pickle.load(open(os.path.join(BASE_DIR, "landslide_threshold.pkl"), "rb"))

# =========================
# LOGIN PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")

# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("disaster.db")  
    cursor = conn.cursor()

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0] or 0

    # High risk cases
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result LIKE 'High%'")
    high_risk = cursor.fetchone()[0] or 0

    # Disaster types
    cursor.execute("SELECT COUNT(DISTINCT disaster_type) FROM predictions")
    disaster_types = cursor.fetchone()[0] or 0

    # Recent history
    cursor.execute("""
        SELECT disaster_type, probability, result, date 
        FROM predictions 
        ORDER BY id DESC LIMIT 5
    """)
    history = cursor.fetchall()

    # Chart data
    cursor.execute("""
        SELECT disaster_type, COUNT(*) 
        FROM predictions 
        GROUP BY disaster_type
    """)
    data = cursor.fetchall()

    conn.close()

    labels = [row[0] for row in data] if data else []
    values = [row[1] for row in data] if data else []

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        high_risk=high_risk,
        disaster_types=disaster_types,
        history=history,
        labels=labels,
        values=values
    )

# =========================
# FLOOD PAGE
# =========================
@app.route("/flood")
def flood_page():
    if "user" in session:
        return render_template("flood.html")
    return redirect(url_for("login"))

# =========================
# LANDSLIDE PAGE
# =========================
@app.route("/landslide")
def landslide_page():
    if "user" in session:
        return render_template("landslide.html")
    return redirect(url_for("login"))

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

    save_prediction("Flood", float(flood_prob), flood_result)

    return render_template("flood.html",
                           flood_result=flood_result,
                           flood_prob=round(flood_prob, 3))

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

    save_prediction("Landslide", float(landslide_prob), landslide_result)

    return render_template("landslide.html",
                           landslide_result=landslide_result,
                           landslide_prob=round(landslide_prob, 3))

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
    app.run(debug=True)

