from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import math

app = Flask(__name__)

# Load dataset once at startup
df = pd.read_csv("air_quality.csv")
df["pollutant_avg"] = pd.to_numeric(df["pollutant_avg"], errors="coerce")
df["pollutant_min"] = pd.to_numeric(df["pollutant_min"], errors="coerce")
df["pollutant_max"] = pd.to_numeric(df["pollutant_max"], errors="coerce")

def build_dataset_json():
    records = []
    for _, row in df.iterrows():
        avg = row["pollutant_avg"]
        mn  = row["pollutant_min"]
        mx  = row["pollutant_max"]
        records.append({
            "state":     row["state"].replace("_", " "),
            "city":      row["city"],
            "station":   row["station"],
            "pollutant": row["pollutant_id"],
            "min":  round(mn,  1) if not (isinstance(mn,  float) and math.isnan(mn))  else "-",
            "max":  round(mx,  1) if not (isinstance(mx,  float) and math.isnan(mx))  else "-",
            "avg":  round(avg, 1) if not (isinstance(avg, float) and math.isnan(avg)) else "-",
        })
    return records

DATASET_JSON = build_dataset_json()

from chatbot_engine import generate_response


@app.route("/")
def dashboard():
    return render_template("dashboard.html", title="Dashboard")

@app.route("/data")
def data():
    return render_template("data.html", title="Dataset")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", title="AirBot — Air Quality Assistant")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

@app.route("/ping")
def ping():
    """Keep-alive — prevents Render free tier from sleeping."""
    return {"status": "ok"}, 200

@app.route("/api/dataset")
def api_dataset():
    """Serve full dataset as JSON for the frontend table."""
    return app.response_class(
        response=__import__('json').dumps(DATASET_JSON),
        status=200,
        mimetype='application/json'
    )

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])
    if not messages:
        return {"error": "No messages provided"}, 400
    user_message = messages[-1]["content"]
    response = generate_response(user_message)
    return {"reply": response}

@app.route("/api/emissions")
def get_emissions():
    BASE_URL = "https://api.climatetrace.org/v7/sources/emissions"
    params = {"year": 2021, "sectors": "transportation",
              "subsectors": "road-transportation", "gas": "co2e_100yr"}
    response = requests.get(BASE_URL, params=params, timeout=10)
    return response.json()

if __name__ == "__main__":
    app.run(debug=False)
