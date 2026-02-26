from flask import Flask, render_template, jsonify, request
import requests
import json
import os

app = Flask(__name__)

# Load dataset context once at startup
with open("dataset_context.json", "r") as f:
    DATASET_CTX = json.load(f)

SYSTEM_PROMPT = f"""You are AirBot, an expert data analyst for India's Air Quality dataset (Feb 2026).

DATASET:
- Total Records: {DATASET_CTX['total_records']}
- Cities: {', '.join(DATASET_CTX['cities'])}
- States: {', '.join(DATASET_CTX['states'])}
- Pollutants: {', '.join(DATASET_CTX['pollutants'])}

CITY-LEVEL AVERAGES (city → pollutant → avg μg/m³):
{json.dumps(DATASET_CTX['city_stats'], indent=1)}

STATE-LEVEL AVERAGES:
{json.dumps(DATASET_CTX['state_stats'], indent=1)}

NATIONAL AVERAGES:
{json.dumps(DATASET_CTX['overall_avg'], indent=1)}

TOP POLLUTED CITIES PER POLLUTANT:
{json.dumps(DATASET_CTX['top_cities_per_pollutant'], indent=1)}

STATION COUNTS PER STATE:
{json.dumps(DATASET_CTX['station_count_per_state'], indent=1)}

YOUR CAPABILITIES:
1. Correct spelling mistakes silently (Dehli→Delhi, mumabi→Mumbai, pollustion→pollution, avrage→average)
2. Handle abbreviations (blore=Bengaluru, hyd=Hyderabad, mum=Mumbai, del=Delhi)  
3. Do math: averages, comparisons, rankings, percentages
4. Compare cities/states side-by-side
5. Rank top/bottom N locations for any pollutant
6. Explain pollutants and health impacts

POLLUTANT WHO LIMITS:
- PM2.5: 15 μg/m³ annual (⚠️ >35.4 = Unhealthy)
- PM10: 45 μg/m³ annual  
- NO2: 10 μg/m³ annual
- SO2: 40 μg/m³ per 24hr
- CO: 4 mg/m³ per 24hr
- OZONE: 100 μg/m³ per 8hr
- NH3: 400 μg/m³ per 24hr

Always include specific numbers. Use ⚠️ for concerning levels, ✅ for safe. If data unavailable for a location, say so clearly."""


@app.route("/")
def dashboard():
    return render_template("dashboard.html", title="Dashboard")


@app.route("/about")
def about():
    return render_template("about.html", title="About")


@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", title="AirBot — Air Quality Assistant")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Proxy endpoint for Claude API — keeps API key server-side."""
    data = request.get_json()
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Limit history to last 20 messages to control token usage
    messages = messages[-20:]

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        },
        json={
            "model": "claude-haiku-4-5-20251001",  # Fast + cheap for chatbot
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": messages
        },
        timeout=30
    )

    return jsonify(response.json()), response.status_code


@app.route("/api/emissions")
def get_emissions():
    BASE_URL = "https://api.climatetrace.org/v7/sources/emissions"
    params = {
        "year": 2021,
        "sectors": "transportation",
        "subsectors": "road-transportation",
        "gas": "co2e_100yr"
    }
    response = requests.get(BASE_URL, params=params)
    return response.json()


if __name__ == "__main__":
    app.run(debug=False)
