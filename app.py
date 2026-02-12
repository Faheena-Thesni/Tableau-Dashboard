from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html", title="Dashboard")

@app.route("/about")
def about():
    return render_template("about.html", title="About")

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
    app.run()
