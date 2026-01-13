# Flask web app

from flask import Flask, render_template, request
import pandas as pd
from textblob import TextBlob
import difflib
import os

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
