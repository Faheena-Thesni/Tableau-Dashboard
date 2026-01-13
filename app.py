# Flask web app

from flask import Flask, render_template
import os

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# DO NOT use app.run() on Render
