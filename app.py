# Flask web app

from flask import Flask, render_template, request
import pandas as pd
from textblob import TextBlob
import difflib
import os

app = Flask(__name__)

df = pd.read_csv("netflix_titles.csv")
df.columns = df.columns.str.lower()

def correct_spelling(text):
    return str(TextBlob(text).correct())

def find_best_column(word, columns):
    matches = difflib.get_close_matches(word, columns, n=1, cutoff=0.6)
    return matches[0] if matches else None

def calculate_answer(question):
    question = correct_spelling(question.lower())
    words = question.split()

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    text_columns = df.select_dtypes(include='object').columns.tolist()

    filter_col, filter_val = None, None
    for col in text_columns:
        for val in df[col].dropna().unique():
            if val.lower() in question:
                filter_col, filter_val = col, val
                break
        if filter_col:
            break

    if "how" in words and "many" in words or "count" in words:
        if filter_col:
            return f"Total records with {filter_col} = {filter_val}: {len(df[df[filter_col] == filter_val])}"
        return f"Total records: {len(df)}"

    if "average" in words or "mean" in words:
        for word in words:
            col = find_best_column(word, numeric_columns)
            if col:
                data = df[df[filter_col] == filter_val][col] if filter_col else df[col]
                return f"The average {col} is {data.mean():.2f}"

    return "Sorry, I cannot compute this from the dataset."


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
