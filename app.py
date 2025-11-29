import pandas as pd
from transformers import pipeline
import torch
import streamlit as st

# Detect GPU
device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

# Load HuggingFace pipelines
summary_pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# --- Summarization function ---
def summarize_text(text):
    try:
        # Adjust max/min lengths based on input size
        input_len = len(text.split())
        if input_len < 5:  # very short, return as is
            return text
        max_len = min(60, input_len + 20)
        min_len = min(10, max_len)
        result = summary_pipe(text, max_length=max_len, min_length=min_len, do_sample=False)
        return result[0]["summary_text"]
    except:
        return "Summary unavailable"

# --- Sentiment analysis function ---
def analyze_sentiment(text):
    try:
        result = sentiment_pipe(text)
        label = result[0]["label"]
        score = result[0]["score"]
        numeric_score = score if label == "POSITIVE" else (1 - score)
        return label, numeric_score
    except:
        return "NEUTRAL", 0.5  # fallback

# --- Risk assessment ---
def risk_level(row):
    if row["attrition_rate"] > 0.15 or row["engagement_score"] < 0.7 or row["project_delay_index"] > 0.2:
        return "High"
    elif row["attrition_rate"] > 0.10 or row["engagement_score"] < 0.8 or row["project_delay_index"] > 0.1:
        return "Medium"
    else:
        return "Low"

# --- Main function to load and process data ---
@st.cache_data(show_spinner=True)
def run_smartcorp_ai():
    df = pd.read_csv("data/company_data.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Summarize meetings
    df["meeting_summary"] = df["meeting_transcript"].apply(summarize_text)

    # Sentiment
    sentiments = df["employee_feedback"].apply(analyze_sentiment)
    df["sentiment_label"] = sentiments.apply(lambda x: x[0])
    df["sentiment_score"] = sentiments.apply(lambda x: x[1])

    # Risk
    df["risk_level"] = df.apply(risk_level, axis=1)

    # Trends
    df = df.sort_values(["department", "date"])
    df["attrition_change"] = df.groupby("department")["attrition_rate"].diff().fillna(0)
    df["engagement_change"] = df.groupby("department")["engagement_score"].diff().fillna(0)

    return df
