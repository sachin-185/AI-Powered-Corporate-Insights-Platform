import os
import pandas as pd
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from transformers import pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
device = 0 if torch.cuda.is_available() else -1

sentiment_pipe = None
summary_pipe = None

def get_sentiment_pipeline():
    global sentiment_pipe
    if sentiment_pipe is None:
        sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    return sentiment_pipe

def get_summary_pipeline():
    global summary_pipe
    if summary_pipe is None:
        summary_pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    return summary_pipe

def analyze_sentiment(text: str):
    if not text or not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL", 0.5
    try:
        pipe = get_sentiment_pipeline()
        result = pipe(text)
        label = result[0]["label"]
        score = result[0]["score"]
        numeric_score = score if label == "POSITIVE" else (1 - score)
        return label, numeric_score
    except Exception as e:
        print("Sentiment Analysis Error:", e)
        return "NEUTRAL", 0.5

def risk_level(row):
    if row["attrition_rate"] > 0.15 or row["engagement_score"] < 0.7 or row["project_delay_index"] > 0.2:
        return "High"
    elif row["attrition_rate"] > 0.10 or row["engagement_score"] < 0.8 or row["project_delay_index"] > 0.1:
        return "Medium"
    else:
        return "Low"

def summarize_text(text: str):
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return text
    try:
        pipe = get_summary_pipeline()
        input_text = text[:1024]
        result = pipe(input_text, max_length=50, min_length=10, do_sample=False)
        return result[0]["summary_text"]
    except Exception as e:
        print("Summarization Error:", e)
        return text

class SmartCorpEngine:
    def __init__(self, data_path: str, db_client=None):
        self.data_path = data_path
        self.db_client = db_client
        self.df = None
        self.clf = None
        self.le = LabelEncoder()
        self.risk_features = ["attrition_rate", "engagement_score", "project_delay_index"]
        self.load_data()

    def load_data(self):
        df = None
        if self.db_client:
            try:
                docs = self.db_client.collection("metrics").stream()
                records = []
                for doc in docs:
                    data = doc.to_dict()
                    records.append(data)
                
                if len(records) > 0:
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"])
                else:
                    df = self.load_local_csv()
                    for _, row in df.iterrows():
                        row_dict = {
                            "date": row["date"].strftime("%Y-%m-%d"),
                            "department": str(row["department"]),
                            "meeting_transcript": str(row["meeting_transcript"]),
                            "employee_feedback": str(row["employee_feedback"]),
                            "attrition_rate": float(row["attrition_rate"]),
                            "engagement_score": float(row["engagement_score"]),
                            "project_delay_index": float(row["project_delay_index"])
                        }
                        self.db_client.collection("metrics").add(row_dict)
            except Exception as e:
                print("Firestore Load Error, falling back to local CSV:", e)
                df = self.load_local_csv()
        else:
            df = self.load_local_csv()

        if "sentiment_label" not in df.columns or "sentiment_score" not in df.columns:
            sentiments = df["employee_feedback"].apply(analyze_sentiment)
            df["sentiment_label"] = sentiments.apply(lambda x: x[0])
            df["sentiment_score"] = sentiments.apply(lambda x: x[1])

        if "risk_level" not in df.columns:
            df["risk_level"] = df.apply(risk_level, axis=1)

        if "meeting_summary" not in df.columns:
            df["meeting_summary"] = df["meeting_transcript"]

        df = df.sort_values(["department", "date"])
        df["attrition_change"] = df.groupby("department")["attrition_rate"].diff().fillna(0)
        df["engagement_change"] = df.groupby("department")["engagement_score"].diff().fillna(0)
        
        self.df = df
        self.train_risk_model()

    def load_local_csv(self):
        if not os.path.exists(self.data_path):
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def train_risk_model(self):
        self.df["risk_label_encoded"] = self.le.fit_transform(self.df["risk_level"])
        self.clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.clf.fit(self.df[self.risk_features], self.df["risk_label_encoded"])

    def get_dashboard_data(self, selected_depts=None, start_date=None, end_date=None):
        df_filtered = self.df.copy()
        
        if selected_depts:
            df_filtered = df_filtered[df_filtered["department"].isin(selected_depts)]
        
        if start_date:
            df_filtered = df_filtered[df_filtered["date"] >= pd.to_datetime(start_date)]
            
        if end_date:
            df_filtered = df_filtered[df_filtered["date"] <= pd.to_datetime(end_date)]

        if len(df_filtered) > 0:
            df_filtered["predicted_risk_encoded"] = self.clf.predict(df_filtered[self.risk_features])
            df_filtered["predicted_risk"] = self.le.inverse_transform(df_filtered["predicted_risk_encoded"])
        else:
            df_filtered["predicted_risk"] = []

        return df_filtered

    def get_forecast(self, department: str, kpi: str):
        dept_df = self.df[self.df["department"] == department].copy()
        if len(dept_df) == 0:
            return []

        lr = LinearRegression()
        X = np.arange(len(dept_df)).reshape(-1, 1)
        y = dept_df[kpi].values
        lr.fit(X, y)
        forecast = lr.predict(X)

        result = []
        for i, row in enumerate(dept_df.itertuples()):
            result.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "actual": float(getattr(row, kpi)),
                "forecast": float(forecast[i])
            })
        return result
