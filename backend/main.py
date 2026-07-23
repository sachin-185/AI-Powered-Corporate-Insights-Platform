import os
import sys
import shutil
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import SmartCorpEngine, analyze_sentiment, summarize_text
app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)

data_path = os.path.join(base_dir, "data", "company_data.csv")
if not os.path.exists(data_path):
    fallback_path = os.path.join(parent_dir, "data", "company_data.csv")
    if os.path.exists(fallback_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        shutil.copy(fallback_path, data_path)
    else:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, "w") as f:
            f.write("date,department,meeting_transcript,employee_feedback,attrition_rate,engagement_score,project_delay_index\n")
            f.write("2026-07-01,Engineering,Dummy meeting,Happy team,0.10,0.80,0.05\n")

db_client = None
cred_path = os.path.join(base_dir, "firebase-service-account.json")
if os.path.exists(cred_path):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        db_client = firestore.client()
        print("Firebase Admin successfully initialized!")
    except Exception as e:
        print("Firebase Admin initialization error:", e)

engine = SmartCorpEngine(data_path, db_client)

@app.route("/api/departments", methods=["GET"])
def get_departments():
    try:
        depts = engine.df["department"].unique().tolist()
        return jsonify({"departments": depts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    try:
        departments = request.args.get("departments")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        
        dept_list = departments.split(",") if departments else None
        filtered_df = engine.get_dashboard_data(
            selected_depts=dept_list,
            start_date=start_date,
            end_date=end_date
        )
        
        records = []
        for row in filtered_df.itertuples():
            records.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "department": row.department,
                "meeting_transcript": getattr(row, "meeting_transcript", ""),
                "meeting_summary": getattr(row, "meeting_summary", ""),
                "employee_feedback": getattr(row, "employee_feedback", ""),
                "sentiment_label": getattr(row, "sentiment_label", "NEUTRAL"),
                "sentiment_score": float(getattr(row, "sentiment_score", 0.5)),
                "predicted_risk": getattr(row, "predicted_risk", "Low"),
                "attrition_rate": float(row.attrition_rate),
                "engagement_score": float(row.engagement_score),
                "project_delay_index": float(row.project_delay_index),
                "attrition_change": float(row.attrition_change),
                "engagement_change": float(row.engagement_change)
            })
            
        min_date = engine.df["date"].min().strftime("%Y-%m-%d") if not engine.df.empty else ""
        max_date = engine.df["date"].max().strftime("%Y-%m-%d") if not engine.df.empty else ""
        all_depts = engine.df["department"].unique().tolist()
        
        return jsonify({
            "records": records,
            "min_date": min_date,
            "max_date": max_date,
            "all_departments": all_depts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast", methods=["POST"])
def get_forecast():
    try:
        data = request.get_json()
        department = data.get("department")
        kpi = data.get("kpi")
        forecast_data = engine.get_forecast(department, kpi)
        return jsonify({"forecast": forecast_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze-sentiment", methods=["POST"])
def post_analyze_sentiment():
    try:
        data = request.get_json()
        text = data.get("text")
        label, score = analyze_sentiment(text)
        return jsonify({"label": label, "score": score})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/summarize", methods=["POST"])
def post_summarize():
    try:
        data = request.get_json()
        text = data.get("text")
        summary = summarize_text(text)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400
    try:
        file.save(data_path)
        global engine
        if engine.db_client:
            docs = engine.db_client.collection("metrics").stream()
            for doc in docs:
                doc.reference.delete()
        engine = SmartCorpEngine(data_path, engine.db_client)
        return jsonify({"status": "success", "message": "CSV uploaded and AI engine re-trained successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)