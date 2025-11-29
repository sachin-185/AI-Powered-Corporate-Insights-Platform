# AI-Powered-Corporate-Insights-Platform
Developed an AI-powered corporate insights dashboard to summarize meeting transcripts, analyze employee sentiment, and assess departmental risks, enabling faster and data-driven decision-making.

## Features

- **Data Analysis**: Load and process corporate data from CSV files
- **Sentiment Analysis**: Analyze employee feedback using DistilBERT model
- **Risk Prediction**: Assess departmental risk levels using Decision Tree classification
- **KPI Trends**: Visualize key performance indicators with forecasting using Linear Regression
- **Interactive Filters**: Filter data by department, KPIs, and date ranges
- **Sentiment Distribution**: Bar charts showing sentiment breakdown by department
- **Download Functionality**: Export filtered data as CSV

## Dataset

The application uses a synthetic dataset (`data/company_data.csv`) containing:
- Date, Department, Meeting transcripts, Employee feedback
- KPIs: Attrition rate, Engagement score, Project delay index

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-corporate-insights.git
   cd ai-corporate-insights
   ```
2. Run the application:
   ```bash
   streamlit run visual.py
   ```

## Requirements

- Python 3.8+
- streamlit
- pandas
- transformers
- torch
- scikit-learn
- altair

## Usage

1. Launch the app with `streamlit run visual.py`
2. Upload or use the provided dataset
3. Use sidebar filters to explore data
4. View sentiment analysis, risk predictions, and KPI trends
5. Download filtered results

## AI Models Used
- **Summarization**: Facebook BART-large-CNN (removed in current version)
- **Sentiment Analysis**: DistilBERT SST-2
- **Risk Classification**: Decision Tree Classifier
- **Trend Forecasting**: Linear Regression

## Project Structure

```
├── app.py              # Backend logic and AI processing
├── visual.py           # Streamlit frontend
├── data/
│   └── company_data.csv # Sample dataset
└── README.md           # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request
