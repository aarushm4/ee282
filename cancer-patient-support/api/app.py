import requests
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import random

app = Flask(__name__)

NCI_API_BASE_URL = "https://clinicaltrialsapi.cancer.gov/v1/"

# Expanded mock patient data
MOCK_PATIENT_DATA = pd.DataFrame({
    'patient_id': range(1000),
    'age': np.random.randint(20, 80, 1000),
    'cancer_type': np.random.choice(['breast', 'lung', 'colon', 'prostate', 'melanoma'], 1000),
    'stage': np.random.choice(['I', 'II', 'III', 'IV'], 1000),
    'treatment': np.random.choice(['chemotherapy', 'radiation', 'surgery', 'immunotherapy'], 1000),
    'outcome': np.random.choice(['improved', 'stable', 'deteriorated'], 1000),
    'genetic_marker': np.random.choice(['BRCA', 'EGFR', 'KRAS', 'ALK', 'BRAF'], 1000),
    'previous_treatments': np.random.randint(0, 5, 1000)
})

# Mock feedback data
MOCK_FEEDBACK_DATA = pd.DataFrame({
    'patient_id': range(1000),
    'feedback': [f"The treatment was {'great' if np.random.random() > 0.5 else 'challenging'}. {'I feel better' if np.random.random() > 0.5 else 'I have some side effects'}." for _ in range(1000)]
})

# Mock wearable device data


def generate_wearable_data(patient_id, days=30):
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({
        'patient_id': [patient_id] * days,
        'date': dates,
        'heart_rate': np.random.randint(60, 100, days),
        'steps': np.random.randint(1000, 15000, days),
        'sleep_hours': np.random.uniform(5, 9, days),
        'stress_level': np.random.randint(1, 6, days)
    })


MOCK_WEARABLE_DATA = pd.concat(
    [generate_wearable_data(i) for i in range(1000)])


def get_clinical_trials(cancer_type, zip_code):
    params = {
        "disease.name": cancer_type,
        "sites.org_postal_code": zip_code,
        "record_verification_date_gte": datetime.now().strftime("%Y-%m-%d"),
        "current_trial_status": "Active",
        "size": 10
    }
    response = requests.get(
        NCI_API_BASE_URL + "clinical-trials", params=params)
    if response.status_code == 200:
        return response.json().get('trials', [])
    else:
        return []


def get_treatment_info(cancer_type):
    params = {
        "query": f"{cancer_type} treatment",
        "size": 5
    }
    response = requests.get(NCI_API_BASE_URL + "resources", params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return []


def train_outcome_prediction_model():
    X = pd.get_dummies(MOCK_PATIENT_DATA[['age', 'cancer_type', 'stage',
                       'treatment', 'genetic_marker', 'previous_treatments']], drop_first=True)
    y = (MOCK_PATIENT_DATA['outcome'] == 'improved').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Outcome prediction model accuracy: {accuracy:.2f}")

    joblib.dump(model, 'outcome_prediction_model.joblib')
    return model


def predict_outcome(patient_data):
    model = joblib.load('outcome_prediction_model.joblib')
    patient_features = pd.get_dummies(patient_data[[
                                      'age', 'cancer_type', 'stage', 'treatment', 'genetic_marker', 'previous_treatments']], drop_first=True)
    missing_cols = set(model.feature_names_in_) - set(patient_features.columns)
    for col in missing_cols:
        patient_features[col] = 0
    patient_features = patient_features[model.feature_names_in_]

    prediction = model.predict_proba(patient_features)[0]
    return {"improvement_probability": prediction[1]}


def train_sentiment_analysis_model():
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(MOCK_FEEDBACK_DATA['feedback'])
    y = MOCK_FEEDBACK_DATA['feedback'].apply(
        lambda x: TextBlob(x).sentiment.polarity > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Sentiment analysis model accuracy: {accuracy:.2f}")

    joblib.dump(model, 'sentiment_analysis_model.joblib')
    joblib.dump(vectorizer, 'sentiment_vectorizer.joblib')
    return model, vectorizer


def analyze_sentiment(feedback):
    model = joblib.load('sentiment_analysis_model.joblib')
    vectorizer = joblib.load('sentiment_vectorizer.joblib')

    feedback_vector = vectorizer.transform([feedback])
    sentiment = model.predict_proba(feedback_vector)[0]
    return {"positive_sentiment_probability": sentiment[1]}


def train_recommendation_model():
    X = pd.get_dummies(MOCK_PATIENT_DATA[[
                       'age', 'cancer_type', 'stage', 'genetic_marker', 'previous_treatments']], drop_first=True)
    model = NearestNeighbors(n_neighbors=5, metric='cosine')
    model.fit(X)

    joblib.dump(model, 'recommendation_model.joblib')
    return model


def get_treatment_recommendations(patient_data):
    model = joblib.load('recommendation_model.joblib')
    patient_features = pd.get_dummies(patient_data[[
                                      'age', 'cancer_type', 'stage', 'genetic_marker', 'previous_treatments']], drop_first=True)
    missing_cols = set(model.feature_names_in_) - set(patient_features.columns)
    for col in missing_cols:
        patient_features[col] = 0
    patient_features = patient_features[model.feature_names_in_]

    _, indices = model.kneighbors(patient_features)
    similar_patients = MOCK_PATIENT_DATA.iloc[indices[0]]
    successful_treatments = similar_patients[similar_patients['outcome']
                                             == 'improved']['treatment'].value_counts()

    return successful_treatments.to_dict()


def analyze_wearable_data(patient_id):
    patient_data = MOCK_WEARABLE_DATA[MOCK_WEARABLE_DATA['patient_id'] == patient_id].sort_values(
        'date')

    latest_data = patient_data.iloc[-1]
    avg_data = patient_data.mean()

    analysis = {
        "latest": {
            "heart_rate": latest_data['heart_rate'],
            "steps": latest_data['steps'],
            "sleep_hours": latest_data['sleep_hours'],
            "stress_level": latest_data['stress_level']
        },
        "average": {
            "heart_rate": avg_data['heart_rate'],
            "steps": avg_data['steps'],
            "sleep_hours": avg_data['sleep_hours'],
            "stress_level": avg_data['stress_level']
        },
        "trends": {
            "heart_rate": "stable" if abs(latest_data['heart_rate'] - avg_data['heart_rate']) < 5 else ("increasing" if latest_data['heart_rate'] > avg_data['heart_rate'] else "decreasing"),
            "steps": "stable" if abs(latest_data['steps'] - avg_data['steps']) < 1000 else ("increasing" if latest_data['steps'] > avg_data['steps'] else "decreasing"),
            "sleep_hours": "stable" if abs(latest_data['sleep_hours'] - avg_data['sleep_hours']) < 1 else ("increasing" if latest_data['sleep_hours'] > avg_data['sleep_hours'] else "decreasing"),
            "stress_level": "stable" if abs(latest_data['stress_level'] - avg_data['stress_level']) < 1 else ("increasing" if latest_data['stress_level'] > avg_data['stress_level'] else "decreasing")
        }
    }

    return analysis


@app.route('/api/clinical-trials', methods=['GET'])
def clinical_trials():
    cancer_type = request.args.get('cancer_type')
    zip_code = request.args.get('zip_code')
    if not cancer_type or not zip_code:
        return jsonify({"error": "Both cancer_type and zip_code are required"}), 400

    trials = get_clinical_trials(cancer_type, zip_code)
    return jsonify(trials)


@app.route('/api/treatment-info', methods=['GET'])
def treatment_info():
    cancer_type = request.args.get('cancer_type')
    if not cancer_type:
        return jsonify({"error": "cancer_type is required"}), 400

    info = get_treatment_info(cancer_type)
    return jsonify(info)


@app.route('/api/predict-outcome', methods=['POST'])
def predict_patient_outcome():
    patient_data = request.json
    if not patient_data or not all(key in patient_data for key in ['age', 'cancer_type', 'stage', 'treatment', 'genetic_marker', 'previous_treatments']):
        return jsonify({"error": "Invalid patient data"}), 400

    prediction = predict_outcome(pd.DataFrame([patient_data]))
    return jsonify(prediction)


@app.route('/api/analyze-feedback', methods=['POST'])
def analyze_feedback_sentiment():
    feedback = request.json.get('feedback')
    if not feedback:
        return jsonify({"error": "Feedback is required"}), 400

    sentiment = analyze_sentiment(feedback)
    return jsonify(sentiment)


@app.route('/api/treatment-recommendations', methods=['POST'])
def get_recommendations():
    patient_data = request.json
    if not patient_data or not all(key in patient_data for key in ['age', 'cancer_type', 'stage', 'genetic_marker', 'previous_treatments']):
        return jsonify({"error": "Invalid patient data"}), 400

    recommendations = get_treatment_recommendations(
        pd.DataFrame([patient_data]))
    return jsonify(recommendations)


@app.route('/api/wearable-data-analysis', methods=['GET'])
def wearable_data_analysis():
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    analysis = analyze_wearable_data(int(patient_id))
    return jsonify(analysis)


@app.route('/api/patient-dashboard', methods=['GET'])
def patient_dashboard():
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    patient_data = MOCK_PATIENT_DATA[MOCK_PATIENT_DATA['patient_id'] == int(
        patient_id)].iloc[0]

    clinical_trials = get_clinical_trials(patient_data['cancer_type'], '90210')[
        :3]  # Using a mock ZIP code
    treatment_info = get_treatment_info(patient_data['cancer_type'])[:3]
    outcome_prediction = predict_outcome(
        pd.DataFrame([patient_data.to_dict()]))

    recent_feedback = MOCK_FEEDBACK_DATA[MOCK_FEEDBACK_DATA['patient_id'] == int(
        patient_id)].iloc[-1]['feedback']
    sentiment_analysis = analyze_sentiment(recent_feedback)

    treatment_recommendations = get_treatment_recommendations(
        pd.DataFrame([patient_data.to_dict()]))
    wearable_analysis = analyze_wearable_data(int(patient_id))

    dashboard = {
        "patient_info": patient_data.to_dict(),
        "clinical_trials": clinical_trials,
        "treatment_info": treatment_info,
        "outcome_prediction": outcome_prediction,
        "recent_feedback": recent_feedback,
        "sentiment_analysis": sentiment_analysis,
        "treatment_recommendations": treatment_recommendations,
        "wearable_data_analysis": wearable_analysis
    }

    return jsonify(dashboard)


@app.before_first_request
def initialize_models():
    if not os.path.exists('outcome_prediction_model.joblib'):
        train_outcome_prediction_model()
    if not os.path.exists('sentiment_analysis_model.joblib'):
        train_sentiment_analysis_model()
    if not os.path.exists('recommendation_model.joblib'):
        train_recommendation_model()


if __name__ == '__main__':
    app.run(debug=True)
