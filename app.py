from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
import re


def extract_symptoms(text):
    text = text.lower()

    symptoms = {
        "fever": 1 if ("fever" in text or "temperature" in text or "104" in text) else 0,
        "high_fever": 1 if re.search(r"(102|103|104|high fever|very high)", text) else 0,
        "cough": 1 if "cough" in text else 0,
        "dry_cough": 1 if "dry cough" in text else 0,
        "weakness": 1 if ("weak" in text or "fatigue" in text or "tired" in text) else 0,
        "headache": 1 if "headache" in text else 0,
        "breath": 1 if ("breath" in text or "breathing" in text) else 0,
        "chest_pain": 1 if "chest pain" in text else 0,
        "body_pain": 1 if ("body pain" in text or "bodyache" in text) else 0,
        "nausea": 1 if "nausea" in text else 0
    }

    symptoms["symptom_count"] = sum(symptoms.values())
    return symptoms


def rule_based_diagnosis(s):
    if (s["high_fever"] and s["dry_cough"] and s["breath"]):
        return "Pneumonia"

    if (s["fever"] and s["headache"] and s["weakness"]):
        return "Viral Fever"

    if (s["fever"] and s["body_pain"] and "rash" in s):
        return "Dengue"

    if (s["dry_cough"] and s["breath"]):
        return "Bronchitis"

    if (s["fever"] and s["weakness"] and s["nausea"]):
        return "Typhoid"

    if s["symptom_count"] <= 1:
        return "Normal Condition"

    return "General Viral Infection"


def hybrid_predict(user_text, ml_model):
    s = extract_symptoms(user_text)

    ml_input = np.array([[s["fever"], s["cough"], s["weakness"], s["headache"],
                          s["breath"], s["nausea"], s["body_pain"], s["chest_pain"]]])
    ml_pred = ml_model.predict(ml_input)[0]

    rule_pred = rule_based_diagnosis(s)

    if rule_pred != "Normal Condition":
        return rule_pred
    
    return ml_pred

# Load Models
log_model = pickle.load(open("models/logistic.pkl", "rb"))
rf_model = pickle.load(open("models/randomforest.pkl", "rb"))
svm_model = pickle.load(open("models/svm.pkl", "rb"))
nb_model = pickle.load(open("models/naivebayes.pkl", "rb"))

# Disease Mapping (VERY IMPORTANT FIX)
disease_map = {
    0: "Normal",
    1: "Flu",
    2: "Covid",
    3: "Malaria",
    4: "Dengue",
    5: "Typhoid",
    6: "Pneumonia",
    7: "Asthma",
    8: "Food Poisoning",
    9: "Unknown Disease"
}


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    name = request.form["name"]
    age = int(request.form["age"])

    # Collect Inputs
    input_features = [
        int(request.form["fever"]),
        int(request.form["cough"]),
        int(request.form["headache"]),
        int(request.form["weakness"]),
        int(request.form["chestpain"]),
        int(request.form["breath"]),
        int(request.form["nausea"]),
        int(request.form["bodypain"]),
        age
    ]

    final_input = np.array(input_features).reshape(1, -1)

    # Predictions (returning integer labels)
    pred1 = log_model.predict(final_input)[0]
    pred2 = rf_model.predict(final_input)[0]
    pred3 = svm_model.predict(final_input)[0]
    pred4 = nb_model.predict(final_input)[0]

    # Convert integer → disease name
    pred1_name = disease_map[pred1]
    pred2_name = disease_map[pred2]
    pred3_name = disease_map[pred3]
    pred4_name = disease_map[pred4]

    # Overall prediction by majority vote
    predictions = [pred1, pred2, pred3, pred4]
    final_pred = max(set(predictions), key=predictions.count)
    final_pred_name = disease_map[final_pred]

    # Suggestions
    suggestions = {
        "Normal": "You are fine! Maintain healthy routine.",
        "Flu": "Drink warm water, take rest and light food.",
        "Covid": "Isolate yourself and monitor oxygen levels.",
        "Malaria": "Consult doctor and test for malaria immediately.",
        "Dengue": "Increase fluid intake and avoid painkillers.",
        "Typhoid": "Take doctor’s advice and avoid outside food.",
        "Pneumonia": "Seek medical attention immediately.",
        "Asthma": "Use inhaler and avoid dust exposure.",
        "Food Poisoning": "Drink ORS and avoid solid food.",
        "Unknown Disease": "Consult a doctor for proper evaluation."
    }.get(final_pred_name, "Consult a doctor.")

    return render_template("result.html",
                           name=name,
                           pred1=pred1_name,
                           pred2=pred2_name,
                           pred3=pred3_name,
                           pred4=pred4_name,
                           overall=final_pred_name,
                           suggestions=suggestions)


if __name__ == "__main__":
    app.run(debug=True)
