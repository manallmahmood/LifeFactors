from flask import Flask, render_template, request
import joblib
import numpy as np
import os

# Load the trained model
model_filename = "highestTestModelHeartBlood.joblib"
model_directory = r'/Users/manalmahmood/Desktop/SP1 updated/joblib test'
full_path = os.path.join(model_directory, model_filename)
model = joblib.load(full_path)

from app import app

# Define the route for the Diabetes prediciton page
@app.route("/heartblood.html")
def heartblood():
    return render_template("heartblood.html")


# Define prediction route for heart blood model
@app.route("/predictheartblood", methods=["POST"])
def predict_heartblood():
    # Get user input from the form
    age = int(request.form["age"])
    sex = int(request.form["sex_blood"])
    angina = int(request.form["cp"])
    if angina == 0:
        cp_0 = 1
        cp_1 = 0
        cp_2 = 0
        cp_3 = 0
    elif angina == 1:
        cp_0 = 0
        cp_1 = 1
        cp_2 = 0
        cp_3 = 0
    elif angina == 2:
        cp_0 = 0
        cp_1 = 0
        cp_2 = 1
        cp_3 = 0
    elif angina == 3:
        cp_0 = 0
        cp_1 = 0
        cp_2 = 0
        cp_3 = 1
    trestbps = int(request.form["trestbps"])
    chol = int(request.form["chol"])
    fbs = int(request.form["fbs"])
    ecg = int(request.form["restecg"])
    if ecg == 0:
        restecg_0 = 1
        restecg_1 = 0
        restecg_2 = 0
    elif ecg == 1:
        restecg_0 = 0
        restecg_1 = 1
        restecg_2 = 0
    elif ecg == 2:
        restecg_0 = 0
        restecg_1 = 0
        restecg_2 = 1
    thalach = int(request.form["thalach"])
    exang = int(request.form["exang"])
    oldpeak = int(request.form["oldpeak"])
    slope = int(request.form["slope"])
    ca = int(request.form["ca"])
    thalnum = int(request.form["thal"])
    thal_0 = 0
    thal_1 = 0
    thal_2 = 0
    thal_3 = 0
    if thalnum == 0:
        thal_0 = 1
    elif thalnum == 1:
        thal_1 = 1
    elif thalnum == 2:
        thal_2 = 1
    elif thalnum == 3:
        thal_3 = 1
    age = int(request.form["age"]) 
    # Prepare the user input as a feature vector
    user_input = [age, sex, trestbps, chol, fbs, 
                  thalach, exang, oldpeak, slope, 
                  ca, cp_0, cp_1, cp_2, cp_3, 
                  restecg_0, restecg_1, restecg_2, 
                  thal_0, thal_1, thal_2, thal_3]

    # Make prediction using the model
    prediction = model.predict(np.array([user_input]))
    decision_function_scores = model.decision_function(np.array([user_input]))

    # Manually calculate the probability based on decision function scores
    probability = 1 / (1 + np.exp(-decision_function_scores))
    probability = probability[0]

   # Interpret the prediction (0: no diabetes, 1: diabetes)
    if prediction == 0:
        result = "Negative. The model predicts you MIGHT NOT have a myocardial infraction or Coronary Heart Disease."
        result2 = "The model predicts that the probability of a heart disease existing is "
    else:
         result = "Positive. The model predicts you MIGHT have a myocardial infraction or Coronary Heart Disease."
         result2 = "The model predicts that the probability of a heart disease existing is "

    prob = round(probability * 100,2)

     # Render the prediction page with the result
    return render_template("prediction.html", result=result, result2=result2, probability=prob)


