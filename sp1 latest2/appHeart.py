from flask import Flask, render_template, request
import joblib
import numpy as np
import os

model_filename = "highestTestModelHeart.joblib"
model_directory = r'/Users/manalmahmood/Desktop/SP1 updated/joblib test'
full_path = os.path.join(model_directory, model_filename)
model = joblib.load(full_path)

from app import app



# Define the route for the Diabetes prediciton page
@app.route("/heartpred.html")
def heartpred():
    return render_template("heartpred.html")


# Define the route for the Diabetes prediciton page
@app.route("/hearthome.html")
def hearthome():
    return render_template("hearthome.html")

def map_age_category(age_range):
    if age_range == '18-24':
        return 1
    elif age_range == '25-29':
        return 2
    elif age_range == '30-34':
        return 3
    elif age_range == '35-39':
        return 4
    elif age_range == '40-44':
        return 5
    elif age_range == '45-49':
        return 6
    elif age_range == '50-54':
        return 7
    elif age_range == '55-59':
        return 8
    elif age_range == '60-64':
        return 9
    elif age_range == '65-69':
        return 10
    elif age_range == '70-74':
        return 11
    elif age_range == '75-79':
        return 12
    else:
        return 13
    
@app.route("/predictheart", methods=["POST"])
def predict_heart():
    # Get user input from the form
    Sex = int(request.form["Sex"])
    Diabetic = int(request.form["Diabetic"])
    KidneyDisease = int(request.form["KidneyDisease"])
    Height = float(request.form["Height"])
    Weight = float(request.form["Weight"])
    heightInM = Height/100
    BMI = Weight/((heightInM)**2)  
    SkinCancer = int(request.form["SkinCancer"])
    Asthma = int(request.form["Asthma"])
    Smoking = int(request.form["Smoking"])
    PhysicalActivity = int(request.form["PhysicalActivity"])
    Stroke = int(request.form["Stroke"])
    AlcoholDrinking = int(request.form["AlcoholDrinking"])
    GenHealth = int(request.form["GenHealth"])
    MentalHealth = int(request.form["MentalHealth"])
    PhysicalHealth = int(request.form["PhysicalHealth"])
    DiffWalking = int(request.form["DiffWalk"])
    age_discreet = int(request.form["AgeCatValue"])
    age = 0  # Initialize age variable
    if 18 <= age_discreet <= 24:
        age = 1
    elif 25 <= age_discreet <= 29:
        age = 2
    elif 30 <= age_discreet <= 34:
        age = 3
    elif 35 <= age_discreet <= 39:
        age = 4
    elif 40 <= age_discreet <= 44:
        age = 5
    elif 45 <= age_discreet <= 49:
        age = 6
    elif 50 <= age_discreet <= 54:
        age = 7
    elif 55 <= age_discreet <= 59:
        age = 8
    elif 60 <= age_discreet <= 64:
        age = 9
    elif 65 <= age_discreet <= 69:
        age = 10
    elif 70 <= age_discreet <= 74:
        age = 11
    elif 75 <= age_discreet <= 79:
        age = 12
    else:
        age = 13
    Race = int(request.form["Race"])
    SleepTime = int(request.form["SleepTime"])
    # Prepare the user input as a feature vector
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    
    data = pd.read_csv(r'/Users/manalmahmood/Desktop/SP1 updated/sp1 latest2/heart_2020_cleaned.csv')
    
    df= data
    df = df[df.columns].replace({'Yes':1, 'No':0, 'Male':0, 'Female':1,
                             'No, borderline diabetes':'0',
                             'Yes (during pregnancy)':'1' })
    RaceMap = {'White': 0, 'Black': 1, 'Hispanic': 2,
            'Asian':3, 'American Indian/Alaskan Native':4,
            'Other':5}
    df['Race'] = df['Race'].map(RaceMap)
    GenHealthMap = {'Excellent':5, 'Very good': 4, 'Good':3, 'Fair':2, 'Poor':1}
    df['GenHealth'] = df['GenHealth'].map(GenHealthMap)
    df['AgeCatValue'] = df['AgeCategory'].map(map_age_category)
    df = df.drop(['AgeCategory'], axis = 1)
    df_features = df.drop(columns = ['HeartDisease'], axis = 1)
    
    df_features = scaler.fit_transform(df_features)
    


    
    user_input = {'BMI':BMI, 'Smoking':Smoking, 'AlcoholDrinking':AlcoholDrinking, 
             'Stroke':Stroke,'PhysicalHealth':PhysicalHealth, 'MentalHealth':MentalHealth, 
             'DiffWalking':DiffWalking, 'Sex':Sex, 'Race':Race,
             'Diabetic':Diabetic, 'PhysicalActivity':PhysicalActivity, 'GenHealth':GenHealth, 
             'SleepTime':SleepTime, 'Asthma':Asthma,'KidneyDisease':KidneyDisease, 
             'SkinCancer':SkinCancer, 'AgeCatValue':age}
    
    scaledInput = scaler.transform([list(user_input.values())])
    user_input=scaledInput
    
    

    # Make prediction using the model
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[:,1]

   # Interpret the prediction (0: no diabetes, 1: diabetes)
    if prediction == 0:
        result = "Negative. The model predicts you MIGHT NOT have a myocardial infraction or Coronary Heart Disease."
        result2 = "The model predicts that the probability of a heart disease existing is "
    else:
         result = "Positive. The model predicts you MIGHT have a myocardial infraction or Coronary Heart Disease."
         result2 = "The model predicts that the probability of a heart disease existing is "

    prob = round(probability[0]*100, 2)
    


     # Render the prediction page with the result
    return render_template("prediction.html", result=result, result2=result2, probability=prob)

