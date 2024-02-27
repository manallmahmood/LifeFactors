from flask import Flask, render_template, request
import joblib
import numpy as np
import os

model_filename = "highestTestModelDiabetes.joblib"
model_directory = r'/Users/manalmahmood/Desktop/SP1 updated/joblib test'
full_path = os.path.join(model_directory, model_filename)
model = joblib.load(full_path)

from app import app

# Define the homepage route
@app.route("/")
def home():
    return render_template("index.html")

# Define the route for the Diabetes home page
@app.route("/diabeteshome.html")
def diabeteshome():
    return render_template("diabeteshome.html")

# Define the route for the Diabetes prediciton page
@app.route("/diabetespred.html")
def diabetespred():
    return render_template("diabetespred.html")

#Define the about route
@app.route("/aboutus.html")
def aboutus():
    return render_template("aboutus.html")

# Define the prediction route
@app.route("/predictdiabetes", methods=["POST"])
def predict_diabetes():
    # Get user input from the form
    highbp = int(request.form["highbp"])
    highchol = int(request.form["highchol"])
    cholcheck = int(request.form["cholcheck"])
    Height = float(request.form["Height"])
    Weight = float(request.form["Weight"])
    heightInM = Height/100
    bmi = Weight/((heightInM)**2) 
    smoker = int(request.form["smoker"])
    stroke = int(request.form["stroke"])
    heartdisease = int(request.form["heartdisease"])
    physactivity = int(request.form["physactivity"])
    veggies = int(request.form["veggies"])
    hvalcohol = int(request.form["hvalcohol"])
    genhlth = int(request.form["genhlth"])
    menthlth = int(request.form["menthlth"])
    physhlth = int(request.form["physhlth"])
    diffwalk = int(request.form["diffwalk"])
    age_discreet = int(request.form["age"])
    age = 0  # Initialize age variable
    if 18 <= age_discreet <= 24:
        age = 1
    elif age_discreet<18:
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
    education = int(request.form["education"])
    income_discreet = int(request.form["income"])
    income = 0  # Initialize income variable
    if income_discreet < 10000:
        income = 1
    elif income_discreet < 15000:
        income = 2
    elif income_discreet < 25000:
        income = 3
    elif income_discreet < 35000:
        income = 4
    elif income_discreet < 50000:
        income = 5
    elif income_discreet < 75000:
        income = 6
    else:
        income = 8 
    # Prepare the user input as a feature vector
    
    
    user_input = {'HighBP' : highbp, 'HighChol': highchol, 'CholCheck':cholcheck, 
           'BMI':bmi, 'Smoker':smoker, 'Stroke':stroke, 
           'HeartDiseaseorAttack':heartdisease, 'PhysActivity':physactivity, 'Veggies':veggies,
           'HvyAlcoholConsump':hvalcohol, 'GenHlth':genhlth, 'MentHlth':menthlth,
           'PhysHlth':physhlth, 'DiffWalk':diffwalk, 'Age':age,
           'Education':education, 'Income':income}
    
    


    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    data = data = pd.read_csv(r'/Users/manalmahmood/Desktop/SP1 updated/sp1 latest2/diabetes.csv')
    df1 = data
    df_0 = df1[df1['Diabetes_binary'] == 0].sample(n=8000, random_state=42, replace=True)
    df_1 = df1[df1['Diabetes_binary'] == 1].sample(n=12000, random_state=42, replace= True)
    df1 = pd.concat([df_0,df_1], ignore_index = True)
    
    df1 = df1.drop('Diabetes_binary', axis=1)
    df1 = df1[['HighBP', 'HighChol', 'CholCheck', 
           'BMI', 'Smoker', 'Stroke', 
           'HeartDiseaseorAttack', 'PhysActivity', 'Veggies',
           'HvyAlcoholConsump', 'GenHlth', 'MentHlth',
           'PhysHlth', 'DiffWalk', 'Age',
           'Education', 'Income']]
    df1= scaler.fit_transform(df1)

    scaledInput = scaler.transform([list(user_input.values())])
    user_input=scaledInput
    
    


    # Make prediction using the model
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)[:,1]

   # Interpret the prediction (0: no diabetes, 1: diabetes)
    if prediction == 0:
        result = "Negative. The model predicts you MIGHT NOT be prediabetic or diabetic."
        result2 = "The model predicts that the probability of you being prediabetic or diabetic is:  "
    else:
         result = "Positive. The model predicts you MIGHT be prediabetic or diabetic."
         result2 = "The model predicts that the probability of you being prediabetic or diabetic is:  "

    prob = round(probability[0] * 100, 2)



     # Render the prediction page with the result
    return render_template("prediction.html", result=result, result2=result2, probability=prob)


