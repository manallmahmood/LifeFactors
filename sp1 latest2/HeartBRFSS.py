import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics


data = pd.read_csv(r'/Users/manalmahmood/Desktop/SP1 updated/sp1 latest2/heart_2020_cleaned.csv')

df = data

print(df.columns)
df = df[df.columns].replace({'Yes':1, 'No':0, 'Male':0, 'Female':1,
                             'No, borderline diabetes':'0',
                             'Yes (during pregnancy)':'1' })
df['Diabetic'] = df['Diabetic'].astype(int)

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

df['AgeCatValue'] = df['AgeCategory'].map(map_age_category)
df = df.drop(['AgeCategory'], axis = 1)

RaceMap = {'White': 0, 'Black': 1, 'Hispanic': 2,
            'Asian':3, 'American Indian/Alaskan Native':4,
            'Other':5}
df['Race'] = df['Race'].map(RaceMap)

GenHealthMap = {'Excellent':5, 'Very good': 4, 'Good':3, 'Fair':2, 'Poor':1}
df['GenHealth'] = df['GenHealth'].map(GenHealthMap)

df_unseen = df.tail(70000)
df_unseen_0 = df_unseen[df_unseen['HeartDisease'] == 0].sample(n=500, random_state=42, replace=True)
df_unseen_1 = df_unseen[df_unseen['HeartDisease'] == 1].sample(n=500, random_state=42, replace= True)

#unseen is balanced

#This is survey
df_unseen = pd.concat([df_unseen_0,df_unseen_1], ignore_index = True)



df_0 = df[df['HeartDisease'] == 0].sample(n=4000, random_state=42, replace=True)
df_1 = df[df['HeartDisease'] == 1].sample(n=6000, random_state=42, replace= True)

df = pd.concat([df_0,df_1], ignore_index = True)

print(df.head(5))
print()
print(df_unseen.head(5))
print(df.shape)
print(df_unseen.shape)
print(df.columns)

"""import seaborn as sns
import matplotlib.pyplot as plt
correlation = df.corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'YlOrBr')

sns.set_style('white')
sns.set_palette('YlOrBr')
plt.figure(figsize = (13,6))
plt.title('Distribution of correlation of features')
abs(correlation['HeartDisease']).sort_values()[:-1].plot.barh()
plt.show()
"""

print(df.shape)

df_unseen_features = df_unseen.drop(columns = ['HeartDisease'], axis = 1)
df_unseen_target = df_unseen['HeartDisease']
df_features = df.drop(columns = ['HeartDisease'], axis = 1)
df_target = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(df_features,df_target,shuffle=True, test_size=0.2, random_state =44)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

df_unseen_features = scaler.fit_transform(df_unseen_features)

model_Name = ['RandomForest', 'KNeighbors', 'GaussianNB', 'LogisticRegression', 'GradientBoosting', 'Support vector Machine']

import joblib
from sklearn.model_selection import GridSearchCV


filename = 'trainedmodel.joblib'
print()


rdfModelbase = RandomForestClassifier(random_state=508312)
KNNModel = KNeighborsClassifier(n_neighbors= 3)
gauss_nbModel = GaussianNB()
lrModel = LogisticRegression(solver='saga', max_iter=1000, random_state=508312)
gbModel = GradientBoostingClassifier(random_state=508312)
svmModel = SVC(probability=True)
print()
print("Fitting Model")
parameterRDF = {'max_features':np.arange(2,5,10), 'n_estimators':[500,1000,1500], 'max_depth':[2,4,8,16,32]}
rdfModel = GridSearchCV(rdfModelbase, parameterRDF, cv=5)
rdfModel.fit(X_train,y_train)
KNNModel.fit(X_train,y_train)
gauss_nbModel.fit(X_train,y_train)
lrModel.fit(X_train,y_train)
gbModel.fit(X_train,y_train)
svmModel.fit(X_train,y_train)

print("training and Testing....")
print()

models = [rdfModel, KNNModel, gauss_nbModel, lrModel, gbModel, svmModel]
modelName =['RandomForestClassifier', 'KNearestNeighbors', 'GaussNaiveBayes', 'LogisticRegression', 'GradientBoosting', 'SupportVectorMachine']
train_acc=[]
accuracytable=[]
precision_yes_list = []
precision_no_list = []
correct_percentage_list = []
incorrect_percentage_list = []
acc_array = []
from sklearn.metrics import confusion_matrix, precision_score

for i in range (len(models)):
    trainingacc = accuracy_score(y_train,models[i].predict(X_train))
    acc= accuracy_score(y_test,models[i].predict(X_test))
    
    

    train_acc.append(np.round(trainingacc,5))
    acc_array.append(np.round(acc, 5))
    precision_yes = precision_score(y_test, models[i].predict(X_test), pos_label=1) #precision of data label 1
    precision_no = precision_score(y_test, models[i].predict(X_test), pos_label=0) #precision of data label 0
    confusion_mat = confusion_matrix(y_test, models[i].predict(X_test))
    correct_percentage = (confusion_mat[0, 0] + confusion_mat[1, 1]) / len(y_test) * 100 #True Values/Total Instances
    incorrect_percentage = (confusion_mat[0, 1] + confusion_mat[1, 0]) / len(y_test) * 100 #False Values/Total Instances
    precision_yes_list.append(precision_yes)
    precision_no_list.append(precision_no)
    correct_percentage_list.append(correct_percentage)
    incorrect_percentage_list.append(incorrect_percentage)
    
print()

print("For Heart Dataset using BFRSS data, the accuracy table is: ")
Performance_table = pd.DataFrame({
    'Algorithm': modelName,
    'Precision_Yes': precision_yes_list,
    'Precision_No': precision_no_list,
    'Correctly_Classified%': correct_percentage_list,
    'Incorrectly_Classified%' : incorrect_percentage_list,
    'Accuracy' : acc_array
})

print(Performance_table)

z = 0
for i in range(len(acc_array)):
#    print("Model: ", modelName[i], " Training Accuracy: " , train_acc[i])
#    print("Model: ", modelName[i], " Testing Accuracy:  ", acc_array[i])
    
    if acc_array[i]>z:
        z=acc_array[i]
        highestTest= models[i]

print()
unseen_acc_score = accuracy_score(df_unseen_target, models[i].predict(df_unseen_features))
print("Highest testing accuracy Model in testing is: ")

print(highestTest, " with Accuracy of ", z)
print(highestTest, " has an accuracy score of", unseen_acc_score *100, " on unseen Data")

print()


#if highestTest = RandomForestClassifier(criterion='gini', max_depth=2, min_samples_leaf=3, min_samples_split=2, n_estimators=30)
highestTest.fit(X_train,y_train)
    
from joblib import dump, load
model_filename = "highestTestModelHeart.joblib"
dump(highestTest, model_filename)

def get_user_input():
    input_columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
       'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race',
       'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
       'KidneyDisease', 'SkinCancer', 'AgeCatValue']
    
    user_input = {}

#USER INPUT 

    for column in input_columns:
        print()
        
        if column == 'BMI':
            value = float(input("Enter your BMI: "))
            value = np.round(value,0)
            value = int(value)
            while not isinstance(value, int):
                value = int(input("Please input a valid BMI number: "))
        elif column == 'Sex':
            value = int(input("Enter your gender  (0 for Male and 1 for Female): "))
            while not isinstance(value, int) or value not in [0, 1]:
                value = int(input("Please enter a value 0 for Male and 1 for Female: "))
        elif column == 'Diabetic':
            value = int(input(f"Have you ever been told or diagnosed with diabetes?  ? 0 for NO | 1 for YES "))
            while value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'KidneyDisease':
            value = int(input("Have you ever had a kidney disease? 0 for NO | 1 for YES "))
            while not isinstance(value, int) or value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'SkinCancer':
            value = int(input("Have you ever had Skin Cancer? 0 for NO | 1 for YES "))
            while not isinstance(value, int) or value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'Asthma':
            value = int(input("Do you have Asthma? 0 for NO | 1 for YES "))
            while not isinstance(value, int) or value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'Smoking':
            value = int(input("Have you smoked more than 100 cigarettes in your Life? 0 for NO | 1 for YES"))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'Stroke':
            print(f"Have you ever had a Stroke? 0 for NO | 1 for YES: ")
            while value not in [0, 1]:
                value = int(input("Please enter 0 for NO and 1 for YES: "))
        elif column == 'PhysicalActivity':
            value = int(input("""In the past 30 days, have you been doing physical activity?
                              Not including your job? 0 for NO | 1 for YES"""))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'PhysicalHealth':
            value = int(input("""In the past 30 days, How many days
                              have you been sick?"""))
        elif column == 'AlcoholDrinking':
            print(f"""DO you consume heavy alcoholic drinks per week?
                  more than 14 if youre a MALE
                  more than 7 if youre a FEMALE
                  0 for NO | 1 for YES""")
            while value not in [0, 1]:
                value = int(input("Please enter 0 for NO and 1 for YES: "))
        elif column == 'GenHealth':
            value = int(input("""Rate your General Health :
                              1 = Excellent
                              2 = Very Good
                              3 = Good
                              4 = Fair 
                              5 = Poor
                              Give the rating :"""))
            while value not in range(6):
                value = int(input("Please enter a value from 0 to 5: "))
        elif column == 'MentalHealth':
            print(f"""How many days in the past 30 days, do you think
                  you have had poor mental health""")
            while value not in range(31):
                value = int(input("Please enter a value from 0 to 30: "))
        elif column == 'SleepTime':
            print(f"""How many hours in a day do you sleep?""")
            while value not in range(23):
                value = int(input("Please enter a value from 0 to 22: "))
        elif column == 'Race':
            race = int(input("""What is your Race bacground? 
            'White' 'Black' 'Hispanic'
            'Asian' 'American Indian/Alaskan Native'
            'Other'"""))
            racelist = ['White', 'Black', 'Hispanic',
            'Asian', 'American Indian/Alaskan Native',
            'Other']
            while value not in racelist:
                value = int(input("Please enter valid race"))
            if race == 'White': value = 0 
            elif race =='Black': value = 1 
            elif race == 'Hispanic': value = 2
            elif race == 'Asian': value = 3 
            elif race == 'American Indian/Alaskan Native': value == 4
            elif race == 'Other': value = 5    
        elif column == 'DiffWalk':
            value = int(input("""Do you have serious physical difficulty walking or climbing stairs? 
                              0 for NO | 1 for YES"""))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'AgeCatValue': 
            age = int(input("Enter your Age: "))
            if 18 <= age <= 24:
                value = 1
            elif 25 <= age <= 29:
                value = 2
            elif 30 <= age <= 34:
                value = 3
            elif 35 <= age <= 39:
                value = 4
            elif 40 <= age <= 44:
                value = 5
            elif 45 <= age <= 49:
                value = 6
            elif 50 <= age <= 54:
                value = 7
            elif 55 <= age <= 59:
                value = 8
            elif 60 <= age <= 64:
                value = 9
            elif 65 <= age <= 69:
                value = 10
            elif 70 <= age <= 74:
                value = 11
            elif 75 <= age <= 79:
                value = 12
            else:
                value = 13
        # You might want to add further validation for input values here
        user_input[column] = value

    return user_input




def scaleInput(user_input, scaler):
    scaledInput = scaler.transform([list(user_input.values())])
    return scaledInput


#user_input = get_user_input() #Delete the hashtag and delete the user_input below for manual input

user_input ={'BMI':28.87, 'Smoking':1, 'AlcoholDrinking':0, 
             'Stroke':1,'PhysicalHealth':6, 'MentalHealth':0, 
             'DiffWalking':1, 'Sex':1, 'Race':1,
             'Diabetic':0, 'PhysicalActivity':0, 'GenHealth':3, 
             'SleepTime':12, 'Asthma':0,'KidneyDisease':0, 
             'SkinCancer':0, 'AgeCatValue':12}

scaler = StandardScaler()
scaler.fit(df_features)

scaledInput = scaleInput(user_input, scaler)

def make_prediction(user_input,model):
    # Convert user input to a NumPy array for prediction
    input_array = np.array(list(user_input.values())).reshape(1, -1)
    feature_names = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
       'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race',
       'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
       'KidneyDisease', 'SkinCancer', 'AgeCatValue']
    input_df = pd.DataFrame(input_array, columns=feature_names)
    # Make prediction using the trained model
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]
    
    return prediction, probability

loadedModel = load(model_filename)

if isinstance(loadedModel, SVC):
    feature_names = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
       'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race',
       'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma',
       'KidneyDisease', 'SkinCancer', 'AgeCatValue']
    svm_params = {
        'C': 0.0011,
        'degree': 2,
        'gamma': 'scale',
        'kernel': 'linear'
    }

    # Create the pipeline
    svmPipe = Pipeline([
        ('scalar', StandardScaler()), 
        ('svm', SVC(probability=True, **svm_params))
    ])
# Create the pipeline
    svc =SVC(probability=True, **svm_params)
    svc.fit(X_train[feature_names], y_train)
    # Scale the user input
    user_input_array = np.array([user_input[feature] for feature in feature_names]).reshape(1, -1)
   
     
    # Make prediction
    prediction= svc.predict(user_input_array)
    probability = svc.predict_proba(user_input_array)[:, 1]
    print("Prediction:", prediction)
else:
    print("The loaded model is not an SVC model.")
    prediction,probability = make_prediction(user_input, loadedModel)
   
 
# The 'user_probability' value now contains the predicted probability for having diabetes
# You can present this information to the user
print(f"Probability of having Heart Disease: {probability[0]:.2%}")
 
print("\nPrediction of Heart Disease existing is: ", prediction)
if prediction == 1:
   
 
  print("""Positive. The Model Predicts there might be a heart disease.
         This IS NOT DIRECT MEDICAL ADVICE. PLEASE ADVISE WITH YOUR DOCTOR""")
else:
    print("""Negative. The Model Predicts there might NOT be a heart disease.
          This IS NOT DIRECT MEDICAL ADVICE. PLEASE ADVISE WITH YOUR DOCTOR""")
print("Success")