import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



# Load the dataset (assuming 'data' contains your dataset)
# Replace 'data.csv' with your actual file name or path
data = pd.read_csv(r'C:\Users\User\Desktop\SP1_Input\diabetes_binary_health_indicators_BRFSS2015.csv')
data2 = pd.read_csv(r'C:\Users\User\Desktop\SP1_Input\diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
df1=data
df2 = data2 #This is unseen data later used for prediction accuracy

df_0 = df1[df1['Diabetes_binary'] == 0].sample(n=4000, random_state=42, replace=True)
df_1 = df1[df1['Diabetes_binary'] == 1].sample(n=6000, random_state=42, replace= True)
df1 = pd.concat([df_0,df_1], ignore_index = True)

print(df1.columns)


#Unseen balanced dataset
df_unseen_0 = df2[df2['Diabetes_binary'] == 0].sample(n=400, random_state=42, replace=True)
df_unseen_1 = df2[df2['Diabetes_binary'] == 1].sample(n=400, random_state=42, replace= True)
df_2 = pd.concat([df_unseen_0,df_unseen_1], ignore_index = True)

# Split the data into features and target variable
df_X = data.drop('Diabetes_binary', axis=1)
df_y = data['Diabetes_binary']

print(df1.shape)
print(df2.shape)

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
px_template = "simple_white"

print(data.head())

#Check Duplicates
duplicates = df1[df1.duplicated()]
print("Duplicate Rows : ",len(duplicates))

df1.drop_duplicates(inplace=True)

print(df1.shape)

#Check Continous and Discreet features
#df1.hist(figsize=(15, 15))
#plt.tight_layout()  # Adjust layout for better visualization
#plt.show()

plt.figure(figsize=(20,10))
sns.heatmap(df1.corr(), annot=True)
plt.title("correlation of features")
#plt.show()

correlation = df1.drop('Diabetes_binary', axis=1).corrwith(df1['Diabetes_binary'])
correlation.plot(kind='bar', grid=True, figsize=(20, 8), title='Correlation Bar Graph', color='blue')
plt.ylabel('Correlation')
plt.xlabel('Features')
#plt.show()

# major feature variables for Diabetes are : 
# HIghBP , HighChol , BMI , Stroke , GenHlth , 
# MentHlth , PhysHlth , Age , Eduation and Income.

#Combined Features
# High BP and High Cholestorol

percentageBPchol = df1.groupby(["HighBP", "HighChol"])["Diabetes_binary"].mean() * 100
print(percentageBPchol)

#30% of the patients with Diabets have High BP and High Bood Pressure

#Combined feature of Alcohol and Smoking
percentageSmokeDrink = df1.groupby(["Smoker", "HvyAlcoholConsump"])["Diabetes_binary"].mean() * 100
print(percentageSmokeDrink)

sns.catplot(x="Smoker", y="HvyAlcoholConsump", data=df1, hue="Diabetes_binary", kind="bar")
plt.title("Relation between Smoker, HvyAlcoholConsump, and Diabetes")
#plt.show()  # Display the plot
#Smoking and Alcohol togethere increase the chance of diabetes

percentageStrokeHeart = df1.groupby(["Stroke", "HeartDiseaseorAttack"])["Diabetes_binary"].mean() * 100
print(percentageStrokeHeart)
#Those who have had Stroke and Heart Disease have more chances of diabetes

#LifeStyle Factors

FruitCT= pd.crosstab(df1.Fruits,df1.Diabetes_binary)
VegCT= pd.crosstab(df1.Veggies,df1.Diabetes_binary)
Fitness = pd.crosstab(df1.PhysActivity, df1.Diabetes_binary)


print()
print(FruitCT)
print()
print(VegCT)
print()
print(Fitness)

def iv_woe(data, target, bins=10, show_woe=True):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE (Weight of Evidence) and IV (Information Value) on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        #print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

iv, woe = iv_woe(df1, 'Diabetes_binary')
print()
print()
print(iv.sort_values(by='IV', ascending=False))



from scipy.stats import chi2_contingency

chi_squared_scores={}
for col in df1.columns:
    contingency_table = pd.crosstab(df1[col], df1['Diabetes_binary']) 
    chi2, pval, dof, expected = chi2_contingency(contingency_table)
    chi_squared_scores[col] = chi2

usefulFeatures = [feature for feature, score in chi_squared_scores.items() if score > 250]
print()
print("Useful Features are:")
#for i in (usefulFeatures):
 #   print (i)
    
print(usefulFeatures)


    
#Model Training

def drop_duplicates_and_reset_index(df1):
    dataframe = df1.copy()
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)
    return dataframe

colList = ['HighBP', 'HighChol', 'CholCheck', 
           'BMI', 'Smoker', 'Stroke', 
           'HeartDiseaseorAttack', 'PhysActivity', 'Veggies',
           'HvyAlcoholConsump', 'GenHlth', 'MentHlth',
           'PhysHlth', 'DiffWalk', 'Age',
           'Education', 'Income']

X = df1[colList]
y = df1['Diabetes_binary']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

df2_Xy = df2.drop_duplicates()
df2_y = df2_Xy['Diabetes_binary']
df2_X = df2_Xy[colList]

df2_ScaledX = scaler.transform(df2_X)


#Models to be used and judged:
#Judging according to accuracy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC





X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=508312)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print()
print(y.value_counts(ascending=True))
print()

model_Name = ['RandomForest', 'KNeighbors', 'GaussianNB', 'LogisticRegression', 'GradientBoosting', 'Support vector Machine']

import joblib

filename = 'trainedmodel.joblib'
print()


rdfModelbase = RandomForestClassifier(random_state=508312)
KNNModel = KNeighborsClassifier(n_neighbors= 3)
gauss_nbModel = GaussianNB()
lrModel = LogisticRegression(solver='saga', max_iter=1000, random_state=508312)
gbModel = GradientBoostingClassifier(random_state=508312)
svmModel = SVC()



testscores=[]
trainscores=[]
print()
print("Training Data........")
print()
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

parameterRDF = {'max_features':np.arange(2,5,10), 'n_estimators':[500,1000,1500], 'max_depth':[2,4,8,16,32]}
rdfModel = GridSearchCV(rdfModelbase, parameterRDF, cv=5)
rdfModel.fit(X_train,y_train)
KNNModel.fit(X_train,y_train)
gauss_nbModel.fit(X_train,y_train)
lrModel.fit(X_train,y_train)
gbModel.fit(X_train,y_train)
svmModel.fit(X_train,y_train)

model_List = [rdfModel,KNNModel, gauss_nbModel, lrModel, gbModel, svmModel]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_svm_model = grid_search.best_estimator_



y_predRDF = rdfModel.predict(X_train)
RDFacc_score = accuracy_score(y_train,y_predRDF)
trainscores.append(RDFacc_score)

y_predKNN = KNNModel.predict(X_train)
KNNacc_score = accuracy_score(y_train,y_predKNN)
trainscores.append(KNNacc_score)

y_predGauss = gauss_nbModel.predict(X_train)
Gaussacc_score = accuracy_score(y_train,y_predGauss)
trainscores.append(Gaussacc_score)

y_predLR = lrModel.predict(X_train)
LRacc_score = accuracy_score(y_train,y_predLR)
trainscores.append(LRacc_score)


y_predGB = gbModel.predict(X_train)
GBacc_score = accuracy_score(y_train,y_predGB)
trainscores.append(GBacc_score)

y_predsvm = best_svm_model.predict(X_train)
svmacc_score = accuracy_score(y_train,y_predsvm)
trainscores.append(svmacc_score)

print()
print("Testing Data Learning............")
print()

from sklearn.metrics import precision_score, confusion_matrix

algorithm_list = []
precision_yes_list = []
precision_no_list = []
correct_percentage_list = []
incorrect_percentage_list = []

y_testpredRDF = rdfModel.predict(X_test)
RDFtestacc_score = accuracy_score(y_test,y_testpredRDF)
RDFtestrmse = np.sqrt(mean_squared_error(y_test, y_testpredRDF))
RDFprecision_yes = precision_score(y_test, y_testpredRDF, pos_label=1) #precision of data label 1
RDFprecision_no = precision_score(y_test, y_testpredRDF, pos_label=0) #precision of data label 0

RDFconfusion_mat = confusion_matrix(y_test, y_testpredRDF)
RDFcorrect_percentage = (RDFconfusion_mat[0, 0] + RDFconfusion_mat[1, 1]) / len(y_test) * 100 #True Values/Total Instances
RDFincorrect_percentage = (RDFconfusion_mat[0, 1] + RDFconfusion_mat[1, 0]) / len(y_test) * 100 #False Values/Total Instances

print()
print("Accuracy of RDF ", RDFtestacc_score)
print()
testscores.append(RDFtestacc_score)
algorithm_list.append('RandomForestClassifier')
precision_yes_list.append(RDFprecision_yes)
precision_no_list.append(RDFprecision_no)
correct_percentage_list.append(RDFcorrect_percentage)
incorrect_percentage_list.append(RDFincorrect_percentage)

y_testpredKNN = KNNModel.predict(X_test)
KNNtestacc_score = accuracy_score(y_test,y_testpredKNN)


KNNprecision_yes = precision_score(y_test, y_testpredKNN, pos_label=1)
KNNprecision_no = precision_score(y_test, y_testpredKNN, pos_label=0)

KNNconfusion_mat = confusion_matrix(y_test, y_testpredKNN)
KNNcorrect_percentage = (KNNconfusion_mat[0, 0] + KNNconfusion_mat[1, 1]) / len(y_test) * 100
KNNincorrect_percentage = (KNNconfusion_mat[0, 1] + KNNconfusion_mat[1, 0]) / len(y_test) * 100

print()
print("Accuracy of KNN ", KNNtestacc_score)
print()
testscores.append(KNNtestacc_score)

algorithm_list.append('K-Nearest Neighbors')
precision_yes_list.append(KNNprecision_yes)
precision_no_list.append(KNNprecision_no)
correct_percentage_list.append(KNNcorrect_percentage)
incorrect_percentage_list.append(KNNincorrect_percentage)

y_testpredGauss = gauss_nbModel.predict(X_test)
Gausstestacc_score = accuracy_score(y_test,y_testpredGauss)


Gaussprecision_yes = precision_score(y_test, y_testpredGauss, pos_label=1)
Gaussprecision_no = precision_score(y_test, y_testpredGauss, pos_label=0)

Gaussconfusion_mat = confusion_matrix(y_test, y_testpredGauss)
Gausscorrect_percentage = (Gaussconfusion_mat[0, 0] + Gaussconfusion_mat[1, 1]) / len(y_test) * 100
Gaussincorrect_percentage = (Gaussconfusion_mat[0, 1] + Gaussconfusion_mat[1, 0]) / len(y_test) * 100

print()
print("Accuracy of Gaussian ", Gausstestacc_score)
print()
testscores.append(Gausstestacc_score)

algorithm_list.append('Gaussian Naive Bayes')
precision_yes_list.append(Gaussprecision_yes)
precision_no_list.append(Gaussprecision_no)
correct_percentage_list.append(Gausscorrect_percentage)
incorrect_percentage_list.append(Gaussincorrect_percentage)

y_testpredLR = lrModel.predict(X_test)
LRtestacc_score = accuracy_score(y_test,y_testpredLR)


LRprecision_yes = precision_score(y_test, y_testpredLR, pos_label=1)
LRprecision_no = precision_score(y_test, y_testpredLR, pos_label=0)

LRconfusion_mat = confusion_matrix(y_test, y_testpredLR)
LRcorrect_percentage = (LRconfusion_mat[0, 0] + LRconfusion_mat[1, 1]) / len(y_test) * 100
LRincorrect_percentage = (LRconfusion_mat[0, 1] + LRconfusion_mat[1, 0]) / len(y_test) * 100

print()
print("Accuracy of Logistic Regression ", LRtestacc_score)
print()
testscores.append(LRtestacc_score)

algorithm_list.append('Logistic Regression')
precision_yes_list.append(LRprecision_yes)
precision_no_list.append(LRprecision_no)
correct_percentage_list.append(LRcorrect_percentage)
incorrect_percentage_list.append(LRincorrect_percentage)

y_testpredGB = gbModel.predict(X_test)
GBtestacc_score = accuracy_score(y_test,y_testpredGB)


GBprecision_yes = precision_score(y_test, y_testpredGB, pos_label=1)
GBprecision_no = precision_score(y_test, y_testpredGB, pos_label=0)

GBconfusion_mat = confusion_matrix(y_test, y_testpredGB)
GBcorrect_percentage = (GBconfusion_mat[0, 0] + GBconfusion_mat[1, 1]) / len(y_test) * 100 
GBincorrect_percentage = (GBconfusion_mat[0, 1] + GBconfusion_mat[1, 0]) / len(y_test) * 100 

print()
print("Accuracy of Gradient Boosting", GBtestacc_score)
print()
testscores.append(GBtestacc_score)


algorithm_list.append('Gradient Boost Classifier')
precision_yes_list.append(GBprecision_yes)
precision_no_list.append(GBprecision_no)
correct_percentage_list.append(GBcorrect_percentage)
incorrect_percentage_list.append(GBincorrect_percentage)

y_testpredsvm = best_svm_model.predict(X_test)
svmtestacc_score = accuracy_score(y_test,y_testpredsvm)


svmprecision_yes = precision_score(y_test, y_testpredsvm, pos_label=1)
svmprecision_no = precision_score(y_test, y_testpredsvm, pos_label=0)

svmconfusion_mat = confusion_matrix(y_test, y_testpredsvm)
svmcorrect_percentage = (svmconfusion_mat[0, 0] + svmconfusion_mat[1, 1]) / len(y_test) * 100
svmincorrect_percentage = (svmconfusion_mat[0, 1] + svmconfusion_mat[1, 0]) / len(y_test) * 100

print()
print("Accuracy of SVM", svmtestacc_score)
print()
testscores.append(svmtestacc_score)

algorithm_list.append('Support Vector Machine')
precision_yes_list.append(svmprecision_yes)
precision_no_list.append(svmprecision_no)
correct_percentage_list.append(svmcorrect_percentage)
incorrect_percentage_list.append(svmincorrect_percentage)


print(len(testscores))

print("On Training: ")
j=0
a=str
b=str
for i in range(len(trainscores)):
    print(model_Name[i], end='\t')
    print("Accuracy: ", trainscores[i])
    if trainscores[i]>j:
        j=trainscores[i]
        a=model_Name[i]
        indNum=i
print(j)
print()

k=0
print("On Testing: ")
print()
for z in range(len(testscores)):
    print(model_Name[z], end='\t')
    print("Accuracy: ", testscores[z])
    if testscores[z]>k:
        k=testscores[z]
        b=model_Name[z]
        highestModel = model_List[z]
print()
print()
print("Highest accuracy for TRAINING is: ", a, " at Accuracy score: ", j*100, "%")        
print("Highest accuracy for TESTING is: ", b, " at Accuracy score: ", k*100, "%")

Performance_table = pd.DataFrame({
    'Algorithm': algorithm_list,
    'Precision_Yes': precision_yes_list,
    'Precision_No': precision_no_list,
    'Correctly_Classified%': correct_percentage_list,
    'Incorrectly_Classified%' : incorrect_percentage_list,
    'Accuracy': testscores
})



print()
print(Performance_table)
print()


filename = 'trainedmodel.joblib'
joblib.dump(highestModel,filename)


y_pred = joblib.load(filename).predict(df2_ScaledX)
unseen_acc_score = accuracy_score(df2_y,y_pred)
unseen_rmse = np.sqrt(mean_squared_error(df2_y, y_pred))

print("On Unseen Data: ", joblib.load(filename), " model has an accuracy of: ", unseen_acc_score*100, "%")
print("And a RMSE of: ", unseen_rmse)

print("Success")

print()
print("For User Input: ")

def get_user_input():
    import numpy as np
    input_columns = ['HighBP', 'HighChol', 'CholCheck', 
           'BMI', 'Smoker', 'Stroke', 
           'HeartDiseaseorAttack', 'PhysActivity', 'Veggies',
           'HvyAlcoholConsump', 'GenHlth', 'MentHlth',
           'PhysHlth', 'DiffWalk', 'Age',
           'Education', 'Income']

    user_input = {}

    for column in input_columns:
        print()
        if column == 'HighBP':
            value = int(input(f"Do you have High Blood Pressure? 0 for NO | 1 for YES "))
            while value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'HighChol':
            value = int(input(f"Do you have High Cholesterol? 0 for NO | 1 for YES "))
            while value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'CholCheck':
            value = int(input("Have you Checked your cholesterol in the last 5 years? 0 for NO | 1 for YES "))
            while not isinstance(value, int) or value not in [0, 1]:
                value = int(input("Please enter a value 0 for NO and 1 for YES: "))
        elif column == 'BMI':
            value = float(input("Enter your BMI: "))
            value = np.round(value,0)
            value = int(value)
            while not isinstance(value, int):
                value = int(input("Please input a valid Cholesterol number: "))
        elif column == 'Smoker':
            value = int(input("Have you smoked more than 100 cigarettes in your Life? 0 for NO | 1 for YES"))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'Stroke':
            print(f"Have you ever had a Stroke? 0 for NO | 1 for YES: ")
            while value not in [0, 1]:
                value = int(input("Please enter 0 for NO and 1 for YES: "))
        elif column == 'HeartDiseaseorAttack':    
            value = int(input("Have you ever had a Coronory Heard Disease or Myocardial Infraction? 0 for NO | 1 for YES: "))
            while value not in [0, 1]:
                value = int(input("Please enter 0 for NO and 1 for YES: "))
        elif column == 'PhysActivity':
            value = int(input("""In the past 30 days, have you been doing physical activity?
                              Not including your job? 0 for NO | 1 for YES"""))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'Veggies':
            value = float(input("""Do you consume vegetables once or more than once per day? 
                                0 for NO | 1 for YES"""))
            while value not in [0, 1]:
                value = int(input("Please enter 0 for NO and 1 for YES: "))
        elif column == 'HvyAlcoholConsump':
            print(f"""DO you consume heavy alcoholic drinks per week?
                  more than 14 if youre a MALE
                  more than 7 if youre a FEMALE
                  0 for NO | 1 for YES""")
            while value not in [0, 1]:
                value = int(input("Please enter 0 for NO and 1 for YES: "))
        elif column == 'GenHlth':
            value = int(input("""Rate your General Health :
                              1 = Excellent
                              2 = Very Good
                              3 = Good
                              4 = Fair 
                              5 = Poor
                              Give the rating :"""))
            while value not in range(6):
                value = int(input("Please enter a value from 0 to 5: "))
        elif column == 'MenHlth':
            print(f"""How many days in the past 30 days, do you think
                  you have had poor mental health""")
            while value not in range(31):
                value = int(input("Please enter a value from 0 to 30: "))
        elif column == 'DiffWalk':
            value = int(input("""Do you have serious physical difficulty walking or climbing stairs? 
                              0 for NO | 1 for YES"""))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'Age': 
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
        elif column == 'Education':
            education_level = int(input("""Enter your education level based on the scale: 
                                        1 : Never attended school or only kindergarten
                                        2 : Attended Elementary
                                        3 : Attended High School
                                        4 : Attended College, no degree
                                        5 : Bachelors Degree
                                        6 : Advanced Degree
                                        Provide your Education Level: """))
            value  = education_level
            while education_level not in range(7):
                value = int(input("Enter a valid education level: "))
        elif column == 'Income': 
            income = int(input("Enter your Income: "))
            if income < 10000:
                value = 1
            elif income < 15000:
                value = 2
            elif income < 25000:
                value = 3
            elif income < 35000:
                value = 4
            elif income < 50000:
                value = 5
            elif income < 75000:
                value = 6
            else:
                value = 8      
        # You might want to add further validation for input values here
        user_input[column] = value

    return user_input


user_input = {'HighBP' : 1, 'HighChol': 1, 'CholCheck':1, 
           'BMI':40, 'Smoker':1, 'Stroke':0, 
           'HeartDiseaseorAttack':0, 'PhysActivity':0, 'Veggies':1,
           'HvyAlcoholConsump':0, 'GenHlth':5, 'MentHlth':18,
           'PhysHlth':15, 'DiffWalk':1, 'Age':1,
           'Education':4, 'Income':3}


def scaleInput(user_input, scaler):
    scaledInput = scaler.transform([list(user_input.values())])
    return scaledInput

scaledInput = scaleInput(user_input, scaler)

loaded_model = joblib.load(filename)

y_pred_user = loaded_model.predict(scaledInput)
if y_pred_user[0] == 0:
    user_probability = loaded_model.predict_proba(scaledInput)[:, 0]
else:
    user_probability = loaded_model.predict_proba(scaledInput)[:, 1]

# The 'user_probability' value now contains the predicted probability for having diabetes
# You can present this information to the user
print(f"Probability of having diabetes: {user_probability[0]:.2%}")

print("\nPrediction of Diabetes existing is: ", y_pred_user[0])
if y_pred_user == 1: 
    

  print("""Positive. The Model Predicts you MIGHT have DIABETES.
         This IS NOT DIRECT MEDICAL ADVICE. PLEASE ADVISE WITH YOUR DOCTOR""")
else:
    print("""Negative. The Model Predicts there MIGHT NOT have diabetes.
          This IS NOT DIRECT MEDICAL ADVICE. PLEASE ADVISE WITH YOUR DOCTOR""")

print(confusion_matrix(y_test, y_testpredGB))