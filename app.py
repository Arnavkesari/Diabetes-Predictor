import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(
    page_title="Healthy Life",
    page_icon="👨‍⚕️",
)



st.write("# Welcome to Diabetics Predictor 💉")


df = pd.read_csv('diabetes.csv')

df_minority=df[df['diabetes']==1]
df_majority=df[df['diabetes']==0]
from sklearn.utils import resample
df_minority_upsampled=resample(df_minority,replace=True, #Sample With replacement
         n_samples=len(df_majority),
         random_state=42
        )
df=pd.concat([df_majority,df_minority_upsampled])


# HEADINGS
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

df['bmi'] = df['bmi'].astype(int)
df['age']=df['age'].astype(int)
from pandas.core.dtypes.common import is_numeric_dtype
for column in df.columns:
    if is_numeric_dtype(df[column]):
        continue
    else:
        df[column]=LabelEncoder().fit_transform(df[column])

# X AND Y DATA
X=df.drop('diabetes',axis=1)
y=df["diabetes"]
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  gender = st.sidebar.radio("What is your Gender",["Male", "Female"],index=None,)
  age = st.sidebar.slider('Age', 0,80, 0 )
  hypertension = st.sidebar.slider('Slide if have Hypertension', 0,1,0 )
  heart_disease = st.sidebar.slider('Slide if have Heart_disease', 0,1,0 )
  HbA1c_level = st.sidebar.slider('HbA1c_level',3.5,9.0,3.5)
  blood_glucose_level = st.sidebar.slider('Blood_glucose_level', 80,300, 80 )
  bmi = st.sidebar.slider('BMI', 10,95, 10 )
  smoking_history = st.sidebar.radio("Select Your Smoking History",["never", "No Info","current","former","ever","not current"],index=None,)

  user_report_data = {
      'gender':gender,
      'age':age,
      'hypertension':hypertension,
      'heart_disease':heart_disease,
      'smoking_history':smoking_history,
      'bmi':bmi,
      'HbA1c_level':HbA1c_level,
      'blood_glucose_level':blood_glucose_level
      
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data



# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

if user_data.loc[0,'gender']=='Male':
  user_data.loc[0,'gender']=1
else:
  user_data.loc[0,'gender']=0
  
if user_data.loc[0,'smoking_history']=='never':
  user_data.loc[0,'smoking_history']=4
elif user_data.loc[0,'smoking_history']=='No Info':
  user_data.loc[0,'smoking_history']=0
elif user_data.loc[0,'smoking_history']=='current':
  user_data.loc[0,'smoking_history']=1
elif user_data.loc[0,'smoking_history']=='former':
  user_data.loc[0,'smoking_history']=3
elif user_data.loc[0,'smoking_history']=='ever':
  user_data.loc[0,'smoking_history']=2
else:
  user_data.loc[0,'smoking_history']=5



# MODEL
gb  = GradientBoostingClassifier()
gb.fit(x_train, y_train)
user_result = gb.predict(user_data)




st.subheader('Your Report: ')
output=''
if user_result==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, gb.predict(x_test))*100)+'%')
