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

# FUNCTION
def user_report():
  gender = st.sidebar.radio("What is your Gender",["Male", "Female"],index=None,)
  age = st.sidebar.slider('Age', 0,80, 0 )
  hypertension_input = st.sidebar.radio('Do you have Hypertension?', ('Yes', 'No'))
  hypertension = 1 if hypertension_input == 'Yes' else 0
  heart_disease_input = st.sidebar.radio("Do you have Heart Disease?", ("Yes", "No"))
  heart_disease = 1 if heart_disease_input == "Yes" else 0
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

description = st.empty()
with description:
    st.markdown("""
    # Welcome to Diabetes Risk Predictor 💉

    ### Why This Matters?
    Diabetes is a serious health condition affecting millions worldwide. Early detection can be life-changing:

    - 🚨 **Prevention First**: Identify your risk before symptoms develop
    - 🏥 **Avoid Complications**: Early awareness helps prevent serious health issues like:
      - Heart disease
      - Vision problems
      - Kidney damage
      - Nerve damage

    ### How This Tool Helps You:
    - 📊 Uses advanced machine learning to analyze your health data
    - ⚡ Provides instant risk assessment
    - 📈 Shows prediction accuracy
    - 🎯 Helps make informed health decisions

    ### Get Started:
    Enter your health details in the sidebar to receive your personalized diabetes risk assessment. Remember, this tool supports but doesn't replace professional medical advice.
    """)
submitbutton = st.empty()
with submitbutton:
     submit = st.button("Submit")  

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

# PATIENT DATA
user_data = user_report()


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
if submit:
    # Clear the welcome description
    description.empty()
    submitbutton.empty()
    
    if st.button("Back"):
      st.experimental_rerun()
    # Show results in a clean layout
    st.title("Your Diabetes Risk Assessment")
    st.subheader("📋 Your Entered Data")
    
    # Transform user_data to be displayed vertically
    user_data_display = user_data.T.copy()  # Transpose the dataframe and create a copy
    user_data_display.columns = ['Value']  # Rename the column
    
    # Map smoking history numbers back to labels
    smoking_map = {
        4: 'never',
        0: 'No Info',
        1: 'current',
        3: 'former',
        2: 'ever',
        5: 'not current'
    }
    
    # Map binary values to Yes/No for hypertension and heart disease
    binary_map = {1: 'Yes', 0: 'No'}
    binary_map1 = {1: 'Male', 0: 'Female'}
    # Apply mappings to respective columns
    if 'gender' in user_data_display.index:
        user_data_display.loc['gender', 'Value'] = binary_map1[user_data_display.loc['gender', 'Value']]
    if 'smoking_history' in user_data_display.index:
        user_data_display.loc['smoking_history', 'Value'] = smoking_map[user_data_display.loc['smoking_history', 'Value']]
    if 'hypertension' in user_data_display.index:
        user_data_display.loc['hypertension', 'Value'] = binary_map[user_data_display.loc['hypertension', 'Value']]
    if 'heart_disease' in user_data_display.index:
        user_data_display.loc['heart_disease', 'Value'] = binary_map[user_data_display.loc['heart_disease', 'Value']]
    
    # Create a styled display of the data
    st.dataframe(
        user_data_display,
        column_config={
            "Value": st.column_config.Column(
                width="medium"
            )
        },
        hide_index=False
    )
    with st.spinner("Training model, please wait..."):
      gb = GradientBoostingClassifier()
      gb.fit(x_train, y_train)
    # Fit model and get prediction
    user_result = gb.predict(user_data)
    accuracy = accuracy_score(y_test, gb.predict(x_test))*100
    
    st.subheader("🔍 Analysis Results")
    if user_result[0] == 0:
        st.success("**Prediction:** You might not be Diabetic")
    else:
        st.error("**Prediction:** You might be Diabetic!")
    st.metric("Model Accuracy", f"{accuracy:.2f}%")
    
    st.info("⚠️ Note: This is a prediction tool and should not replace professional medical advice.")

