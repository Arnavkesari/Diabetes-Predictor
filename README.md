# Diabetes Risk Predictor 💉

## Overview
The Diabetes Risk Predictor is an intelligent healthcare application that helps identify potential diabetes risk in patients. Using advanced machine learning algorithms, it analyzes various health parameters to provide quick and accurate risk assessments.

## 🌟 Key Features
- **User-Friendly Interface**: Easy-to-use web interface for inputting patient data
- **Real-Time Predictions**: Instant diabetes risk assessment
- **High Accuracy**: Utilizes Gradient Boosting for reliable predictions
- **Data Visualization**: Clear presentation of results and insights
- **Privacy-Focused**: No data storage, all calculations done in real-time

## 📊 Input Parameters
The model considers several key health indicators:
- Gender
- Age
- Hypertension status
- Heart disease history
- Smoking history
- BMI (Body Mass Index)
- HbA1c level
- Blood glucose level

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML Libraries**: 
  - scikit-learn
  - pandas
  - numpy
- **Visualization**: 
  - matplotlib
  - seaborn

## 🚀 Live Demo
Try the live application: [Diabetes Predictor App](https://diabeticspredictor.streamlit.app/)

## 📌 Installation & Local Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Diabetes-Predictor.git
```
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run app.py
```

## 🎯 Model Details
- **Algorithm**: Gradient Boosting Classifier
- **Training Data**: Balanced dataset of diabetic and non-diabetic patients
- **Accuracy**: ~85-90% (varies with data)
- **Validation**: Cross-validation techniques applied

## ⚠️ Disclaimer
This tool is designed for preliminary screening only and should not replace professional medical advice. Always consult healthcare professionals for proper diagnosis and treatment.


