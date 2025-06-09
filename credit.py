import pandas as pd
import numpy as np
import streamlit as st
import pickle
import re
from PIL import Image
from streamlit_lottie import st_lottie
import json
st.set_page_config(page_title="Credit Score Status", layout="wide",page_icon=r"C:\Users\anson\Downloads\speedometer.png")
from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    html {
        scroll-behavior: smooth;
    }
    * {
        font-size: 20px !important;
    }
    h1 {
        font-size: 44px !important;
    }
    h2 {
        font-size: 34px !important;
    }
    h3 {
        font-size: 36px !important;
    }
    .stButton>button {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    page = option_menu(
        "Navigation",
        ["Home", "Prediction", "About"],
        icons=["house", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

if page == "Home":
  st.markdown("""
    <div style="text-align:center; padding-bottom:60px;">
        <h1>🏠 Welcome to CreditStatus AI App</h1>
    </div>""", unsafe_allow_html=True)
  col1, col2 = st.columns(2)
  with col1:
    with open("credit_animation.json", "r") as f:
      lottie_data = json.load(f)
    st_lottie(lottie_data, speed=1, height=500, key="credit")
  with col2:
    st.write('An intelligent web app designed to predict your credit score status based on your financial profile. Get instant, personalized feedback powered by machine learning to help you make smarter financial decisions.')
    st.markdown("""
    ### ✨ Key Features:
    - 🔍 Predict credit score status as **Good**, **Standard**, or **Bad**
    - 📊 Easy-to-use input interface
    - ⚡ Instant prediction results with visual feedback
    - 🧠 Powered by a 96% accurate machine learning model
    - 🌐 Deployed using Streamlit for seamless web access""")
    st.markdown("""
    ### 🛠️ How It Works:
    1. Go to the **Prediction** page
    2. Enter your financial information
    3. Click on **Predict**
    4. Get your **Credit Score Status** instantly""")
    
elif page == "Prediction":
  col1,col2,col3 = st.columns([1,2,1])
  with col2:
    with open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\credit.pkl','rb') as obj2:
      dict1 = pickle.load(obj2)
    st.title("Credit Score Status Prediction")
    home_page_image=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\homepage2.png').resize((768,512))
    st.image(home_page_image)
    st.write("Upload your data and get predictions.")
    Num_Bank_Accounts=st.selectbox("🏦Number of bank accounts",range(1,11))
    Num_Credit_Card=st.selectbox('💳Number of credit cards',range(1,11))
    Interest_Rate=st.number_input('％✨Interest rate',1.0,35.0,step=0.1)
    Num_of_Loan=st.select_slider('📄Number of loans',range(1,11))
    Delay_from_due_date=st.number_input('📅⚠️Due date delay',1,70,help='Represents the number of days delayed from the payment date')
    Num_of_Delayed_Payment=st.number_input('💸🚨Number of delayed payment',0,30)
    Changed_Credit_Limit=st.number_input('🔁Changed credit limit',-7.0,37.0,step=0.01,help='Represents the percentage change in credit card limit')
    Outstanding_Debt=st.number_input('💰❗Outstanding debt',10.0,5000.0,help='Represents the remaining debt to be paid')
    Credit_History_Age=st.text_input('🗓️Credit history age',placeholder='e.g., 2 years 6 months')
    year_match=re.search(r'(\d+)\s*[yY]ears?',Credit_History_Age)
    month_match=re.search(r'(\d+)\s*[mM]onths?',Credit_History_Age)
    years=int(year_match.group(1)) if year_match else 0
    months=int(month_match.group(1)) if month_match else 0
    Credit_History_Age=(12*years)+months
    Payment_of_Min_Amount=st.selectbox('✅ / ❌Mininum amount paid or not', ['Yes', 'No'])
    Payment_of_Min_Amount=dict1['Payment_of_Min_Amount'].transform([Payment_of_Min_Amount])[0]
    col1,col2,col3=st.columns([2,1,2])
    with col2:
      button=st.button('🤖Predict')
    if button:
      data=[[Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date,Num_of_Delayed_Payment,Changed_Credit_Limit,Outstanding_Debt,Credit_History_Age,Payment_of_Min_Amount]]
      scaled=dict1['scaler'].transform(data)
      res=dict1['model'].predict(scaled)
      result_label=dict1['output'].inverse_transform([res])[0][0]
      bad_zone=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\bad_zone.png').resize((350,350))
      good_zone=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\good_zone.png').resize((350,350))
      standard_zone=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\stndard_zone.png').resize((350,350))
      if result_label=='Good':
        col1,col2=st.columns([1,1])
        with col1:
          st.image(good_zone)
        with col2:
          st.success("🎉 **Your credit score is GOOD(700-850). You're eligible for better credit offers and lower interest rates.**")
      elif result_label=='Standard':
        col1,col2=st.columns([1,1])
        with col1:
          st.image(standard_zone)
        with col2:
          st.warning("⚠️ **Your score is STANDARD(600-699). Try to reduce debt and pay bills on time to boost it.**")
      else:
        col1,col2=st.columns([1,1])
        with col1:
          st.image(bad_zone)
        with col2:
          st.error("🚨 **Your credit score is BAD(300-599). You may face challenges in loan approvals or interest rates.**")
elif page == "About":
    st.title("About")
    st.subheader('📌 About the Project')
    st.write("This application predicts Credit Score Status (Good, Standard, or Bad) based on multiple financial indicators provided by the user. It aims to help individuals or financial institutions assess the creditworthiness of a user quickly and efficiently using a machine learning model.")
    st.subheader('🧾 Dataset Details')
    st.write('The dataset used in this project was sourced from Kaggle.')
    st.write('Link: https://www.kaggle.com/datasets/parisrohan/credit-score-classification')
    st.write('Before training the model, the dataset underwent preprocessing steps including:')
    st.markdown("""
    - Handled missing values by dropping and filling.
    - Encoding categorical variables (e.g., payment of minimum amount)
    - Scaling numerical features to ensure consistency
    - Feature selection to improve model performance""")
    st.subheader('🔍 Features Considered')
    st.markdown(""" 
    - Number of Bank Accounts  
    - Number of Credit Cards  
    - Interest Rate  
    - Number of Loans  
    - Delay from Due Date  
    - Number of Delayed Payments  
    - Changed Credit Limit  
    - Outstanding Debt  
    - Credit History Age  
    - Payment of Minimum Amount(Yes/No)""")
    st.write("These features represent common financial behaviors and account management practices that impact a person's credit status.")
    st.subheader('🧠 Model Information')
    st.write('The model was trained using:')
    st.markdown("""
    - Algorithm: Random Forest Classifier
    - Libraries: pandas, numpy, scikit-learn, streamlit, regular expression(re), pickle.
    - Accuracy: **96.43%** on the test dataset""")
    st.write('The model was trained and tested in a Jupyter Notebook environment and deployed using Streamlit for interactive web access.')
    st.subheader('✅ Why This Matters')
    st.write('Understanding your credit score status can:')
    st.markdown("""
    - Help you qualify for loans and credit cards
    - Unlock better interest rates
    - Improve your financial planning""")
    st.write('By using this app, users can get an instant evaluation of their credit score status and take informed steps to improve it.')
    st.subheader('⚠️ Disclaimer')
    st.write('This tool is for educational and demonstration purposes only. The predictions should not be used as a substitute for official credit reports from financial institutions or agencies.')
    st.subheader('🧑‍💻 Behind the scene')
    st.write("For detailed reference and access to the complete source code, please visit the GitHub repository linked below.")
    st.markdown("[🐙 Credit score project](https://github.com/Anson369/Credit-card-status-prediction)")
# samples
# Bad - [[6,7,17,5,51,18,9.95,2430.21,226(18y and 10m),1],[10,7,17,7,29,20,15.72,4523.30,123(12y and 3m),1]]
# Standard - [[5,5,20,3,16,19,11,1328.93,238(19y and 10m),1],[8,7,14,5,16,15,1.83,758.44,229(19y and 1m),1]]
# Good - [[1,5,8,3,6,3,2.1,1303.01,222(18y and 6m),0],[2,5,4,1,5,6,1.99,632.46,217(18y and 1m),0]]