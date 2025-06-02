sdjhfbgsjkdfngdfjg
msffnvkjsdnmfklamfkj
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import re
from PIL import Image
import base64
st.set_page_config(page_title="Credit Score Status", layout="wide")
from streamlit_option_menu import option_menu

with st.sidebar:
    page = option_menu(
        "Navigation",
        ["Home", "Prediction", "About"],
        icons=["house", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

if page == "Home":
  st.title("🔹 Welcome to CrediVision: Your Trusted Credit Score Status Prediction Platform")
  home_page=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\home page.png')
  st.image(home_page)
elif page == "Prediction":
  with open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\credit.pkl','rb') as obj2:
    dict1 = pickle.load(obj2)
  st.title("Credit Score Status Prediction")
  st.write("Upload your data and get predictions.")
  Annual_Income=st.number_input('💰Annual Income',5000.0,30000000.0,step=0.01)
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
  button=st.button('🤖Predict')
  if button:
    data=[[Annual_Income,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Delay_from_due_date,Num_of_Delayed_Payment,Changed_Credit_Limit,Outstanding_Debt,Credit_History_Age,Payment_of_Min_Amount]]
    scaled=dict1['scaler'].transform(data)
    res=dict1['model'].predict(scaled)
    result_label=dict1['output'].inverse_transform([res])[0][0]
    bad_zone=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\bad_zone.png').resize((350,350))
    good_zone=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\good_zone.png').resize((350,350))
    standard_zone=Image.open(r'C:\Users\anson\OneDrive\Desktop\Machine Learning\ML project datasets\credit score\stndard_zone.png').resize((350,350))
    if result_label=='Good':
      st.success("🎉 Your credit score is GOOD. You're eligible for better credit offers and lower interest rates.")
      st.image(good_zone)
    elif result_label=='Standard':
      st.warning("⚠️ Your score is STANDARD. Try to reduce debt and pay bills on time to boost it.")
      st.image(standard_zone)
    else:
      st.error("🚨 Your credit score is BAD. You may face challenges in loan approvals or interest rates.")
      st.image(bad_zone)
elif page == "About":
    st.title("About")
    st.write("This app was developed to demonstrate multi-page navigation.")

# samples
# Bad - [[19300.340,6,7,17,5,51,18,9.95,2430.21,226,1],[81093.160,10,7,17,7,29,20,15.72,4523.30,123,1]]
# Standard - [[33751.27,5,5,20,3,16,19,11,1328.93,238,1],[25546.26,8,7,14,5,16,15,1.83,758.44,229,1]]
# Good - [[143162.64,1,5,8,3,6,3,2.1,1303.01,222,0],[30689.89,2,5,4,1,5,6,1.99,632.46,217,0]]

