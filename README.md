**📊 Credit Score Status Prediction**
This project presents a machine learning web application that predicts a user's credit score category (Good, Standard, or Bad) based on financial and personal features. The model was built using a Random Forest Classifier and deployed using Streamlit for real-time predictions.

**🔍 Features**
Data Preprocessing: Handled missing values, removed outliers using the IQR method, and encoded categorical variables.

Exploratory Data Analysis (EDA): Visualized trends and relationships using Seaborn and Matplotlib to gain insights into credit score distributions.

Feature Selection: Applied Recursive Feature Elimination (RFE) to select the most impactful features.

Model Training: Trained a Random Forest Classifier with hyperparameter tuning to achieve a test accuracy of 96.37%.

Deployment: Used pickle to save the trained model and deployed it on a Streamlit-based web app for real-time predictions.

**🚀 Tech Stack**
Python, Pandas, NumPy

Scikit-learn (Random Forest, RFE, Scaling)

Streamlit

Pickle

Google Colab (for development and experimentation)

**📂 How to Run Locally**

- Clone the repository:
git clone https://github.com/your-username/credit-score-prediction.git
cd credit-score-prediction

- Run the Streamlit app:
streamlit run app.py


📌 Use Case
This app can assist banks, NBFCs, and financial institutions to preliminarily assess creditworthiness before full credit reports are requested.

