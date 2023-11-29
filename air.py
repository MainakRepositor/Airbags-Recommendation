import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset (assuming the dataset is in a CSV file)
# Replace 'your_dataset.csv' with the actual file name and path
data = pd.read_csv('airbag_recommendation.csv')

# Split data into features and target
X = data.drop('result', axis=1)
y = data['result']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Airbag Recommendation System')

st.write("""
Enter the car parameters to get airbag recommendation:
""")

# Input fields for car parameters
car_weight = st.slider('Car Weight', min_value=1000, max_value=5000, value=2500)
car_length = st.slider('Car Length', min_value=100, max_value=300, value=150)
bonnet_length = st.slider('Bonnet Length', min_value=50, max_value=200, value=100)
dashboard_strength = st.slider('Dashboard Strength', min_value=1, max_value=10, value=5)
max_speed = st.slider('Max Speed', min_value=50, max_value=200, value=100)
deceleration_time = st.slider('Deceleration Time', min_value=1, max_value=10, value=5)
response_time = st.slider('Response Time', min_value=10, max_value=50, value=25)
impact_force = st.slider('Impact Force', min_value=100, max_value=1000, value=500)

# Predict function
def predict_airbag():
    features = [[car_weight, car_length, bonnet_length, dashboard_strength, max_speed, 
                 deceleration_time, response_time, impact_force]]
    prediction = model.predict(features)
    return prediction

# Get recommendation
if st.button('Get Recommendation'):
    prediction = predict_airbag()
    if prediction[0] == 4:
        st.info('Recommendation: Install 2 front airbag.')
    elif prediction[0] == 3:
        st.warning('Recommendation: Install 2 front + 2 side airbag.')
    elif prediction[0] == 2:
        st.warning('Recommendation: Install 2 front + 4 side airbag.')
    elif prediction[0] == 1:
        st.error('Recommendation: Install 2 front + 4 side + 2 rear airbag.')
    else:
        st.success('Recommendation: Be extra precautious while drive and use proper air-bags')
