from types import prepare_class

import streamlit as st
import pandas as pd
import joblib
import sys
import pickle

model = joblib.load('model.pkl')

with open('model_features.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.title('Titanic Survival Prediction')

st.write('This is a simple web app to predict if a passenger survived the Titanic disaster. Funny thing in the csv file only two people are missing from embarking. From deduction you can assume that they were the original people boarding before Jack and his friend won the tickets in poker. But that should still mean that these two people should have been labelled embarked under the previous owner''s name. Hence, we can conclude that Jack and the friend were most likely hallucinations of Rose...')
st.sidebar.header('Write the Passenger Information')

if st.sidebar.button('Stop Application'):
    sys.exit()


def user_input_features():
    pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
    sex= st.sidebar.selectbox('Sex', ['male', 'female'])
    age = st.sidebar.slider('Age', 0.0, 100.0, 20.0)
    sibsp = st.sidebar.number_input('Siblings/Spouses Aboard',0,8,0)
    parch = st.sidebar.number_input('Parents/Children Aboard', 0, 8, 0)
    fare = st.sidebar.number_input('Fare', 0.0, 600.0, 32.0)
    embarked = st.sidebar.selectbox('Port of embarkation', ['S', 'C', 'Q'])
    title = st.sidebar.selectbox('Title', ['Mrs,Miss,Sir'])
    
    data = {
        'Pclass': pclass,
        'Sex': 0 if sex == 'male' else 1,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        f'Embarked_{embarked}': 1,
        **{f'Embarked_{e}': 0 for e in ['C', 'Q', 'S']},
        **{f'Title_{t}': 1 for t in ['Mrs', 'Miss', 'Sir']},
        **{f'Title_{t}': 0 for t in ['Mrs', 'Miss', 'Sir']}
    }
    return pd.DataFrame([data])
    
    
input_df = user_input_features()

input_df= input_df.reindex(columns=feature_names, fill_value=0)
st.subheader('User Input parameters')
st.write(input_df)

prediction = model.predict(input_df)
probability = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('The passenger survived ' if prediction==1 else 'The passenger did not survive')

st.subheader('Probability')
st.write(f'Probability of survival: {probability[0][1]:.2f}')


        
    