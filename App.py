import numpy as np
import pandas as pd

import streamlit as st 
import joblib
import pickle

model = joblib.load('ridge_model_ipynb.pkl')

st.set_page_config(layout="wide")


st.title('Rate the Player')


column_name = ['Reactions',
 'Composure',
 'Short passing',
 'Ball control',
 'Heading accuracy',
 'Age',
 'Vision',
 'Standing tackle',
 'Positioning',
 'Marking']

Player_name = st.text_input('Input a player name', 'Kevin')
Reactions = st.slider('Choose Reaction',0,100)
Composure = st.slider('Choose Composure',0,100)
Short_passing = st.slider('Choose Short Passing',0,100)
Ball_control = st.slider('Choose Ball Control',0,100)
Heading_accuracy = st.slider('Choose Heading Accuracy',0,100)
Age = st.slider('Choose Age',15,50)
Vision = st.slider('Choose Vision',0,100)
Standing_tackle = st.slider('Choose Standing Tackle',0,100)
Positioning = st.slider('Choose Positioning',0,100)
Marking = st.slider('Choose Marking',0,100)

def predict():
    row_values = np.array([Reactions,
                            Composure,
                            Short_passing,
                            Ball_control,
                            Heading_accuracy,
                            Age,
                            Vision,
                            Standing_tackle,
                            Positioning,
                            Marking])
    
    input_x = pd.DataFrame([row_values], columns=column_name)
    pred = model.predict(input_x)

    st.success(f"{Player_name} has an estimated overall score of {pred}")

trigger = st.button('Predict', on_click=predict)

