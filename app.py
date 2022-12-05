import numpy as np
import pandas as pd
import catboost as cat
import pickle
import streamlit as st

st.set_page_config(page_title="Video Game Sale Prediction Web App", page_icon='👾')
#leading the saved model
loaded_model = pickle.load(open('game_model.sav','rb')) 

def videogamesale_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return prediction


if __name__ == '__main__':

    #giving a title
    st.title('Video Game Sale Prediction Web App 👾')

    #getting the input data from the user
    platform_options = ['3DS', 'DC', 'DS', 'GBA', 'GC', 'PC', 'PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV', 'Wii', 'WiiU', 'X', 'X360', 'XOne']
    Platform = st.selectbox("Select your Platform:", options = platform_options)
    Year_of_Release = st.slider('Release Date:', min_value=1976, max_value=2017)
    genre_options = ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy']
    Genre = st.selectbox("Select the genre:", options = genre_options)
    publisher_options = ['505 Games', 'Acclaim Entertainment', 'Activision', 'Atari', 'Bethesda Softworks', 'Capcom', 'Codemasters', 'Deep Silver', 'Disney Interactive Studios', 'Eidos Interactive', 'Electronic Arts', 'Konami Digital Entertainment', 'LucasArts', 'Microsoft Game Studios', 'Midway Games', 'Namco Bandai Games', 'Nintendo', 'Nippon Ichi Software', 'Rising Star Games', 'Sega', 'Small Publisher', 'Sony Computer Entertainment', 'Square Enix', 'THQ', 'Take-Two Interactive', 'Tecmo Koei', 'Ubisoft', 'Vivendi Games', 'Warner Bros. Interactive Entertainment']
    Publisher = st.selectbox("Select the Publisher:", options = publisher_options)
    rating_options = ['E', 'E10+', 'M', 'T']
    Critic_Score = st.slider('Score of the game (Critic):', min_value=0, max_value=100)
    User_Score = st.slider('Score of the game (User):', min_value=0.0, max_value=10.0,step=0.1,format="%.2f")
    Rating = st.selectbox("Select the Rating:", options = rating_options)

    #code for Prediction
    sales = ''
    predicted_value = videogamesale_prediction([Year_of_Release,Genre,Critic_Score,Publisher,Platform,User_Score,Rating])

    # creating a button for Prediction
    if st.button('Video Game Sale Prediction'):
        if predicted_value > 0:
            sales = predicted_value
        else:
            sales = 'Too Low'
        st.success('Predicted Global Sales in millions: {0}'.format(sales))

