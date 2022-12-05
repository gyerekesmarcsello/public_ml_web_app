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
    publisher_options = ['10TACLE Studios', '1C Company', '2D Boy', '3DO', '505 Games', '506 Games', '5p', '7Sixty LLC', 'AQ Interactive', 'ASCII Entertainment', 'Acclaim Entertainment', 'Ackkstudios', 'Acquire', 'Activision', 'Activision Blizzard', 'Activision Value', 'Agatsuma Entertainment', 'Agetec', 'Aksys Games', 'Alternative Software', 'Aqua Plus', 'Arc System Works', 'Ascaron Entertainment GmbH', 'Aspyr', 'Atari', 'Atlus', 'Avalon Interactive', 'Avanquest', 'BAM! Entertainment', 'Banpresto', 'Bethesda Softworks', 'Bigben Interactive', 'Black Bean Games', 'Black Label Games', 'Blue Byte', 'Bohemia Interactive', 'Brash Entertainment', 'CCP', 'CDV Software Entertainment', 'Capcom', 'Cave', 'City Interactive', 'Cloud Imperium Games Corporation', 'Codemasters', 'Codemasters Online', 'Compile Heart', 'Conspiracy Entertainment', 'Crave Entertainment', 'Crimson Cow', 'D3Publisher', 'DHM Interactive', 'DSI Games', 'DTP Entertainment', 'Deep Silver', 'Destination Software, Inc', 'Destineer', 'Devolver Digital', 'Digital Jesters', 'Disney Interactive Studios', 'DreamCatcher Interactive', 'Dusenberry Martin Racing', 'EA Games', 'ESP', 'Eidos Interactive', 'Electronic Arts', 'Empire Interactive', 'En Masse Entertainment', 'Encore', 'Enix Corporation', 'Enterbrain', 'Evolved Games', 'Falcom Corporation', 'Flashpoint Games', 'Focus Home Interactive', 'Fox Interactive', 'From Software', 'FuRyu', 'FuRyu Corporation', 'Funbox Media', 'Funcom', 'Funsta', 'G.Rev', 'GOA', 'GT Interactive', 'Game Factory', 'Gamebridge', 'Gamecock', 'Gaslamp Games', 'Gathering of Developers', 'Gearbox Software', 'Genki', 'Ghostlight', 'Global A Entertainment', 'Global Star', 'Gotham Games', 'Graffiti', 'Graphsim Entertainment', 'Groove Games', 'GungHo', 'Gust', 'HMH Interactive', 'Harmonix Music Systems', 'Hasbro Interactive', 'Havas Interactive', 'Hello Games', 'Her Interactive', 'Hip Interactive', 'Home Entertainment Suppliers', 'Hudson Entertainment', 'Hudson Soft', 'Human Entertainment', 'Iceberg Interactive', 'Idea Factory', 'Idea Factory International', 'Ignition Entertainment', 'Illusion Softworks', 'Indie Games', 'Infogrames', 'Insomniac Games', 'Interplay', 'Introversion Software', 'Irem Software Engineering', 'Jaleco', 'Jester Interactive', 'JoWood Productions', 'Just Flight', 'Kadokawa Shoten', 'Kalypso Media', 'Kemco', 'Koch Media', 'Konami Digital Entertainment', 'Kool Kizz', 'Level 5', 'Lexicon Entertainment', 'Lighthouse Interactive', 'Little Orbit', 'LucasArts', 'MC2 Entertainment', 'MTV Games', 'Mad Catz', 'Majesco Entertainment', 'Marvelous Entertainment', 'Marvelous Interactive', 'Mastertronic', 'Mastiff', 'Max Five', 'Maxis', 'Media Rings', 'Mercury Games', 'Metro 3D', 'Microids', 'Microsoft Game Studios', 'Midas Interactive Entertainment', 'Midway Games', 'Milestone S.r.l', 'Milestone S.r.l.', 'Milkstone Studios', 'Mindscape', 'Monster Games', 'Monte Christo Multimedia', 'Moss', 'Myelin Media', 'NCSoft', 'NDA Productions', 'NIS America', 'Namco Bandai Games', 'Natsume', 'NaturalMotion', 'Navarre Corp', 'NewKidCo', 'Nihon Falcom Corporation', 'Nintendo', 'Nippon Ichi Software', 'Nobilis', 'Nordic Games', 'NovaLogic', 'Number None', 'O-Games', 'O3 Entertainment', 'Oovee Game Studios', 'Oxygen Interactive', 'P2 Games', 'PM Studios', 'PQube', 'Pacific Century Cyber Works', 'Paradox Interactive', 'Phantagram', 'Phantom EFX', 'Pinnacle', 'Play It', 'Playlogic Game Factory', 'PopCap Games', 'Popcorn Arcade', 'Psygnosis', 'RTL', 'Rage Software', 'Rebellion', 'Rebellion Developments', 'RedOctane', 'Reef Entertainment', 'Revolution Software', 'Rising Star Games', 'Rondomedia', 'Russel', 'SCi', 'SNK', 'Sammy Corporation', 'Scholastic Inc.', 'Screenlife', 'Sega', 'Sierra Entertainment', 'Slightly Mad Studios', 'Sold Out', 'Sony Computer Entertainment', 'Sony Computer Entertainment America', 'Sony Computer Entertainment Europe', 'Sony Online Entertainment', 'SouthPeak Games', 'Spike', 'Square', 'Square EA', 'Square Enix', 'Square Enix ', 'SquareSoft', 'Stainless Games', 'Sting', 'Strategy First', 'Success', 'Sunflowers', 'Sunsoft', 'Swing! Entertainment', 'System 3', 'System 3 Arcade Software', 'TDK Mediactive', 'THQ', 'THQ Nordic', 'Taito', 'Takara Tomy', 'Take-Two Interactive', 'TalonSoft', 'Team Meat', 'Team17 Software', 'Tecmo Koei', 'Telegames', 'Telltale Games', 'Tetris Online', 'The Adventure Company', 'Titus', 'Tomy Corporation', 'TopWare Interactive', 'Touchstone', 'Trion Worlds', 'Tripwire Interactive', 'Tru Blu Entertainment', 'Ubisoft', 'Ubisoft Annecy', 'Universal Interactive', 'Unknown', 'Valcon Games', 'Valve', 'Valve Software', 'Virgin Interactive', 'Visco', 'Vivendi Games', 'Wanadoo', 'Wargaming.net', 'Warner Bros. Interactive Entertainment', 'White Park Bay Software', 'XS Games', 'Xicat Interactive', 'Xplosiv', 'Xseed Games', 'Yacht Club Games', 'Zoo Digital Publishing', 'Zoo Games', 'Zushi Games', 'bitComposer Games', 'id Software', 'inXile Entertainment']
    Publisher = st.selectbox("Select the Publisher:", options = publisher_options)
    rating_options = ['AO', 'E', 'E10+', 'K-A', 'M', 'RP', 'T']
    Critic_Score = st.slider('Score of the game (Critic):', min_value=0, max_value=100)
    User_Score = st.slider('Score of the game (User):', min_value=0.0, max_value=10.0,step=0.1,format="%.2f")
    Rating = st.selectbox("Select the Rating:", options = rating_options)

    #code for Prediction
    sales = ''
    predicted_value = videogamesale_prediction([Genre,Year_of_Release,Critic_Score,Publisher,Platform,Rating,User_Score])

    # creating a button for Prediction
    if st.button('Video Game Sale Prediction'):
        if predicted_value > 0:
            sales = predicted_value
        else:
            sales = 'Too Low'
        st.success('Predicted Global Sales in millions: {0}'.format(sales))

