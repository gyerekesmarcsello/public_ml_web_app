import numpy as np
import pandas as pd
import catboost as cat
import pickle
import streamlit as st

#leading the saved model
loaded_model = pickle.load(open('game_model.sav','rb')) 

def videogamesale_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)


if __name__ == '__main__':

    #giving a title
    st.title('Video Game Sale Prediction Web App')

    #getting the input data from the user
    platform_options = ['Wii', 'DS', 'X360', 'PS3', 'PS2', 'PS4', '3DS', 'PS', 'X', 'PC',
       'PSP', 'WiiU', 'GC', 'GBA', 'XOne', 'PSV', 'DC']
    Platform = st.selectbox("Select your Platform:", options = platform_options)
    Year_of_Release = st.text_input('Release Date:')
    genre_options = ['Sports', 'Racing', 'Platform', 'Misc', 'Action', 'Puzzle','Shooter', 'Fighting', 'Simulation', 'Role-Playing', 'Adventure','Strategy']
    Genre = st.selectbox("Select the genre:", options = genre_options)
    publisher_options = ['Nintendo', 'Microsoft Game Studios', 'Take-Two Interactive',
       'Sony Computer Entertainment', 'Activision', 'Ubisoft',
       'Electronic Arts', 'Bethesda Softworks', 'SquareSoft',
       'GT Interactive', 'Konami Digital Entertainment', 'Square Enix',
       'Sony Computer Entertainment Europe', 'Virgin Interactive',
       'LucasArts', '505 Games', 'Warner Bros. Interactive Entertainment',
       'Capcom', 'Universal Interactive', 'RedOctane', 'Atari',
       'Eidos Interactive', 'Namco Bandai Games', 'Vivendi Games',
       'MTV Games', 'Sega', 'THQ', 'Disney Interactive Studios',
       'Acclaim Entertainment', 'Midway Games', 'Deep Silver', 'NCSoft',
       'Tecmo Koei', 'Valve Software', 'Infogrames', 'Hello Games',
       'Mindscape', 'Valve', 'Global Star', 'Gotham Games',
       'Crave Entertainment', 'Hasbro Interactive', 'Codemasters',
       'TDK Mediactive', 'Zoo Games', 'Sony Online Entertainment', 'RTL',
       'D3Publisher', 'Black Label Games', 'SouthPeak Games',
       'Zoo Digital Publishing', 'City Interactive', 'Empire Interactive',
       'Atlus', 'Slightly Mad Studios', 'Russel', 'Mastertronic',
       'Play It', 'Tomy Corporation', 'Focus Home Interactive',
       'Game Factory', 'Titus', 'Marvelous Entertainment', 'Genki',
       'TalonSoft', 'Square Enix ', 'SCi', 'Rage Software',
       'Ubisoft Annecy', 'Rising Star Games', 'Enix Corporation',
       'Level 5', 'Koch Media', 'Square EA', 'Touchstone',
       'Nippon Ichi Software', 'Sony Computer Entertainment America',
       'Spike', 'Illusion Softworks', 'Interplay', 'Trion Worlds',
       'Metro 3D', 'Rondomedia', 'Ghostlight', 'Majesco Entertainment',
       'Monster Games', 'Xseed Games', 'PQube', 'Natsume',
       'Ignition Entertainment', 'Kadokawa Shoten',
       'Harmonix Music Systems', 'Square', 'Gamebridge',
       'Midas Interactive Entertainment', 'ASCII Entertainment',
       'System 3 Arcade Software', 'Rebellion', 'Activision Blizzard',
       'Xplosiv', 'Wanadoo', 'NovaLogic', 'BAM! Entertainment',
       'Tetris Online', 'Psygnosis', 'En Masse Entertainment',
       'Screenlife', 'GungHo', 'Jester Interactive', 'Black Bean Games',
       '3DO', 'Takara Tomy', 'Sammy Corporation', 'Kalypso Media',
       'Hudson Soft', 'Marvelous Interactive', 'Arc System Works',
       'Home Entertainment Suppliers', 'Banpresto', 'Wargaming.net',
       'Destineer', 'Unknown', 'FuRyu', 'Pacific Century Cyber Works',
       'PopCap Games', 'Indie Games', 'Nihon Falcom Corporation',
       'Gathering of Developers', 'Oxygen Interactive',
       'DTP Entertainment', 'Sierra Entertainment', 'Milestone S.r.l.',
       'Falcom Corporation', 'Kemco', 'AQ Interactive', 'Telltale Games',
       'Agetec', 'XS Games', 'Activision Value', 'Zushi Games', 'CCP',
       'Agatsuma Entertainment', 'Compile Heart', 'Mad Catz', 'Gust',
       'Media Rings', 'JoWood Productions', 'Mastiff', 'NaturalMotion',
       'Brash Entertainment', 'Funcom', 'Jaleco',
       'Playlogic Game Factory', 'Human Entertainment', 'Fox Interactive',
       '7Sixty LLC', 'Scholastic Inc.', 'System 3', 'Nordic Games',
       'Yacht Club Games', 'White Park Bay Software', '506 Games',
       'NIS America', 'EA Games', 'Acquire', 'Paradox Interactive',
       'Swing! Entertainment', 'Idea Factory', 'Havas Interactive',
       'Hip Interactive', 'Tripwire Interactive', 'Enterbrain', 'Sting',
       'Funsta', 'Tru Blu Entertainment', 'Bigben Interactive',
       'Idea Factory International', 'Moss', 'From Software',
       'NDA Productions', 'PM Studios', 'inXile Entertainment', 'O-Games',
       'Funbox Media', 'Valcon Games', 'Insomniac Games',
       'Bohemia Interactive', 'Aqua Plus', 'Ackkstudios',
       'HMH Interactive', 'Cave', 'Microids', 'Phantom EFX',
       'Evolved Games', 'Aksys Games', 'O3 Entertainment', 'Aspyr',
       'Nobilis', 'Sunsoft', 'DSI Games', 'The Adventure Company',
       'Little Orbit', 'Telegames', 'Dusenberry Martin Racing',
       'Popcorn Arcade', 'Irem Software Engineering', 'Taito',
       'Reef Entertainment', 'Myelin Media', 'Success',
       'Rebellion Developments', 'SNK', 'Avalon Interactive',
       'Revolution Software', 'Gamecock', 'Groove Games',
       'Hudson Entertainment', 'Mercury Games',
       'Ascaron Entertainment GmbH', '1C Company',
       'Destination Software, Inc', 'Gearbox Software', 'Graffiti',
       'Phantagram', 'DreamCatcher Interactive', 'Navarre Corp', 'ESP',
       'Team17 Software', 'Gaslamp Games', 'Max Five',
       'Conspiracy Entertainment', 'FuRyu Corporation', 'Milestone S.r.l',
       'Kool Kizz', 'Monte Christo Multimedia', '5p',
       'Alternative Software', 'Cloud Imperium Games Corporation',
       'Flashpoint Games', 'Sold Out', 'Introversion Software',
       'DHM Interactive', 'Iceberg Interactive', 'Devolver Digital',
       'MC2 Entertainment', '2D Boy', 'Global A Entertainment',
       'Just Flight', 'bitComposer Games', 'Sunflowers', 'id Software',
       'Maxis', 'Pinnacle', 'Xicat Interactive', 'Number None',
       'TopWare Interactive', 'Strategy First', 'Stainless Games',
       'Lexicon Entertainment', 'GOA', 'Avanquest',
       'Graphsim Entertainment', 'Codemasters Online', '10TACLE Studios',
       'Visco', 'Crimson Cow', 'Lighthouse Interactive',
       'CDV Software Entertainment', 'Encore', 'Blue Byte', 'THQ Nordic',
       'NewKidCo', 'Digital Jesters', 'Oovee Game Studios', 'P2 Games',
       'G.Rev', 'Milkstone Studios', 'Her Interactive', 'Team Meat']
    Publisher = st.selectbox("Select the Publisher:", options = publisher_options)
    rating_options = ['E', 'M', 'T', 'E10+', 'AO', 'K-A', 'RP']
    Critic_Score = st.text_input('Score of the game (Critic):')
    User_Score = st.text_input('Score of the game (User):')
    Rating = st.selectbox("Select the Rating:", options = rating_options)

    #code for Prediction
    sales = ''

    # creating a button for Prediction
    if st.button('Video Game Sale Prediction'):
        sales = videogamesale_prediction([Platform,Year_of_Release,Genre,Publisher,Critic_Score,User_Score,Rating])

    st.success('This is a success message!')

    


