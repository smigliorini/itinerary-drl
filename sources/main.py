import holidays
import os
import sys
import numpy as np
import math
import tensorflow as tf
import pandas as pd
import random
from tensorflow import keras
from tensorflow.keras import layers
from datetime import *
from collections import deque

# import di libs
#module_path = os.path.abspath(os.path.join('./libs'))
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)
from PoiEnv import poi_env
from utils import *

# map POI->action
map_from_poi_to_action, map_from_action_to_poi = neural_poi_map()


def initialization_dn(layer_size, input_layer, output_layer):
    model = keras.Sequential()
    model.add(layers.Dense(layer_size, input_dim=input_layer, activation="relu"))  # input layer + 1 hidden layer
    model.add(layers.Dense(layer_size, activation="relu"))  # 2
    model.add(layers.Dense(layer_size, activation="relu"))  # 3
    model.add(layers.Dense(layer_size, activation="relu"))  # 4
    model.add(layers.Dense(layer_size, activation="relu"))  # 5
    # model.add(layers.Dense(layer_size, activation="relu")) #6
    model.add(layers.Dense(output_layer, activation="linear"))  # output layer
    model.compile(loss="mean_squared_error", optimizer='adam')
    return model


def train_model(model, memory, batch_size, poi_set_len, last_set, gamma=0.96):
    size = min(batch_size, len(memory))
    mb = random.sample(memory, size)
    # print(memory[-1])
    # if len(memory)>batch_size:
    # mb.append(last_set)
    # print(f"lastset:{last_set}")
    # print(f"mb:{mb}")
    for [s, a, s_1, r, done, s_act_space, s1_act_space] in mb:
        state = s.reshape(1, poi_set_len + 2)
        target = model.predict(state, verbose=0)
        target = target[0]
        for i in range(poi_set_len):
            if map_from_action_to_poi[i] not in s_act_space:
                target[i] = 0
        if done == True:
            target[a] = r
        else:
            state_1 = s_1.reshape(1, poi_set_len + 2)
            predict_state_1 = model.predict(state_1, verbose=0)[0]
            for i in range(poi_set_len):
                if map_from_action_to_poi[i] not in s1_act_space:
                    predict_state_1[i] = 0
            max_q = max(predict_state_1)
            target[a] = r + max_q * gamma
        model.fit(state, np.array([target]), epochs=15, verbose=0)

    return model


def DQN(environment, neural_network, trials, batch_size, time_input, poi_start, date_input, experience_buffer,
        epsilon_decay=0.997):
    epsilon = 1;
    epsilon_min = 0.01
    score = 0;
    score_queue = []
    # experience_buffer = deque(maxlen=700)
    score_max = 0
    best_journey = []
    best_trial = -1

    for trial in range(trials):
        s = environment.reset(poi_start, timedelta(hours=time_input), date_input)
        s_act_space = environment.action_space.copy()
        done = False
        score = 0
        visited_poi = []

        while done == False:  # controllare se ci sono ancora azioni da fare
            if np.random.random() < epsilon:
                a = random.choices(list(environment.action_space), k=1)[0]

            else:
                state = s.reshape(1, len(environment.poi_set) + 2)
                prediction = neural_network.predict(state, verbose=0)
                for i in range(len(environment.poi_set)):
                    if map_from_action_to_poi[i] not in environment.action_space:
                        prediction[0][i] = -1000000
                a_index = prediction.argmax()
                a = map_from_action_to_poi[a_index]
            visited_poi.append(a)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            s_1, r, done = environment.step(a)
            a = map_from_poi_to_action[a]
            s1_act_space = environment.action_space.copy()
            experience_buffer.append([s, a, s_1, r, done, s_act_space, s1_act_space])
            last_set = [s, a, s_1, r, done, s_act_space, s1_act_space]
            train_model(neural_network, experience_buffer, batch_size, len(environment.poi_set), last_set)
            s = s_1
            score += r
            s_act_space = s1_act_space.copy()

        if score > score_max:
            score_max = score
            best_journey = visited_poi.copy()
            best_trial = trial
        score_queue.append(score)

        # stampo il journey
        print(f"Percorso svolto dall'episode: {trial}  =  {visited_poi}")
        print("Episode: {:7.0f}, Score: {}, EPS: {:3.2f}".format(trial, score_queue[trial], epsilon))

    print("Il percorso migliore è: {best_journey}  con reward: {score_max} all'episodio numero: {best_trial}")
    return neural_network, score_queue, best_journey


### Caricamento dataframes tramite panda

# Latitudine,Longitudine e Tempo di Visita
df_poi_it = pd.read_csv('../data/poi_it.csv', usecols=['id', 'latitude', 'longitude', 'Time_Visit', 'max_crowd'])
# df_poi_it_complete = pd.read_csv('../data/poi_it_complete.csv', usecols= ['id','latitude','longitude','Time_Visit','max_crowd'] ) #ToDo: remove, not used anymore

# Informazioni sull'occupazione dei poi divisi in giorni e fasce orarie
df_crowding = pd.read_csv('../data/log_crowd.csv', usecols=['data', 'val_stim', 'poi']).sort_values(by=['data', 'poi'])
# Distanza in tempo(minuti) per ogni coppia di poi
df_poi_time_travel = pd.read_csv('../data/poi_time_travel.csv', usecols=['poi_start', 'poi_dest', 'time_travel'])

# Apro il csv Veronacard_2022_opendata contenente le visite già fatte degli utenti e creo un dictionary Poi_name->Poi_number
df_poi_vr2022 = pd.read_csv('../data/veronacard_2022_original.csv',
                            usecols=['id_veronacard', 'data_visita', 'ora_visita', 'sito_nome']).sort_values(
    ['id_veronacard', 'data_visita', 'ora_visita'])
# Csv con le condizioni meteo del 2022 [temperatura,precipitazione]
df_weather_2022 = pd.read_csv('../data/Verona 2022-01-01 to 2022-12-31_Precipitazioni.csv')

# Dictionary Poi number - > nome Poi
poi_dict = {'Arena': 49, 'Palazzo della Ragione': 58, 'Casa Giulietta': 61, 'Castelvecchio': 71, 'Teatro Romano': 42,
            'San Fermo': 62,
            'Torre Lamberti': 59, 'Duomo': 52, 'Santa Anastasia': 54, 'San Zeno': 63, 'Museo Storia': 201,
            'Tomba Giulietta': 202, 'Giardino Giusti': 75, 'Museo Lapidario': 76, 'Museo Miniscalchi': 300,
            'Palazzo Maffei Casa Museo': 301, 'Museo Nazionale': 302, 'Eataly Verona': 303}

df_poi_vr2022['poi'] = df_poi_vr2022['sito_nome'].map(poi_dict)

#df_poi_vr2022 = df_poi_vr2022.loc[df_poi_vr2022["data_visita"] == '08/11/22']
df_poi_vr2022 = df_poi_vr2022.loc[df_poi_vr2022["data_visita"] == '28/12/22']

# Inserisco gli input per l'ambiente
date_input = datetime(2022, 12, 28, 10, 00)
time_input = 4
poi_start = 63

reward_globale = 0
i = 0
global_time_wasted_cam = 0
global_time_wasted_cod = 0
global_time = 0
global_poi_len = 0

# Raggruppamento per ogni id_veronacard e data visita
grouped_df_2022 = df_poi_vr2022.groupby(['id_veronacard', 'data_visita'])
# inizializzo l'ambiente
poi_env_2022 = poi_env(date_input, df_poi_it, df_crowding, df_poi_time_travel)
experience_buffer = deque(maxlen=700)

# Ogni gruppo è formato da id veronacard e data visita
for (group_id, group_date), group_data in grouped_df_2022:

    date_loop = datetime.strptime(group_date, "%d/%m/%y")
    print(date_loop)
    print( "POI START: " + str( poi_start ) )
    print_date_type(date_loop, df_weather_2022, group_id)

    # reset dell'ambiente
    state = poi_env_2022.reset(poi_start, timedelta(hours=14), date_input)
    reward_tot = 0
    relative_start_time = 0
    insertable = True
    first_visit_time = None
    last_visit_time = None

    poi_len = 0

    for index, row in group_data.iterrows():
        print(f"Ora visita: {row['ora_visita']},  POI {row['poi']} ")
        poi_len += 1
        if (row['poi'] not in poi_env_2022.explored):

            if first_visit_time is None:
                first_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                relative_start_time = int(first_visit_time.hour * 60 + first_visit_time.minute - (
                        date_input.hour * 60 + date_input.minute + poi_env_2022.calc_distance(
                    row['poi']) + poi_env_2022.crowding_wait(row['poi'])))
                # relative_start_time = 0
                # print(f"relative start time: {relative_start_time}")

                poi_env_2022.state[1] = relative_start_time
            else:
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')

            act_space = poi_env_2022.action_space.copy()
            # print(state)
            # print(act_space)

            new_state, reward, done = poi_env_2022.step(row['poi'])
            if new_state[1] >= time_input * 60:
                insertable = True
            if insertable:
                act_space_2 = poi_env_2022.action_space.copy()
                a = map_from_poi_to_action[row['poi']]
                experience_buffer.append([state, a, new_state, reward, done, act_space, act_space_2])
            state = new_state
            reward_tot += reward
    poi_env_2022.timeleft = timedelta(minutes=0)
    print(f"REWARD: {reward_tot}")
    print("STATS")
    print(f"Tempo viaggio: {last_visit_time - first_visit_time}")
    total_time_visit, total_time_distance, total_time_crowd, time_left = poi_env_2022.time_stats()
    print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left,
                (total_time_visit + total_time_distance + total_time_crowd) / 60)
    global_time_wasted_cam += total_time_distance
    global_time_wasted_cod += total_time_crowd
    global_time += total_time_distance + total_time_crowd + total_time_visit

    if ((total_time_visit + total_time_distance + total_time_crowd) <= 420):
        reward_globale += reward_tot
        i += 1
        global_poi_len += poi_len

    print("\n\n\n\n")
print(f"Experience Buffer: {experience_buffer}")
print(f"Reward Medio: {reward_globale/i}")
print(f"Tempo medio sprecato: { global_time_wasted_cod/global_time * 100 }")
print(f"TEMPO TOTALE = {global_time}     TEMPO SPRECATO= {global_time_wasted_cod}  ")
print(f" POI LEN = {global_poi_len/i}")


###################################################
###################### MAIN #######################
###################################################

# meteo e temperatura
print_date_type(date_input,df_weather_2022,"admin")

# Inizializzo l'ambiente
env=poi_env(date_input,df_poi_it,df_crowding,df_poi_time_travel)
start_state = env.reset(poi_start , timedelta( hours = time_input ) , date_input )

# mappatura poi -> action
map_from_poi_to_action,map_from_action_to_poi = neural_poi_map()

# 20 il numero di neuroni in un layer, 15 è il numero di campi dello stato, 13 è il numero di ouput(POI)
#neural_network = initialization_dn(20,15,13)
neural_network = initialization_dn(15,20,18)

# lancio DQN algoritmo deep q learning
neural_network, score, best_journey = DQN(env, neural_network,1, 32 ,time_input,poi_start,date_input, experience_buffer)
score
print(f"Media Reward  = {np.array([score]).mean()}")

# Stampo le statistiche del tour
best_journey=  [300, 52, 76, 61]
start_state = env.reset(poi_start , timedelta( hours = time_input ) , date_input )
for a in best_journey:
   env.step(a)
total_time_visit, total_time_distance, total_time_crowd, time_left=env.time_stats()
print_stats(189, total_time_distance, 21, time_left,time_input)

# aggiorno il file crowd
# df_crowding = pd.read_csv('data/log_crowd.csv', usecols=['data','val_stim','poi']).sort_values(by=['data','poi'])

# best_journey= [59, 58, 71, 49, 76]
start_state = env.reset(poi_start, timedelta(hours=time_input), date_input)
for a in best_journey:
    current_time = env.current_time()
    if current_time.hour < 12:
        crowd_range = current_time.replace(hour=8, minute=0, second=0)
    elif current_time.hour >= 12 and current_time.hour < 16:
        crowd_range = current_time.replace(hour=12, minute=0, second=0)
    else:
        crowd_range = current_time.replace(hour=16, minute=0, second=0)

    estimated_crowd = df_crowding.loc[(df_crowding['poi'] == a) & (df_crowding['data'] == str(crowd_range))]
    print(estimated_crowd)
    env.step(a)

    if estimated_crowd.empty:
        # print("empty")
        new_row = pd.DataFrame({'data': [str(crowd_range)], 'val_stim': [1], 'poi': [a]})
        df_crowding = pd.concat([df_crowding, new_row], ignore_index=True)
    else:
        df_crowding.loc[(df_crowding['poi'] == a) & (df_crowding['data'] == str(crowd_range)), 'val_stim'] += 1

    estimated_crowd2 = df_crowding.loc[(df_crowding['poi'] == a) & (df_crowding['data'] == str(crowd_range))]
    print(estimated_crowd2)

# df_crowding.to_csv('data/log_crowd.csv', index=False)
# print(df_crowding)

# BaseLine Casuale

# Inizializzo l'ambiente
env_random = poi_env(date_input,df_poi_it,df_crowding,df_poi_time_travel)
start_state = env_random.reset(poi_start , timedelta( hours = time_input ) , date_input)
r_tot = 0
trials_rand_number = 400
global_time = 0
global_time_wasted_cam = 0
global_time_wasted_cod = 0
global_poi_len = 0

# scelgo una serie di POI casuali finche ho tempo per visitarli
for i in range(trials_rand_number):
    done = False
    poi_len = 0
    env_random.reset(poi_start , timedelta( hours = time_input ) ,date_input)
    while done==False:   #controllare prima se ci sono ancora azioni da fare
            a = random.choices(list(env_random.action_space), k=1)[0]
            _ , r, done = env_random.step(a)
            r_tot += r
            poi_len += 1
    total_time_visit, total_time_distance, total_time_crowd, time_left=env_random.time_stats()


    global_time_wasted_cam += total_time_distance
    global_time_wasted_cod += total_time_crowd
    global_time += total_time_visit + total_time_distance + total_time_crowd
    global_poi_len += poi_len

print(f"REWARD MEDIO: {r_tot/trials_rand_number} con {trials_rand_number} Episodi")
global_time_wasted += total_time_distance + total_time_crowd
global_time += total_time_visit + total_time_distance + total_time_crowd
print(f" TEMPO BUTTATO {global_time_wasted_cam / 400} ")
print(f" TEMPO BUTTATO {global_time_wasted_cod / 400} ")

print(f" media poi = {global_poi_len/400} ")

