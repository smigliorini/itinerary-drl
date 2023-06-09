import numpy as np
import pandas as pd
import random
from tensorflow import keras
from tensorflow.keras import layers
from datetime import *
from collections import deque

from PoiEnv import poi_env
from utils import *

#print each path and stats in history and random baseline
print_each_path = False

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
    epsilon = 1
    epsilon_min = 0.01
    score = 0
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

        while done == False:  # check if all actions are done
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

        print(f"Path from episode: {trial}  =  {visited_poi}")
        print("Episode: {:7.0f}, Score: {}, EPS: {:3.2f}".format(trial, score_queue[trial], epsilon))

    print("\n\n")
    print(f"[DRL] The better POI sequence is: {best_journey}")
    print(f"[DRL] Reward: {score_max}")
    print(f"[DRL] Episode number: {best_trial}")

    return neural_network, score_queue, best_journey

#############################################
############### ENV # INFO ##################
#############################################

# Environment input
date_input = datetime(2022, 12, 2, 9, 00)
time_input = 4
poi_start = 300

### Load data
# Latitude, Longitude, Time visit
df_poi_it = pd.read_csv('../data/poi_it.csv', usecols=['id', 'latitude', 'longitude', 'Time_Visit', 'max_crowd'])

# POI occupation grouped by day and time slots
df_crowding = pd.read_csv('../data/log_crowd.csv', usecols=['data', 'val_stim', 'poi']).sort_values(by=['data', 'poi'])

# Time travel in minutes of each pair of poi
df_poi_time_travel = pd.read_csv('../data/poi_time_travel.csv', usecols=['poi_start', 'poi_dest', 'time_travel'])

# weather contidion during 2022 [temperature,rain]
df_weather_2022 = pd.read_csv('../data/Verona 2022-01-01 to 2022-12-31_Precipitazioni.csv')


#############################################
############ HISTORY # BASELINE #############
#############################################

print("\n################## HISTORY ##################\n")

### Load historical data

# Veronacard_2022_opendata contains user visits
df_poi_vr2022 = pd.read_csv('../data/veronacard_2022_original.csv',
                            usecols=['id_veronacard', 'data_visita', 'ora_visita', 'sito_nome']).sort_values(
    ['id_veronacard', 'data_visita', 'ora_visita'])

# Dictionary Poi number - > nome Poi
poi_dict = {'Arena': 49, 'Palazzo della Ragione': 58, 'Casa Giulietta': 61, 'Castelvecchio': 71, 'Teatro Romano': 42,
            'San Fermo': 62,
            'Torre Lamberti': 59, 'Duomo': 52, 'Santa Anastasia': 54, 'San Zeno': 63, 'Museo Storia': 201,
            'Tomba Giulietta': 202, 'Giardino Giusti': 75, 'Museo Lapidario': 76, 'Museo Miniscalchi': 300,
            'Palazzo Maffei Casa Museo': 301, 'Museo Nazionale': 302, 'Eataly Verona': 303}

df_poi_vr2022['poi'] = df_poi_vr2022['sito_nome'].map(poi_dict)

# Filter by date fro experiments
df_poi_vr2022 = df_poi_vr2022.loc[df_poi_vr2022["data_visita"] == '02/12/22']

global_reward = 0
i = 0
global_total_time_visit = 0
global_total_time_distance = 0
global_total_time_crowd = 0
global_time_left = 0
global_time = 0
global_poi_len = 0

# Grouped by Verona card and date of visit
grouped_df_2022 = df_poi_vr2022.groupby(['id_veronacard', 'data_visita'])
# Initialization of the environment
poi_env_2022 = poi_env(date_input, df_poi_it, df_crowding, df_poi_time_travel)
experience_buffer = deque(maxlen=700)

for (group_id, group_date), group_data in grouped_df_2022:

    date_loop = datetime.strptime(group_date, "%d/%m/%y")
    if print_each_path:
        print_date_type(date_loop, df_weather_2022, group_id)
        print( "POI START: " + str( poi_start ) )

    # reset environment for each new user
    state = poi_env_2022.reset(poi_start, timedelta(hours=14), date_input)
    reward_tot = 0
    relative_start_time = 0
    insertable = True
    first_visit_time = None
    last_visit_time = None

    poi_len = 0

    for index, row in group_data.iterrows():
        if print_each_path:
            print(f"Begin visit time: {row['ora_visita']},  POI {row['poi']}")

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
    poi_env_2022.timeleft = timedelta(minutes=0)  # reset time left for each real user, he or she has no more time for visits
    
    total_time_visit, total_time_distance, total_time_crowd, time_left = poi_env_2022.time_stats()
    
        
    global_total_time_visit += total_time_visit
    global_total_time_distance += total_time_distance
    global_total_time_crowd += total_time_crowd
    global_time_left += time_left
    global_time += total_time_distance + total_time_crowd + total_time_visit

    global_reward += reward_tot
    i += 1
    global_poi_len += poi_len
    if print_each_path:
        print("\n")
        print("STATS")
        print(f"[BH] REWARD: {reward_tot}")
        print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left,
                (total_time_visit + total_time_distance + total_time_crowd) / 60, '[BH] ')
        print("\n\n\n\n")

print(f"\n[BH SUM UP] Sum up history baseline:")
print(f"[BH SUM UP] AVG Reward: {global_reward/i}")
print_stats(global_total_time_visit/i, global_total_time_distance/i, global_total_time_crowd/i, global_time_left/i, global_time/i/60, '[BH SUM UP] ')
#print(f"[BH SUM UP] AVG Wasted time: { global_total_time_crowd/global_time * 100 }")
#print(f"[BH SUM UP] TOTAL TIME = {global_time}     TIME WASTED = {global_total_time_crowd}  ")
print(f"[BH SUM UP] POI LEN = {global_poi_len/i}")


###################################################
###################### DQN ########################
###################################################

print("\n#################### DQN ####################\n")

# Weather and time
print_date_type(date_input, df_weather_2022, "admin")

# Initialization of the environment
env = poi_env(date_input,df_poi_it,df_crowding,df_poi_time_travel)
start_state = env.reset(poi_start , timedelta( hours = time_input ) , date_input )

# Initialization of the neural network
neural_network = initialization_dn(layer_size=15, input_layer=20, output_layer=18)

# run DQN algorithm (deep q learning)
neural_network, score, best_journey = DQN(env, neural_network, 300, 32, time_input, poi_start, date_input, experience_buffer)
score
print(f"[DRL] AVG Reward  = {np.array([score]).mean()}")

print("\n#################### TOUR ###################\n")
# Print tour stats
best_journey=  [300, 52, 76, 61]
start_state = env.reset(poi_start , timedelta( hours = time_input ) , date_input )
reward_best = 0
for a in best_journey:
    _ , r, _ = env.step(a)
    reward_best += r
print(f"[BEST] Reward: {reward_best}")
total_time_visit, total_time_distance, total_time_crowd, time_left=env.time_stats()
print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left,time_input, "[BEST] ")

#############################################
########### BASELINE # RANDOM ###############
#############################################

print("\n################## RANDOM ##################\n")

# Initialization of the environment
env_random = poi_env(date_input,df_poi_it,df_crowding,df_poi_time_travel)
start_state = env_random.reset(poi_start , timedelta( hours = time_input ) , date_input)
global_reward = 0
trials_rand_number = 400
global_time = 0
global_total_time_visit = 0
global_total_time_distance = 0
global_total_time_crowd = 0
global_time_left = 0
global_poi_len = 0

# choose one POI to vist randomly untile time is over
for i in range(trials_rand_number):
    time_tot = 0
    done = False
    poi_len = 0
    env_random.reset(poi_start , timedelta( hours = time_input ) ,date_input)
    visited_poi = []
    reward = 0   
    
    while done==False:   # check if all possible actions are done
            a = random.choices(list(env_random.action_space), k=1)[0]
            _ , r, done = env_random.step(a)
            #r_tot += r
            reward += r
            poi_len += 1
            visited_poi.append(a)
    total_time_visit, total_time_distance, total_time_crowd, time_left=env_random.time_stats()

    global_total_time_visit += total_time_visit
    global_total_time_distance += total_time_distance
    global_total_time_crowd += total_time_crowd
    global_time_left += time_left
    global_time += total_time_distance + total_time_crowd + total_time_visit

    global_reward += reward

    global_poi_len += poi_len

    if print_each_path:
        print(f"[BR] Random sequence: {visited_poi}")
        print(f"[BR] Reward: {reward}")
        print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left,
            (total_time_visit + total_time_distance + total_time_crowd) / 60, '[BR] ')
        print("\n\n\n\n")
    

print(f"\n[BR SUM UP] Sum up random baseline:")
print(f"[BR SUM UP] AVG Reward ({trials_rand_number} episodes): {global_reward/trials_rand_number} ")
print_stats(global_total_time_visit/trials_rand_number, global_total_time_distance/trials_rand_number, global_total_time_crowd/trials_rand_number, global_time_left/trials_rand_number, global_time/trials_rand_number/60, '[BR SUM UP] ')
#print(f"[BR SUM UP] AVG Wasted time: { global_total_time_crowd/global_time * 100 }")
#print(f"[BR SUM UP] TOTAL TIME = {global_time}     TIME WASTED = {global_total_time_crowd}  ")
print(f"[BR SUM UP] POI LEN = {global_poi_len/trials_rand_number}")
