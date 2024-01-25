import numpy as np
import pandas as pd
import random
from tensorflow import keras
from tensorflow.keras import layers
from datetime import *
import time
from collections import deque

from PoiEnv import poi_env
from utils import *

#############################################
################ LOAD # DATA ################
#############################################

# Latitude, Longitude, Time visit
df_poi_it = pd.read_csv('../data/poi_it.csv', usecols=['id', 'latitude', 'longitude', 'Time_Visit', 'max_crowd'])

# POI occupation grouped by day and time slots
df_crowding = pd.read_csv('../data/log_crowd.csv', usecols=['data', 'val_stim', 'poi']).sort_values(by=['data', 'poi'])

# Time travel in minutes of each pair of poi
df_poi_time_travel = pd.read_csv('../data/poi_time_travel.csv', usecols=['poi_start', 'poi_dest', 'time_travel'])

# weather contidion during 2022 [temperature,rain]
df_weather = pd.read_csv('../data/weather_2022_processed.csv',usecols=['date','temp','rain'])
df_weather_test = pd.read_csv('../data/weather_2023_processed.csv',usecols=['date','temp','rain'])

# data_all contains user visits
df_poi_train = pd.read_csv('../data/data_train.csv',
                            usecols=["id_veronacard","profilo","data_visita","ora_visita","sito_nome","poi"]).sort_values(
    ['id_veronacard', 'data_visita', 'ora_visita'])

df_poi_test = pd.read_csv('../data/data_2023.csv',
                            usecols=["id_veronacard","profilo","data_visita","ora_visita","sito_nome","poi"]).sort_values(
    ['id_veronacard', 'data_visita', 'ora_visita'])

# POI popularity 
df_poi_popularity = pd.read_csv('../data/poi_popularity_train.csv', usecols=['poi', 'popularity','position'])
df_poi_popularity_context = pd.read_csv('../data/poi_popularity_ctx_train.csv', usecols=['temp','rain','poi','popularity','position'])

df_poi_popularity_test = pd.read_csv('../data/poi_popularity_2023.csv', usecols=['poi', 'popularity','position'])
df_poi_popularity_context_test = pd.read_csv('../data/poi_popularity_ctx_2023.csv', usecols=['temp','rain','poi','popularity','position'])

popular_poi = df_poi_popularity.sort_values(by=['popularity'], ascending=False)['poi'].values[:3]

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


def DQN(environment, neural_network, trials, batch_size, time_input, poi_start, date_input, 
        experience_buffer, epsilon_decay=0.997):
    
    if len(experience_buffer) > 100:
        experience_buffer = random.sample(experience_buffer, 100)
    epsilon = 1
    epsilon_min = 0.01
    score = 0
    score_queue = []
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

def context_distance_popularity_baseline(date_input, df_weather, time_input, poi_start, context=False):
    if context:
        str_summary = "BCDP"
        poi_popularity_ctx = df_poi_popularity_context.copy()
    else:
        str_summary = "BDP"
        poi_popularity = df_poi_popularity.copy()

        # choose the nearest and most popular POI to vist untile time is over
    
    done = False
    poi_len = 0

    # Initialization of the environment -> reset for each date
    env = poi_env(df_poi_it,df_crowding,df_poi_time_travel)
    env.reset(poi_start , timedelta( hours = time_input ) ,date_input)
    visited_poi = [poi_start]
    reward = 0  

    if context:
        temp, rain = get_weather(date_input, df_weather)
        poi_popularity = poi_popularity_ctx[(poi_popularity_ctx['temp'] == temp) & 
                                            (poi_popularity_ctx['rain'] == rain)]
    
    while done==False:   # check if all possible actions are done
            #last poi visited
            last_poi_visited = env.state[0]
            distance_poi = df_poi_time_travel[(df_poi_time_travel['poi_start'] == last_poi_visited) & 
                                                (df_poi_time_travel['poi_dest'] != last_poi_visited) & df_poi_time_travel['poi_dest'].isin(env.action_space)].copy()
            distance_values = list(distance_poi['time_travel'].drop_duplicates().sort_values())

            distance_poi['points'] = distance_poi['time_travel'].apply(lambda x: distance_values.index(x)+1)
            distance_poi.rename(columns={'poi_dest': 'poi'}, inplace=True)
            choiche_poi = pd.merge(distance_poi, poi_popularity, on='poi', how='left')
            choiche_poi['tot'] = choiche_poi['points'] + choiche_poi['position']
            choiche_poi = choiche_poi.sort_values(by=['tot'], ascending=True)

            a = choiche_poi['poi'].values[0]
            
            _ , r, done = env.step(a)
            reward += r
            poi_len += 1
            visited_poi.append(a)

    total_time_visit, total_time_distance, total_time_crowd, time_left=env.time_stats()
    
    popular_poi_visited = len(set(visited_poi) & set(popular_poi))
        
    return reward, total_time_visit, total_time_distance, total_time_crowd, time_left, poi_len, popular_poi_visited, visited_poi

#############################################
############ TRAINING EXPBUFFER #############
#############################################

print("\n################## EXP BUFFER ##################\n")

df_poi_h = df_poi_train.copy()
print(f"Number of sample: {len(df_poi_h)}")

# Grouped by Verona card and date of visit
grouped_df_h = df_poi_h.groupby(['id_veronacard', 'data_visita'])

# Initialization of the environment
env_h = poi_env(df_poi_it, df_crowding, df_poi_time_travel)

# One experience buffer for each context
poi_exp_b = df_poi_it['id'].values.astype(str).tolist()
time_input_exp_b =  [str(x) for x in range(0, 6)]
rain_exp_b = ['rain', 'no_rain']
temp_exp_b = [str(x) for x in range(0, 5)]

df_key_exp_buffer_h = pd.DataFrame({'poi': poi_exp_b})
df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'time':time_input_exp_b}), how='cross')
df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'temp':temp_exp_b}), how='cross')
df_key_exp_buffer_h = df_key_exp_buffer_h.merge( pd.DataFrame({'rain':rain_exp_b}), how='cross')

exp_buffer_h = {}
for i in range(len(df_key_exp_buffer_h)):
    exp_buffer_h['+'.join(df_key_exp_buffer_h.loc[i].tolist())] = deque(maxlen=700)

#ctx_experience_buffer_h = {[]:deque(maxlen=700)}

print(f"Number of users: {len(grouped_df_h)}")
i_train = 0
for (group_id, group_date), group_data in grouped_df_h:
    poi_start_h = group_data['poi'].iloc[0]
    date_input_h = datetime.strptime(group_date, '%Y-%m-%d')

    # reset environment for each new user
    state = env_h.reset(poi_start_h, timedelta(hours=14), date_input_h)
    relative_start_time = 0
    insertable = True
    first_visit_time = None
    last_visit_time = None

    poi_len = 0

    for index, row in group_data.iterrows():
        if first_visit_time is None:
            first_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
            last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
            last_poi = row['poi']
        else:
            last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
            last_poi = row['poi']
    time_input_h = (last_visit_time + timedelta(minutes=env_h.poi_time_visit[last_poi])).hour - first_visit_time.hour

    #'303+10+2+no_rain'
    #poi+time/2+temp+rain
    temp_h, rain_h = get_weather(date_input_h, df_weather)
    key_exp_buff = '+'.join([str(poi_start_h), str(int(time_input_h/2)), str(temp_h), str(rain_h)])
    if len(exp_buffer_h[key_exp_buff]) >= 200:
        continue
    for index, row in group_data.iterrows():
        if (row['poi'] not in env_h.explored):
            act_space = env_h.action_space.copy()

            new_state, reward, done = env_h.step(row['poi'])
            act_space_2 = env_h.action_space.copy()
            a = map_from_poi_to_action[row['poi']]
            exp_buffer_h[key_exp_buff].append([state, a, new_state, reward, done, act_space, act_space_2]) 
            state = new_state

    env_h.timeleft = timedelta(minutes=0)  # reset time left for each real user, he or she has no more time for visits

    exit = True
    for v in exp_buffer_h.values():
        if len(v) < 100:#--> finire quando arrivo almeno a 100 per ogni contesto
            exit = False
    
    if exit:
        print('\nEnxperience buffer filled for all context\n')
        break

    i_train += 1
    
    if i_train % 1000 == 0:
        print(f"Processed {i_train}/{len(grouped_df_h)} - {i_train/len(grouped_df_h)*100}")


#############################################
################# BASELINES #################
#############################################

print("\n################## BASELINES ##################\n")

### Load historical data
df_poi_h = df_poi_test.copy()

itinerary_h = pd.DataFrame(columns=['id_veronacard', 'itinerary', 'reward', 'time_visit', 'time_distance',
                                    'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len'])
itinerary_dp = pd.DataFrame(columns=['id_veronacard', 'itinerary', 'reward', 'time_visit', 'time_distance',
                                     'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len'])
itinerary_cdp = pd.DataFrame(columns=['id_veronacard', 'itinerary', 'reward', 'time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len'])
itinerary_dqn = pd.DataFrame(columns=['id_veronacard', 'itinerary', 'reward', 'best_reward','time_visit', 'time_distance',
                                      'time_crowd', 'time_left', 'time_input', 'popular_poi_visited', 'poi_len','time_process'])

global_reward_h = global_reward_cdp = global_reward_dp = global_reward_dqn = global_best_reward_dqn = 0
global_total_time_visit_h = global_total_time_visit_cdp = global_total_time_visit_dp = global_total_time_visit_dqn = 0
global_total_time_distance_h = global_total_time_distance_cdp = global_total_time_distance_dp = global_total_time_distance_dqn = 0
global_total_time_crowd_h = global_total_time_crowd_cdp = global_total_time_crowd_dqn = global_total_time_crowd_dp = 0
global_time_left_h = global_time_left_cdp = global_time_left_dp = global_time_left_dqn = 0
global_poi_len_h = global_poi_len_cdp = global_poi_len_dqn = global_poi_len_dp = 0
popular_poi_visited_h = global_popular_poi_visited_cdp = global_popular_poi_visited_dp = global_popular_poi_visited_dqn =  0

global_time_process_dqn = 0
global_time_h = 0
i_h = i_dqn = 0

# Grouped by Verona card and date of visit
grouped_df_h = df_poi_h.groupby(['id_veronacard', 'data_visita'])
len_vc = len(grouped_df_h)
print(f"Number of sample: {len_vc}")
# Initialization of the environment
env_h = poi_env(df_poi_it, df_crowding, df_poi_time_travel)

for (group_id, group_date), group_data in grouped_df_h:
    if len(list(group_data['poi'].values)) != len(set(group_data['poi'].values)): #check if there are duplicates
        continue
    poi_start = group_data['poi'].iloc[0]

    date_input = datetime.strptime(group_date, "%Y-%m-%d")

    # reset environment for each new user
    state = env_h.reset(poi_start, timedelta(hours=14), date_input)
    reward_tot = 0
    relative_start_time = 0
    insertable = True
    first_visit_time = None
    last_visit_time = None

    poi_len = 0

    for index, row in group_data.iterrows():
        poi_len += 1
        if (row['poi'] not in env_h.explored):

            if first_visit_time is None:
                first_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_poi = row['poi']

                env_h.state[1] = relative_start_time
            else:
                last_visit_time = datetime.strptime(row['ora_visita'], '%H:%M:%S')
                last_poi = row['poi']

            act_space = env_h.action_space.copy()
            # print(state)
            # print(act_space)
            new_state, reward, done = env_h.step(row['poi'])

            act_space_2 = env_h.action_space.copy()
            a = map_from_poi_to_action[row['poi']]
            state = new_state
            reward_tot += reward
    env_h.timeleft = timedelta(minutes=0)  # reset time left for each real user, he or she has no more time for visits

    time_input = (last_visit_time + timedelta(minutes=env_h.poi_time_visit[last_poi])).hour - first_visit_time.hour

    total_time_visit_h, total_time_distance_h, total_time_crowd_h, time_left_h = env_h.time_stats()

    itinerary_h.loc[i_h] = [group_id, list(group_data['poi'].values), reward_tot, total_time_visit_h, total_time_distance_h, total_time_crowd_h, time_left_h, time_input, len(set(env_h.explored) & set(popular_poi)), poi_len]
       
    global_total_time_visit_h += total_time_visit_h
    global_total_time_distance_h += total_time_distance_h
    global_total_time_crowd_h += total_time_crowd_h
    global_time_left_h += time_left_h
    #global_time_h += total_time_distance_h + total_time_crowd_h + total_time_visit_h
    global_time_h += time_input*60

    visited_poi = env_h.explored.copy()
    popular_poi_visited_h += len(set(visited_poi) & set(popular_poi))

    global_reward_h += reward_tot
    i_h += 1
    global_poi_len_h += poi_len
    
    #at this point evaluate the historical path, use the same start poi, time and date to query other baselines

    ################### DQN ###################

    # Initialization of the environment
    env_dqn = poi_env(df_poi_it,df_crowding,df_poi_time_travel)
    start_state_dqn = env_dqn.reset(poi_start, timedelta( hours = time_input ) , date_input)

    # Initialization of the neural network
    neural_network = initialization_dn(layer_size=15, input_layer=20, output_layer=18)

    # run DQN algorithm (deep q learning)
    
    temp, rain = get_weather(date_input, df_weather_test)
    key_exp_buff = '+'.join([str(poi_start), str(int(time_input/2)), str(temp), str(rain)])
    experience_buffer_h = exp_buffer_h[key_exp_buff]

    if experience_buffer_h == 'DONE':
        continue
    else:
        i_dqn += 1

    experience_buffer_dqn = experience_buffer_h 
    start_dqn = time.time()
    neural_network, score_dqn, best_journey = DQN(env_dqn, neural_network, 300, 32, time_input, poi_start, date_input, experience_buffer_dqn)
    time_process_dqn = time.time() - start_dqn
    reward_dqn = np.array([score_dqn]).mean()
    best_reward_dqn = np.array([score_dqn]).max()
    global_reward_dqn += reward_dqn
    global_best_reward_dqn += best_reward_dqn

    exp_buffer_h[key_exp_buff] = 'DONE'

    total_time_visit_dqn, total_time_distance_dqn, total_time_crowd_dqn, time_left_dqn=env_dqn.time_stats()
    popular_poi_visited_dqn = len(set(best_journey) & set(popular_poi))
    global_popular_poi_visited_dqn += popular_poi_visited_dqn
    global_total_time_visit_dqn += total_time_visit_dqn
    global_total_time_distance_dqn += total_time_distance_dqn
    global_total_time_crowd_dqn += total_time_crowd_dqn
    global_time_left_dqn += time_left_dqn
    global_time_process_dqn += time_process_dqn

    itinerary_dqn.loc[i_dqn] = [group_id, list(best_journey), reward_dqn, best_reward_dqn, total_time_visit_dqn, total_time_distance_dqn, total_time_crowd_dqn, 
                             time_left_dqn, time_input, popular_poi_visited_dqn, len(best_journey), time_process_dqn]


    ########## DISTANCE-P # BASELINE ##########

    reward_dp, total_time_visit_dp, total_time_distance_dp, total_time_crowd_dp, time_left_dp, poi_len_dp, popular_poi_visited_dp, poi_visited_dp = context_distance_popularity_baseline(date_input, df_weather_test, time_input, poi_start, False)

    itinerary_dp.loc[i_h] = [group_id, poi_visited_dp, reward_dp, total_time_visit_dp, total_time_distance_dp, total_time_crowd_dp, time_left_dp, time_input, popular_poi_visited_dp, poi_len_dp]

    global_reward_dp += reward_dp
    global_total_time_visit_dp += total_time_visit_dp
    global_total_time_distance_dp += total_time_distance_dp
    global_total_time_crowd_dp += total_time_crowd_dp
    global_time_left_dp += time_left_dp
    global_poi_len_dp += poi_len_dp
    global_popular_poi_visited_dp += popular_poi_visited_dp


    ######### CTX-DISTANCE-P # BASELINE #########

    reward_cdp, total_time_visit_cdp, total_time_distance_cdp, total_time_crowd_cdp, time_left_cdp, poi_len_cdp, popular_poi_visited_cdp, poi_visited_cdp = context_distance_popularity_baseline(date_input, df_weather_test, time_input, poi_start, True)

    itinerary_cdp.loc[i_h] = [group_id, poi_visited_cdp, reward_cdp, total_time_visit_cdp, total_time_distance_cdp, total_time_crowd_cdp, time_left_cdp, time_input, popular_poi_visited_cdp, poi_len_cdp]

    global_reward_cdp += reward_cdp
    global_total_time_visit_cdp += total_time_visit_cdp
    global_total_time_distance_cdp += total_time_distance_cdp
    global_total_time_crowd_cdp += total_time_crowd_cdp
    global_time_left_cdp += time_left_cdp
    global_poi_len_cdp += poi_len_cdp
    global_popular_poi_visited_cdp += popular_poi_visited_cdp

    if i_h % 100 == 0:
        print(f'Processed {i_h}/{len_vc} users')


# itinerary_h.to_csv('../itineraries/itinerary_h.csv', index=False)
# itinerary_dp.to_csv('../itineraries/itinerary_dp.csv', index=False)
# itinerary_cdp.to_csv('../itineraries/itinerary_cdp.csv', index=False)
# itinerary_dqn.to_csv('../itineraries/itinerary_dqn.csv', index=False)

list_poi = list(df_poi_it['id'].values)
arp_h = arp_measure(list_poi, list(itinerary_h['itinerary'].values))
gini_h = gini_measure(list_poi, list(itinerary_h['itinerary'].values))

arp_dp = arp_measure(list_poi, list(itinerary_dp['itinerary'].values))
gini_dp = gini_measure(list_poi, list(itinerary_dp['itinerary'].values))

arp_cdp = arp_measure(list_poi, list(itinerary_cdp['itinerary'].values))
gini_cdp = gini_measure(list_poi, list(itinerary_cdp['itinerary'].values))

arp_dqn = arp_measure(list_poi, list(itinerary_dqn['itinerary'].values))
gini_dqn = gini_measure(list_poi, list(itinerary_dqn['itinerary'].values))


print(f"\n\tB-History SUM UP")
print(f"[BH SUM UP] AVG Reward: {global_reward_h/i_h}")
print_stats(global_total_time_visit_h/i_h, global_total_time_distance_h/i_h, 
            global_total_time_crowd_h/i_h, global_time_left_h/i_h, 
            global_time_h/i_h/60, popular_poi_visited_h/i_h, global_poi_len_h/i_h, arp_h, gini_h, '[BH SUM UP] ')

print(f"\n\tB-Distance+Popularity SUM UP")
print(f"[BDP SUM UP] AVG Reward: {global_reward_dp/i_h}")
print_stats(global_total_time_visit_dp/i_h, global_total_time_distance_dp/i_h, 
            global_total_time_crowd_dp/i_h, global_time_left_dp/i_h,
            global_time_h/i_h/60,global_popular_poi_visited_dp/i_h, global_poi_len_dp/i_h, arp_dp, gini_dp,"[BDP SUM UP] ")

print(f"\n\tB-Context+Distance+Popularity SUM UP")
print(f"[BCDP SUM UP] AVG Reward: {global_reward_cdp/i_h}")
print_stats(global_total_time_visit_cdp/i_h, global_total_time_distance_cdp/i_h, 
            global_total_time_crowd_cdp/i_h, global_time_left_cdp/i_h,
            global_time_h/i_h/60,global_popular_poi_visited_cdp/i_h, global_poi_len_cdp/i_h, arp_cdp, gini_cdp, "[BCDP SUM UP] ")

print(f"\n\tB-DQN SUM UP")
print(f"[BDQN SUM UP] AVG Reward: {global_reward_dqn/i_dqn}")
print(f"[BDQN SUM UP] AVG Best Reward: {global_best_reward_dqn/i_dqn}")
print_stats(global_total_time_visit_dqn/i_dqn, global_total_time_distance_dqn/i_dqn, 
            global_total_time_crowd_dqn/i_dqn, global_time_left_dqn/i_dqn,
            global_time_h/i_h/60,global_popular_poi_visited_dqn/i_dqn, global_poi_len_dqn/i_dqn, arp_dqn, gini_dqn, "[BDQN SUM UP] ")
print(f'[BDQN SUM UP] Time process: {global_time_process_dqn/i_dqn}')
