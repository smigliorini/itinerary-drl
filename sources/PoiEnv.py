# CLASSE POI_ENV
"""
Questa classe genera l'ambiente che utilizzeremo per gestire il percorso svolto da un turista attraverso vari POI(Poin of Interest) 
all'interno della nostra rete neurale.

METODI

__init__
    inizializza l'ambiente 
    INPUT: data di inizio viaggio , dataframe con id del poi e alcuni campi, dataframe del crowding per fascia oraria, dataframe col tempo di visita da un Poi ad un altro

reset
    resetta l'ambiente 
    INPUT: id del poi di partenza , tempo totale del viaggio
    OUTPUT: stato aggiornato coi dati inseriti

step
    step dal poi di partenza a quello di destinazione
    INPUT: id del poi di destinzazione
    OUTPUT: stato aggiornato, reward dello step , Booleano che definisce se lo stato è terminale 

poi_available
    Posso andare dal poi di partenza a quello di destinazione nel tempo che mi rimane ?
    INPUT: poi di destinazione 
    OUTPUT: stato terminale? , True: posso raggiungere il poi di destinazione , fattore di conversione del reward-> solo se non ho abbastanza tempo per visitare il Poi
 
crowding_wait
    calcolo l'attesa per un determinato Poi
    INPUT: poi destinazione
    OUTPUT: tempo di attesa per il poi di destinazione

calc_distance
    calcolo la distanza tra poi partenza e destinaazione
    INPUT: poi destinazione
    OUTPUT: distanza dal poi attuale a quello di destinazione

time_stats
    stampa le statistiche che riguardano il tempo sprecato, utilizzato, speso in coda e speso in viaggio 

current_time 
    calcola la data utilizzando la data inserita inizialmente + il tempo speso finora all'interno dell'ambiente
"""

from datetime import *
from math import ceil
import numpy as np
import utils


class poi_env:
    def __init__(self, df_poi_it, df_crowding, df_poi_time_travel):#(self, date, df_poi_it, df_crowding, df_poi_time_travel):

        self.action_space = set(df_poi_it.id)  # dove posso andare, tutti i poi
        self.state = None  # stato ->[poi,timestamp] [array di poi visitati] ,None siccome lo stato iniziale è vuoto
        self.timeleft = None  # Tempo rimasto per il tour
        #self.poi_date = date  # data nel formato datetime(year, month, day, hour, minute, second, microsecond)
        self.explored = set()  # poi già visitati, all'inizio vuoto
        self.map_from_poi_to_action, self.map_from_action_to_poi = utils.neural_poi_map()

        # dataframes
        self.df_poi_it = df_poi_it
        self.df_crowding = df_crowding
        self.df_poi_time_travel = df_poi_time_travel

        # Creo Il Dictionary con latitudine e longitudine-> poi_position
        # E quello per per il tempo di visita -> poi_time_visit
        poi_position = {}
        poi_time_visit = {}
        for row in df_poi_it.itertuples():
            poi_position[row[1]] = (row[2], row[3])
            poi_time_visit[row[1]] = row[4]

        self.poi_time_visit = poi_time_visit  # tempo per visitare ogni poi
        self.poi_position = poi_position  # latitudine e longitudine
        self.poi_set = set(df_poi_it.id)  # set di tutti i poi

        # STATS
        self.total_time_visit = 0
        self.total_time_crowd = 0
        self.total_time_distance = 0

    def reset(self, initial_poi, total_time, date):
        # reset dell'environment, inizializzo tutti i campi
        self.action_space = self.poi_set.copy()  # set dei poi visitabili
        self.state = np.array([initial_poi, 0])  # stato [int poi, int timestamp ]
        self.poi_date = date  # data del tour

        self.state = np.pad(self.state, (0, len(self.poi_set)),
                            mode='constant')  # stato  [0..1.....0...1] 1 se ho visitato il poi 2+n esimo, 0 altrimenti
        self.timeleft = total_time  # tempo rimanente per il tour
        self.explored = set()  # poi già visitati

        self.total_time_visit = 0
        self.total_time_crowd = 0
        self.total_time_distance = 0

        return self.state.copy()

    # Mi muovo da un poi A ad un poi B
    def step(self, poi_dest):

        # Calcolo la distanza da A a B
        time_distance = self.calc_distance(poi_dest)
        # Calcolo il tempo di attesa per il poi B
        time_crowd = self.crowding_wait(poi_dest)
        # Tempo rimanente come int
        time_left_int = self.timeleft.total_seconds() / 60

        # Calcolo time visit, se sfora tengo solo la parte di tempo che riesco ad usare
        if ((time_crowd + time_distance + self.poi_time_visit[poi_dest]) > time_left_int):
            if time_crowd + time_distance > time_left_int:
                time_visit = time_left_int
            else:
                time_visit = time_left_int - (time_crowd + time_distance)
        else:
            time_visit = self.poi_time_visit[poi_dest]

            # STATS
        self.total_time_distance += time_distance
        self.total_time_crowd += time_crowd
        self.total_time_visit += time_visit

        ##### Calcolo il reward ####
        factor_base = 0  # todo: remove, non lo useremo più
        reward = factor_base + ((time_visit)) / (time_crowd + time_distance + time_visit) * time_visit / 5

        #### Aggiorno lo stato ####
        self.action_space.remove(poi_dest)  # rimuovo il nodo B da quelli visitabili
        self.explored.add(poi_dest)  # lo inserisco in quelli visitati
        self.state[0] = poi_dest  # aggiorno lo stato del Poi corrente
        self.state[2 + self.map_from_poi_to_action[poi_dest]] = 1  # aggiungo anche nello stato che ho visitato il Poi

        ### Calcolo tempi di viaggio , visita e attesa
        time_total = time_distance + time_crowd + time_visit

        # Aggiorno il tempo rimanente
        self.timeleft = timedelta(minutes=(time_left_int - time_total))
        self.state[1] = int(self.state[1]) + int(time_total)

        # Controllo se ci sono altri poi da visitare, se non ci sono allora è uno stato terminale
        done_temp = False
        actual_act_space = self.action_space.copy()
        for poi in actual_act_space:
            self.poi_available(poi)
        if len(self.action_space) == 0:
            done_temp = True
            if self.timeleft.total_seconds() > 0:  # Mi avanza tempo ? tolgo un po' di reward
                reward = reward - self.timeleft.total_seconds() / 1200

        # RETURN stato attuale, reward, se lo stato è terminale
        return self.state.copy(), reward, done_temp

    def poi_available(self, poi_dest):

        time_travel = self.calc_distance(poi_dest)
        time_queue = self.crowding_wait(poi_dest)
        minimum_time = ceil((self.poi_time_visit[
            poi_dest]) / 3)  # se un Poi non è visitabile per 1/3 del tempo di visita non ha senso entrarci
        time_t_q = time_travel + time_queue + minimum_time

        # se il tempo rimasto non permette di entrare nel POI destinazione, allora lo rimuovo da quelli visitabili
        if (self.timeleft.total_seconds() / 60 < time_t_q):
            self.action_space.remove(poi_dest)

    def crowding_wait(self, poi_dest):
        # trova la fascia di crowd del poi prima in basea al giorno e dopo in base all'orario
        date_c = timedelta(minutes=int(self.state[1])) + self.poi_date

        if date_c.hour < 12:
            crowd_range = date_c.replace(hour=8, minute=0, second=0)
        elif date_c.hour >= 12 and date_c.hour < 16:
            crowd_range = date_c.replace(hour=12, minute=0, second=0)
        else:
            crowd_range = date_c.replace(hour=16, minute=0, second=0)

        # cerco la entry corrispondente in df_crowding-> persone stimate in coda
        estimated_crowd = \
        self.df_crowding.loc[(self.df_crowding['poi'] == poi_dest) & (self.df_crowding['data'] == str(crowd_range))][
            "val_stim"].values
        # se la riga non c'è o è vuota, assumo che non ci sia coda
        if len(estimated_crowd) == 0:
            return 15
        else:
            estimated_crowd = estimated_crowd[0]
            if np.isnan(estimated_crowd):
                return 15
            # calcolo quanto tempo dovrò aspettare -> (tempo visita/2 * persone in coda) / capienza massima del poi
            crowd_wait = ((self.poi_time_visit[poi_dest] / 2) * estimated_crowd) / \
                         self.df_poi_it.loc[(self.df_poi_it['id'] == poi_dest)]["max_crowd"].values[0]
            # ritorno i minuti di attesa arrotondati al minuto successivo
            return ceil(crowd_wait)

    def calc_distance(self, poi_dest):
        # calcolo il time travel da A a B...mi viene restituito da df_poi_time_travel in minuti
        df_tmp = self.df_poi_time_travel.loc[(self.df_poi_time_travel['poi_start'] == self.state[0])]
        df_tmp = df_tmp.loc[(df_tmp['poi_dest'] == poi_dest)]
        return int(df_tmp.to_numpy()[0][2])

    def time_stats(self):
        return self.total_time_visit, self.total_time_distance, self.total_time_crowd, self.timeleft.total_seconds() / 60

    def current_time(self):
        return timedelta(minutes=int(self.state[1])) + self.poi_date