# Questa funzione associa ad ogni poi un numero da 0 a [numero di poi -1]
def neural_poi_map():
    # Array con tutti i poi
    # all_poi = [42, 43, 44, 45, 46, 47, 48, 49 , 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 200, 201, 202, 203, 204, 100, 101, 102, 110]
    # all_poi = [42,49,52,54,58,59,61,62,63,71,75,76,202] #13 poi_it
    # all_poi = [49,58,61,50,47,62,54,59,52,63,201,202,75,76,55] #open_data_2019
    # all_poi = [49,58,61,50,47,62,54,59,52,63,201,202,75,76,55,300,301,302,303]
    all_poi = [42, 49, 52, 54, 58, 59, 61, 62, 63, 71, 75, 76, 201, 202, 300, 301, 302, 303]

    map_from_poi_to_action = {}  # Poi to Action
    i = 0
    for x in all_poi:
        map_from_poi_to_action[x] = i
        i += 1
    map_from_action_to_poi = {}
    for x in range(len(all_poi)):
        map_from_action_to_poi[x] = all_poi[x]
    return map_from_poi_to_action, map_from_action_to_poi


# Tipologia del giorno Temperatura e condizione atmosferica
import holidays
from datetime import datetime


def print_date_type(date_input, df_weather, id_verona_card):
    giorni_festivi = holidays.Italy()

    # stampo tipologia del giorno
    row_weather = df_weather.loc[df_weather['datetime'] == date_input.strftime('%Y-%m-%d')]
    if not row_weather.empty:
        temperatura = row_weather['temp'].iloc[0]
        condizione = str(row_weather['preciptype'].iloc[0])
        if condizione == "nan": condizione = "sun"
    else:
        temperatura = None
        condizione = None
    if date_input in giorni_festivi or date_input.weekday() == 6 or date_input.weekday() == 5 or date_input == datetime(
            2022, 2, 14):
        print(
            f"ID Verona Card: {id_verona_card}  Data visita: {date_input.strftime('%d-%m-%Y')} è un giorno Festivo con Temperatura°: {temperatura}  e condizione meteo: {condizione} ")
    else:
        print(
            f"ID Verona Card: {id_verona_card}  Data visita: {date_input.strftime('%d-%m-%Y')} è un giorno Feriale con Temperatura°: {temperatura}  e condizione meteo: {condizione} ")


# Statistiche di utilizzo del tempo
def print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left, time_input):
    print(f"Tempo sfruttato: {total_time_visit} min")
    print(f"Tempo di camminata: {total_time_distance} min")
    print(f"Tempo in coda: {total_time_crowd} min")
    print(f"Tempo rimasto: {time_left} min")

    percentage_visit = total_time_visit * 100 / (time_input * 60)
    percentage_distance = total_time_distance * 100 / (time_input * 60)
    percentage_crowd = total_time_crowd * 100 / (time_input * 60)
    percentage_final = time_left * 100 / (time_input * 60)
    percentage_lost_time = percentage_distance + percentage_crowd + percentage_final
    print("Percentuale di tempo sfruttato: {:3.2f}%".format(percentage_visit))
    print("Percentuale di tempo di camminata: {:3.2f}%".format(percentage_distance))
    print("Percentuale di tempo in coda: {:3.2f}%".format(percentage_crowd))
    print("Percentuale di tempo rimasto: {:3.2f}%".format(percentage_final))
    print("Percentuale di tempo sprecato: {:3.2f}%".format(percentage_lost_time))
