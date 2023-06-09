# Tipologia del giorno Temperatura e condizione atmosferica
import holidays
from datetime import datetime

# map each poi to an action with a number from 0 to (#POI - 1)
def neural_poi_map():
    # POI ids
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

def print_date_type(date_input, df_weather, id_verona_card):
    giorni_festivi = holidays.Italy()

    # print day informations (weather, temperature, holidays)
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
            f"ID Verona Card: {id_verona_card}  Date of the visit: {date_input.strftime('%d-%m-%Y')} (public holiday)\t  Temperature°°: {temperatura}\t Weather: {condizione} ")
    else:
        print(
            f"ID Verona Card: {id_verona_card}  Date of the visit: {date_input.strftime('%d-%m-%Y')} (weekday)\t Temperature°: {temperatura}\t Weather: {condizione} ")


# Print stats about the itinerary
def print_stats(total_time_visit, total_time_distance, total_time_crowd, time_left, time_input, prefix=""):
    print(f"{prefix}Total time used: {total_time_visit} min")
    print(f"{prefix}Time walked: {total_time_distance} min")
    print(f"{prefix}Time in queue: {total_time_crowd} min")
    print(f"{prefix}Time left: {time_left} min")

    percentage_visit = total_time_visit * 100 / (time_input * 60)
    percentage_distance = total_time_distance * 100 / (time_input * 60)
    percentage_crowd = total_time_crowd * 100 / (time_input * 60)
    percentage_final = time_left * 100 / (time_input * 60)
    percentage_lost_time = percentage_distance + percentage_crowd + percentage_final
    print("{}Percentage of total time used (VT): {:3.2f}%".format(prefix,percentage_visit))
    print("{}Percentage of time walked     (MT): {:3.2f}%".format(prefix,percentage_distance))
    print("{}Percentage of time in queue   (QT): {:3.2f}%".format(prefix,percentage_crowd))
    print("{}Percentage of time left       (RT): {:3.2f}%".format(prefix,percentage_final))
    print("{}Percentage of time wasted         : {:3.2f}%".format(prefix,percentage_lost_time))
