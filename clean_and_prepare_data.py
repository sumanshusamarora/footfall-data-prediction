import pandas as pd
import holidays
import numpy as np
import requests
import datetime

def get_weather_data(date, woeid=1105779):
    response = requests.get(f"https://www.metaweather.com/api/location/{woeid}/{date.year}/{date.month}/{date.day}")
    out = response.json()
    if datetime.date.today() > date.date():
        out = [dictn for dictn in out if pd.to_datetime(dictn["applicable_date"]).date() == date.date()]
    else:
        out = [out[0]]

    weather_state_abbr = pd.Series([this_dict["weather_state_abbr"] for this_dict in out]).value_counts().index[0]
    min_temp = np.mean([this_dict["min_temp"] for this_dict in out])
    max_temp = np.mean([this_dict["max_temp"] for this_dict in out])
    humidity = np.mean([this_dict["humidity"] for this_dict in out])
    return {"weather": weather_state_abbr, "min_temp": min_temp, "max_temp": max_temp, "humidity": humidity}

def preprocess(data):
    # Populate holiday
    data["holiday"] = data["Date"].apply(lambda x: int(x.date() in holidays.Australia()))
    data = data.rename(columns={"Date":"ds", "Footfall":"y"})

    # Week day
    data["dayofweek"] = data['ds'].apply(lambda x: x.weekday())
    week_day_dict = {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4, "Sat":5, "Sun":6}
    week_day_rev_dict = {val:key for key, val in week_day_dict.items()}
    for value in week_day_dict.keys():
        data[f"{value}day"] = 0

    for i in range(len(data)):
        data.loc[i, f"{week_day_rev_dict[data['dayofweek'][i]]}day"] = 1

    # Weather data
    weather_type_dict = {"sn":0, "sl":1, "h":2, "t":3, "hr":4, "lr":5, "s":6, "hc":7, "lc":8, "c":9}
    for value in weather_type_dict.keys():
        data[f"weather_{value}"] = 0

    data["min_temp"] = 0
    data["max_temp"] = 0
    data["humidity"] = 0

    for i in range(len(data)):
        weather_data = get_weather_data(data["ds"][i])
        data.loc[i, f"weather_{value}"] = 1
        data.loc[i, "min_temp"] = weather_data["min_temp"]
        data.loc[i, "max_temp"] = weather_data["max_temp"]
        data.loc[i, "humidity"] = weather_data["humidity"]

    return data


if __name__=="__main__":
    # Import data
    data = pd.read_excel("data/dataset.xls")
    data = preprocess(data)
    data.to_pickle("data/data.pkl", protocol=3)

    #data_test = pd.read_excel("data/dataset.xls", sheet_name="Test")
    #data_test = preprocess(data_test)
    #data_test.to_pickle("data/data_test.pkl", protocol=3)
