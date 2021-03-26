"""

"""
from datetime import datetime, timedelta

import argparse
import holidays
import joblib
import pandas as pd

from clean_and_prepare_data import get_weather_data

def get_prediction(days_plus_today):
    """

    :param days_plus_today:
    :return:
    """
    prediction_date = datetime.today() + timedelta(days=days_plus_today)
    data_dict = get_weather_data(prediction_date)

    week_day_dict = {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4, "Sat":5, "Sun":6}
    week_day_rev_dict = {val:key for key, val in week_day_dict.items()}

    data_dict["holiday"] = int(prediction_date in holidays.Australia())
    dayofweek = prediction_date.weekday()

    for value in week_day_dict.keys():
        data_dict[f"{value}day"] = 0

    data_dict[f"{week_day_rev_dict[dayofweek]}day"] = 1

    weather_type_dict = {"sn":0, "sl":1, "h":2, "t":3, "hr":4, "lr":5, "s":6, "hc":7, "lc":8, "c":9}
    for value in weather_type_dict.keys():
        data_dict[f"weather_{value}"] = 0

    data_dict[f"weather_{data_dict['weather']}"] = 1

    del data_dict["weather"]

    df = pd.DataFrame(columns=['holiday', 'Monday', 'Tueday', 'Wedday', 'Thuday', 'Friday', 'Satday',
               'Sunday', 'weather_sn', 'weather_sl', 'weather_h', 'weather_t',
               'weather_hr', 'weather_lr', 'weather_s', 'weather_hc', 'weather_lc',
               'weather_c', 'min_temp', 'max_temp', 'humidity'])

    for col in df.columns:
        df.loc[0, col] = data_dict[col]

    model = joblib.load('models/linear_regression.pkl')
    return (prediction_date, model.predict(df)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Input Values")
    parser.add_argument(
        "--days-in-future",
        dest="days_in_future",
        default=0,
        type=int,
        help="How many days in future",
    )

    args = parser.parse_args()
    prediction_date, prediction_value = get_prediction(args.days_in_future)
    print(f"Expected footfall for {prediction_date.date()} is {int(prediction_value)}")

# python predict.py
