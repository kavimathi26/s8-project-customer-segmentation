import csv
import pandas as pd

# new imports start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
from prophet.plot import plot_plotly as py

# new imports end

from flask import Flask,request
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

@app.route('/',methods=["POST","GET"])
def read_file():
    for file in request.files.getlist('file'):
        data=pd.read_csv(file)
        data.isnull().sum()
        data.dropna(inplace=True)
        my_model = Prophet(interval_width=0.95)
        data['Month'] = pd.DatetimeIndex(data['Month'])
        data = data.rename(columns={'Month': 'ds','Sales': 'y'})
        my_model.fit(data)
        future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
        forecast = my_model.predict(future_dates)
        my_model.plot(forecast, uncertainty=True)
        my_model.plot_components(forecast)
        fig1 = my_model.plot_components(forecast)
        from prophet.plot import add_changepoints_to_plot
        fig = my_model.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), my_model, forecast)
        my_model.changepoints
        pro_change= Prophet(changepoint_range=0.9)
        forecast = pro_change.fit(data).predict(future_dates)
        fig= pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
        pro_change= Prophet(n_changepoints=20, yearly_seasonality=True)
        forecast = pro_change.fit(data).predict(future_dates)
        fig= pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
        pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
        forecast = pro_change.fit(data).predict(future_dates)
        fig= pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
        pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.001)
        forecast = pro_change.fit(data).predict(future_dates)
        fig= pro_change.plot(forecast)
        a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)
        forecast.drop(["trend","yhat_lower","yhat_upper","trend_lower", "trend_upper", "additive_terms","additive_terms_lower","additive_terms_upper","yearly","yearly_lower","yearly_upper","multiplicative_terms","multiplicative_terms_lower","multiplicative_terms_upper"], inplace = True, axis = 1)
        forecast = forecast.rename(columns={'ds': 'Month', 'yhat': 'Sales'})
        train = forecast.drop(forecast.index[-36:])
        from sklearn.metrics import mean_absolute_error
        y_true = data['y'].values
        y_pred = train['Sales'].values
        mae = mean_absolute_error(y_true, y_pred)
        print('MAE: %.3f' % mae)       
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.show()
    # print(data.head(100))
    return data.to_dict()
if __name__ == "__main__":
    app.run(debug=True)