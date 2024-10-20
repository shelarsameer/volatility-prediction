import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
from datetime import timedelta

def process_data(file):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'])
    prices = data['Close Price'].values
    returns = np.diff(np.log(prices))
    return data, prices, returns

def fit_model(returns, model_type):
    if model_type == 'GARCH':
        model = arch_model(returns, vol='GARCH', p=1, q=1, mean='AR', lags=1, dist='studentst')
    elif model_type == 'EGARCH':
        model = arch_model(returns, vol='EGARCH', p=1, q=1, mean='AR', lags=1, dist='studentst')
    elif model_type == 'TGARCH':
        model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, power=1.0, mean='AR', lags=1, dist='studentst')
    
    results = model.fit(disp='off')
    fitted_returns = results.params['Const'] + results.params['y[1]'] * np.roll(returns, 1)
    fitted_returns[0] = results.params['Const']
    rmse = np.sqrt(np.mean((returns - fitted_returns)**2))
    return results, rmse

def forecast(results, prices, dates):
    forecast_horizon = 10
    forecasts = results.forecast(horizon=forecast_horizon, method='simulation', simulations=10000)
    forecasted_mean = forecasts.mean.iloc[-1]
    forecasted_vol = np.sqrt(forecasts.variance.iloc[-1])
    last_price = prices[-1]
    predicted_prices = last_price * np.exp(np.cumsum(forecasted_mean))
    pred_dates = [dates.iloc[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]
    ci_lower = last_price * np.exp(np.cumsum(forecasted_mean - 1.96 * forecasted_vol))
    ci_upper = last_price * np.exp(np.cumsum(forecasted_mean + 1.96 * forecasted_vol))
    return pred_dates, predicted_prices, ci_lower, ci_upper

st.title('Stock Price Forecast')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data, prices, returns = process_data(uploaded_file)
    
    models = ['GARCH', 'EGARCH', 'TGARCH']
    results = {}
    
    for model in models:
        model_results, rmse = fit_model(returns, model)
        results[model] = {'results': model_results, 'rmse': rmse}
    
    best_model = min(results, key=lambda x: results[x]['rmse'])
    
    st.subheader('RMSE Values:')
    for model, result in results.items():
        st.write(f"{model}: {result['rmse']:.6f}")
    
    st.subheader(f'Best Model: {best_model}')
    
    pred_dates, predicted_prices, ci_lower, ci_upper = forecast(results[best_model]['results'], prices, data['Date'])
    
    st.subheader('Predictions for Next 10 Days:')
    prediction_df = pd.DataFrame({
        'Date': pred_dates,
        'Predicted Price': predicted_prices,
        'Lower CI': ci_lower,
        'Upper CI': ci_upper
    })
    st.dataframe(prediction_df)