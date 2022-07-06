import os

import streamlit as st

from utils.config import FORECAST_FEATURES, TIME_STEPS
from utils.plotters import Plotter
from utils.readers import DataReader, ModelReader, Preprocessor

st.set_page_config(layout='wide',
                   page_title='Forecasting',
                   page_icon=os.path.join(os.path.dirname(__file__), 'images','fav.png'))
DEV = False


def main():
    st.header('Forecasting test data')
    raw = st.checkbox('Read raw data', value=False)
    df = DataReader.get_processed_data_from_gcs(raw=raw)
    with st.expander('Plot heatmap of features'):
        st.plotly_chart(Plotter.plot_cov_matrix(df, FORECAST_FEATURES,
                                             show=False),
                        use_container_width=True)
    uploaded_file = st.file_uploader('Upload raw data file', type=['csv'])
    if uploaded_file is not None:
        with st.expander('Show new forecast'):
            window = int(
                st.number_input('Window size of moving average, seconds',
                                value=3600,
                                min_value=1,
                                max_value=7200,
                                step=1))
            new_df = DataReader.read_newcoming_data(uploaded_file)
            new_df = Preprocessor.remove_step_zero(new_df)
            new_df = Preprocessor.feature_engineering(new_df)
            st.write('Forecast on the new data')
            forecast_bar = st.progress(0)
            for idx, feature in enumerate(FORECAST_FEATURES, 1):
                forecast_bar.progress(idx / len(FORECAST_FEATURES))
                scaler = ModelReader.read_model_from_gcs(
                    f'RNN_{feature}_scaler')
                forecaster = ModelReader.read_model_from_gcs(f'RNN_{feature}')
                scaled_new_data = scaler.transform(
                    new_df[feature].values.reshape(-1, 1))
                sequenced_scaled_new_data = Preprocessor.create_sequences(
                    scaled_new_data, lookback=TIME_STEPS, inference=True)
                forecast = forecaster.predict(sequenced_scaled_new_data)
                forecast = scaler.inverse_transform(forecast)
                st.plotly_chart(Plotter.plot_forecast(df,
                                                      forecast,
                                                      feature,
                                                      new=new_df,
                                                      plot_ma_all=True,
                                                      window=window,
                                                      show=False),
                                use_container_width=True)
                if DEV: break
    else:
        with st.expander('Show current forecast'):
            plot_each_unit = st.checkbox('Plot each unit', value=True)
            window = int(
                st.number_input('Window size of moving average, seconds',
                                value=3600,
                                min_value=1,
                                max_value=7200,
                                step=1))
            forecast_bar = st.progress(0)
            for idx, feature in enumerate(FORECAST_FEATURES, 1):
                forecast_bar.progress(idx / len(FORECAST_FEATURES))
                scaler = ModelReader.read_model_from_gcs(
                    f'RNN_{feature}_scaler')
                forecaster = ModelReader.read_model_from_gcs(f'RNN_{feature}')
                scaled_data = scaler.transform(
                    df.iloc[-window:][feature].values.reshape(-1, 1))
                sequenced_scaled_data = Preprocessor.create_sequences(
                    scaled_data, lookback=TIME_STEPS, inference=True)
                current_forecast = forecaster.predict(sequenced_scaled_data)
                current_forecast = scaler.inverse_transform(current_forecast)
                st.plotly_chart(Plotter.plot_forecast(
                    df,
                    current_forecast,
                    feature,
                    new=None,
                    plot_ma_all=True,
                    window=window,
                    plot_each_unit=plot_each_unit,
                    show=False),
                                use_container_width=True)
                if DEV: break


if __name__ == '__main__':
    main()