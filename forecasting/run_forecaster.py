import streamlit as st
from utils.config import FORECAST_FEATURES, TIME_STEPS
from utils.readers import DataReader, ModelReader, Preprocessor
from utils.plotters import Plotter

st.set_page_config(layout='wide')


def main():
    st.header('Forecasting test data')
    df = DataReader.get_processed_data_from_gcs(raw=False)
    if st.button('Plot heatmap of features'):
        st.plotly_chart(Plotter.plot_heatmap(df, FORECAST_FEATURES,
                                             show=False),
                        use_container_width=True)
    uploaded_file = st.file_uploader('Upload raw data file', type=['csv'])
    window = int(st.number_input('Window size of moving average, seconds',
                             value=3600,
                             min_value=1,
                             max_value=7200,
                             step=1))
    if uploaded_file is not None:
        new_df = DataReader.read_newcoming_data(uploaded_file)
        new_df = Preprocessor.remove_step_zero(new_df)
        new_df = Preprocessor.feature_engineering(new_df)
        st.write('Forecast on the new data')
        for feature in FORECAST_FEATURES:
            scaler = ModelReader.read_model_from_gcs(f'RNN_{feature}_scaler')
            forecaster = ModelReader.read_model_from_gcs(f'RNN_{feature}')
            scaled_new_data = scaler.transform(new_df[feature].values.reshape(
                -1, 1))
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
    else:
        st.write('Current forecast')
        for feature in FORECAST_FEATURES:
            scaler = ModelReader.read_model_from_gcs(f'RNN_{feature}_scaler')
            forecaster = ModelReader.read_model_from_gcs(f'RNN_{feature}')
            scaled_data = scaler.transform(
                df.iloc[-window:][feature].values.reshape(-1, 1))
            sequenced_scaled_data = Preprocessor.create_sequences(
                scaled_data, lookback=TIME_STEPS, inference=True)
            current_forecast = forecaster.predict(sequenced_scaled_data)
            current_forecast = scaler.inverse_transform(current_forecast)
            st.plotly_chart(Plotter.plot_forecast(df,
                                                  current_forecast,
                                                  feature,
                                                  new=None,
                                                  plot_ma_all=True,
                                                  window=window,
                                                  show=False),
                            use_container_width=True)


if __name__ == '__main__':
    main()