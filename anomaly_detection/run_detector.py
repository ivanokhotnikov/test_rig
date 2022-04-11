import numpy as np
import streamlit as st

from utils.readers import DataReader, Preprocessor, ModelReader, create_sequences
from utils.plotters import Plotter
from utils.config import FEATURES_FOR_ANOMALY_DETECTION, ENGINEERED_FEATURES, PRESSURE_TEMPERATURE_FEATURES, TIME_STEPS

st.set_page_config(layout='wide')

def main():
    st.header('Anomaly Detection in the unit test data')
    uploaded_file = st.file_uploader('Upload raw data file', type=['csv'])
    if uploaded_file is not None:
        df = DataReader.read_newcoming_data(uploaded_file)
        df = Preprocessor.remove_step_zero(df)
        df = Preprocessor.feature_engineering(df)
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Plot raw data'):
                for feature in FEATURES_FOR_ANOMALY_DETECTION:
                    if 'TIME' not in feature:
                        st.plotly_chart(
                            Plotter.plot_unit_per_test_feature(
                                df,
                                unit=df.iloc[0]['UNIT'],
                                feature=feature,
                                save=False,
                                show=False))
        with col2:
            algorithm = st.radio(
                'Select algorithm', (None, 'IsolationForest', 'LocalOutlierFactor',
                                    'ConvolutionalAutoencoder'),
                help=
                'References on isolation forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html, local outlier factor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html, convolutional autoencoder: https://keras.io/examples/timeseries/timeseries_anomaly_detection/',
                index=0)
            if algorithm is not None:
                for feature in ENGINEERED_FEATURES + PRESSURE_TEMPERATURE_FEATURES:
                    detector = ModelReader.read_model(f'{algorithm}_{feature}',
                                                    task='anomaly_detection')
                    if algorithm == 'ConvolutionalAutoencoder':
                        scaler = ModelReader.read_model(
                            f'{algorithm}_{feature}_scaler',
                            task='anomaly_detection')
                        scaled_data = scaler.transform(df[feature].values.reshape(
                            -1, 1))
                        x = create_sequences(scaled_data, time_steps=TIME_STEPS)
                        pred = detector.predict(x)
                        threshold = ModelReader.read_model(
                            f'{algorithm}_{feature}_threshold',
                            task='anomaly_detection',
                            extension='.txt')
                        test_mae_loss = np.mean(np.abs(pred - x), axis=1)
                        test_mae_loss = test_mae_loss.reshape((-1))
                        anomalies = test_mae_loss > threshold
                        anomalous_data_indices = []
                        for data_idx in range(TIME_STEPS - 1,
                                            len(scaled_data) - TIME_STEPS + 1):
                            if np.all(anomalies[data_idx - TIME_STEPS +
                                                1:data_idx]):
                                anomalous_data_indices.append(data_idx)
                        df.loc[:, f'ANOMALY_{feature}'] = 1
                        df.loc[anomalous_data_indices, f'ANOMALY_{feature}'] = -1
                    else:
                        df[f'ANOMALY_{feature}'] = detector.predict(
                            df[feature].values.reshape(-1, 1))
                    try:
                        st.plotly_chart(
                            Plotter.plot_anomalies_per_unit_test_feature(
                                df,
                                unit=df.iloc[0]['UNIT'],
                                test=1,
                                feature=feature,
                                show=False))
                    except:
                        st.write(
                            f'No anomalies found in unit {df.iloc[0]["UNIT"]} for {feature} with {algorithm}'
                        )
                        continue

if __name__ == '__main__':
    main()