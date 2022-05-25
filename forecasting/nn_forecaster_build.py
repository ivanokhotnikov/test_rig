import os

import plotly.graph_objects as go
from google.cloud import storage
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from utils.config import (EARLY_STOPPING, FORECAST_FEATURES, MODELS_PATH,
                          TIME_STEPS, VERBOSITY)
from utils.readers import DataReader, Preprocessor


def main():
    df = DataReader.get_processed_data_from_gcs(raw=True)
    trained_forecasters = []
    storage_client = storage.Client()
    forecasting_models_bucket = storage_client.get_bucket("models_forecasting")
    for feature in FORECAST_FEATURES:
        model = f'RNN_{feature}'
        train_data, test_data = train_test_split(df[feature],
                                                 train_size=0.8,
                                                 shuffle=False)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
        blob = forecasting_models_bucket.blob(f'{model}_scaler.joblib')
        dump(scaler, os.path.join(MODELS_PATH, f'{model}_scaler.joblib'))
        blob.upload_from_filename(os.path.join(MODELS_PATH,
                                               f'{model}_scaler.joblib'),
                                  content_type='application/joblib')
        scaled_test = scaler.transform(test_data.values.reshape(-1, 1))
        x_train, y_train = Preprocessor.create_sequences(scaled_train,
                                                         lookback=TIME_STEPS,
                                                         inference=False)
        x_test, y_test = Preprocessor.create_sequences(scaled_test,
                                                       lookback=TIME_STEPS,
                                                       inference=False)

        forecaster = keras.models.Sequential()
        forecaster.add(
            keras.layers.LSTM(5,
                              input_shape=(x_train.shape[1], x_train.shape[2]),
                              return_sequences=False))
        forecaster.add(keras.layers.Dense(1))
        forecaster.summary()

        forecaster.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        )
        history = forecaster.fit(
            x_train,
            y_train,
            epochs=200,
            batch_size=256,
            validation_split=0.2,
            verbose=VERBOSITY,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=EARLY_STOPPING,
                                              monitor='val_loss',
                                              mode='min',
                                              verbose=VERBOSITY,
                                              restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.75,
                                                  patience=EARLY_STOPPING // 2,
                                                  verbose=VERBOSITY,
                                                  mode='min')
            ])
        trained_forecasters.append(forecaster)
        forecaster.save(os.path.join(MODELS_PATH, f'{model}.h5'))

        fig = go.Figure(go.Scatter(y=history.history['loss'], name='Training'))
        fig.add_scatter(y=history.history['val_loss'], name='Validation')
        fig.update_layout(xaxis=dict(title='Epochs'),
                          yaxis=dict(title=f'{feature} loss'),
                          template='none',
                          legend=dict(orientation='h',
                                      yanchor='bottom',
                                      xanchor='right',
                                      x=1,
                                      y=1.01))
        fig.show()

        test_predict = forecaster.predict(x_test)
        test_predict = scaler.inverse_transform(test_predict)

        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        score_rmse = mean_squared_error(y_test, test_predict, squared=False)
        score_mae = mean_absolute_error(y_test, test_predict)

        x_train_ticks = list(train_data.index)
        x_test_ticks = list(test_data.index)
        fig = go.Figure(
            go.Scatter(x=x_train_ticks,
                       y=y_train.reshape(-1),
                       name='Train',
                       line=dict(color='lightgray', width=1)))
        fig.add_scatter(x=x_test_ticks,
                        y=test_predict.reshape(-1),
                        name='Prediction',
                        line=dict(color='indianred', width=1))
        fig.add_scatter(x=x_test_ticks,
                        y=y_test.reshape(-1),
                        name='Ground truth',
                        line=dict(color='steelblue', width=1))
        fig.update_layout(
            template='none',
            yaxis_title=feature,
            title=
            f'{feature} forecast MAE: {score_mae:.2f}, RMSE: {score_rmse:.2f}',
            legend=dict(orientation='h',
                        yanchor='bottom',
                        xanchor='right',
                        x=1,
                        y=1.01))
        fig.show()


if __name__ == '__main__':
    main()
