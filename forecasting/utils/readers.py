import gc
import io
import os

import gcsfs
import h5py

import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import storage
from joblib import load
from tensorflow import keras

from utils.config import (DATA_PATH, FEATURES_NO_TIME, FORECAST_FEATURES,
                          LOCAL_DATA_PATH, MODELS_PATH, PREDICTIONS_PATH,
                          RAW_FORECAST_FEATURES, TIME_FEATURES, TIME_STEPS)


class DataReader:

    @staticmethod
    def get_processed_data_from_gcs(raw=False):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('rig_data')
        if raw:
            print(f'Reading raw data from gcs')
            final_df = pd.DataFrame()
            units = []
            for blob in bucket.list_blobs(prefix='raw'):
                data_bytes = blob.download_as_bytes()
                try:
                    if blob.name.endswith('.csv'):
                        current_df = pd.read_csv(io.BytesIO(data_bytes),
                                                 usecols=RAW_FORECAST_FEATURES,
                                                 infer_datetime_format=True,
                                                 index_col=False)
                    elif blob.name.endswith('.xlsx') or blob.name.endswith(
                            '.xls'):
                        current_df = pd.read_excel(
                            io.BytesIO(data_bytes),
                            usecols=RAW_FORECAST_FEATURES,
                            index_col=False)
                except:
                    print(f'Can\'t read {blob.name}')
                    continue
                current_df[FEATURES_NO_TIME] = current_df[
                    FEATURES_NO_TIME].apply(pd.to_numeric,
                                            errors='coerce',
                                            downcast='float')
                if len(current_df['STEP'].unique()) == 36:
                    current_df.dropna(inplace=True)
                    name_list = blob.name.split('-')
                    try:
                        unit = np.uint8(name_list[0][-3:].lstrip('0D'))
                    except ValueError:
                        unit = np.uint8(
                            name_list[0].split('_')[0][-3:].lstrip('0D'))
                    units.append(unit)
                    current_df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
                    current_df['ARMANI'] = current_df['ARMANI'].astype(
                        np.uint8)
                    current_df['UNIT'] = unit
                    current_df['TEST'] = np.uint8(units.count(unit))
                    current_df['STEP'] = current_df['STEP'].astype(np.uint8)
                    current_df['TIME'] = pd.to_datetime(
                        current_df['TIME'], errors='coerce').dt.time
                    current_df[' DATE'] = pd.to_datetime(current_df[' DATE'],
                                                         errors='coerce')
                    final_df = pd.concat((final_df, current_df),
                                         ignore_index=True)
                del current_df
                gc.collect()
            try:
                final_df.sort_values(by=[' DATE', 'TIME'],
                                     inplace=True,
                                     ignore_index=True)
            except:
                print('Can\'t sort dataframe')
            final_df = Preprocessor.remove_step_zero(final_df)
            final_df['TIME'] = pd.to_datetime(
                range(len(final_df)),
                unit='s',
                origin=f'{final_df[" DATE"].min()} 00:00:00')
            final_df['DURATION'] = pd.to_timedelta(range(len(final_df)),
                                                   unit='s')
            final_df['TOTAL SECONDS'] = (pd.to_timedelta(
                range(len(final_df)),
                unit='s').total_seconds()).astype(np.uint64)
            final_df['RUNNING HOURS'] = (final_df['TOTAL SECONDS'] /
                                         3600).astype(np.float64)
            final_df = Preprocessor.feature_engineering(final_df)
            return final_df

        else:
            print(f'Reading processed data from gcs')
            blob = bucket.get_blob('processed/forecast_data.csv')
            forecast_data_bytes = blob.download_as_bytes()
            final_df = pd.read_csv(io.BytesIO(forecast_data_bytes))
            final_df[FORECAST_FEATURES] = final_df[FORECAST_FEATURES].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            return final_df

    @classmethod
    def get_processed_data(cls,
                           raw=False,
                           local=True,
                           features_to_read=RAW_FORECAST_FEATURES,
                           verbose=False):
        if local:
            if verbose:
                print(f'Reading processed local data from {LOCAL_DATA_PATH}\\')
            df = pd.read_csv(os.path.join(LOCAL_DATA_PATH,
                                          'forecast_data.csv'))
            df[FORECAST_FEATURES] = df[FORECAST_FEATURES].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            if verbose:
                print('Reading done!')
            return df
        if not raw:
            if verbose:
                print(f'Reading processed data from {DATA_PATH}\\processed\\')
            df = pd.read_csv(os.path.join(DATA_PATH, 'processed',
                                          'combined_timed_data.csv'),
                             parse_dates=True,
                             infer_datetime_format=True,
                             dtype=dict(
                                 zip(FEATURES_NO_TIME,
                                     [np.float32] * len(FEATURES_NO_TIME))))
            df[['STEP', 'UNIT', 'TEST',
                'ARMANI']] = df[['STEP', 'UNIT', 'TEST',
                                 'ARMANI']].astype(np.uint8)
            df['TIME'] = pd.to_datetime(df['TIME'])
            df[' DATE'] = pd.to_datetime(df[' DATE'])
            df[FORECAST_FEATURES] = df[FORECAST_FEATURES].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            if verbose:
                print('Reading done!')
            return df
        if verbose:
            print(f'Reading raw data from {DATA_PATH}\\raw\\')
        df = cls.read_all_raw_data(features_to_read=features_to_read)
        df = Preprocessor.remove_step_zero(df)
        df['TIME'] = pd.to_datetime(range(len(df)),
                                    unit='s',
                                    origin=f'{df[" DATE"].min()} 00:00:00')
        df['DURATION'] = pd.to_timedelta(range(len(df)), unit='s')
        df['TOTAL SECONDS'] = (pd.to_timedelta(range(
            len(df)), unit='s').total_seconds()).astype(np.uint64)
        df['RUNNING HOURS'] = (df['TOTAL SECONDS'] / 3600).astype(np.float64)
        df = Preprocessor.feature_engineering(df)
        if verbose:
            print('Reading done!')
        return df

    @staticmethod
    def read_newcoming_data(csv_file):
        df = pd.read_csv(csv_file,
                         usecols=RAW_FORECAST_FEATURES,
                         index_col=False)
        df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                          errors='coerce',
                                                          downcast='float')
        df = df.dropna(axis=0)
        name_list = csv_file.name.split('-')
        try:
            unit = np.uint8(name_list[0][-3:].lstrip('0D'))
        except ValueError:
            unit = np.uint8(name_list[0].split('_')[0][-3:].lstrip('0D'))
        df['UNIT'] = unit
        df['STEP'] = df['STEP'].astype(np.uint8)
        #TODO retesting case needs to be addressed
        df['TEST'] = np.uint8(1)
        return df

    @staticmethod
    def read_ma_data():
        df = pd.read_csv(os.path.join(LOCAL_DATA_PATH, 'forecast_ma_data.csv'),
                         dtype=np.float32)
        return df

    @staticmethod
    def read_all_raw_data(features_to_read=RAW_FORECAST_FEATURES):
        print(f'Reading raw data from {DATA_PATH}\\raw\\')
        final_df = pd.DataFrame()
        units = []
        for file in os.listdir(os.path.join(DATA_PATH, 'raw')):
            print(f'Reading {file}')
            try:
                if file.endswith('.csv'):
                    current_df = pd.read_csv(os.path.join(
                        DATA_PATH, 'raw', file),
                                             usecols=features_to_read,
                                             infer_datetime_format=True,
                                             index_col=False)
                elif file.endswith('.xlsx') or file.endswith('.xls'):
                    current_df = pd.read_excel(os.path.join(
                        DATA_PATH, 'raw', file),
                                               usecols=features_to_read,
                                               index_col=False)
            except:
                print(f'Can\'t read {file}')
                continue
            current_df[FEATURES_NO_TIME] = current_df[FEATURES_NO_TIME].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            if len(current_df['STEP'].unique()) == 36:
                current_df.dropna(inplace=True)
                name_list = file.split('-')
                try:
                    unit = np.uint8(name_list[0][-3:].lstrip('0D'))
                except ValueError:
                    unit = np.uint8(
                        name_list[0].split('_')[0][-3:].lstrip('0D'))
                units.append(unit)
                current_df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
                current_df['ARMANI'] = current_df['ARMANI'].astype(np.uint8)
                current_df['UNIT'] = unit
                current_df['TEST'] = np.uint8(units.count(unit))
                current_df['STEP'] = current_df['STEP'].astype(np.uint8)
                current_df['TIME'] = pd.to_datetime(current_df['TIME'],
                                                    errors='coerce').dt.time
                current_df[' DATE'] = pd.to_datetime(current_df[' DATE'],
                                                     errors='coerce')
                final_df = pd.concat((final_df, current_df), ignore_index=True)
            del current_df
            gc.collect()
        try:
            final_df.sort_values(by=[' DATE', 'TIME'],
                                 inplace=True,
                                 ignore_index=True)
        except:
            print('Can\'t sort dataframe')
        print('Reading done!')
        return final_df

    @staticmethod
    def read_raw_unit_data(unit='HYD000091-R1_RAW',
                           features_to_read=RAW_FORECAST_FEATURES):
        try:
            unit_df = pd.read_csv(os.path.join(DATA_PATH, 'raw',
                                               unit + '.csv'),
                                  usecols=features_to_read,
                                  index_col=False)
        except:
            try:
                unit_df = pd.read_excel(os.path.join(DATA_PATH, 'raw',
                                                     unit + '.xlsx'),
                                        usecols=features_to_read,
                                        index_col=False)
            except:
                print(f'No {unit} file found')
                return None
        unit_df[FEATURES_NO_TIME] = unit_df[FEATURES_NO_TIME].astype(
            np.float32)
        unit_df = unit_df.dropna(axis=0)
        return unit_df

    @staticmethod
    def read_combined_data():
        print('Reading "combined_data.csv"')
        df = pd.read_csv(os.path.join(DATA_PATH, 'processed',
                                      'combined_timed_data.csv'),
                         usecols=RAW_FORECAST_FEATURES,
                         dtype=dict(
                             zip(FEATURES_NO_TIME,
                                 [np.float32] * len(FEATURES_NO_TIME))),
                         index_col=False)
        print('Reading done')
        return df

    @staticmethod
    def read_summary_data(verbose=True):
        try:
            if verbose:
                print(f'Reading the summary file.')
            xl = pd.ExcelFile(
                os.path.join(DATA_PATH, 'processed',
                             'report template-V4.xlsx'))
            units = {}
            for sheet in xl.sheet_names:
                if 'HYD' in sheet:
                    if verbose:
                        print(f'Reading {sheet}')
                    units[f'{sheet}'] = pd.read_excel(xl, sheet_name=sheet)
            if verbose:
                print(f'Done')
            return units
        except:
            print('No "report template-V4.xlsx" found')
            return None

    @classmethod
    def load_data(cls,
                  read_all=True,
                  raw=False,
                  unit=None,
                  verbose=True,
                  features_to_read=RAW_FORECAST_FEATURES):
        if read_all:
            if raw:
                return cls.read_all_raw_data(features_to_read=features_to_read)
            else:
                return cls.read_combined_data()
        else:
            if raw:
                return cls.read_raw_unit_data(unit=unit)
            else:
                return pd.DataFrame(cls.read_summary_data())

    @staticmethod
    def read_predictions(file):
        return pd.read_csv(os.path.join(PREDICTIONS_PATH, file),
                           index_col=False)


class Preprocessor:

    @staticmethod
    def create_sequences(values, lookback=TIME_STEPS, inference=False):
        X, Y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i - lookback:i])
            Y.append(values[i])
        if inference:
            return np.stack(X)
        return np.stack(X), np.stack(Y)

    @staticmethod
    def remove_step_zero(df):
        return df.drop(df[df['STEP'] == 0].index,
                       axis=0).reset_index(drop=True)

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def get_warm_up_steps(df):
        return df[(df['STEP'] >= 1) & (df['STEP'] <= 11)]

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def get_break_in_steps(df):
        return df[(df['STEP'] >= 12) & (df['STEP'] <= 22)]

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def get_performance_check_steps(df):
        return df[(df['STEP'] >= 23) & (df['STEP'] <= 33)]

    @staticmethod
    def feature_engineering(df):
        df['DRIVE POWER'] = (df['M1 SPEED'] * df['M1 TORQUE'] * np.pi / 30 /
                             1e3).astype(np.float32)
        df['LOAD POWER'] = abs(df['D1 RPM'] * df['D1 TORQUE'] * np.pi / 30 /
                               1e3).astype(np.float32)
        df['CHARGE MECH POWER'] = (df['M2 RPM'] * df['M2 Torque'] * np.pi /
                                   30 / 1e3).astype(np.float32)
        df['CHARGE HYD POWER'] = (df['CHARGE PT'] * 1e5 * df['CHARGE FLOW'] *
                                  1e-3 / 60 / 1e3).astype(np.float32)
        df['SERVO MECH POWER'] = (df['M3 RPM'] * df['M3 Torque'] * np.pi / 30 /
                                  1e3).astype(np.float32)
        df['SERVO HYD POWER'] = (df['Servo PT'] * 1e5 * df['SERVO FLOW'] *
                                 1e-3 / 60 / 1e3).astype(np.float32)
        df['SCAVENGE POWER'] = (df['M5 RPM'] * df['M5 Torque'] * np.pi / 30 /
                                1e3).astype(np.float32)
        df['MAIN COOLER POWER'] = (df['M6 RPM'] * df['M6 Torque'] * np.pi /
                                   30 / 1e3).astype(np.float32)
        df['GEARBOX COOLER POWER'] = (df['M7 RPM'] * df['M7 Torque'] * np.pi /
                                      30 / 1e3).astype(np.float32)
        return df


class ModelReader:

    @staticmethod
    def read_model(model):
        if 'scaler' in model:
            return load(os.path.join(MODELS_PATH, model + '.joblib'))
        return keras.models.load_model(os.path.join(MODELS_PATH,
                                                    model + '.h5'))

    @staticmethod
    def read_model_from_gcs(model):
        print('Reading model from GCS')
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('forecasters')
        if 'scaler' in model:
            blob = bucket.get_blob(model + '.joblib')
            data_bytes = blob.download_as_bytes()
            return load(io.BytesIO(data_bytes))
        fs = gcsfs.GCSFileSystem()
        with fs.open(f'gs://forecasters/{model}.h5', 'rb') as model_file:
            model_gcs = h5py.File(model_file, 'r')
            return keras.models.load_model(model_gcs)


if __name__ == '__main__':
    raw_data_df = DataReader.get_processed_data(
        raw=True, features_to_read=RAW_FORECAST_FEATURES)
    combined_data_df = DataReader.get_processed_data(
        raw=False, features_to_read=RAW_FORECAST_FEATURES)
