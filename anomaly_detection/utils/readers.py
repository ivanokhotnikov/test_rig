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

from .config import (FEATURES_FOR_ANOMALY_DETECTION, FEATURES_NO_TIME,
                     MODELS_PATH, DATA_PATH, PREDICTIONS_PATH, TIME_STEPS)
from joblib import load


def read_data(mode='processed'):
    if mode == 'raw':
        df = DataReader.read_all_raw_data(
            verbose=True, features_to_read=FEATURES_FOR_ANOMALY_DETECTION)
        df = Preprocessor.remove_step_zero(df)
    if mode == 'processed':
        df = pd.read_csv(os.path.join(DATA_PATH, 'processed',
                                      'combined_data.csv'),
                         parse_dates=True,
                         infer_datetime_format=True,
                         dtype=dict(
                             zip(FEATURES_NO_TIME,
                                 [np.float32] * len(FEATURES_NO_TIME))))
        df[['STEP', 'UNIT', 'TEST',
            'ARMANI']] = df[['STEP', 'UNIT', 'TEST',
                             'ARMANI']].astype(np.uint8)
        df['TIME'] = pd.to_datetime(df['TIME'])
        df['DATE'] = pd.to_datetime(df['DATE'])
    df['DURATION'] = pd.to_timedelta(range(len(df)), unit='s')
    df['TOTAL SECONDS'] = (pd.to_timedelta(range(
        len(df)), unit='s').total_seconds()).astype(np.uint32)
    df['HOURS'] = (df['TOTAL SECONDS'] // 3600).astype(np.uint16)
    df['MINUTES'] = (df['TOTAL SECONDS'] % 3600 // 60).astype(np.uint8)
    df['SECONDS'] = (df['TOTAL SECONDS'] % 60).astype(np.uint8)
    return df


def get_preprocessed_data(raw=False,
                          features_to_read=FEATURES_FOR_ANOMALY_DETECTION):
    df = DataReader.load_data(raw=raw, features_to_read=features_to_read)
    df = Preprocessor.remove_step_zero(df)
    df = Preprocessor.feature_engineering(df)
    return df


class DataReader:

    @staticmethod
    def read_all_raw_data(verbose=True,
                          features_to_read=FEATURES_FOR_ANOMALY_DETECTION):
        current_df = pd.DataFrame()
        final_df = pd.DataFrame()
        units = []
        for file in os.listdir(os.path.join(DATA_PATH, 'raw')):
            if verbose:
                print(f'Reading {file}')
            try:
                if file.endswith('.csv'):
                    current_df = pd.read_csv(os.path.join(
                        DATA_PATH, 'raw', file),
                                             usecols=features_to_read,
                                             infer_datetime_format=True,
                                             index_col=False)
                elif file.endswith('.xlsx'):
                    current_df = pd.read_excel(os.path.join(
                        DATA_PATH, 'raw', file),
                                               usecols=features_to_read,
                                               index_col=False)
            except ValueError:
                if verbose:
                    print(f'{file} got a faulty header')
                continue
            current_df[FEATURES_NO_TIME] = current_df[FEATURES_NO_TIME].apply(
                pd.to_numeric, errors='coerce', downcast='float')
            current_df = current_df.dropna(axis=0)
            name_list = file.split('-')
            try:
                unit = np.uint8(name_list[0][-3:].lstrip('0D'))
            except ValueError:
                unit = np.uint8(name_list[0].split('_')[0][-3:].lstrip('0D'))
            units.append(unit)
            current_df['ARMANI'] = 1 if name_list[0][3] == '2' else 0
            current_df['ARMANI'] = current_df['ARMANI'].astype(np.uint8)
            current_df['UNIT'] = unit
            current_df['TEST'] = np.uint8(units.count(unit))
            current_df['STEP'] = current_df['STEP'].astype(np.uint8)
            current_df['TIME'] = pd.to_datetime(current_df['TIME'],
                                                errors='coerce').dt.time
            final_df = pd.concat((final_df, current_df), ignore_index=True)
        if verbose: print('Reading done!')
        return final_df

    @staticmethod
    def read_raw_unit_data(unit='HYD000091-R1_RAW',
                           features_to_read=FEATURES_FOR_ANOMALY_DETECTION):
        try:
            unit_df = pd.read_csv(os.path.join(DATA_PATH, 'raw',
                                               unit + '.csv'),
                                  usecols=features_to_read,
                                  index_col=False,
                                  header=0)
        except:
            try:
                unit_df = pd.read_excel(os.path.join(DATA_PATH, 'raw',
                                                     unit + '.xlsx'),
                                        usecols=features_to_read,
                                        index_col=False,
                                        header=0)
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
        df = pd.read_csv(
            os.path.join(DATA_PATH, 'processed', 'combined_data.csv'),
            usecols=[
                f for f in FEATURES_FOR_ANOMALY_DETECTION + ['UNIT', 'TEST']
                if f not in ('Vibration 1', ' Vibration 2', ' DATE')
            ],
            index_col=False)
        df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].apply(pd.to_numeric,
                                                          errors='coerce')
        df[FEATURES_NO_TIME] = df[FEATURES_NO_TIME].astype(np.float32)
        df[['STEP', 'UNIT', 'TEST']] = df[['STEP', 'UNIT',
                                           'TEST']].astype(np.uint8)
        print('Reading done')
        return df

    @staticmethod
    def read_summary_data(verbose=True):
        try:
            if verbose:
                print(f'Reading the summsry file.')
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
                  features_to_read=FEATURES_FOR_ANOMALY_DETECTION):
        if read_all:
            if raw:
                return cls.read_all_raw_data(verbose=verbose,
                                             features_to_read=features_to_read)
            else:
                return cls.read_combined_data()
        else:
            if raw:
                return cls.read_raw_unit_data(unit=unit)
            else:
                return pd.DataFrame(cls.read_summary_data())

    @staticmethod
    def check_if_in_bucket(csv_file):
        storage_client = storage.Client()
        try:
            bucket = storage_client.get_bucket('test_rig_data')
        except:
            bucket = storage_client.get_bucket('rig_data')
        raw_folder_content = {
            blob.name[4:]
            for blob in list(bucket.list_blobs(prefix='raw'))
        }
        return bucket, csv_file.name in raw_folder_content

    @classmethod
    def read_newcoming_data(cls, csv_file):
        bucket, in_bucket = cls.check_if_in_bucket(csv_file)
        if in_bucket:
            st.write(f'{csv_file.name} in the GCS bucket {bucket.name}')
        else:
            st.write(f'{csv_file.name} not in the GCS bucket {bucket.name}')
            blob = bucket.blob(f'raw/{csv_file.name}')
            blob.upload_from_file(csv_file, content_type='text/csv')
            st.write(
                f'{csv_file.name} uploaded to the GCS bucket {bucket.name}')
        blob = bucket.get_blob(f'raw/{csv_file.name}')
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                         usecols=FEATURES_FOR_ANOMALY_DETECTION,
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
    def read_predictions(file):
        return pd.read_csv(os.path.join(PREDICTIONS_PATH, file),
                           index_col=False)


class Preprocessor:

    @staticmethod
    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i:(i + time_steps)])
        return np.stack(output)

    @staticmethod
    def remove_step_zero(df):
        return df.drop(df[df['STEP'] == 0].index,
                       axis=0).reset_index(drop=True)

    @staticmethod
    def get_warm_up_steps(df):
        return df[(df['STEP'] >= 1) & (df['STEP'] <= 11)]

    @staticmethod
    def get_break_in_steps(df):
        return df[(df['STEP'] >= 12) & (df['STEP'] <= 22)]

    @staticmethod
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
    def read_model(model, task='anomaly_detection', extension='.joblib'):
        if task == 'anomaly_detection':
            if 'Autoencoder' in model:
                if 'scaler' in model:
                    return load(os.path.join(MODELS_PATH, model + '.joblib'))
                if 'threshold' in model:
                    with open(os.path.join(MODELS_PATH, model + '.txt'),
                              'r') as file:
                        threshold = np.float16(file.read().rstrip())
                    return threshold
                return keras.models.load_model(
                    os.path.join(MODELS_PATH, model + '.h5'))
            return load(os.path.join(MODELS_PATH, model + extension))
        elif task == 'decomopose':
            return load(
                os.path.join(MODELS_PATH, 'decomposers', model + extension))
        elif task == 'forecast':
            return load(
                os.path.join(MODELS_PATH, 'forecasters', model + extension))

    @staticmethod
    def read_model_from_gcs(model):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('models_detection')
        if 'Autoencoder' in model:
            if 'scaler' in model:
                blob = bucket.get_blob(model + '.joblib')
                data_bytes = blob.download_as_bytes()
                return load(io.BytesIO(data_bytes))
            if 'threshold' in model:
                blob = bucket.get_blob(model + '.txt')
                return np.float16(blob.download_as_string())
            fs = gcsfs.GCSFileSystem()
            with fs.open(f'gs://models_detection/{model}.h5',
                         'rb') as model_file:
                model_gcs = h5py.File(model_file, 'r')
                return keras.models.load_model(model_gcs)
        blob = bucket.get_blob(model + '.joblib')
        data_bytes = blob.download_as_bytes()
        return load(io.BytesIO(data_bytes))


if __name__ == '__main__':
    scalers = {}
    thresholds = {}
    detectors = {}
    for file in os.listdir(MODELS_PATH):
        if 'Autoencoder' in file:
            if 'scaler' in file:
                scalers[file[:-14]] = ModelReader.read_model_from_gcs(
                    file[:-14])
            elif 'threshold' in file:
                thresholds[file[:-14]] = ModelReader.read_model_from_gcs(
                    file[:-14])
            else:
                detectors[file[:-3]] = ModelReader.read_model_from_gcs(
                    file[:-3])
        else:
            detectors[file[:-7]] = ModelReader.read_model_from_gcs(file[:-7])