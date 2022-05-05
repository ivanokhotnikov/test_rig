from utils.readers import DataReader, Preprocessor

def gcs_preprocess_function():
    return DataReader.get_processed_data_from_gcs(raw=False)