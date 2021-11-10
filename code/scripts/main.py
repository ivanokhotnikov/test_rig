import os
from utils.config import MODELS_PATH, IMAGES_PATH
from utils.readers import DataReader, Preprocessor, ModelReader
from utils.plotters import Plotter

from joblib import dump, load


def main():
    pass


if __name__ == '__main__':
    os.chdir('..\\..')
    clf = load('outputs\\models\\isolation_forest.joblib')