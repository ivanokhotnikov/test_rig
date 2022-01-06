import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from utils.config import FEATURES, FEATURES_NO_TIME_AND_COMMANDS
from utils.readers import DataReader, Preprocessor

if __name__ == '__main__':
    os.chdir('..\\..')
    df = DataReader.read_all_raw_data(verbose=True, features_to_read=FEATURES)
    df = Preprocessor.remove_step_zero(df, inplace=False)
    df.sort_values(by=['DATE', 'TIME'], inplace=True, ignore_index=True)
    df['RUNNING TIME'] = pd.date_range(start=f'00:00:00 {df["DATE"].min()}',
                                       periods=len(df),
                                       freq='S')
    df.set_index('RUNNING TIME')
    plot_all = False
    plot_only_trends = True
    model = 'stl'
    decompositions = {feature: [] for feature in FEATURES_NO_TIME_AND_COMMANDS}
    for feature in FEATURES_NO_TIME_AND_COMMANDS:
        if model == 'seasonal_decompose':
            decomposition = seasonal_decompose(
                df[feature],
                model='additive',
                period=5400,
            )
        if model == 'stl':
            decomposition = STL(
                df[feature],
                period=5400,
                seasonal_jump=1000,
                trend_jump=1000,
            ).fit()
        decompositions[feature] = decomposition
        if plot_all:
            for plot in ('observed', 'trend', 'seasonal', 'resid'):
                fig = go.Figure(
                    go.Scatter(x=df['RUNNING TIME'],
                               y=getattr(decomposition, plot)))
                fig.update_layout(title=f'{plot.capitalize()} {feature}',
                                  xaxis_title='Running time',
                                  template='none')
                fig.show()
        if plot_only_trends:
            fig = go.Figure(
                go.Scatter(x=df['RUNNING TIME'],
                           y=getattr(decomposition, 'trend')))
            fig.update_layout(title=f'Trend {feature}',
                              xaxis_title='Running time',
                              template='none')
            fig.show()