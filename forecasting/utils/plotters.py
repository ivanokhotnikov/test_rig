import os
import gc
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (ENGINEERED_FEATURES, IMAGES_PATH,
                     FEATURES_NO_TIME_AND_COMMANDS, PRESSURE_TEMPERATURE)


class Plotter:

    @staticmethod
    def plot_forecast(historical: pd.DataFrame,
                      forecast: np.ndarray,
                      feature: str,
                      new=None,
                      plot_ma_all=False,
                      window=None,
                      plot_each_unit=False,
                      show=False) -> go.Figure | None:
        fig = go.Figure()
        if plot_each_unit:
            for unit in historical['UNIT'].unique():
                for test in historical[historical['UNIT'] ==
                                       unit]['TEST'].unique():
                    fig.add_scatter(
                        x=np.array(
                            (historical[(historical['UNIT'] == unit)
                                        & (historical['TEST'] == test)].index))
                        / 3600,
                        y=historical[(historical['UNIT'] == unit)
                                     & (historical['TEST'] == test)]
                        [feature].values.reshape(-1),
                        line=dict(width=1),
                        opacity=0.2,
                        name=f'{unit}-{test}',
                        showlegend=False)
        else:
            fig.add_scatter(x=np.arange(len(historical)) / 3600,
                            y=historical[feature].values.reshape(-1),
                            line=dict(width=1, color='gray'),
                            opacity=0.2,
                            name='Historical',
                            showlegend=True)
        if new is not None:
            fig.add_scatter(x=np.arange(len(historical),
                                        len(historical) + len(new) + 1) / 3600,
                            y=new[feature].values.reshape(-1),
                            name='New data',
                            line=dict(color='steelblue', width=1.25))
            fig.add_scatter(x=np.arange(
                len(historical) + len(new),
                len(historical) + len(new) + len(forecast) + 1) / 3600,
                            y=forecast.reshape(-1),
                            name='Forecast',
                            line=dict(color='indianred', width=1.25))
            if plot_ma_all and window:
                fig.add_scatter(x=np.arange(
                    len(historical) + len(new) + len(forecast) + 1) / 3600,
                                y=pd.Series(
                                    np.concatenate(
                                        (historical[feature].values,
                                         new[feature].values,
                                         forecast.reshape(-1)
                                         ))).rolling(window).mean().values,
                                name='Moving average trend',
                                line=dict(color='orange', width=1.5))
        else:
            fig.add_scatter(x=np.arange(len(historical),
                                        len(historical) + len(forecast) + 1) /
                            3600,
                            y=forecast.reshape(-1),
                            name='Current forecast',
                            line=dict(color='indianred', width=1.25))
            if plot_ma_all and window:
                fig.add_scatter(
                    x=np.arange(len(historical) + len(forecast) + 1) / 3600,
                    y=pd.Series(
                        np.concatenate(
                            (historical[feature].values, forecast.reshape(-1)
                             ))).rolling(window).mean().values,
                    name='Moving average trend',
                    line=dict(color='orange', width=1.5))

        fig.update_layout(template='none',
                          xaxis=dict(title='Total running time, hours'),
                          yaxis_title=feature,
                          title=f'{feature} forecast',
                          legend=dict(orientation='h',
                                      yanchor='bottom',
                                      xanchor='right',
                                      x=1,
                                      y=1.01))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_ma_trend(df: pd.DataFrame,
                      feature: str,
                      window: int,
                      show=False,
                      ma_df=None):
        fig = go.Figure()
        fig.add_scatter(x=df['TIME'],
                        y=df[feature],
                        name='Observed',
                        line=dict(color='lightgray', width=.25))
        if ma_df is not None:
            fig.add_scatter(x=df.loc[window:, 'TIME'],
                            y=ma_df[f'MA {feature}'],
                            name='Moving average',
                            line=dict(color='orange', width=1.5))
        else:
            fig.add_scatter(x=df['TIME'],
                            y=df[feature].rolling(window).mean(),
                            name='Moving average',
                            line=dict(color='orange', width=1.5))
        fig.update_layout(yaxis_title=feature,
                          xaxis_title='TIME',
                          template='none',
                          legend=dict(orientation='h',
                                      yanchor='bottom',
                                      xanchor='right',
                                      x=1,
                                      y=1.01))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_heatmap(df, features, show=False):
        fig = go.Figure(
            go.Heatmap(x=features,
                       y=features,
                       z=df[features].cov().values,
                       colorscale='inferno'))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    def plot_unit_from_summary_file(unit_id='HYD000091-R1'):
        from readers import DataReader

        reader = DataReader()
        units = reader.read_summary_data()
        unit = units[unit_id]
        cols = unit.columns.to_list()
        num_cols = [
            col for col in cols
            if unit[col].dtype in ('float64',
                                   'int64') and col not in ('NOT USED',
                                                            'M4 SETPOINT')
        ]
        fig = make_subplots(rows=len(num_cols),
                            cols=1,
                            subplot_titles=num_cols)
        for i, col in enumerate(num_cols, start=1):
            fig.append_trace(go.Scatter(x=unit['TIME'], y=unit[col]),
                             row=i,
                             col=1)
        fig.update_layout(template='none',
                          height=8000,
                          title=f'{unit_id}',
                          showlegend=False)
        fig.show()

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_unit(df, unit=89, save=False, show=True):
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure()
            for test in df[df['UNIT'] == unit]['TEST'].unique():
                fig.add_scatter(x=df[(df['UNIT'] == unit)
                                     & (df['TEST'] == test)]['TIME'],
                                y=df[(df['UNIT'] == unit)
                                     & (df['TEST'] == test)][feature],
                                name=f'{unit}-{test}')
            fig.update_layout(template='none',
                              title=f'{feature}',
                              xaxis_title='TIME',
                              yaxis_title=feature,
                              showlegend=True)
            if save:
                fig.write_image(
                    os.path.join(IMAGES_PATH,
                                 f'{feature}_unit_{unit}' + '.png'))
            if show: fig.show()

    @staticmethod
    def plot_unit_per_feature(df,
                              unit=89,
                              feature='M4 ANGLE',
                              save=False,
                              show=True):
        for test in df[(df['UNIT'] == unit)]['TEST'].unique():
            Plotter.plot_unit_per_test_feature(df,
                                               unit=unit,
                                               test=test,
                                               feature=feature,
                                               save=save,
                                               show=show)

    @staticmethod
    def plot_unit_per_test_feature(df,
                                   unit=89,
                                   test=1,
                                   feature='M4 ANGLE',
                                   save=False,
                                   show=True):
        fig = go.Figure()
        fig.add_scatter(x=df[(df['UNIT'] == unit)
                             & (df['TEST'] == test)]['TIME'],
                        y=df[(df['UNIT'] == unit)
                             & (df['TEST'] == test)][feature],
                        name=f'{unit}-{test}')
        fig.update_layout(template='none',
                          title=f'{feature}',
                          xaxis_title='TIME',
                          yaxis_title=feature,
                          showlegend=True)
        if save:
            fig.write_image(
                os.path.join(IMAGES_PATH,
                             f'{feature}_unit_{unit}-{test}' + '.png'))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_unit_per_test_step_feature(df,
                                        unit=89,
                                        test=1,
                                        step=23,
                                        feature='M4 ANGLE',
                                        save=False,
                                        show=True):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df[(df['UNIT'] == unit)
                            & (df['STEP'] == step)
                            & (df['TEST'] == test)]['TIME'],
                       y=df[(df['UNIT'] == unit)
                            & (df['STEP'] == step)
                            & (df['TEST'] == test)][feature],
                       name=f'{unit}-{test}'))
        fig.update_layout(template='none',
                          title=f'Step {step}, {feature}',
                          xaxis_title='TIME',
                          yaxis_title=feature,
                          showlegend=True)
        if save:
            fig.write_image(
                os.path.join(IMAGES_PATH,
                             f'{feature}_unit_{unit}-{test}-{step}' + '.png'))
        if show:
            fig.show()
            return None
        return fig

    @staticmethod
    def plot_all_per_step_feature(df,
                                  step=18,
                                  feature='M4 ANGLE',
                                  single_plot=True):
        if single_plot:
            fig = go.Figure()
            for unit in df['UNIT'].unique():
                for test in df[(df['UNIT'] == unit)]['TEST'].unique():
                    if not df[(df['TEST'] == test) & (df['STEP'] == step) &
                              (df['UNIT'] == unit)].empty:
                        fig.add_trace(
                            go.Scatter(y=df[(df['UNIT'] == unit)
                                            & (df['STEP'] == step)
                                            & (df['TEST'] == test)][feature],
                                       name=f'{unit}-{test}'))
            fig.update_layout(template='none', title=f'Step {step}, {feature}')
        else:
            titles = [
                f'{unit}-{test}' for unit in df['UNIT'].unique()
                for test in df[df['UNIT'] == unit]['TEST'].unique()
                if not df[(df['TEST'] == test) & (df['STEP'] == step)
                          & (df['UNIT'] == unit)][feature].empty
            ]
            fig = make_subplots(rows=len(titles),
                                cols=1,
                                subplot_titles=titles)
            current_row = 1
            for unit in df['UNIT'].unique():
                for test in df[(df['UNIT'] == unit)]['TEST'].unique():
                    if not df[(df['TEST'] == test) & (df['STEP'] == step) &
                              (df['UNIT'] == unit)].empty:
                        fig.append_trace(go.Scatter(
                            x=df[(df['UNIT'] == unit)
                                 & (df['STEP'] == step)
                                 & (df['TEST'] == test)]['TIME'],
                            y=df[(df['UNIT'] == unit)
                                 & (df['STEP'] == step)
                                 & (df['TEST'] == test)][feature]),
                                         row=current_row,
                                         col=1)
                        current_row += 1
            fig.update_layout(template='none',
                              height=20000,
                              title=f'Step {step}, {feature}',
                              showlegend=False)
        fig.show()

    @staticmethod
    def plot_all_per_step(df, step):
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure()
            for unit in df['UNIT'].unique():
                for test in df[(df['UNIT'] == unit)
                               & (df['STEP'] == step)]['TEST'].unique():
                    fig.add_trace(
                        go.Scatter(y=df[(df['UNIT'] == unit)
                                        & (df['STEP'] == step)
                                        & (df['TEST'] == test)][feature],
                                   name=f'{unit}-{test}'))
            fig.update_layout(template='none',
                              title=f'Step {step}, {feature}',
                              xaxis_title='Time')
            fig.show()

    @staticmethod
    def plot_all_means_per_step(df, step):
        units = []
        features_means = {
            feature: []
            for feature in FEATURES_NO_TIME_AND_COMMANDS
        }
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            for unit in df['UNIT'].unique():
                for test in df[(df['STEP'] == step)
                               & (df['UNIT'] == unit)]['TEST'].unique():
                    features_means[feature].append(
                        df[(df['STEP'] == step)
                           & (df['UNIT'] == unit) &
                           (df['TEST'] == test)][feature].mean())
                    units.append(unit)
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure(data=go.Scatter(
                x=units, y=features_means[feature], mode='markers'))
            fig.update_layout(template='none',
                              title=f'Step {step}, {feature} means',
                              xaxis_title='Unit')
            fig.show()

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_anomalies_per_unit(df, unit=89):
        for test in df[(df['UNIT'] == unit)]['TEST'].unique():
            for feature in ENGINEERED_FEATURES + PRESSURE_TEMPERATURE:
                Plotter.plot_anomalies_per_unit_test_feature(df,
                                                             unit=unit,
                                                             test=test,
                                                             feature=feature)

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_anomalies_per_unit_feature(df, unit=89, feature='PT4', show=True):
        for test in df[(df['UNIT'] == unit)]['TEST'].unique():
            Plotter.plot_anomalies_per_unit_test_feature(df,
                                                         unit=unit,
                                                         test=test,
                                                         feature=feature,
                                                         show=show)

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_anomalies_per_unit_test_feature(df,
                                             unit=89,
                                             test=1,
                                             feature='M4 ANGLE',
                                             show=True):
        if any(df[(df['UNIT'] == unit)
                  & (df['TEST'] == test)][f'ANOMALY_{feature}'] == -1):
            fig = go.Figure()
            fig.add_scatter(
                x=df[(df['UNIT'] == unit)
                     & (df['TEST'] == test)]['TIME'],
                y=df[(df['UNIT'] == unit)
                     & (df['TEST'] == test)][feature],
                mode='lines',
                name='Data',
                showlegend=False,
                line={'color': 'steelblue'},
            )
            fig.add_scatter(
                x=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
                     (df[f'ANOMALY_{feature}'] == -1)]['TIME'],
                y=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
                     (df[f'ANOMALY_{feature}'] == -1)][feature],
                mode='markers',
                name='Anomaly',
                line={'color': 'indianred'},
            )
            fig.update_layout(yaxis={'title': feature},
                              template='none',
                              title=f'Unit {unit}-{test}, {feature}')
            if show:
                fig.show()
                return None
            return fig
        else:
            print(
                f'No anomalies found in {feature} during {unit}-{test} test.')
            return None

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_anomalies_per_unit_test_step_feature(df,
                                                  unit=89,
                                                  test=1,
                                                  step=23,
                                                  feature='M4 ANGLE',
                                                  show=False):
        try:
            if any(df[(df['UNIT'] == unit)
                      & (df['TEST'] == test)
                      & (df['STEP'] == step)]['ANOMALY'] == -1):
                fig = go.Figure()
                fig.add_scatter(x=df[(df['UNIT'] == unit)
                                     & (df['TEST'] == test) &
                                     (df['STEP'] == step)]['TIME'],
                                y=df[(df['UNIT'] == unit)
                                     & (df['TEST'] == test) &
                                     (df['STEP'] == step)][feature],
                                mode='lines',
                                showlegend=False,
                                line={'color': 'steelblue'})
                fig.add_scatter(
                    x=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
                         (df['STEP'] == step) & (df['ANOMALY'] == -1)]['TIME'],
                    y=df[(df['UNIT'] == unit) & (df['TEST'] == test) &
                         (df['STEP'] == step) &
                         (df['ANOMALY'] == -1)][feature],
                    mode='markers',
                    name='Anomaly',
                    line={'color': 'indianred'})
                fig.update_layout(yaxis={'title': feature},
                                  template='none',
                                  title=f'Unit {unit}-{test}, step {step}')
                if show:
                    fig.show()
                return fig
            else:
                print(
                    f'No anomalies found in {unit}-{test}, step {step} test.')
                return None
        except KeyError:
            print('No "ANOMALY" column found in the dataframe')

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def plot_all_running_features(df, show=True):
        for feature in FEATURES_NO_TIME_AND_COMMANDS:
            fig = go.Figure()
            fig.add_scatter(x=df['RUNNING TIME'], y=df[feature])
            fig.update_layout(template='none',
                              xaxis_title='RUNNING TIME',
                              yaxis_title=feature)
            if show:
                fig.show()
                return None
            return fig


if __name__ == '__main__':
    import pandas as pd
    from readers import DataReader, Preprocessor
    from config import PREDICTIONS_PATH

    print(os.getcwd())
    os.chdir('..\\..\\..')
    print(os.getcwd())
    data = DataReader.load_data()
    data = Preprocessor.remove_step_zero(data)
    data['ANOMALY'] = pd.read_csv(
        os.path.join(PREDICTIONS_PATH, 'IsolationForest_0212_1742.csv'))
    Plotter.plot_unit_raw_data(data, unit=91, in_time=False, save=False)
    Plotter.plot_unit_from_summary_file(unit_id='HYD000091-R1')
    Plotter.plot_covariance(data)
    Plotter.plot_all_per_step_feature(data, single_plot=False)
    Plotter.plot_all_per_step(data, step=18)
    Plotter.plot_all_means_per_step(data, step=18)
    Plotter.plot_kdes_per_step(data, step=18)
    Plotter.plot_unit_per_step_feature(data,
                                       unit=91,
                                       step=23,
                                       feature='M4 ANGLE')
    Plotter.plot_unit_per_feature(data, unit=91, feature='M4 ANGLE')
    Plotter.plot_anomalies_per_unit_feature(data, unit=91, feature='M4 ANGLE')
    params = {'unit': 18, 'step': 12, 'test': 2}
    for feature in FEATURES_NO_TIME_AND_COMMANDS:
        Plotter.plot_anomalies_per_unit_test_step_feature(data,
                                                          feature=feature,
                                                          show=True,
                                                          **params)
