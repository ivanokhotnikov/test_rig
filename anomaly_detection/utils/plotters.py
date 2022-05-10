import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ENGINEERED_FEATURES, IMAGES_PATH, FEATURES_NO_TIME_AND_COMMANDS, PRESSURE_TEMPERATURE_FEATURES


class Plotter:

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
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
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
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
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
            for feature in ENGINEERED_FEATURES + PRESSURE_TEMPERATURE_FEATURES:
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
                              title=f'Unit {unit}-{test}, {feature}',
                              legend=dict(orientation='h',
                                          yanchor='bottom',
                                          xanchor='right',
                                          x=1,
                                          y=1.01))
            if show:
                fig.show()
                return None
            return fig
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
                                  title=f'Unit {unit}-{test}, step {step}',
                                  legend=dict(orientation='h',
                                              yanchor='bottom',
                                              xanchor='right',
                                              x=1,
                                              y=1.01))
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
