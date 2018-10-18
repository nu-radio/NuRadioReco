from __future__ import absolute_import, division, print_function  # , unicode_literals
from dash.dependencies import Input, Output, State
import dash
import radiotools.helper as hp
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
from plotly import tools
import json
from app import app
import dataprovider
from NuRadioReco.utilities import units
from NuRadioReco.utilities import templates
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
from NuRadioReco.eventbrowser.apps.common import get_point_index
import numpy as np
import logging
logger = logging.getLogger('traces')

provider = dataprovider.DataProvider()
template_provider = templates.Templates()

xcorr_options = [{'label': 'maximum cr x-corr all channels', 'value': 'cr_max_xcorr'},
                {'label': 'maximum of avg cr x-corr in parallel cr channels', 'value': 'cr_avg_xcorr_parallel_crchannels'},
                {'label': 'maximum cr x-corr cr channels', 'value': 'cr_max_xcorr_crchannels'},
                {'label': 'average cr x-corr cr channels', 'value': 'cr_avg_xcorr_crchannels'},
                {'label': 'maximum nu x-corr all channels', 'value': 'nu_max_xcorr'},
                {'label': 'maximum of avg nu x-corr in parallel nu channels', 'value': 'nu_avg_xcorr_parallel_nuchannels'},
                {'label': 'maximum nu x-corr nu channels', 'value': 'nu_max_xcorr_nuchannels'},
                {'label': 'average nu x-corr nu channels', 'value': 'nu_avg_xcorr_nuchannels'}]

layout = html.Div([
    #Sim Traces Plot
    html.Div([
        html.Div([
            html.Div('Correlations', className='panel-heading'),
            html.Div([
                dcc.Dropdown(
                    id='cr-xcorrelation-dropdown',
                    options=xcorr_options,
                    value=xcorr_options[0]['value']
                ),
                html.Div([
                    html.Div([
                        dcc.Graph(id='cr-xcorrelation')
                    ], style={'flex': '1'}),
                    html.Div([
                        dcc.Graph(id='cr-xcorrelation-amplitude')
                    ], style={'flex': '1'})
                ], style={'display': 'flex'})
            ], className='panel-body')
        ], className = 'panel panel-default'),
        html.Div([
            html.Div('Polarization', className='panel-heading'),
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(id='cr-polarization-zenith')
                    ], style={'flex': '1'}),
                    html.Div([], style={'flex': '1'})
                ], style={'display': 'flex'})
            ], className='panel-body')
        ], className='panel panel-default')
    ])
])


@app.callback(Output('cr-xcorrelation', 'figure'),
              [Input('cr-xcorrelation-dropdown', 'value'),
               Input('filename', 'value'),
               Input('event-ids', 'children'),
               Input('station_id', 'children')],
              [State('user_id', 'children')])
def plot_cr_xcorr(xcorr_type, filename, jcurrent_selection, jstation_id, juser_id):
    if filename is None or jstation_id is None:
        return {}
    user_id = json.loads(juser_id)
#     filename = json.loads(jfilename)
    station_id = json.loads(jstation_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if not stnp.cr_xcorrelations in keys:
        return {}
    xcorrs = ariio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.station_time in keys:
        traces.append(go.Scatter(
            x=ariio.get_header()[station_id][stnp.station_time],
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            mode='markers',
            opacity=1
        ))
    else:
        return {}
    # update with current selection
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = get_point_index(trace['text'], current_selection)
    return {
        'data': traces,
        'layout': go.Layout(
#             xaxis={'type': 'linear', 'title': ''},
            yaxis={'title': xcorr_type, 'range': [0, 1]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }

@app.callback(Output('cr-xcorrelation-amplitude', 'figure'),
              [Input('cr-xcorrelation-dropdown', 'value'),
               Input('filename', 'value'),
               Input('event-ids', 'children'),
               Input('station_id', 'children')],
              [State('user_id', 'children')])
def plot_cr_xcorr_amplitude(xcorr_type, filename, jcurrent_selection, jstation_id, juser_id):
    if filename is None:
        return {}
    user_id = json.loads(juser_id)
#     filename = json.loads(jfilename)
    station_id = json.loads(jstation_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    xcorrs = ariio.get_header()[station_id][stnp.cr_xcorrelations]
    if stnp.channels_max_amplitude in keys:
        traces.append(go.Scatter(
            x=ariio.get_header()[station_id][stnp.channels_max_amplitude] / units.mV,
            y=[xcorrs[i][xcorr_type] for i in range(len(xcorrs))],
            text=[str(x) for x in ariio.get_event_ids()],
            mode='markers',
            opacity=1
        ))
    else:
        return {}
    # update with current selection
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = get_point_index(trace['text'], current_selection)

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'maximum amplitude [mV]'},
            yaxis={'title': xcorr_type, 'range': [0, 1]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }

@app.callback(Output('cr-polarization-zenith', 'figure'),
              [Input('filename', 'value'),
               Input('btn-open-file', 'value'),
               Input('event-ids', 'children'),
               Input('station_id', 'children')],
              [State('user_id', 'children')])
def plot_cr_polarization_zenith(filename, btn, jcurrent_selection, jstation_id, juser_id):
    if filename is None or jstation_id is None:
        return {}
    user_id = json.loads(juser_id)
#     filename = json.loads(jfilename)
    station_id = json.loads(jstation_id)
    ariio = provider.get_arianna_io(user_id, filename)
    traces = []
    keys = ariio.get_header()[station_id].keys()
    if stnp.polarization_angle not in keys or stnp.polarization_angle_expectation not in keys or stnp.zenith not in keys:
        return {}
    pol = ariio.get_header()[station_id][stnp.polarization_angle]
    pol = np.abs(pol)
    pol[pol > 0.5 * np.pi] = np.pi - pol[pol > 0.5 * np.pi]
    pol_exp = ariio.get_header()[station_id][stnp.polarization_angle_expectation]
    pol_exp = np.abs(pol_exp)
    pol_exp[pol_exp > 0.5 * np.pi] = np.pi - pol_exp[pol_exp > 0.5 * np.pi]
    traces.append(go.Scatter(
        x=ariio.get_header()[station_id][stnp.zenith] / units.deg,
        y=np.abs(pol - pol_exp) / units.deg,
        text=[str(x) for x in ariio.get_event_ids()],
        mode='markers',
        opacity=1
    ))

    # update with current selection
    current_selection = json.loads(jcurrent_selection)
    if current_selection != []:
        for trace in traces:
            trace['selectedpoints'] = get_point_index(trace['text'], current_selection)

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'linear', 'title': 'zenith angle [deg]'},
            yaxis={'title': 'polarization angle [deg]', 'range': [0, 90]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }