#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:00:13 2019

@author: vincent roy
"""

# Dashboard
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_daq as daq


# Classicals
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_validate

# from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


from os import getcwd, path
data_path = path.join(getcwd(), 'data')
# to join a file with a path : path.join(data_path, 'name.csv')

# Preparation
df_1 = pd.read_csv(path.join(data_path, 'dashboard_airlinequality.csv'))
df_2 = pd.read_csv(path.join(data_path, 'dashboard_tab2.csv'))
df_2_2 = pd.read_csv(path.join(data_path, 'wordcloud_data.csv'))
df_3_1 = pd.read_csv(path.join(data_path, 'dashboard_tab3.csv'))
df_3_2 = pd.read_csv(path.join(data_path, 'query_tab3.csv'))
df_journey = pd.read_csv(path.join(data_path, 'customerjourney.csv'))
df_products = pd.read_csv(path.join(data_path, 'Products.csv'))

cols_15 = [
    'Seat Comfort',
    'Cabin Staff Service',
    'Ground Service',
    'Value For Money',
    'Food & Beverages',
    'Inflight Entertainment'
]
li_seat_type = [
    'Economy Class',
    'Business Class',
    'Premium Economy',
    'First Class'
]
li_type_of_traveller = [
    'Solo Leisure',
    'Business',
    'Family Leisure',
    'Couple Leisure'
]
li_flight_length = [
    'Short-Haul',
    'Middle-Haul',
    'Long-Haul'
]
li_companies = df_journey['Company'].values
li_airplanes = df_journey['Airplane'].values
# Running the dashboard

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# A function to output the chart
def output_chart_1_1(df = df_1,
                 seat_type = 'Economy Class',
                 type_of_travaller = 'Solo Leisure',
                 stop = False,
                 flight_length = 'Short-Haul'):

    # Select data by category
    mask = (df['Seat Type'].isin([seat_type])) &\
            (df['Type Of Traveller'].isin([type_of_travaller])) &\
            (df['Is_stop'].isin([stop])) &\
            (df['Flight Length'].isin([flight_length]))
    df = df[mask]

    # Fill nans with mean
    df = df.fillna(df.mean())

    # Make X and y
    X = df[cols_15]
    y = df['Recommended']

    # Calculate coefficients
    model = linear_model.LogisticRegression(solver='lbfgs', C=0.05)
    try :
        model.fit(X, y)
        df_coef = pd.DataFrame({'Feature':X.columns.to_list(),
                        'Coef':model.coef_.tolist()[0]})
        df_coef = df_coef.sort_values(['Coef'])
        df_coef['Importance'] = np.exp(df_coef['Coef']) - 1
        # Calculate some statistics
        model_accuracy = cross_validate(model, X, y, cv=5, return_train_score=True)['test_score'].mean()
        n_samples = X.shape[0]


        # Visualize coefficients
        # Intepretation: If the coefficient for Inflight Entertainment is 0.2,
        #                an increase in a star in Inflight Entertainment will make a customer 22% (exp(0.2)=1.22)
        #                more likely to recommend the flight
        colors_list = [
            'rgb(74, 197, 255)',
            'rgb(97, 236, 255)',
            'rgb(110, 255, 218)',
            'rgb(216, 255, 128)',
            'rgb(255, 165, 67)',
            'rgb(255, 86, 65)',
        ]
        df_coef['Feature'] = df_coef['Feature'].map(lambda x: x + '  ')
        fig = px.bar(
            df_coef,
            y='Feature',
            x='Importance',
            orientation='h',
        )
        fig.update_layout(
            xaxis_title = '',
            yaxis_title = '',
            xaxis_tickformat = ',.0%',
            showlegend=False,
            font={'size': 16},
            margin=dict(l=1, r=1, t=1, b=1),)
        fig.update(
            layout=dict(
                title=dict(
                    x=0.5,
                    y = 0.98
                )
            )
        )
        fig.update_yaxes(
            tickfont=dict(
                family='Arial',
                color='rgb(159, 166, 183)',
                size= 10,
            )
        )
        fig.update_xaxes(
            tickfont=dict(
                family='Arial',
                color='rgb(159, 166, 183)',
                size=20
            )
        )

        # fig.update_xaxes(
        #     title_font=dict(
        #         size=18,
        #         family='Courier',
        #         color='crimson'))
        # fig.update_yaxes(title_font=dict(size=18, family='Courier', color='crimson'))
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        fig.update_traces(
            marker_color= colors_list,
            marker_line_color='rgb(240,240,240)',
            marker_line_width=1.5,
            opacity=0.9
        )
    except:
        fig = go.Figure()
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        annotations = []
        annotations.append(dict(xref='paper',
                                yref='paper',
                                x=0.5,
                                y=0.5,
                                xanchor='center',
                                yanchor='middle',
                                text='No data for this combination',
                                font=dict(family='Arial',
                                          size=20,
                                          color='rgb(255, 86, 65)'),
                                          showarrow=False))
        fig.update_layout(annotations=annotations)
        fig.update_layout(
            xaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                linecolor='rgba(0, 0, 0, 0)',
                linewidth=2,
                tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgba(0, 0, 0, 0)',
                ),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                linecolor='rgba(0, 0, 0, 0)',
                tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgba(0, 0, 0, 0)',
                )
            )
        )
    return fig

def output_chart_2(df = df_2,
                 choice_class = 'Economy Class',
                 choice_flight_length = ['Short-Haul']):

    # Select data by category
    mask = (df['Class'].isin([choice_class])) &\
           (df['Flight Length'].isin(choice_flight_length))
    df = df[mask]

    # Calculate the mean ratings
    cols_ratings = ['Cabin Staff Service', 'Food & Beverages','Ground Service',
                    'Inflight Entertainment', 'Seat Comfort', 'Value for Money']
    df_for_viz = df.groupby('Flight Length')[cols_ratings].mean().reset_index()
    df_for_viz = df_for_viz.melt(id_vars=['Flight Length'], var_name='Feature', value_name='Stars')



    # Visualize ratings
    try:
        fig = px.line_polar(df_for_viz, r='Stars', theta='Feature', color='Flight Length', line_close=True, template="plotly_dark")\
        .for_each_trace(lambda t: t.update(name=t.name.replace("Flight Length=","")))
        #fig.update_traces(fill='toself')
        fig.update_layout(
          polar=dict(
            radialaxis=dict(
              visible=True,
              range=[0, 5]
            )),
          showlegend=True
        )
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        fig.update(
            layout=dict(
                title=dict(
                    x=0.5,
                    y = 0.98
                )
            )
        )
    except:
        fig = go.Figure()
        fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        annotations = []
        annotations.append(dict(xref='paper',
                                yref='paper',
                                x=0.5,
                                y=0.5,
                                xanchor='center',
                                yanchor='middle',
                                text='Please select at least one distance',
                                font=dict(family='Arial',
                                          size=20,
                                          color='rgb(255, 86, 65)'),
                                          showarrow=False))
        fig.update_layout(annotations=annotations)
        fig.update_layout(
            xaxis=dict(
                showline=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                linecolor='rgba(0, 0, 0, 0)',
                linewidth=2,
                tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgba(0, 0, 0, 0)',
                ),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
                linecolor='rgba(0, 0, 0, 0)',
                tickfont=dict(
                    family='Arial',
                    size=16,
                    color='rgba(0, 0, 0, 0)',
                )
            )
        )
    return fig
    df = df_2
    # Select data by category
    mask = (df['Class'].isin([choice_class])) &\
           (df['Flight Length'].isin(choice_flight_length))
    df = df[mask]

    # Calculate the mean ratings
    cols_ratings = [
        'Cabin Staff Service',
        'Food & Beverages',
        'Ground Service',
        'Inflight Entertainment',
        'Seat Comfort',
        'Value for Money'
    ]
    df_for_viz = df.groupby('Flight Length')[cols_ratings].mean().reset_index()
    df_for_viz = df_for_viz.melt(id_vars=['Flight Length'],
                                 var_name='Feature',
                                 value_name='Stars')

    # Calculate some statistics
    n_samples = df.shape[0]


    # Visualize ratings
    fig = px.line_polar(df_for_viz,
                        r='Stars',
                        theta='Feature',
                        color='Flight Length',
                        line_close=True)\
    .for_each_trace(lambda t: t.update(name=t.name.replace("Flight Length=", "")))
    #fig.update_traces(fill='toself')
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 5]
        )),
      showlegend=True
    )
    fig.update_layout(title='What do Customers Make of the Flights',
                      title_x=0.5,
                      font={'size': 16})
    return fig


# Layout
tab1 = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'Ratings per topic',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                html.Label('Class'),
                                dcc.Dropdown(
                                    id='class_tab12',
                                    options=[{'label': i, 'value': i} for i in li_seat_type],
                                    multi=False,
                                    value='Economy Class'
                                )
                             ],
                                md=6
                            ),
                            dbc.Col(
                                [
                                    daq.ToggleSwitch(
                                        id='short_haul_bt',
                                        label='Short Haul',
                                        labelPosition='bottom',
                                        value= True
                                    )
                                ],
                                md=6,
                                className = 'toggleswitch_tab1'
                         )
                        ]
                    ),
                    dbc.Row(
                        [
                             dbc.Col(
                                  [
                                     daq.ToggleSwitch(
                                         id='middle_haul_bt',
                                         label='Middle Haul',
                                         labelPosition='bottom',
                                         value= False
                                     )
                                 ],
                                  md=6,
                                 className = 'toggleswitch_tab1'
                             ),
                             dbc.Col(
                                 [
                                    daq.ToggleSwitch(
                                        id='long_haul_bt',
                                        label='Long Haul',
                                        labelPosition='bottom',
                                        value= False
                                    )
                                ],
                                 md=6,
                                 className = 'toggleswitch_tab1'
                             )
                        ]
                    ),
                    dcc.Graph(
                                id='chart_spider'
                            )
                ],
                md=6
            ),
            dbc.Col(
                [
                    dbc.Row(
                        html.Div(
                            'What Contribute to Customer Satisfaction',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label('Class'),
                                    dcc.Dropdown(
                                        id='class_tab1',
                                        options=[{'label': i, 'value': i} for i in li_seat_type],
                                        multi=False,
                                        value='Economy Class'
                                    )
                                ],
                                md=6
                            ),
                            dbc.Col(
                                [
                                    html.Label('Travel Purpose'),
                                    dcc.Dropdown(
                                        id='purpose_tab1',
                                        options=[{'label': i, 'value': i} for i in li_type_of_traveller],
                                        multi=False,
                                        value='Solo Leisure'
                                    ),
                                ],
                                md=6
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label('Flight Length'),
                                    dcc.Dropdown(
                                        id='length_tab1',
                                        options=[{'label': i.replace('-',' '), 'value': i} for i in li_flight_length],
                                        multi=False,
                                        value='Short-Haul'
                                    ),
                               ],
                                md=6
                            ),
                            dbc.Col(
                                [
                                    html.Label('Is there a stop'),
                                    dcc.Dropdown(
                                        id='stop_tab1',
                                        options=[{'label': 'Non-stop', 'value': True},
                                                 {'label': 'Stop', 'value': False}],
                                        multi=False,
                                        value=True
                                    )
                                ],
                                md=6
                            )
                        ]
                    ),
                    dbc.Row(
                        html.Div(
                            ' ',
                            className = 'chart_tab1_title'
                        )
                    ),
                    dcc.Graph(
                        id='chart_tab1'
                    )
                ],
                md=6
            )
        ]
    )
)
# A helper function to create a world cloud
def createWordcloud(wc):

    # Limit the length of the text for the sake of speed
    # Comment out this line when you are on a server
    wc = wc[:752041]

    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color='white').generate(wc)

    # Display the generated image:
    fig = plt.gcf()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.figure(figsize=(10, 5))

    image_path = path.join(data_path, 'cloud.png')
    fig.savefig(image_path)

def output_chart_2_2(df = df_2_2,
                     choice_class = 'Economy Class',
                     choice_flight_length = ['Short-Haul'],
                     choice_polarity = True,
                     choice_topic = 'Seat Comfort'):
    # A dictionary mapping topics to its boolean attributes
    cols_topic = ['Cabin Staff Service', 'Food & Beverages','Ground Service',
                    'Inflight Entertainment', 'Seat Comfort']
    cols_is = ['is_staff', 'is_food', 'is_ground',
               'is_entertainment', 'is_seat']
    cols_pol = ['Staff_Service_Polarity', 'FoodBeverages_Polarity', 'GroundService_Polarity',
                'InflightEntertainment_Polarity','Seat_Polarity']
    topic_to_is = {cols_topic[i]:cols_is[i] for i in range(len(cols_topic))}
    topic_to_pol = {cols_topic[i]:cols_pol[i] for i in range(len(cols_topic))}

    # Select data by category
    mask = (df['Class'].isin([choice_class])) &\
            (df['Flight Length'].isin(choice_flight_length)) &\
            (df[topic_to_is[choice_topic]] == True) &\
            (df[topic_to_pol[choice_topic]].isin([choice_polarity]))

    df = df[mask]

    # Build a corpus
    text = ' '.join(df['Text'].to_list());print(len(text))

    # Visualize
    createWordcloud(text)

# Layout
tab2 = html.Div([
    dbc.Row([
            dbc.Col([
            html.Label('Class'),
            dcc.Dropdown(
                id='class_tab2',
                options=[{'label': i, 'value': i} for i in li_seat_type],
                multi=False,
                value='Economy Class'
            )],md=4),

            dbc.Col([
            html.Label('Flight Length'),
            dcc.Checklist(
            id='length_tab2',
            options=[{'label': i.replace('-',' '), 'value': i} for i in li_flight_length],
            value=['Short-Haul'])]),
    ]),

    dbc.Row([dbc.Col(html.Img(id='chart_cloud_tab2'),md=5),
            dbc.Col(dcc.Graph(id='chart_spider_tab2'),md=7)]),

    dbc.Col([
            html.Label('Positive or Negative Words?'),
            dcc.RadioItems(
            id='pos_tab2',
            options=[{'label': 'Positive', 'value': True},
                     {'label': 'Negative', 'value': False}],
            value=True)])

])


def output_chart_3(df = df_3_1):
    labels = ['Seat',
              'Staff Service',
              'Food & Beverage',
              'Inflight Entertainment',
              'Ground Service']

    colors = ['rgb(110, 255, 218)',
              'rgb(115,115,115)',
              'rgb(49,130,189)',
              'rgb(189,189,189)',
              'rgb(40,140,140)']

    mode_size = [10, 10, 10, 10, 10]
    line_size = [4, 4, 4, 4, 4]

    x_data = np.array([df['day']]*5)

    y_data = np.array([df['seat'],
                       df['staff_service'],
                       df['foodbeverage'],
                       df['inflightent'],
                       df['gserv']])

    fig = go.Figure()

    for i in range(0, 5):
        # endpoints
        fig.add_trace(go.Scatter(
            x=[x_data[i][0], x_data[i][-1]],
            y=[y_data[i][0], y_data[i][-1]],
            mode='markers',
            marker=dict(color=colors[i], size=mode_size[i])
        ))
        # lines
        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
        ))



    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(159, 166, 183)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(159, 166, 183)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            showline=False,
            showticklabels=False,
            linecolor='rgb(255, 86, 65)',
        ),
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=40,
            r=30,
            t=40,
            b=70
        ),
        showlegend=False,
    )
    fig.update_yaxes(range=[-0.2, 0.32])
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    annotations = []
    legend_places_x = [
        0.2,
        0.5,
        0.8,
        0.38,
        0.68,
    ]
    legend_places_y = [
        0.31,
        0.31,
        0.31,
        0.28,
        0.28,
    ]
    # Adding labels
    for y_trace, label, color, legend_place_x, legend_place_y in zip(y_data, labels, colors, legend_places_x, legend_places_y):
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper',
                                x=0.05,
                                y=y_trace[0],
                                xanchor='right',
                                yanchor='middle',
                                text=' {}‰'.format(round(y_trace[0]*10, 2)),
                                font=dict(family='Arial',
                                          size=16,
                                          color=color),
                                showarrow=False))
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper',
                                x=0.95,
                                y=y_trace[10],
                                xanchor='left',
                                yanchor='middle',
                                text='{}‰'.format(round(y_trace[10]*10, 2)),
                                font=dict(family='Arial',
                                          size=16,
                                          color=color),
                                showarrow=False))

        # legend
        annotations.append(dict(xref='paper',
                                x=legend_place_x,
                                y=legend_place_y,
                                xanchor='center',
                                yanchor='middle',
                                text=label,
                                font=dict(family='Arial',
                                          size=14,
                                          color=color),
                                showarrow=False))
    # Title
    # annotations.append(dict(xref='paper',
    #                         yref='paper',
    #                         x=0.5,
    #                         y=1.05,
    #                         xanchor='left',
    #                         yanchor='bottom',
    #                         text='Social Media Sentiment over the last 7 days',
    #                         font=dict(family='Arial',
    #                                   size=20,
    #                                   color='rgb(159, 166, 183)'),
    #                                   showarrow=False))
    # Source
    annotations.append(dict(xref='paper',
                            yref='paper',
                            x=0.5,
                            y=-0.1,
                            xanchor='center',
                            yanchor='top',
                            text='Source: Twitter',
                            font=dict(family='Arial',
                                      size=12,
                                      color='rgb(159, 166, 183)'),
                                      showarrow=False))
    fig.update_layout(annotations=annotations)

    return fig

def output_query_3(df = df_3_2,
                   day='2020-01-17',
                   subject_nb = 0 ):
    if subject_nb in [0, 1]:
        subject = 'Seat'
        subject_clean = 'Seat'
    elif subject_nb in [2, 3]:
        subject = 'Staff_Service'
        subject_clean = 'Staff Service'
    elif subject_nb in [4, 5]:
        subject = 'FoodBeverages'
        subject_clean = 'Food Beverages'
    elif subject_nb in [6, 7]:
        subject = 'InflightEntertainment'
        subject_clean = 'Inflight Entertainment'
    else:
        subject = 'GroundService'
        subject_clean = 'Ground Service'
    # Select by day and subject
    df2 = df[df['day'] == day][df['subject'] == subject]['avg_tweet']
    list_tweet = str(list(df2)).split('[')
    list_tweet_clean = []
    for k in range(len(list_tweet)):
        list_tweet[k] = list_tweet[k]\
            .replace('\n', '')\
            .replace('\\', '')\
            .replace('"', '')\
            .replace("'", '')\
            .replace(']n', '')\
            .replace(']', '')\
            .replace(':', '')\
            .replace('[', '')\
            .replace('(', '')\
            .replace(')', ' ')\
            .replace('ax6', ' ')\
            .replace('  ', ' ')\
            .strip()
        if list_tweet[k] != '':
            list_tweet_clean.append(list_tweet[k])
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth = 100,
                header=dict(
                    values=[
                        '<b>Representative tweets</b> of {} about the {}'.format(
                            day,
                            subject_clean
                        )
                    ],
                    align = 'center',
                    line_color='rgb(159, 166, 183)',
                    fill_color='#171b26',
                    font=dict(
                        family='Arial',
                        color='rgb(159, 166, 183)',
                        size=16),
                ),
                cells=dict(
                    values=[list_tweet_clean],
                    line_color='rgb(159, 166, 183)',
                    font=dict(
                        family='Arial',
                        color='rgb(159, 166, 183)',
                        size=13),
                    height=50,
                    align = 'center'
                )
            )
        ]
    )
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_layout(
        # xaxis_title = '',
        # yaxis_title = '',
        # xaxis_tickformat = ',.0%',
        # showlegend=False,
        # font={'size': 16},
        margin=dict(l=5, r=15))
    return fig


# Layout
tab3 = html.Div(
    dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                html.Div(
                                    'Social Media Sentiment over the last 7 days',
                                    className = 'chart_tab3_title'
                                ),
                            ),
                            dbc.Row(
                                dbc.Col(
                                    dcc.Graph(
                                        id='chart_1_tab3',
                                        figure=output_chart_3()
                                    ),
                                    md= 12
                                )
                            )
                        ],
                        md=8
                    ),
                    dbc.Col(
                         [
                             dbc.Row(
                                html.Div(
                                    '',
                                    className = 'chart_tab3_title'
                                ),
                            ),
                             dcc.Graph(
                                id='chart_2_tab3',
                                figure=output_query_3(
                                    df = df_3_2,
                                    day='2020-01-17',
                                    subject_nb = 0
                                )
                            ),
                        ],
                        md=4
                    )
                ]
            ),
            dbc.Row(
                daq.ToggleSwitch(
                    id='toggleswitch',
                    label=['Twitter', 'TripAdvisor'],
                    style={'width': '50%', 'margin': 'auto'},
                    value=False
                )
            )
        ],
        md=12
    )
)

def output_chart_4(df = df_journey,
                   class_ = 'Economy Class',
                   company = 'Aegean Airlines',
                   airplane = 'Airbus A320'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df[df['Class'] == class_]
    query = query[query['Company'] == company]
    query = query[query['Airplane'] == airplane]
    query = query[[
        'Booking_Polarity',
        'PreTrip_Polarity',
        'Departure_Polarity',
        'Flying_Polarity',
        'Arrival_Polarity' ]].mean()
    # print(query)
    x_data = [
        'Booking',
        'Pre-trip',
        'Departure',
        'Flying',
        'Arrival'
    ]
    y_data = query.values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            marker=dict(color='rgb(255, 86, 65)', size=10)
        ))
    # lines
    fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            line=dict(color='rgb(255, 86, 65)', width=4),
            connectgaps=True,
        ))
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_layout(
        # xaxis_title = '',
        # yaxis_title = '',
        # xaxis_tickformat = ',.0%',
        # showlegend=False,
        # font={'size': 16},
        margin=dict(l=5, r=15))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),

        ),
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=40,
            r=30,
            t=40,
            b=70
        ),
        showlegend=False,
    )
    annotations = []
    annotations.append(dict(
                            x=-0.1,
                            y=0.5,
                            xanchor='right',
                            yanchor='top',
                            text='Good feeling',
                            font=dict(family='Arial',
                                      size=16,
                                      color='rgb(159, 166, 183)'),
                                      showarrow=False))
    annotations.append(dict(
                            x=-0.1,
                            y=-0.5,
                            xanchor='right',
                            yanchor='top',
                            text='Bad feeling',
                            font=dict(family='Arial',
                                      size=16,
                                      color='rgb(159, 166, 183)'),
                                      showarrow=False))

    fig.update_layout(annotations=annotations)
    #fig.update_yaxes(range=[-0.2, 0.32])
    return fig



def output_query_4(df = df_journey,
                   class_ = 'Economy Class',
                   company = 'Aegean Airlines',
                   airplane = 'Airbus A320',
                   subject = 'Booking'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df[df['Class'] == class_]
    query = query[query['Company'] == company]
    query = query[query['Airplane'] == airplane]
    query = query[query[subject] != '[]']
    query = query[subject]
    list_query = str(list(query)).split('[')
    list_query_clean = []
    for k in range(len(list_query)):
        list_query[k] = list_query[k]\
            .replace('\n', '')\
            .replace('\\', '')\
            .replace('"', '')\
            .replace("'", '')\
            .replace(']n', '')\
            .replace(']', '')\
            .replace(':', '')\
            .replace('[', '')\
            .replace('(', '')\
            .replace(')', ' ')\
            .replace('ax6', ' ')\
            .replace('  ', ' ')\
            .strip()
        if list_query[k] != '':
            list_query_clean.append(list_query[k])
        list_query_clean = list_query_clean[:5]

    fig = go.Figure(
        data=[
            go.Table(
                cells=dict(
                    values=[list_query_clean],
                    line_color='#171b26',
                    fill_color='#171b26',
                    font=dict(
                        family='Arial',
                        color='rgb(159, 166, 183)',
                        size=13),
                    height=50,
                    align = 'center'
                )
            )
        ]
    )
    fig.layout['template']['data']['table'][0]['header']['fill']['color']='rgba(0,0,0,0)'
    fig.layout['template']['data']['table'][0]['header']['line']['color']='rgba(0,0,0,0)'
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_layout(
        # xaxis_title = '',
        # yaxis_title = '',
        # xaxis_tickformat = ',.0%',
        # showlegend=False,
        # font={'size': 16},
        margin=dict(l=5, r=15, t = 0, b = 0))
    return fig



# Layout
tab4 = html.Div(
    dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                'Class',
                                className = 'CJ_button'
                            ),
                            dcc.Dropdown(
                                id='class_tab4',
                                options=[{'label': i, 'value': i} for i in li_seat_type],
                                multi=False,
                                value='Economy Class')
                        ],
                        md=4
                    ),
                    dbc.Col(
                        [
                             html.Label(
                                'Company',
                                className = 'CJ_button'
                            ),
                            dcc.Dropdown(
                                id='company_tab4'
                            ),
                        ],
                        md=4
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                'Airplane',
                                className = 'CJ_button'
                            ),
                            dcc.Dropdown(
                                id='airplane_tab4'
                            )
                        ],
                        md=4
                    ),
                ]
            ),
            dbc.Row(
                html.H6(
                    'Average Passenger Satisfaction',
                    className="passenger_satisfaction_title"
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id='chart_tab4',
                        figure=output_chart_4()
                        ),
                    className="Customer_journey_text"
                )
            ),
            dbc.Row(
                html.H6(
                    'Some examples of reviews for this type of flight',
                    className="passenger_satisfaction_title"
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Booking.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Booking',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='table_booking_CJ',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Pre-trip.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Pre-trip',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='table_pretrip_CJ',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Departure.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Departure',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='table_departure_CJ',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Flying.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Flying',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='table_flying_CJ',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Arrival.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Arrival',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='table_arrival_CJ',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            )
        ]
    )
)

def output_chart_5(df = df_products,
                   class_ = 'Economy Class',
                   company = 'Aegean Airlines',
                   airplane = 'Airbus A320'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df[df['Class'] == class_]
    query = query[query['Company'] == company]
    query = query[query['Airplane'] == airplane]
    query = query[[
        'Lighting_Polarity',
        'SeatingTextile_Polarity',
        'Lavatory_Polarity',
        'Safety_Polarity',
        'Catering_Polarity',
        'Connectivity_Polarity',
        'Storage_Polarity', ]].mean()
    # print(query)
    x_data = [
        'Lighting',
        'Seating & Textile',
        'Lavatory',
        'Safety',
        'Catering',
        'Connectivity',
        'Storage',
    ]
    y_data = query.values
    data_df = pd.DataFrame([x_data, y_data]).T
    data_df.columns = ['x_data', 'y_data']
    fig = px.bar(
            x=x_data,
            y=y_data)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_layout(
        # xaxis_title = '',
        # yaxis_title = '',
        # xaxis_tickformat = ',.0%',
        # showlegend=False,
        # font={'size': 16},
        margin=dict(l=5, r=15))
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=False,
            showticklabels=True,
            linecolor='rgb(115,115,115)',
            tickfont=dict(
                family='Arial',
                size=16,
                color='rgb(159, 166, 183)',
            ),

        ),
        autosize=True,
        margin=dict(
            autoexpand=False,
            l=40,
            r=30,
            t=40,
            b=70
        ),
        showlegend=False,
    )
    annotations = []
    annotations.append(dict(
                            x=-0.1,
                            y=0.5,
                            xanchor='right',
                            yanchor='top',
                            text='Good feeling',
                            font=dict(family='Arial',
                                      size=16,
                                      color='rgb(159, 166, 183)'),
                                      showarrow=False))
    annotations.append(dict(
                            x=-0.1,
                            y=-0.5,
                            xanchor='right',
                            yanchor='top',
                            text='Bad feeling',
                            font=dict(family='Arial',
                                      size=16,
                                      color='rgb(159, 166, 183)'),
                                      showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.update_traces(
        marker_color= 'rgb(255, 86, 65)',
        marker_line_color='rgb(240,240,240)',
        marker_line_width=1.5,
        opacity=0.9
    )
    #fig.update_yaxes(range=[-0.2, 0.32])
    return fig


def output_query_5(df = df_products,
                   class_ = 'Economy Class',
                   company = 'Aegean Airlines',
                   airplane = 'Airbus A320',
                   subject = 'Lighting'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df[df['Class'] == class_]
    query = query[query['Company'] == company]
    query = query[query['Airplane'] == airplane]
    query = query[query[subject] != '[]']
    query = query[subject]
    list_query = str(list(query)).split('[')
    list_query_clean = []
    for k in range(len(list_query)):
        list_query[k] = list_query[k]\
            .replace('\n', '')\
            .replace('\\', '')\
            .replace('"', '')\
            .replace("'", '')\
            .replace(']n', '')\
            .replace(']', '')\
            .replace(':', '')\
            .replace('[', '')\
            .replace('(', '')\
            .replace(')', ' ')\
            .replace('ax6', ' ')\
            .replace('  ', ' ')\
            .strip()
        if list_query[k] != '':
            list_query_clean.append(list_query[k])
        list_query_clean = list_query_clean[:5]

    fig = go.Figure(
        data=[
            go.Table(
                cells=dict(
                    values=[list_query_clean],
                    line_color='#171b26',
                    fill_color='#171b26',
                    font=dict(
                        family='Arial',
                        color='rgb(159, 166, 183)',
                        size=13),
                    height=50,
                    align = 'center'
                )
            )
        ]
    )
    fig.layout['template']['data']['table'][0]['header']['fill']['color']='rgba(0,0,0,0)'
    fig.layout['template']['data']['table'][0]['header']['line']['color']='rgba(0,0,0,0)'
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    fig.update_layout(
        # xaxis_title = '',
        # yaxis_title = '',
        # xaxis_tickformat = ',.0%',
        # showlegend=False,
        # font={'size': 16},
        margin=dict(l=5, r=15, t = 0, b = 0))
    return fig

# Layout
tab5 = html.Div(
    dbc.Col(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(
                                'Class',
                                className = 'CJ_button'
                            ),
                            dcc.Dropdown(
                                id='class_tab5',
                                options=[{'label': i, 'value': i} for i in li_seat_type],
                                multi=False,
                                value='Economy Class')
                        ],
                        md=4
                    ),
                    dbc.Col(
                        [
                             html.Label(
                                'Company',
                                className = 'CJ_button'
                            ),
                            dcc.Dropdown(
                                id='company_tab5'
                            ),
                        ],
                        md=4
                    ),
                    dbc.Col(
                        [
                            html.Label(
                                'Airplane',
                                className = 'CJ_button'
                            ),
                            dcc.Dropdown(
                                id='airplane_tab5'
                            )
                        ],
                        md=4
                    ),
                ]
            ),
            dbc.Row(
                html.H6(
                    'Sentiment score per product',
                    className="passenger_satisfaction_title"
                )
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id='chart_tab5',
                        figure=output_chart_5()
                        ),
                    className="Customer_journey_text"
                )
            ),
            dbc.Row(
                html.H6(
                    'Some examples of reviews for this type of flight',
                    className="passenger_satisfaction_title"
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Lighting.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Lighting',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Lighting',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Seating.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Seating & Textile',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='SeatingTextile',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Lavatory.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Lavatory',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Lavatory',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Safety.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Safety',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Safety',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Catering.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Catering',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Catering',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Connectivity.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Connectivity',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Connectivity',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Col(
                                html.Img(
                                    src=app.get_asset_url(
                                        "Storage.png"
                                    ),
                                    className="Customer_journey_step"
                                )
                            ),
                            dbc.Row(
                                html.H6(
                                    'Storage',
                                    className="Customer_journey_step"
                                ),
                            )
                        ],
                        md=2,
                        className = "Customer_journey_step"
                    ),
                    dbc.Col(
                        dcc.Graph(
                            id='Storage',
                            figure=output_query_4()

                        ),
                        className="Customer_journey_text",
                    )
                ]
            )

        ]
    )
)

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'backgroundColor': "#171b26"
}

tab_selected_style = {
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H6(
                        '🚀 Passenger Tracker',
                        className="banner"
                    ),
                    md=9
                ),
                dbc.Col(
                    html.Img(
                        src=app.get_asset_url(
                            "essec_logo.png"
                        ),
                        className="banner"
                    ),
                    md=1
                ),
                dbc.Col(
                    html.Img(
                        src=app.get_asset_url(
                            "centralesupelec_logo.png"
                        ),
                        className="banner"
                    ),
                    md=1
                )


            ]),
        dcc.Tabs(
            [
                dcc.Tab(
                    label='Customer Satisfaction Keys',
                    children=tab1,
                    style=tab_style,
                    selected_style=tab_selected_style

                ),
                dcc.Tab(
                    label='Product line',
                    children=tab5,
                    style=tab_style,
                    selected_style=tab_selected_style
                ),
                dcc.Tab(
                    label='Customer Journey',
                    children=tab4,
                    style=tab_style,
                    selected_style=tab_selected_style
                ),
                dcc.Tab(
                    label='Social Media Sentiment Barometer',
                    children=tab3,
                    style=tab_style,
                    selected_style=tab_selected_style
                )
            ]
        )
    ]
)


# Tab 1 interactivity 1
@app.callback(
    Output('chart_tab1', 'figure'),
    [Input('class_tab1', 'value'),
     Input('purpose_tab1', 'value'),
     Input('length_tab1', 'value'),
     Input('stop_tab1', 'value')])
def update_figure(seat_type, type_of_travaller, flight_length, stop):
    return output_chart_1_1(df_1,
                            seat_type,
                            type_of_travaller,
                            stop,
                            flight_length)


# # Tab 2 interactivity 2
# @app.callback(
#     Output('chart_tab2', 'figure'),
#     [Input('class_tab1', 'value'),
#      Input('purpose_tab1', 'value'),
#      Input('length_tab1', 'value'),
#      Input('stop_tab1', 'value')])
# def update_figure(seat_type, type_of_travaller, flight_length, stop):
#     return output_chart_2(df_1,
#                             seat_type,
#                             type_of_travaller,
#                             stop,
#                             flight_length)


# Tab 2 interactivity 1
@app.callback(
    Output('chart_spider', 'figure'),
    [Input('class_tab12', 'value'),
     Input('short_haul_bt', 'value'),
     Input('middle_haul_bt', 'value'),
     Input('long_haul_bt', 'value')])
def update_figure(choice_class, short_haul_bt = True,
                  middle_haul_bt = False, long_haul_bt = False):
    choice_flight_length = []
    if short_haul_bt:
        choice_flight_length.append('Short-Haul')
    if middle_haul_bt:
        choice_flight_length.append('Middle-Haul')
    if long_haul_bt:
        choice_flight_length.append('Long-Haul')
    return output_chart_2(df_2, choice_class, choice_flight_length)
#
# # Tab 2 interactivity 2
# @app.callback(
#     Output('chart_cloud_tab2', 'src'),
#     [Input('class_tab2', 'value'),
#      Input('length_tab2', 'value'),
#      Input('chart_spider_tab2', 'clickData'),
#      Input('pos_tab2', 'value')])
# def update_figure(choice_class, choice_flight_length, clickData, choice_polarity):
#     try:
#         choice_topic = clickData['points'][0]['theta']
#     except:
#         choice_topic = 'Seat Comfort'
#     print(choice_topic)
#
#     try:
#         output_chart_2_2(df_2_2, choice_class, choice_flight_length, choice_polarity=choice_polarity, choice_topic=choice_topic)
#     except:
#         pass
#
#     image_path = path.join(data_path, 'cloud.png')
#     encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
#     return 'data:image/png;base64,{}'.format(encoded_image)

# Tab 3 interactivity
@app.callback(
    Output('chart_2_tab3', 'figure'),
    [Input('chart_1_tab3', 'clickData')])
def update_figure(clickData):
    return output_query_3(df=df_3_2,
                          day=clickData['points'][0]['x'],
                          subject_nb = clickData['points'][0]['curveNumber'])


# tab4
@app.callback(
    Output(component_id='table_booking_CJ',
           component_property='figure'),
    [Input(component_id='class_tab4', component_property='value'),
     Input(component_id='company_tab4', component_property='value'),
     Input(component_id='airplane_tab4', component_property='value')]
)
def table_booking_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_4(df = df_journey,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Booking')
    return fig

@app.callback(
    Output(component_id='table_pretrip_CJ',
           component_property='figure'),
    [Input(component_id='class_tab4', component_property='value'),
     Input(component_id='company_tab4', component_property='value'),
     Input(component_id='airplane_tab4', component_property='value')]
)
def table_pretrip_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_4(df = df_journey,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Pre-trip')
    return fig

@app.callback(
    Output(component_id='table_departure_CJ',
           component_property='figure'),
    [Input(component_id='class_tab4', component_property='value'),
     Input(component_id='company_tab4', component_property='value'),
     Input(component_id='airplane_tab4', component_property='value')]
)
def table_departure_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_4(df = df_journey,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Departure')
    return fig

@app.callback(
    Output(component_id='table_flying_CJ',
           component_property='figure'),
    [Input(component_id='class_tab4', component_property='value'),
     Input(component_id='company_tab4', component_property='value'),
     Input(component_id='airplane_tab4', component_property='value')]
)
def table_flying_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_4(df = df_journey,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Flying')
    return fig

@app.callback(
    Output(component_id='table_arrival_CJ',
           component_property='figure'),
    [Input(component_id='class_tab4', component_property='value'),
     Input(component_id='company_tab4', component_property='value'),
     Input(component_id='airplane_tab4', component_property='value')]
)
def table_arrival_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_4(df = df_journey,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Arrival')
    return fig


@app.callback(
    Output(component_id='chart_tab4',
           component_property='figure'),
    [Input(component_id='class_tab4', component_property='value'),
     Input(component_id='company_tab4', component_property='value'),
     Input(component_id='airplane_tab4', component_property='value')]
)
def chart_tab4_update(class_ = 'Economy Class',
                                         company = 'Aegean Airlines',
                                         airplane = 'Airbus A320'):
    fig = output_chart_4(df = df_journey,
                   class_ = class_,
                   company = company,
                   airplane = airplane)
    return fig

@app.callback(
    dash.dependencies.Output('company_tab4', 'options'),
    [dash.dependencies.Input('class_tab4', 'value')]
)
def update_date_dropdown(class_ = 'Economy Class'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df_journey[df_journey['Class'] == class_]
    query = query['Company'].drop_duplicates().values
    return [{'label': i, 'value':i} for i in query]

@app.callback(
    dash.dependencies.Output('airplane_tab4', 'options'),
    [dash.dependencies.Input('class_tab4', 'value'),
     dash.dependencies.Input('company_tab4', 'value')]
)

def update_date_dropdown(class_ = 'Economy Class',
                         company = 'Aegean Airlines'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df_journey[df_journey['Class'] == class_]
    query = query[query['Company'] == company]
    query = query['Airplane'].drop_duplicates().values
    return [{'label': i, 'value':i} for i in query]

# tab5
@app.callback(
    Output(component_id='Lighting',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_Lighting_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Lighting')
    return fig

@app.callback(
    Output(component_id='SeatingTextile',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_SeatingTextile_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Seating & Textile')
    return fig

@app.callback(
    Output(component_id='Lavatory',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_Lavatory_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Lavatory')
    return fig

@app.callback(
    Output(component_id='Safety',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_Safety_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Safety')
    return fig

@app.callback(
    Output(component_id='Catering',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_Catering_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Catering')
    return fig

@app.callback(
    Output(component_id='Connectivity',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_Connectivity_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Connectivity')
    return fig

@app.callback(
    Output(component_id='Storage',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def table_storage_CJ_update(class_ = 'Economy Class',
                            company = 'Aegean Airlines',
                            airplane = 'Airbus A320'):
    fig = output_query_5(df = df_products,
                         class_ = class_,
                         company = company,
                         airplane = airplane,
                         subject = 'Storage')
    return fig

@app.callback(
    Output(component_id='chart_tab5',
           component_property='figure'),
    [Input(component_id='class_tab5', component_property='value'),
     Input(component_id='company_tab5', component_property='value'),
     Input(component_id='airplane_tab5', component_property='value')]
)
def chart_tab5_update(class_ = 'Economy Class',
                                         company = 'Aegean Airlines',
                                         airplane = 'Airbus A320'):
    fig = output_chart_5(df = df_products,
                   class_ = class_,
                   company = company,
                   airplane = airplane)
    return fig

@app.callback(
    dash.dependencies.Output('company_tab5', 'options'),
    [dash.dependencies.Input('class_tab5', 'value')]
)
def update_date_dropdown(class_ = 'Economy Class'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df_journey[df_journey['Class'] == class_]
    query = query['Company'].drop_duplicates().values
    return [{'label': i, 'value':i} for i in query]

@app.callback(
    dash.dependencies.Output('airplane_tab5', 'options'),
    [dash.dependencies.Input('class_tab5', 'value'),
     dash.dependencies.Input('company_tab5', 'value')]
)

def update_date_dropdown(class_ = 'Economy Class',
                         company = 'Aegean Airlines'):
    if class_ == 'Economy Class':
        class_ = 'Economy'
    elif class_ == 'Business Class':
        class_ = 'Business'
    elif class_ == 'Premium Economy':
        class_ = 'Premium Eco'
    else: # First class
        class_ = 'First'
    query = df_journey[df_journey['Class'] == class_]
    query = query[query['Company'] == company]
    query = query['Airplane'].drop_duplicates().values
    return [{'label': i, 'value':i} for i in query]


if __name__ == '__main__':
    app.run_server(debug=False)
