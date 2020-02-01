#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:00:13 2019

@author: vincent roy
"""
print('-------------------------------')
print('preprocessing TripAdvisors_data')
print('-------------------------------')
import os
import pandas as pd
import numpy as np

df = pd.read_csv("TripAdvisors_data/TripAdvisors_data.csv", sep ='|')

def split_town_country(x):
    try:
        x = x.split(',')
        user_town = x[0].strip()
        user_country = x[1].strip()
    except:
        user_town = np.nan
        user_country = np.nan
    return user_town, user_country
vsplit_town_country = np.vectorize(split_town_country)

user_hometown, user_homecountry = vsplit_town_country(df['user_hometown'].values)
df['user_hometown'] = user_hometown
df['user_homecountry'] = user_homecountry
print('user_hometown preprocessed')

def grade_dict_convert(grade):
    dict_grades = dict(eval(grade))

    return dict_grades
vgrade_dict_convert = np.vectorize(grade_dict_convert)

def get_value_from_key(dict_grades, key):
    try:
        value = dict_grades[key]
    except KeyError:
        value = np.nan
    return value
vget_value_from_key = np.vectorize(get_value_from_key)


grades_dict = vgrade_dict_convert(df['grades'].values)

key_list = [
        'Legroom',
        'Seat comfort',
        'In-flight Entertainment',
        'Customer service',
        'Value for money',
        'Cleanliness',
        'Check-in and boarding',
        'Food and Beverage'
    ]
for key in key_list:
    df['{}_grade'.format(key)] = vget_value_from_key(grades_dict, key)

print('grades preprocessed')
df.to_csv('TripAdvisors_data/TripAdvisors_data_preprocessed.csv', sep ='|')

print('TripAdvisors_data preprocessed and exported')