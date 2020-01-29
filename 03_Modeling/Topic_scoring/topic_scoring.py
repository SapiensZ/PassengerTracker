#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import spacy
import en_core_web_md
nlp = en_core_web_md.load()
import re
import functools
import operator
from spacy.tokens import Doc
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import plotly


dict_topics = {'Seat': ['seat', 'neck', 'back', 'leg', 'comfort'],
              'Staff Service' : ['staff', 'crew', 'service'],
              'Time' : ['delay', 'time'],
              'Ground Service' : ['check', 'boarding', 'ticket', 'access', 'efficiency'],
              'Food & Beverages' : ['drinks', 'food', 'meal', 'catering'],
              'Aircraft' : ['aircraft', 'cabin', 'luggage' 'cleanliness'],
              'Inflight Entertainment' : ['screen', 'movies', 'entertainment', 'wifi' ]
              }


# Set sentiment extensions
sent_analyzer = SentimentIntensityAnalyzer()
def sentiment_scores(docx):
    return sent_analyzer.polarity_scores(docx.text)
Doc.set_extension("sentimenter",getter=sentiment_scores,force=True)


#returns a list of sentences if contains a list of words accoridng to some seperators
def find_sentence_if_l_words(txt, l_words):
    seps = ["? ", ". ", "! ", ", "]
    for sep in seps:
        txt = txt.replace(sep, '. ')
    l = [t for t in txt.split('. ') if any(x in t for x in l_words)]
    return l

#returns new features of list of sentence according to words related to a topic
def sentence_extraction(df_reviews, dict_topics, column_name='review'):
    df = df_reviews.copy()
    for key, l_words in dict_topics.items():
        df[key] = df.apply(lambda row: find_sentence_if_l_words(row[column_name], l_words), axis=1)
    return df

# A helper function to get sentiment of a comment
def get_sentiment(text):
    return nlp(text)._.sentimenter['compound']

def average_score(l_sentences):
    if len(l_sentences) == 0:
        return np.nan
    else:
        l_results = [get_sentiment(text) for text in l_sentences]
        return np.mean(l_results)

#returns new features of list of sentence according to words related to a topic
def sentence_extraction_scoring(df_reviews, dict_topics, column_name='review'):
    df = df_reviews.copy()
    for key, l_words in dict_topics.items():
        df[key + '_sentences'] = df.apply(lambda row: find_sentence_if_l_words(row[column_name], l_words), axis=1)
        df[key + '_score'] = df[key + '_sentences'].apply(average_score)
    return df



