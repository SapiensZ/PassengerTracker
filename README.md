
# 🚀 PassengerTracker


#### Use *topic_scoring.py*

Like any library, use :
````
from topic_scoring import sentence_extraction_scoring, dict_topics
````
It will import all the libraries of the script. Because of nlp library it can last quite long.
An initial dictionnary of topics with related words is already written down.

To use the function with your dataframe reviews:
`````
"""
It returns your dataframe with list of sentences related to each topic (TOPIC_sentences) and the related score (TOPIC_score)
"""
df_results = sentence_extraction_scoring(df, dict_topics, column_name='review')
`````
- df : your dataframe
- dict_topics : dictionnary of topics with related words (loaded with the script)
- column_name : the name of your review header (default='review)


