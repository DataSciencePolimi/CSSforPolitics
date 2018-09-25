from scipy.stats import randint as sp_randint
import numpy as np

remain_hashtag_list = ["#strongerin", "#voteremain", "#intogether", "#labourinforbritain", "#moreincommon",
                       "#greenerin",
                       "#catsagainstbrexit", "#bremain", "#betteroffin", "#leadnotleave", "#remain", "#stay", "#ukineu",
                       "#votein", "#voteyes", "#yes2eu", "#yestoeu", "#sayyes2europe"]

leave_hashtag_list = ["#independenceDay", "#leaveeuofficial", "#leaveeu", "leave", "#labourleave", "#votetoleave",
                      "#voteleave", "#takebackcontrol", "#ivotedleave", "beleave", "#betteroffout", "#britainout",
                      "#nottip", "#takecontrol", "#voteno", "#voteout", "#voteleaveeu"]

WINDOWS_LOG_PATH = "F:/tmp/predictor.log"
UNIX_LOG_PATH = "predictor.log"

ORIGINAL_TEXT_COLUMN = "tweet_text"
PROCESSED_TEXT_COLUMN = "processed_text"

FILE_COLUMNS = ["ID", "nbr_retweet", "nbr_favorite", "nbr_reply", "datetime", "tw_full", "tw_lang", "new_p1",
                        "user_favourites_count", "user_followers_count", "user_friends_count", "user_statuses_count",
                        "api_res","eye_p1"]

DATAFRAME_COLUMNS_INT = ['nbr_retweet', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'new_p1',
                 'hashtag_count', 'mention_count', 'contains_link']

DATAFRAME_COLUMNS = ['tw_full', 'nbr_retweet', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'new_p1',
                 'hashtag_count', 'mention_count', 'contains_link']

TARGET_COLUMN = 'eye_p1'

INPUT_FILE_NAME = "C:/_Documents/POLIMI/Research/Brexit/remain-leave-train-650.txt"

TWITTER_APP_AUTH = {
   'consumer_key': 'YOUR_KEY',
   'consumer_secret': 'YOUR_SECRET',
   'access_token': 'YOUR_TOKEN',
   'access_token_secret': 'YOUR_TOKEN_SECRET',
}

MASHAPE_KEY = "YOUR BOTOMETER API KEY"

GRID_SEARCH_PARAMS_RANDOM_FOREST = {
    'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)),  # unigrams or bigrams
    "clf__max_depth": [3, None],
    "clf__max_features": sp_randint(1, 11),
    "clf__min_samples_split": sp_randint(2, 11),
    "clf__min_samples_leaf": sp_randint(1, 11),
    "clf__bootstrap": [True, False],
    "clf__criterion": ["gini", "entropy"],

            }

GRID_SEARCH_PARAMS_SGD = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__n_iter': (10, 50, 80),
}

SGD_BEST_PARAMS = {
    'alpha' : 0.00001,
    'n_iter' : 10,
    'penalty': 'l2',
}

NGRAM_BEST_PARAMS = {
    'vect__max_df': 0.75,
    'vect__max_features': 10000,
    'vect__ngram_range': (1, 2),
    'tfidf__use_idf': False,
    'tfidf__norm': 'l2',
}
