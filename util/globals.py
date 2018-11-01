from scipy.stats import randint as sp_randint

###########################################################
#In this file, there are only the constant variables#######
###########that may be used only once or more##############
###########################################################

os = "windows"

WINDOWS_LOG_PATH = "F:/tmp/predictor.log"
UNIX_LOG_PATH = "predictor.log"

#RUN_MODE could be TRAIN, TEST, PREDICT_UNLABELED_DATA
RUN_MODE = "PREDICT_UNLABELED_DATA"

FILE_STORE_MODEL = "F:/tmp/model_MLMA.mdl"
ORIGINAL_TEXT_COLUMN = "tweet_text"
PROCESSED_TEXT_COLUMN = "processed_text"


TRAIN_FILE_COLUMNS_MLRB = ["text","r1"]
TRAIN_FILE_COLUMNS_MLMA = ["ID", "nbr_retweet", "nbr_favorite", "nbr_reply", "datetime", "text", "tw_lang", "new_p1",
                        "user_favourites_count", "user_followers_count", "user_friends_count", "user_statuses_count",
                        "api_res","r1"]

TRAIN_FILE_COLUMNS = ["ID", "user_id", "datetime", "text"]

STANCE_FILE_COLUMNS = ["ID", "user_id", "datetime", "text", "r1"]

DISCOVER_FILE_COLUMNS = ["ID", "user_id", "datetime", "text"]


DATAFRAME_COLUMNS_INT = ['nbr_retweet', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'new_p1',
                 'hashtag_count', 'mention_count', 'contains_link']

DATAFRAME_COLUMNS = ['tw_full', 'nbr_retweet', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'new_p1',
                 'hashtag_count', 'mention_count', 'contains_link']

TARGET_COLUMN = 'r1'

INPUT_FILE_NAME_RB = "F:/tmp/full_en3.csv_out.csv"
INPUT_FILE_NAME_TRAIN_MLRB = "F:/tmp/random_stance_1_2_sample10K.csv"
INPUT_FILE_NAME_TRAIN_MLMA = "F:/tmp/test_train.txt"
#INPUT_FILE_NAME_TRAIN_MLMA = "F:/tmp/test.txt"
INPUT_FILE_NAME_TEST = "F:/tmp/test_train.txt"
INPUT_FILE_NAME_DISCOVER_PREDICT_NEUTRALS = "F:/tmp/test_predict.txt"

TWITTER_APP_AUTH = {
    'consumer_key': 'your_data',
    'consumer_secret': 'your_data',
    'access_token': 'your_data',
    'access_token_secret': 'your_data',
}

MASHAPE_KEY = "your_data"

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

GRID_SEARCH_PARAMS_SVM = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)),
    'vect__analyzer': ('char', 'word'),
    'clf__kernel':('rbf', 'linear'),
    'clf__C':(1, 10, 100, 1000),
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
}

HASHTAG_REMAIN = ["strongerin", "voteremain", "intogether", "labourinforbritain", "moreincommon", "greenerin", "catsagainstbrexit", "bremain", "betteroffin", "leadnotleave", "remain", "stay", "ukineu", "votein", "voteyes", "yes2eu", "yestoeu", "sayyes2europe", "fbpe","stopbrexit", "stopbrexitsavebritain","brexitshambles"]
HASHTAG_LEAVE = ["leaveeuofficial", "leaveeu", "leave", "labourleave", "votetoleave", "voteleave", "takebackcontrol", "ivotedleave", "beleave", "betteroffout", "britainout", "nottip", "takecontrol", "voteno", "voteout", "voteleaveeu", "leavers", "vote_leave", "leavetheeu", "voteleave", "takecontrol", "votedleave"]
HASHTAG_NEUTRAL = ["euref", "eureferendum", "eu", "uk"]


INPUT_TWEET_IDS_FILE_NAME = "tweet_ids.csv"

INPUT_FILE_FULL_FEATURES = "F:/tmp/test.txt"

MAX_PROB = 0.9

MIN_PROB = 0.1

ELIMINATE_LOW_PROB = True

p1_times = ['2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06']
p2_times = ['2016-07', '2016-08', '2016-09','2016-10','2016-11','2016-12','2017-01', '2017-02']
p3_times = ['2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09','2017-10','2017-11']
p4_times = ['2017-12', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09']
