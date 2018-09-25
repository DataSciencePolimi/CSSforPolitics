import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import globals
import utils as utils
import ml_utils
import traceback
import logging as logger
from sklearn import svm, naive_bayes, grid_search
import numpy as np

#LOAD_FILE = utils.load_input_file()


class Classifier:
    row_index = 0
    pipeline = None
    clf = None

    def build_model(_self):
        try:
            logger.info("started")

            df = utils.load_input_file(globals.INPUT_FILE_NAME)

            utils.drop_nans(df)
            logger.info("Read & Pre-processing & New columns composition complete. Total rowcount:" + str(len(df)))

            _self.clf = ml_utils.get_model("sgd")

            ##vect = CountVectorizer(ngram_range=(2, 3), analyzer='word', decode_error='replace', encoding='utf-8')

            vect = CountVectorizer()
            tfidf = TfidfTransformer()
            print(np.arange(1, 11, 1))
            _self.pipeline = ml_utils.get_pipeline("single",vect, tfidf, _self.clf)
            logger.info(_self.pipeline.get_params().keys())

            #ml_utils.find_best_parameters(globals.GRID_SEARCH_PARAMS_SGD, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN])

            #ml_utils.run_prob_based_train_test_kfold_roc_curve_plot(_self.pipeline, df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN], True, False)

            #ml_utils.run_and_evaluate_train_test(False, False,_self.pipeline,df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN])
            ml_utils.run_and_evaluate_cross_validation(True, False, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN],
                                                 df[globals.TARGET_COLUMN], True)
            #ml_utils.run_prob_based_train_test_roc_curve_plot(True, False, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN], False)

            logger.info("Model prediction complete.")

            #text_discover = df_discover[globals.PROCESSED_TEXT_COLUMN]
            #fixed_text_discover = text_discover[pd.notnull(text_discover)]

            #logger.info("shape of text_discover: " + str(text_discover.shape()))
            #logger.info("shape of fixed_text_discover: " + str(fixed_text_discover.shape()))

            #x_discover = fixed_text_discover.tolist()
            #y_discover_pred = _self.pipeline.predict(x_discover)


            #if len(x_discover) != len(y_discover_pred):
                #logger.error("major error. df count: " + str(df["tw_full"].size) + " new pred count : " + str(len(y_discover_pred)))
                #exit(-1);

            #df_discover["pred_p1"] = pd.Series(y_discover_pred)
            #logger.info(df_discover["pred_p1"].value_counts())
            #df_discover.to_csv("F:/tmp/pred_data.csv",sep="~")
            #logger.info("complete.")

        except Exception as ex:
            logger.error(traceback.format_exc())


    def init(_self, utils):
        logger.info("P1 prediction started")
        _self.build_model()
