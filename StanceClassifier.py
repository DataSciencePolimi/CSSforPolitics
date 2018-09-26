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
# import logging as logger
from sklearn import svm, naive_bayes, grid_search
import numpy as np


# LOAD_FILE = utils.load_input_file()


class Classifier:
    row_index = 0
    pipeline = None
    clf = None
    logger = None

    def build_model(_self):
        try:
            _self.logger.info("started")
            #mode = "TRAIN-TUNE-THE-CLASSIFIER"
            mode = "PREDICT_NEW_DATA_WITH_CLASSIFIER"


            df = utils.load_input_file(globals.INPUT_FILE_NAME)

            # pre-processing operations
            utils.drop_nans(df)
            utils.normalize_text(df)
            utils.extract_new_features(df)

            _self.logger.info(
                "Read & Pre-processing & New columns composition complete. Total rowcount:" + str(len(df)))

            # determine which ml classifier to use
            _self.clf = ml_utils.get_model("svm-linear-prob")

            # build the pipeline
            vect = CountVectorizer(ngram_range=(1, 3), analyzer='word', decode_error='replace', encoding='utf-8')
            #vect = CountVectorizer()
            tfidf = TfidfTransformer()

            _self.pipeline = ml_utils.get_pipeline("single", vect, tfidf, _self.clf)

            if globals.RUN_MODE == "TUNE_CLASSIFIER":

                # fit the classifier with training data, and perform test to evaluate the classifier performance
                #ml_utils.find_best_parameters(globals.GRID_SEARCH_PARAMS_SVM, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN])

                ml_utils.run_prob_based_train_test_kfold_roc_curve_plot(_self.pipeline, df[globals.PROCESSED_TEXT_COLUMN],
                                                                       df[globals.TARGET_COLUMN], True, True)

                # ml_utils.run_and_evaluate_train_test(False, False,_self.pipeline,df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN])
                #ml_utils.run_and_evaluate_cross_validation(True, False, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN],
                # df[globals.TARGET_COLUMN], True)
                # ml_utils.run_prob_based_train_test_roc_curve_plot(True, False, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN], False)

                _self.logger.info("Model evaluation complete.")
            elif globals.RUN_MODE == "PREDICT_NEW_DATA":

                # train the classifier
                _self.pipeline.fit(df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN])

                # predict new data
                df_discover = utils.load_input_file(globals.INPUT_DISCOVER_NEUTRAL_FILE_NAME)
                utils.normalize_text(df_discover)

                text_discover = df_discover[globals.PROCESSED_TEXT_COLUMN]
                fixed_text_discover = text_discover[pd.notnull(text_discover)]
                X_discover = fixed_text_discover.values

                probas_ = _self.pipeline.predict_proba(X_discover)
                y_discover_pred = probas_[:, 1]

                df_discover["pred_p1_prob"] = pd.Series(y_discover_pred)
                #ml_utils.discard_low_pred_prob_prediction_single(df_discover)
                _self.logger.info(str(df_discover.shape))
                df_discover = df_discover[(df_discover.pred_p1_prob>0.7) | (df_discover.pred_p1_prob <0.3)]
                _self.logger.info(str(df_discover.shape))
                converted = ml_utils.convert_continuous_prob_to_label(df_discover["pred_p1_prob"].tolist())
                df_discover["final_pred_p1"] = pd.Series(converted)

                df_discover.to_csv(globals.OUTPUT_DISCOVER_NEUTRAL_FILE_NAME, sep="~")
                _self.logger.info("complete.")

        except Exception as ex:
            _self.logger.error(traceback.format_exc())

    def init(_self, utils):
        _self.logger.info("P1 prediction started")
        _self.build_model()
