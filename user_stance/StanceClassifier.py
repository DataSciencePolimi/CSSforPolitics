import sys, os
sys.path.append("/home/ubuntu/users/emre/CSSforPolitics/")

import pandas as pd
from util_tools import ml_utils, text_utils, globals, utils
import traceback
import numpy as np
import pickle
import logging as logger
import csv

logger.basicConfig(level=logger.INFO, filename=globals.UNIX_LOG_PATH, format="%(asctime)s %(message)s")


class Classifier:
    pipeline = None
    clf = None

    def __init__(self, clf, pipeline):
        self.clf = clf
        self.pipeline = pipeline
        self.model_remain = ml_utils.restore_model(globals.CLASSIFIER_REMAIN)
        self.model_leave = ml_utils.restore_model(globals.CLASSIFIER_LEAVE)

    def predict_with_remain_classifier(self, processed_text):
        remain_confidence = self.model_remain.predict_proba([processed_text])[:, 1]
        return remain_confidence[0]

    def predict_with_leave_classifier(self, processed_text):
        leave_confidence = self.model_leave.predict_proba([processed_text])[:, 1]
        return leave_confidence[0]


    def build_model(self):
        try:
            logger.info("started")

            if globals.RUN_MODE == "TRAIN":
                trained_model_save_enabled = True

                df_train = utils.read_file(globals.INPUT_FILE_NAME_TRAIN_MLMA, "~", ['ID', 'user_id', 'datetime', 'text', 'r1'], dtype={'ID':object, 'user_id':object,  'datetime':object, 'text':'U','r1':int})

                logger.info("stance distribution of train dataset: " + str(df_train["r1"].value_counts()))

                logger.info(df_train.head())
                # pre-processing operations
                utils.drop_nans(df_train)
                text_utils.preprocess_text(df_train)
                extract_new_features = False
                if extract_new_features:
                    utils.extract_new_features(df_train)

                logger.info(
                    "Read & Pre-processing & New columns composition complete. Total rowcount:" + str(len(df_train)))

                logger.info("Model evaluation started.")

                # fit the classifier with training data, and test the classifier performance

                if globals.TRAIN_TYPE == "GRID_SEARCH":
                    ml_utils.find_best_parameters(globals.GRID_SEARCH_PARAMS_SVM, self.pipeline,
                                                  df_train[globals.PROCESSED_TEXT_COLUMN], df_train[globals.TARGET_COLUMN])

                elif globals.TRAIN_TYPE == "K-FOLD":
                    ml_utils.run_prob_based_train_test_kfold_roc_curve_plot(self.pipeline,df_train["ID"],
                                                                            df_train[globals.PROCESSED_TEXT_COLUMN],
                                                                            df_train[globals.TARGET_COLUMN], is_plot_enabled=True, discard_low_pred=True)

                elif globals.TRAIN_TYPE == "TRAIN-TEST-SPLIT":
                    ml_utils.run_and_evaluate_train_test(False, False, self.pipeline,
                                                         df_train[globals.PROCESSED_TEXT_COLUMN],
                                                         df_train[globals.TARGET_COLUMN])

                elif globals.TRAIN_TYPE == "CROSS-VALIDATION":
                    ml_utils.run_and_evaluate_cross_validation(False, False, self.pipeline,
                                                               df_train[globals.PROCESSED_TEXT_COLUMN],
                                                               df_train[globals.TARGET_COLUMN], True)

                #if trained_model_save_enabled:
                #    pkl_filename = globals.FILE_STORE_MODEL
                #    with open(pkl_filename, 'wb') as file:
                #        pickle.dump(self.pipeline, file)
                logger.info("Model evaluation complete.")

            elif globals.RUN_MODE == "TEST":

                if self.pipeline is None:
                    with open(globals.FILE_STORE_MODEL, 'rb') as file:
                        self.pipeline = pickle.load(file)

                # predict over new data
                df_test = utils.read_file(globals.INPUT_FILE_NAME_TEST, "~", names=['ID', '2', '3', '4', '5', 'text', '6', '7', '8', '9', '10', '11', '12', 'real_res'])

                utils.drop_nans(df_test)
                text_utils.preprocess_text(df_test)

                processed = df_test[globals.PROCESSED_TEXT_COLUMN]
                fixed_processed = processed[pd.notnull(processed)]
                X = fixed_processed.values

                probas_ = self.pipeline.predict_proba(X)
                y_pred = probas_[:, 1]

                df_test["pred_p1_prob"] = pd.Series(y_pred)

                # discard gray area, the records having less probability of being truly predicted
                logger.info(str(df_test.shape))
                eliminate_less_accurate_results = False
                if eliminate_less_accurate_results:
                    df_test = df_test[(df_test.pred_p1_prob>0.9) | (df_test.pred_p1_prob <0.1)]
                logger.info(str(df_test.shape))

                # convert from float to integer
                df_test['pred'] = np.where(df_test['pred_p1_prob']>0.5,1,0)
                ml_utils.print_evaluation_stats(df_test['real_res'].tolist(), df_test['pred'].tolist(), False)
                ml_utils.print_confusion_matrix(True, df_test['real_res'].tolist(), df_test['pred'].tolist())
                ml_utils.draw_confusion_matrix(df_test['real_res'].tolist(), df_test['pred'].tolist())

                #df_test['is_pred_true']=np.where(df_test['pred']==df_test['real_res'],1,0)
                #print(df_test['is_pred_true'].value_counts())
                #print(df_test['real_res'].value_counts())
                #write to file
                df_test.to_csv(globals.INPUT_FILE_NAME_TEST + "_out.csv", sep="~", columns=['ID', 'processed_text', 'real_res', 'pred_p1_prob', 'pred', 'is_pred_true'], line_terminator='\n\n')

                logger.info("complete.")

            elif globals.RUN_MODE == "PREDICT_UNLABELED_DATA":
                try:
                    df_pred = utils.read_file(globals.INPUT_FILE_PRED, "~", globals.PRED_NEW_COLS,lineterminator='\r')

                    df_pred['text_filled'] = np.where(pd.isna(df_pred.tw_full), df_pred.text, df_pred.tw_full)

                    df_pred = df_pred.drop('text', axis=1)
                    df_pred = df_pred.drop('tw_full', axis=1)

                    df_pred['processed_text'] = df_pred['text_filled'].apply(text_utils.apply_preprocessing_to_tweet)

                    print(df_pred.shape)
                    print(df_pred.head())
                    logger.info(df_pred.head())

                    df_pred['remain_confidence'] = df_pred[globals.PROCESSED_TEXT_COLUMN].apply(self.predict_with_remain_classifier)
                    df_pred['leave_confidence'] = df_pred[globals.PROCESSED_TEXT_COLUMN].apply(self.predict_with_leave_classifier)

                    df_pred['stance'] = np.where(((df_pred['remain_confidence']>0.7) & (df_pred["leave_confidence"]<0.7)), "remain", np.where(((df_pred['leave_confidence']>0.7) & (df_pred["remain_confidence"]<0.7)), "leave", "other"))

                    print("total file: " + str(df_pred.shape))
                    logger.info("total file: " + str(df_pred.shape))
                    df_pred.to_csv(globals.INPUT_FILE_PRED + "_pred_sep9.csv", sep="~", index=False, encoding="ISO-8859-1" )

                    logger.info("complete.")

                except Exception as ex:
                    logger.error(ex)


        except Exception as ex:
            logger.error(ex)
            logger.error("Something bad happened: %s", ex)
            logger.info(traceback.format_exc())

