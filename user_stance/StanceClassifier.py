import sys, os
sys.path.append("/home/ubuntu/users/emre/CSSforPolitics/")

import pandas as pd
from util import ml_utils, text_utils, globals, utils
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
                    df_pred = utils.read_file(globals.INPUT_FILE_PRED, "~", globals.PRED_FILE_COLUMNS,lineterminator='\r')
                except Exception as ex:
                    logger.error(ex)
                print(df_pred.shape)
                print(df_pred.head())
                model_remain = ml_utils.restore_model(globals.CLASSIFIER_REMAIN)
                model_leave = ml_utils.restore_model(globals.CLASSIFIER_LEAVE)

                logger.info(df_pred.head())
                # pre-processing operations

                text_utils.preprocess_text_in_column(df_pred, "tw_full")

                df_pred = ml_utils.predict_with_multiple_classifiers(self.pipeline, df_pred, model_remain, model_leave)
                print("total file: " + str(df_pred.shape))

                df_pred_remains = df_pred[(df_pred["y_preds_remain"]>0.7) & (df_pred["y_preds_leave"]<0.7)]
                df_pred_remains["stance_res"] = "remain"
                print("total file remains: " + str(df_pred_remains.shape))

                condition_res = (df_pred["y_preds_remain"]>0.7) & (df_pred["y_preds_leave"]<0.7)
                condition_res_inverted = np.invert(condition_res)

                df_pred_remains_others = df_pred[condition_res_inverted]
                df_pred_remains_others["stance_res"] = "others"
                print("total file remains others: " + str(df_pred_remains_others.shape))

                df_pred_leaves = df_pred[(df_pred["y_preds_leave"]>0.7) & (df_pred["y_preds_remain"]<0.7)]
                df_pred_leaves["stance_res"] = "leave"
                print("total file leaves: " + str(df_pred_leaves.shape))

                condition_res = (df_pred["y_preds_leave"]>0.7) & (df_pred["y_preds_remain"]<0.7)
                condition_res_inverted = np.invert(condition_res)

                df_pred_leaves_others = df_pred[condition_res_inverted]
                df_pred_leaves_others["stance_res"] = "others"
                print("total file leaves others: " + str(df_pred_leaves_others.shape))

                df_incl_others = pd.merge(df_pred_remains_others, df_pred_leaves_others, how='inner')
                print(df_incl_others.shape)
                print("total file all others shape: " + str(df_incl_others.shape))

                df_all_others = df_incl_others.drop_duplicates()
                print(df_all_others.shape)

                df_all_preds = pd.concat([df_pred_remains, df_pred_leaves, df_all_others])
                print(df_all_preds.shape)


                df_all_preds.to_csv(globals.INPUT_FILE_PRED + "_pred_aug11.csv", sep="~", index=False, encoding="ISO-8859-1" )

                #if self.pipeline is None:
                #    with open(globals.FILE_STORE_MODEL, 'rb') as file:
                #        self.pipeline = pickle.load(file)

                #df_discover = utils.read_file(globals.INPUT_FILE_NAME_DISCOVER_PREDICT_NEUTRALS, "~", names=globals.DISCOVER_FILE_COLUMNS)
                #utils.drop_nans(df_discover['text'])
                #text_utils.preprocess_text(df_discover)
#
                #processed = df_discover[globals.PROCESSED_TEXT_COLUMN]
                #fixed_processed = processed[pd.notnull(processed)]
                #X = fixed_processed.values
#
                #prabas_ = self.pipeline.predict_proba(X)
                #y_pred = prabas_[:, 1]
                #df_discover["pred_p1_prob"] = pd.Series(y_pred)
                ## discard gray area, the records having less probability of being truly predicted
                #logger.info(str(df_discover.shape))
                #if globals.ELIMINATE_LOW_PROB:
                #    df_discover = df_discover[(df_discover.pred_p1_prob > globals.MAX_PROB) | (df_discover.pred_p1_prob < globals.MIN_PROB)]
                #    logger.info("after discarding less accurate results:" + str(df_discover.shape))
#
                ## convert from float to integer
                #df_discover['pred'] = np.where(df_discover['pred_p1_prob'] > 0.5, 1, 0)
                #logger.info(df_discover['pred'].value_counts())
#
                ## write to file
                #df_discover.to_csv(globals.INPUT_FILE_NAME_DISCOVER_PREDICT_NEUTRALS + "_out.csv", sep="~",
                #                   columns=['ID', 'user_id', 'datetime', 'text', 'pred'], index=False)
#
                logger.info("complete.")

        except Exception as ex:
            logger.error(ex)
            logger.error("Something bad happened: %s", ex)
            logger.info(traceback.format_exc())

