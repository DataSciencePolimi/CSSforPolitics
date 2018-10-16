import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import globals
import utils as utils
import ml_utils
import traceback
import nltk_utils
import numpy as np
import pickle

class Classifier:
    row_index = 0
    pipeline = None
    clf = None
    logger = None

    def build_model(_self):
        try:
            _self.logger.info("started")

            if globals.RUN_MODE == "TRAIN":
                trained_model_save_enabled = True

                df_train = utils.read_file(globals.INPUT_FILE_NAME_TRAIN_MLMA, "~", ['ID','text','r1'])
                _self.logger.info(df_train.head())
                # pre-processing operations
                utils.drop_nans(df_train)
                nltk_utils.preprocess_text(df_train)
                extract_new_features = False
                if extract_new_features:
                    utils.extract_new_features(df_train)

                _self.logger.info(
                    "Read & Pre-processing & New columns composition complete. Total rowcount:" + str(len(df_train)))

                _self.logger.info("Model evaluation started.")

                # determine which ml classifier to use
                _self.clf = ml_utils.get_model("svm-linear-prob")

                # build the pipeline
                vect = CountVectorizer(ngram_range=(1, 3), analyzer='word', decode_error='replace', encoding='utf-8')
                # vect = CountVectorizer()
                tfidf = TfidfTransformer()

                _self.pipeline = ml_utils.get_pipeline("single", vect, tfidf, _self.clf)
                # fit the classifier with training data, and perform test to evaluate the classifier performance
                ##################
                ### GRID_SEARCH ##
                # ml_utils.find_best_parameters(globals.GRID_SEARCH_PARAMS_SVM, _self.pipeline, df[globals.PROCESSED_TEXT_COLUMN], df[globals.TARGET_COLUMN])
                ##################

                ###############################
                ### K-FOLD-PROBABILITY-BASED###
                ###############################
                ml_utils.run_prob_based_train_test_kfold_roc_curve_plot(_self.pipeline, df_train[globals.PROCESSED_TEXT_COLUMN], df_train[globals.TARGET_COLUMN], True, True)
                ###############################

                ###############################
                ### TRAIN-TEST-SPLIT-BASED#####
                ###############################
                #ml_utils.run_and_evaluate_train_test(False, False,_self.pipeline,df_train[globals.PROCESSED_TEXT_COLUMN], df_train[globals.TARGET_COLUMN])

                ###########################################
                ### TRAIN-TEST-SPLIT-PROBABILITY-BASED#####
                ###########################################
                # ml_utils.run_prob_based_train_test_roc_curve_plot(True, False, _self.pipeline, df_train[globals.PROCESSED_TEXT_COLUMN], df_train[globals.TARGET_COLUMN], False)

                ###############################
                ### CROSS-VALIDATION-BASED#####
                ###############################
                #ml_utils.run_and_evaluate_cross_validation(True, False, _self.pipeline, df_train[globals.PROCESSED_TEXT_COLUMN],df_train[globals.TARGET_COLUMN], True)

                if trained_model_save_enabled:
                    pkl_filename = globals.FILE_STORE_MODEL
                    with open(pkl_filename, 'wb') as file:
                        pickle.dump(_self.pipeline, file)
                _self.logger.info("Model evaluation complete.")

            elif globals.RUN_MODE == "TEST":

                if _self.pipeline is None:
                    with open(globals.FILE_STORE_MODEL, 'rb') as file:
                        _self.pipeline = pickle.load(file)

                # predict over new data
                df_test = utils.read_file(globals.INPUT_FILE_NAME_TEST,"~",names=['ID','2','3','4','5','text','6', '7','8','9','10','11','12','real_res' ])

                utils.drop_nans(df_test)
                nltk_utils.preprocess_text(df_test)

                processed = df_test[globals.PROCESSED_TEXT_COLUMN]
                fixed_processed = processed[pd.notnull(processed)]
                X = fixed_processed.values

                probas_ = _self.pipeline.predict_proba(X)
                y_pred = probas_[:, 1]

                df_test["pred_p1_prob"] = pd.Series(y_pred)

                # discard gray area, the records having less probability of being truly predicted
                _self.logger.info(str(df_test.shape))
                eliminate_less_accurate_results = False
                if eliminate_less_accurate_results:
                    df_test = df_test[(df_test.pred_p1_prob>0.9) | (df_test.pred_p1_prob <0.1)]
                _self.logger.info(str(df_test.shape))

                # convert from float to integer
                df_test['pred'] = np.where(df_test['pred_p1_prob']>0.5,1,0)
                ml_utils.print_evaluation_stats(df_test['real_res'].tolist(),df_test['pred'].tolist(),False)
                ml_utils.print_confusion_matrix(True,df_test['real_res'].tolist(),df_test['pred'].tolist())
                ml_utils.draw_confusion_matrix(df_test['real_res'].tolist(), df_test['pred'].tolist())

                #df_test['is_pred_true']=np.where(df_test['pred']==df_test['real_res'],1,0)
                #print(df_test['is_pred_true'].value_counts())
                #print(df_test['real_res'].value_counts())
                #write to file
                df_test.to_csv(globals.INPUT_FILE_NAME_TEST+"_out.csv", sep="~", columns=['ID','processed_text','real_res','pred_p1_prob','pred','is_pred_true'], line_terminator='\n\n')

                _self.logger.info("complete.")

            elif globals.RUN_MODE == "PREDICT_UNLABELED_DATA":
                if _self.pipeline is None:
                    with open(globals.FILE_STORE_MODEL, 'rb') as file:
                        _self.pipeline = pickle.load(file)

                df_discover = utils.read_file(globals.INPUT_FILE_NAME_DISCOVER_PREDICT_NEUTRALS,"~",names=globals.DISCOVER_FILE_COLUMNS)
                utils.drop_nans(df_discover['text'])
                nltk_utils.preprocess_text(df_discover)

                processed = df_discover[globals.PROCESSED_TEXT_COLUMN]
                fixed_processed = processed[pd.notnull(processed)]
                X = fixed_processed.values

                prabas_ = _self.pipeline.predict_proba(X)
                y_pred = prabas_[:, 1]
                df_discover["pred_p1_prob"] = pd.Series(y_pred)
                # discard gray area, the records having less probability of being truly predicted
                _self.logger.info(str(df_discover.shape))
                if globals.ELIMINATE_LOW_PROB:
                    df_discover = df_discover[(df_discover.pred_p1_prob > globals.MAX_PROB) | (df_discover.pred_p1_prob < globals.MIN_PROB)]
                    _self.logger.info("after discarding less accurate results:" + str(df_discover.shape))

                # convert from float to integer
                df_discover['pred'] = np.where(df_discover['pred_p1_prob'] > 0.5, 1, 0)
                _self.logger.info(df_discover['pred'].value_counts())

                # write to file
                df_discover.to_csv(globals.INPUT_FILE_NAME_DISCOVER_PREDICT_NEUTRALS + "_out.csv", sep="~",
                               columns=['ID', 'user_id', 'datetime', 'text', 'pred'], index=False)

                _self.logger.info("complete.")

        except Exception as ex:
            _self.logger.error(ex)
            _self.logger.error("Something bad happened: %s", ex)
            _self.logger.info(traceback.format_exc())

    def init(_self, utils):
        _self.logger.info("P1 prediction started")
        _self.build_model()
