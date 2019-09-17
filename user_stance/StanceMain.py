import sys, os

from user_stance.StanceClassifier import Classifier

import logging as logger
import traceback
from util_tools import ml_utils, globals, utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

logger.basicConfig(level=logger.INFO, filename="/Users/emrecalisir/git/brexit/CSSforPolitics/logs/predictor.log", format="%(asctime)s %(message)s")

if __name__ == "__main__":

    try:
        #utils.check_common_lines_btw_two_files();
        # determine which ml classifier to use. they are already pre-defined in ml_utils class. add a new one if not exists.
        clf = ml_utils.get_model("svm-linear-prob")

        # build the pipeline
        vect = CountVectorizer(ngram_range=(1, 3), analyzer='word', decode_error='replace', encoding='utf-8')
        tfidf = TfidfTransformer()
        pipeline = ml_utils.get_pipeline("single", vect, tfidf, clf)

        # initialize the classifier
        classifier = Classifier(clf,pipeline)

        # build and run the model
        classifier.build_model()

    except Exception as ex:
          logger.error(traceback.format_exc())
