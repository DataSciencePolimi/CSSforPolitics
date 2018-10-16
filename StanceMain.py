from StanceClassifier import Classifier

import logging as logger
import globals
import traceback
logger.basicConfig(level=logger.INFO, filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")


if __name__ == "__main__":

    try:
        clf = Classifier()
        clf.logger = logger
        clf.build_model()
        print("ok")
    except:
          logger.error(traceback.format_exc())
