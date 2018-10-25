import sys, traceback
import logging as logger
import warnings
from util import ml_utils, text_utils, globals, utils

warnings.filterwarnings("ignore", category=DeprecationWarning)


def discover():
    try:

        logger.info("Started Topic discovery operations")
        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")

        # data = df.tweet.values.tolist()
        logger.info("started LDA related operations")
        filename_read = "F:/tmp/p3_1000.csv"
        df = utils.read_file(filename_read, ",", names=['ID','datetime','text'])

        corpus, id2word, data_words_bigrams = text_utils.prepare_lda_input(df)
        logger.info("building LDA model")

        expected_topic_cnt = 20

        lda_model = ml_utils.build_lda_model(corpus, id2word, expected_topic_cnt)

        ml_utils.evaluate_lda_results(corpus, id2word, data_words_bigrams, lda_model, expected_topic_cnt, filename_read)
        utils.combine_lda_results_with_lda_output(corpus, lda_model, df, filename_read)

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        logger.error(ex)
        logger.error("Something bad happened: %s", ex)

    logger.info("Completed everything. Program is being terminated")


if __name__ == "__main__":
    discover()
