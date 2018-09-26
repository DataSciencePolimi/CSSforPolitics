import logging as logger
import globals
import pandas as pd
import nltk_utils
import traceback


def load_input_file_and_normalize_text(filename):
    df = pd.read_csv(filename, delimiter="~", encoding="ISO-8859-1", error_bad_lines=False,
                     names=globals.FILE_COLUMNS)
    return df


def normalize_text(df):
    df["processed_text"] = pd.Series(nltk_utils.text_preprocessing_for_tweet_texts(df["tw_full"].tolist()))


def extract_new_features(df):
    try:
        hashtag_cnt, mention_cnt, flag_contains_weblink = extract_hashtags_mentions_linkexistences(df["tw_full"].tolist())
        df["hashtag_cnt"] = pd.Series(hashtag_cnt)
        df["mention_cnt"] = pd.Series(mention_cnt)
        df["flag_contains_weblink"] = pd.Series(flag_contains_weblink)
    except Exception as ex:
        logger.error(ex)
        logger.error(traceback.format_exc())


def filter_out_bad_rows(df):
    logger.info(str(df.shape))
    df = df.dropna()

    logger.info(str(df.shape))

    df = df[df.new_p1 != 'undefined']

    logger.info(str(df.shape))

    df = df[df.tw_full != 'undefined']

    logger.info(str(df.shape))

    df = df[df.tw_lang == 'en']

    logger.info(str(df.shape))

    df = df.reset_index()


def drop_nans(df):
    logger.info(df.shape)
    logger.info("dropping nans")
    df = df.dropna()
    logger.info(df.shape)


def extract_hashtags_mentions_linkexistences(tweets):
    sizes_hashtags = []
    sizes_mentions = []
    list_contains_link = []
    # TODO remove this
    for tweet in tweets:
        try:
            hashtag_count = len(tweet.split("#")) - 1
            sizes_hashtags.append(hashtag_count)

            mention_count = len(tweet.split("@")) - 1
            sizes_mentions.append(mention_count)

            contains_link = 0
            if "http" in tweet:
                contains_link = 1
            list_contains_link.append(contains_link)

        except Exception as ex:
            logger.info("error", ex)

    return sizes_hashtags, sizes_hashtags, list_contains_link


def remove_ampercant_first_char_if_exists(input):
    if input[0] == "@":
        input = input[1:len(input)]
    return input


def write_dict_to_file(filename_write, dict):
    print("started to write")
    counter = 0
    file_write = open(filename_write, "w", encoding='utf-8')
    for key,value in dict.items():
        counter += 1
        if counter %1000 == 0:
            print(str(counter))
        file_write.write(str(key)+"~"+str(value))
        file_write.write("\n")
    print("completed writing")