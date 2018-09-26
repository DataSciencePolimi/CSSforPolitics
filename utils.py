import logging as logger
import globals
import pandas as pd
import nltk_utils
import traceback


def load_input_file(filename):
    df = pd.read_csv(filename, delimiter="~", encoding="ISO-8859-1", error_bad_lines=False,
                     names=globals.FILE_COLUMNS)
    df = compose_new_columns(df)
    return df



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
    return df


def drop_nans(df):
    logger.info(df.shape)
    logger.info("dropping nans")
    df = df.dropna()
    logger.info(df.shape)


def compose_new_columns(df):
    try:
        sizes_hashtags = []
        sizes_mentions = []
        list_contains_link = []
        # converting column types to int
        #print(df.columns[df.isna().any()].tolist())
        #print(df.loc[:, df.isna().any()])
        counter = 0
        drop_nans(df)

        for index, row in df['tw_full'].iteritems():
            if (pd.isnull(row)) :
                logger.info(str(index))

        #compose new fields
        sizes_hashtags, sizes_mentions, list_contains_link = extract_hashtags_mentions_linkexistences(df["tw_full"].tolist())

        #add new columns to dataframe
        df["hashtag_count"] = pd.Series(sizes_hashtags)
        df["mention_count"] = pd.Series(sizes_mentions)
        df["contains_link"] = pd.Series(list_contains_link)
        df["processed_text"] = pd.Series(nltk_utils.text_preprocessing_for_tweet_texts(df["tw_full"].tolist()))


    except Exception as ex:
        logger.error(ex)
        logger.error(traceback.format_exc())
    return df


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