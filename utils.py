import logging as logger
import globals
import pandas as pd
import nltk_utils
import traceback
from pymongo import MongoClient


def is_tweet_eligible_for_new_enrichment(res):
    if "r1" in res.keys():
        return False

    if res["tw_lang"] != "en":
        return False

    return True

def preprocess_text_for_topic_discovery(df):
    logger.info("dropping nans")
    drop_nans(df)
    logger.info("removing unwanted words")
    remove_unwanted_words_from_df(df)
    logger.info("nltk preprocessing")
    nltk_utils.preprocess_text(df)
    logger.info("removing word count lower than 2")
    mylist = df['processed_text'].tolist()
    new_list = []
    for line in mylist:
        split = line.split(" ")
        if len(split)>2:
            new_list.append(line)
    df_new = pd.Series(new_list)
    #df['total_words'] = df['processed_text'].str.split().str.len()
    #df_new = df[df['total_words']>2]
    return df_new

def remove_unwanted_words_from_df(df):
    texts_unwanted_eliminated = []
    for text in df['text'].values.tolist():
        text = str(text).rstrip("\r")
        text = text.lower()
        new_text = text.replace("#brexit", "")
        new_text = new_text.replace("#eu", "")
        new_text = new_text.replace("\\", "")

        texts_unwanted_eliminated.append(new_text)
    df['text']=pd.Series(texts_unwanted_eliminated)


def get_mongo_client_db():
    client = MongoClient('localhost:27017')
    db = client.TweetScraper
    return db


def read_file(filename, delimiter=None, names=None):
    df = pd.read_csv(filename, delimiter=delimiter, encoding="ISO-8859-1", error_bad_lines=False,
                     names=names,lineterminator='\n')
    return df


def read_file_to_dict(file, delimiter=None):
    dict ={}
    counter_remain_user = 0
    counter_leave_user = 0
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        for line in ins:
            try:
                if line == '\n':
                    continue;
                fields = line.split("~")
                id = fields[0]
                stance = fields[1].rstrip('\n')
                dict[id]=stance
                if stance == '0':
                    counter_remain_user += 1
                elif stance == '1':
                    counter_leave_user += 1
            except Exception as ex:
                logger.error(ex)
                logger.error(line)
                logger.error(traceback.format_exc())
    logger.info("number of users in stance for remain, leave sides are: " + str(counter_remain_user) + "," + str(counter_leave_user))
    return dict


def calculate_user_stance(count_remain, count_leave):
    user_stance = -1
    if (count_remain + count_leave) == 0:
        return user_stance
    score = count_remain / (count_remain + count_leave)
    if score >= 0.6:
        user_stance = 0
    elif score <= 0.4:
        user_stance = 1
    else:
        user_stance = -1
    return user_stance


def find_label_of_tweet(hashtags):
    label = -1
    has_leave_hashtag = False
    has_remain_hashtag = False

    for hashtag in hashtags:
        hashtag = hashtag.replace("\n", "")
        hashtag = hashtag.lower()

        if hashtag in globals.HASHTAG_REMAIN:
            has_remain_hashtag = True
        elif hashtag in globals.HASHTAG_LEAVE:
            has_leave_hashtag = True
    if has_remain_hashtag and not has_leave_hashtag:
        label = 0
    elif has_leave_hashtag and not has_remain_hashtag:
        label = 1
    return label


def convert_grouped_user_and_stances_to_dict(grouped_df):
    users = {}
    for index, row in grouped_df.iterrows():
        try:
            user_id = index[0]
            stance = index[1]
            cnt = row.text

            if user_id not in users:
                add = ()
                if stance == 0:
                    add = (cnt, 0, 0)
                elif stance == 1:
                    add = (0, cnt, 0)
                elif stance == 2:
                    add = (0, 0, cnt)
                users[user_id] = add
            else:
                old = users[user_id]
                new = ()
                if stance == 0:
                    new = (cnt, old[1], old[2])
                elif stance == 1:
                    new = (old[0], cnt, old[2])
                elif stance == 2:
                    new = (old[0], old[1], cnt)
                users[user_id] = new

        except Exception as ex:
            logger.error(ex)

    return users


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


def extract_hash_tags(text):
    ss = [part[1:] for part in text.split() if part.startswith('#')]
    return ss


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