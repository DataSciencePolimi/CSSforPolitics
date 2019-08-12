import logging as logger

from util import globals
import pandas as pd
import traceback
from pymongo import MongoClient
import random as rnd
import math
###########################################################
#In this file, there are only the general utility methods#
###########that may be used only once or more##############
###########################################################

def check_common_lines_btw_two_files():
    list1 = []
    list2 = []
    with open('/Users/emrecalisir/git/brexit/CSSforPolitics/tweets_stance_remain_0_8.txt') as f:
        for line in f:
            list1.append(line.rstrip())

    with open('/Users/emrecalisir/git/brexit/CSSforPolitics/tweets_stance_leave_0_8.txt') as f:
        for line in f:
            list2.append(line.rstrip())

    common = list(set(list1).intersection(list2));
    for com in common:
        print(com)
    print("fine")


def combine_lda_results_with_lda_output(corpus, lda_model, df, filename_read):
    ids = []
    datetimes = []
    topic_ids = []
    # topic_words = []
    counter_index = 0

    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        topic_counter = 0
        max_prob = 0
        max_prob_topic = None
        for topic in topics:
            prob = topic[1]
            if max_prob < prob:
                max_prob = prob
                max_prob_topic = topic
            else:
                break

        topic_ids.append(max_prob_topic[0])
        tweet_id = df.iloc[counter_index]['ID']
        datetime = df.iloc[counter_index]['datetime']
        ids.append(tweet_id)
        datetimes.append(datetime)
        counter_index += 1

    if len(ids) != len(topic_ids):
        logger.error("FATAL ERROR caused by data mismatch: len other cols: " + len(ids) + " len new topic cols:" + len(
            topic_ids))
        exit(-1)

    newdf = pd.DataFrame(
        {
            'ID': ids,
            'datetime': datetimes,
            'topic_id': topic_ids
        })
    if (df.shape[0] != newdf.shape[0]):
        logger.info("FATAL ERROR caused by data mismatch: the number of lines are not matching in input and output data collections")
    else:
        newdf.to_csv(filename_read + "_topic_out.csv", index=False)
        logger.info("saved succesfully into a file")


def get_random_int(min, max):
    return rnd.randint(min,max)


def convert_nested_dict_to_line_chart_input_and_write(texts, use_random_nums = False):

    counter = 0
    counter_skipped = 0
    exception_datetime = None
    try:
        if use_random_nums:
            dict_keys_random_ints = {}
            for key in texts.keys():

                    rnd_val = get_random_int(1,15000000)
                    while rnd_val in dict_keys_random_ints.values():
                        rnd_val = get_random_int(1,15000000)

                    dict_keys_random_ints[key] = rnd_val

            rnd_key_list = dict_keys_random_ints.values()
        A = pd.DataFrame(index=texts.keys(), columns=["2016-01","2016-02","2016-03","2016-04","2016-05","2016-06","2016-07","2016-08","2016-09","2016-10","2016-11","2016-12","2017-01","2017-02","2017-03","2017-04","2017-05","2017-06","2017-07","2017-08","2017-09","2017-10","2017-11","2017-12","2018-01","2018-02","2018-03","2018-04","2018-05","2018-06","2018-07","2018-08","2018-09"])

        for user_id,value in texts.items():
            if user_id is None or user_id == '':
                continue;
            vals = []
            #if len(value.items())==1:
            #    logger.info("skipping this id, it has only 1 record: " + str(key))
            #    counter_skipped+= 1
            #    continue;
            counter += 1
            if counter % 1000 == 0:
                logger.info(str(counter))
            for datetime, stance in value.items():
                exception_datetime = datetime
                if use_random_nums:
                    rnd_assigned_id = dict_keys_random_ints[user_id]
                    A[datetime].loc[rnd_assigned_id]=stance
                else:
                    A[datetime].loc[user_id] = stance

        logger.info("counter_skipped: " + str(counter_skipped))
    except Exception as ex:
        logger.error(str(ex))
        logger.info(traceback.format_exc())
        logger.error("exception_datetime:" + str(exception_datetime))

    return A




def is_tweet_eligible_for_new_enrichment(res):
    if "r1" in res.keys():
        return False

    if res["tw_lang"] != "en":
        return False

    return True


def write_list_to_file(filename_write, texts):
    logger.info("started to write")
    file_write = open(filename_write, "w", encoding='utf-8')
    for row in texts:
        file_write.write(str(row[0]) + "," + str(row[1]))
        file_write.write("\n")
    logger.info("completed writing")


def write_nested_dict_to_file(filename_write, separator, texts, nested_level=2, discard_1st_key_having_only_1_value=False):
    logger.info("started to write")
    counter = 0
    counter_skipped = 0
    file_write = open(filename_write, "w", encoding='utf-8')
    try:
        for key, value in texts.items():

            if discard_1st_key_having_only_1_value and len(value.items())==1:
                logger.info("skipping this id, it has only 1 record: " + str(key))
                counter_skipped+= 1
                continue;
            counter += 1
            if counter % 1000 == 0:
                logger.info(str(counter))
            for key2, value2 in value.items():
                if(nested_level==2):
                    file_write.write(str(key) + str(separator) + str(key2) + str(separator) + str(value2))
                    file_write.write("\n")
                elif(nested_level==3):
                    for key3, value3 in value2.items():
                        file_write.write(str(key) + str(separator) + str(key3) + str(separator) + str(key2) + str(separator) + str(value3))
                        file_write.write("\n")
        logger.info("counter_skipped: " + str(counter_skipped))
    except Exception as ex:
        logger.error(str(ex))

    logger.info("completed writing")


def write_dict_to_file(filename_write, texts, is_value_list=False):
    logger.info("started to write")
    counter = 0
    file_write = open(filename_write, "w", encoding='utf-8')
    for key, value in texts.items():
        counter += 1
        if counter % 1000 == 0:
            logger.info(str(counter))
        if is_value_list:
            value = ','.join(value)
        if (key == 1512974670):
            print("hop")
        file_write.write(str(key) + "~" + str(value))
        file_write.write("\n")
    logger.info("completed writing")


def write_text_list_to_file(filename_write, texts):
    logger.info("started to write")
    file_write = open(filename_write, "w", encoding='utf-8')
    for key, value in texts.items():
        if type(value) == int:
            file_write.write(str(key) + "," + str(value))
            file_write.write("\n")
            continue

        # file_write.write(str(key) + "~" + str(value))

        if len(value) == 2:
            file_write.write(str(key) + "," + str(value[0]) + "," + str(value[1]))
        if len(value) == 3:
            file_write.write(str(key) + "," + str(value[0]) + "," + str(value[1]) + "," + str(value[2]) + "," + str(
                round((int(value[0]) / int(value[2])), 2)) + "," + str(round((int(value[1]) / int(value[2])), 2)))

        file_write.write("\n")
    logger.info("completed writing")


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


def read_file(filename, delimiter=None, names=None, dtype=None, lineterminator='\n'):
    if dtype is None:
        df = pd.read_csv(filename, delimiter=delimiter, encoding="ISO-8859-1", error_bad_lines=False,
                         names=names, index_col=False, engine='python')
    else:
        df = pd.read_csv(filename, delimiter=delimiter, encoding="ISO-8859-1", error_bad_lines=False,
                     names=names,lineterminator=lineterminator, dtype=dtype, index_col=False)
    return df


def every_col_is_nan(row):
    every_col_nan = True
    for i in range(0, row.size):
        temp = row[i]
        if(type(temp)==str):
            temp = int(temp)
        if not math.isnan(temp):
            every_col_nan = False
            break
    return every_col_nan


def read_file_to_dict(file, delimiter=None):
    dict ={}
    counter_remain_user = 0
    counter_leave_user = 0
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        try:
            linenumber = 0
            for line in ins:
                try:
                    linenumber += 1
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
                    print(ex)
                    print(line)
                    print(traceback.format_exc())
        except Exception as e:
            print(("Error line %d: %s %s" % (linenumber, str(type(e)), e.message)))
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


def convert_consolidate_monthly_stances(df_grouped):
    counter = 0
    dict_user = {}
    try:
        # first, populate the map with daily user based stance results, but discard the duplicate
        dict = {}

        #first, aggregate in monthly basis
        for index, row in df_grouped.iterrows():
            user_id = row[0]
            stance = row[1]
            datetime = row[2][0:7]
            count = row[3]

            if user_id not in dict:
                dict_date = {datetime: {stance: count}}
                dict[user_id] = dict_date
            else:
                dict_date = dict[user_id]
                if datetime not in dict_date:
                    dict_date[datetime] = {stance:count}
                else:
                    dict_stance = dict_date[datetime]
                    if stance not in dict_stance:
                        dict_stance[stance]=count
                    else:
                        dict_stance[stance]+=count

        #second, discard records based on max value in case of having mixed stances in that month
        new_dict = {}
        for key, value in dict.items():
            for key2, value2 in value.items():
                if len(value2)>1:
                    agg_stance = keywithmaxval(value2)
                    if key not in new_dict:
                        new_dict[key]={key2:agg_stance}
                    else:
                        temp = new_dict[key]
                        temp[key2] = agg_stance

                    print("ok")
                else:
                    agg_stance = list(value2)[0]
                    if key not in new_dict:
                        new_dict[key]={key2:agg_stance}
                    else:
                        temp = new_dict[key]
                        temp[key2] = agg_stance

                    print("good")

        # enrich users with constant number of months

        return new_dict

    except Exception as ex:
        print(str(ex))


def keywithmaxval(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

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
    return df

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
