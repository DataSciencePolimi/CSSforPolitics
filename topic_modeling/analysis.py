import sys, os
#sys.path.append("C:/Users/emre2/workspace/CSSforPolitics/TweetAnalyserGit/")
sys.path.append("/home/ubuntu/users/emre/CSSforPolitics/")
import traceback
import random
import logging as logger
from builtins import round
from util import text_utils, globals, ml_utils, utils
import matplotlib.pyplot as plt
import math
import collections
import numpy as np
from datetime import datetime
import json, urllib
import pandas as pd

data_path = "/home/ubuntu/users/emre/CSSforPolitics/topic_modeling/data/"

def extract_tweet_text(file):
    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        count_tweet = 0
        counter_en = 0
        dict = {}
        counter_und = 0
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")
                if (fields[0][0:2] != '20'):
                    continue
                language = fields[len(fields) - 2]
                if language == 'en':
                    counter_en += 1
                else:
                    continue

                tweet_id = fields[1]
                tweet_text = fields[3]

                if tweet_text == "" or tweet_text == "undefined":
                    counter_und += 1
                    continue
                else:
                    dict[tweet_id] = tweet_text

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        logger.info("counter undefined:" + str(counter_und))

        logger.info("count_tweets:" + str(count_tweet))
        logger.info("count english tweets:" + str(counter_en))
        logger.info("count_tweets:" + str(count_tweet))
        return dict


#this method extracts new files based on their time period
def time_period_splitter(file, separator):
    try:
        df = utils.read_file(file, separator, ["ID","datetime","text"], dtype=object)
        df = utils.drop_nans(df)
        df_p2 = df[df["datetime"].str.startswith(tuple(globals.p2_times))]
        df_p3 = df[df["datetime"].str.startswith(tuple(globals.p3_times))]
        df_p4 = df[df["datetime"].str.startswith(tuple(globals.p4_times))]

        df_p2.to_csv("p2.csv",sep=separator, index=False, encoding="utf-8")
        df_p3.to_csv("p3.csv",sep=separator, index=False, encoding="utf-8")
        df_p4.to_csv("p4.csv",sep=separator, index=False, encoding="utf-8")

        print("ok")
    except Exception as ex:
        logger.error(ex)


def transform_topic_results(file):
    try:
        df = utils.read_file(file, names=['id','datetime','topic_id'])
        df = df[['datetime', 'topic_id']]
        df.datetime = df.datetime.str.slice(0, 7)
        df_grouped = df.groupby(['datetime', 'topic_id']).size().reset_index(name='counts')
        columns_list = ['2016-01', '2016-02', '2016-03', '2016-04', '2016-05', '2016-06', '2016-07',
                   '2016-08', '2016-09', '2016-10', '2016-11', '2016-12', '2017-01', '2017-02',
                   '2017-03', '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09',
                   '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03', '2018-04',
                   '2018-05', '2018-06', '2018-07', '2018-08', '2018-09']
        index_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
        df_final = pd.DataFrame(index=index_list,
                                columns=columns_list)
        for index, row in df_grouped.iterrows():
            ind = row["topic_id"]
            col = row["datetime"]
            cnt = row["counts"]
            if(ind in index_list and col in columns_list):
                df_final.loc[ind, col] = cnt
            print("good")

        df_final.to_csv('F:/tmp/ttt')
        print("good")
    except Exception as ex:
        logger.error(str(ex))

def pandas_extract_tweet_text_by_topic_label_random_n_records(file, requested_amount, stance):
    try:
        logger.info("started to read file")

        df = utils.read_file(file,"~",names=['ID','user_id','datetime','text','r1'])
        df_filtered = df[df['r1']==stance]
        df_filtered_sample = df_filtered.sample(n=requested_amount)

        df_filtered_sample.to_csv("F:/tmp/random_stance_" + str(stance) + "_sample" + str(requested_amount) + ".csv", index=False, columns=['text','r1'], sep="~", header=False)
        logger.info("file export operation completed")
    except Exception as ex:
        logger.error(ex)


def extract_convert_lda_input(file):
    #overall process for data preparation as input to LDA algorithm
    #1st, extract records from mongo by executing mongo expert javascript file. mongo_extract_data_script.js

    logger.info("started LDA related operations")
    df = utils.read_file(file, "~", names=['ID', 'datetime', 'text'])
    df_new = utils.preprocess_text_for_topic_discovery(df)

    df_new.to_csv("ssss_out.csv", index=False)


if __name__ == "__main__":
    file = data_path + "p1.csv"

    logger.basicConfig(level=logger.INFO, filename="topic.log", format="%(asctime)s %(message)s")
    time_period_splitter(file, "~")
    #transform_topic_results(file)

