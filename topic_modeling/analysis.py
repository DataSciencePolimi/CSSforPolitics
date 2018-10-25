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
        df_p5 = df[df["datetime"].str.startswith(tuple(globals.p5_times))]
        df_p6 = df[df["datetime"].str.startswith(tuple(globals.p6_times))]
        df_p7 = df[df["datetime"].str.startswith(tuple(globals.p7_times))]
        df_p8 = df[df["datetime"].str.startswith(tuple(globals.p8_times))]

        df_p2.to_csv("p2.csv",sep=separator, index=False, encoding="utf-8")
        df_p3.to_csv("p3.csv",sep=separator, index=False, encoding="utf-8")
        df_p4.to_csv("p4.csv",sep=separator, index=False, encoding="utf-8")
        df_p5.to_csv("p5.csv",sep=separator, index=False, encoding="utf-8")
        df_p6.to_csv("p6.csv",sep=separator, index=False, encoding="utf-8")
        df_p7.to_csv("p7.csv",sep=separator, index=False, encoding="utf-8")
        df_p8.to_csv("p8.csv",sep=separator, index=False, encoding="utf-8")

        print("ok")
    except Exception as ex:
        logger.error(ex)


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
    df_new.to_csv(file+ "_out.csv", index=False)