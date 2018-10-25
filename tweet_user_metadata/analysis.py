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


def analyze_group_by_influence(file):
    try:
        df = utils.read_file(file, "~", ['datetime', 'nb_retweet', 'nb_like'])
        logger.info(df.head())
        grouped_r = df.groupby('datetime')['nb_retweet'].mean()
        grouped_l = df.groupby('datetime')['nb_like'].mean()

        grouped_r.to_csv('F:/tmp/retweet.txt')
        grouped_l.to_csv('F:/tmp/like.txt')

    except Exception as ex:
        logger.info(ex)


def extract_top_mentioned_accounts(filename_read):
    mentioned_accounts = {}

    with open(filename_read, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        for line in ins:
            line = line.rstrip('\n')
            line = line.rstrip('\r')

            counter = 0
            try:
                counter += 1
                fields = line.split("~")
                mentions = fields[2].split(";")
                for mention in mentions:
                    if(mention==""):
                        continue;
                    mention = text_utils.remove_ampercant_first_char_if_exists(mention)
                    if mention in mentioned_accounts.keys():
                        mentioned_accounts[mention] += 1
                    else:
                        mentioned_accounts[mention] = 1
            except Exception as ex:
                logger.error(str(ex) + " " + line)

    utils.write_dict_to_file(filename_read + "_top_mentioned_accounts.csv", mentioned_accounts)



def analyze_duplicate_tweets(file):
    try:
        df = utils.read_file(file,"~",['ID', 'user_id', 'datetime','text'])
        grouped = df.groupby('text').text.count()
        grouped = grouped[grouped>5]
        grouped.sort_index(ascending=False)
        logger.info(grouped.value_counts())
        grouped.to_csv('F:/tmp/tt1.txt')

    except Exception as ex:
        logger.error(ex)

def extract_mentions(filename_read):
    try:
        filename_write = filename_read+"_mentions.csv"
        file_write = open(filename_write, "w", encoding='utf-8')
        counter_lines_not_compatible = 0
        with open(filename_read, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
            for line in ins:
                line = line.rstrip('\n')
                counter = 0
                try:
                    counter += 1
                    fields = line.split("~")
                    if(len(fields)!=4):
                        continue;
                    tweet_text = fields[3]
                    mentions = text_utils.extract_mentions_from_text(tweet_text)
                    if(len(mentions)==0):
                        counter_lines_not_compatible += 1
                        logger.info("counter_lines_not_compatible : " + str(counter_lines_not_compatible))
                        continue
                    datetime = fields[2]
                    ID = fields[0]

                    mentions_str = ";".join(str(x) for x in mentions)

                    file_write.write(str(ID)+"~"+datetime+"~"+mentions_str)
                    file_write.write("\n")
                    if(counter % 1000 == 0):
                        file_write.flush()
                except Exception as ex:
                    logger.error(str(ex) + " " + line)
    except Exception as ex:
        logger.error(str(ex))


def extract_mentions_daily_of_accounts(file):
    dates = {}
    politicians = ["theresa_may","jeremycorbyn","nigel_farage","borisjohnson","realdonaldtrump","david_cameron","daviddavismp","jacob_rees_mogg"]
    counter_lines_not_compatible = 0

    try:
        with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
            for line in ins:
                counter = 0
                try:
                    counter += 1
                    #if counter == 10000:
                    #    break
                    line = line.rstrip('\n')
                    fields = line.split("~")
                    if (len(fields) != 4):
                        continue;
                    tweet_text = fields[3]
                    mentions= text_utils.extract_mentions_from_text(tweet_text)
                    if (len(mentions) == 0):
                        continue
                    pol = {}
                    datetime = fields[2]

                    for mention in mentions:
                        mention = mention.lower()
                        mention = text_utils.remove_ampercant_first_char_if_exists(mention)
                        if mention in politicians:
                            if datetime in dates:
                                pol = dates[datetime]
                                if mention in pol:
                                    pol[mention]+=1
                                else:
                                    pol[mention]=1
                            else:
                                pol[mention]=1
                                dates[datetime]=pol


                except Exception as ex:
                    logger.error(ex)
                    logger.error(line)
                    logger.error(traceback.format_exc())

        utils.write_nested_dict_to_file("daily_politicians_res.csv", ",", dates)

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


def extract_daily_average_retweet_likes(file):
    #not this method is no longer used because it is replaced with the method analyze_group_by_influence
    dict = {}
    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        counter = 0
        for line in ins:
            counter += 1
            try:
                fields = line.split("~")
                if len(fields) != 13:
                    continue
                datetime = fields[0]
                retweet_cnt = int(fields[4])
                like_cnt = int(fields[5])
                if datetime[0:2] != '20' or len(datetime) != 19:
                    continue
                yearmonthday = datetime[0:10]

                if not yearmonthday in dict.keys():
                    dict[yearmonthday] = [retweet_cnt, like_cnt, int(1)]
                else:
                    dict[yearmonthday][0] += retweet_cnt
                    dict[yearmonthday][1] += like_cnt
                    dict[yearmonthday][2] += int(1)

            except Exception as ex:
                logger.error(str(ex))

        logger.info("counter: " + str(counter))
        logger.info("len of dict: " + str(len(dict)))
    return dict


def extract_hashtag_usage(file):
    hashtag_usage = {}
    counter_en = 0

    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        counter_total = 0
        counter_err = 0
        for line in ins:
            line = line.rstrip('\n')
            counter_total += 1
            if counter_total % 1000 == 0:
                logger.info("counter: " + str(counter_total))
            try:
                fields = line.split("~")

                counter_en += 1
                tweet_text = fields[3]
                tweeet_hashtags = utils.extract_hash_tags(tweet_text)
                if len(tweeet_hashtags) == 0:
                    continue

                for tag in tweeet_hashtags:
                    tag = tag.lower()
                    if tag in hashtag_usage.keys():
                        hashtag_usage[tag]+=1
                    else:
                        hashtag_usage[tag]=1

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())
        try:
            logger.info("sorting started")
            sorted_hashtags = [(k, hashtag_usage[k]) for k in sorted(hashtag_usage, key=hashtag_usage.get, reverse=True)]
            logger.info("sorting completed")
            logger.info("unique hashtag count: " + str(len(sorted_hashtags)))
        except Exception as ex:
            counter_err += 1
            if counter_err % 1000 == 0:
                logger.info("counter err: " + str(counter_err))
                logger.info(ex)
                logger.info(traceback.format_exc())

    return sorted_hashtags
