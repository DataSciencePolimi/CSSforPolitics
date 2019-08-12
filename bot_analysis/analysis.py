import sys, os
#sys.path.append("C:/Users/emre2/workspace/CSSforPolitics/TweetAnalyserGit/")
sys.path.append("/home/ubuntu/users/emre/CSSforPolitics/")
data_path = "/home/ubuntu/users/emre/CSSforPolitics/topic_modeling/data/"

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
import user_stance.analysis as stance
#from user_stance import analysis


def combine_write(file_write_name, user_id_post_count, user_screennames_bot_scores, user_id_user_screennames):
    file_write = open(file_write_name, "w")
    for user_id, post_count in user_id_post_count.items():
        user_screenname = user_id_user_screennames(user_id)
        if not user_screenname:
            continue
        if user_screenname is None:
            continue


def get_user_names_ids(file):
    dict = {}
    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        for line in ins:
            fields = line.split(",")
            if len(fields) == 2:
                name = fields[1].rstrip("\r\n").lower()
                if name in dict:
                    continue
                id = fields[0]
                dict[name] = id

    return dict


def get_bot_screennames_scores(file_bot):
    dict_bot_screenname_score = {}
    with open(file_bot, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        for line in ins:
            try:
                line = line.rstrip("\r\n")
                fields = line.split(",")
                screenname = fields[0].lower()
                score = fields[1]
                dict_bot_screenname_score[screenname]=score
            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())
    logger.info("completed reading bot file. size of bot accounts: " + str(len(dict_bot_screenname_score)))
    return dict_bot_screenname_score


def add_stance_to_last_column_for_bots2(file, filename_write, dict_user_names_ids, dict_user_stances):
    file_write = open(filename_write, "w", encoding='utf-8')

    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
        counter_non_exist = 0
        for line in ins:
            try:
                fields = line.split(",")
                # user_screen_name = fields[0].lower()
                user_screen_name = fields[0].lower()
                if not user_screen_name in dict_user_names_ids:
                    print("major error")
                    counter_non_exist += 1
                    continue

                user_id = dict_user_names_ids[user_screen_name]

                if not user_id in dict_user_stances:
                    print("major error")
                    counter_non_exist += 1
                    continue

                stance = dict_user_stances[user_id]
                output_line = ""

                counter = 0
                for field in fields:

                    field = field.rstrip("\r\n")
                    output_line += str(field)
                    output_line += ","
                    counter += 1

                output_line += str(stance)
                file_write.write(output_line)
                file_write.write("\n")
            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())
        print(str(counter_non_exist))


def combine_all_and_export_file(dict_screen_names,dict_bot_scores,dict_user_stance):
    counter_not_existing_screenname = 0
    counter_not_existing_userstance = 0
    counter = 0
    file_write = open(data_path+"tt.csv", "w", encoding='utf-8')

    for screenname,score in dict_bot_scores.items():

        if not screenname in dict_screen_names.keys():
            counter_not_existing_screenname += 1
            continue
        user_id = dict_screen_names[screenname]

        if not user_id in dict_user_stance.keys():
            counter_not_existing_userstance += 1
            continue
        counter += 1
        stance = dict_user_stance[user_id]
        score=score[0:4]
        file_write.write(screenname+"~"+str(user_id)+"~"+str(score)+"~"+str(stance))
        file_write.write("\n")
        if(counter%10000 == 0):
            file_write.flush()


if __name__ == "__main__":
    logger.basicConfig(level="INFO", filename="bot.log", format="%(asctime)s %(message)s")
    #dict_screen_names = get_user_names_ids("F:/tmp/Bots/test_screennames.csv")
    logger.info("started 1")
    dict_screen_names = get_user_names_ids(data_path+"last_screennames_unique.csv")
    #dict_bot_scores = get_bot_screennames_scores("F:/tmp/Bots/test_bot_res.csv")
    logger.info("started 2")
    dict_bot_scores = get_bot_screennames_scores(data_path+"bot_res.csv")
    logger.info("started 3")
    dict_user_stance = stance.pandas_users_stances(data_path+"merged_stance_of_tweets.csv")
    logger.info("started 4")
    combine_all_and_export_file(dict_screen_names,dict_bot_scores,dict_user_stance)
    logger.info("completed all")

