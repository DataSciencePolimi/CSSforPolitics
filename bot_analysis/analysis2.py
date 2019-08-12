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
                screenname = fields[0]
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
