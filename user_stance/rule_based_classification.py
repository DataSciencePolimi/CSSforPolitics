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


def create_ml_r1_file_read_line():
    #calculate_r1()
    f = open(globals.INPUT_FILE_FULL_FEATURES, "r", encoding="utf-8", errors="ignore", newline='\n')
    filename_out = globals.INPUT_FILE_FULL_FEATURES + "_out.csv"
    filename_polarized_out = globals.INPUT_FILE_FULL_FEATURES + "_polarized_out.csv"

    f_write = open(filename_out, "w")
    f_write_polarized = open(filename_polarized_out, "w")

    try:
        counter = 0
        counter_err = 0
        counter_neutral = 0
        counter_remain = 0
        counter_leave = 0
        for line in f:
            try:

                splits = line.split("~")
                if (len(splits) != 4):
                    counter_err += 1
                    if (counter_err % 10000 == 0):
                        logger.info("counter err" + str(counter_err))
                        logger.error("fatal error for line: " + str(line))
                    continue

                id = splits[0]
                if id == "940361425729486848":
                    print("hop")
                user_id = splits[1]
                datetime = splits[2]
                #datetime = datetime[0:10]
                text = splits[3]
                text = text.replace("\n","")
                hashtags = utils.extract_hash_tags(text)

                #hashtags = splits[4]

                if len(hashtags) == -1:
                    counter_neutral += 1
                    r1 = -1
                else:
                    r1 = utils.find_label_of_tweet(hashtags)
                    if r1 == 0:
                        counter_remain += 1
                    elif r1 == 1:
                        counter_leave += 1
                    elif r1 == -1:
                        counter_neutral += 1

                counter += 1
                new_line = id + "~" + user_id + "~" + datetime + "~" + text + "~" + str(r1) + "\n"
                f_write.write(new_line)
                if r1 == 0 or r1 == 1:
                    f_write_polarized.write(new_line)
                if (counter % 10000 == 0):
                    logger.info(str(datetime) + ". counter total, neutral, remain, leave: " + str(counter)+","+str(counter_neutral) + "," + str(counter_remain) + "," + str(counter_leave))
                    f_write.flush()
            except Exception as e:
                counter_err += 1
                if (counter_err % 10000 == 0):
                    logger.info("counter err" + str(counter_err) + " for line: " + line)
                    logger.error(str(e))

    except Exception as e:
        logger.error(str(e))
