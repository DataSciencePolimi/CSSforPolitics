import logging as logger
from util import text_utils, globals, ml_utils, utils

import sys, os
#sys.path.append("/home/ubuntu/users/emre/CSSforPolitics/")

logger.basicConfig(filename=globals.WINDOWS_LOG_PATH + 'rule.log', format="%(asctime)s:%(levelname)s:%(message)s", level=logger.INFO)


def create_ml_r1_file_read_line():
    #calculate_r1()
    f = open(globals.INPUT_FILE_FULL_FEATURES, "r", encoding="utf-8", errors="ignore", newline='\n')
    filename_out = globals.INPUT_FILE_FULL_FEATURES + "_all_out.csv"
    filename_polarized_out = globals.INPUT_FILE_FULL_FEATURES + "_polarized_out.csv"
    filename_neutrals_out = globals.INPUT_FILE_FULL_FEATURES + "_neutrals_out.csv"

    f_write = open(filename_out, "w")
    f_write_polarized = open(filename_polarized_out, "w")
    f_write_neutrals = open(filename_neutrals_out, "w")

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
                words = text.split()
                words = [word.replace(".","") for word in words]
                words = [word.replace(",","") for word in words]
                words = text_utils.to_lowercase(words)
                hashtags = [part[1:] for part in words if part.startswith('#')]

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
                new_line = id + "~" + user_id + "~" + datetime + "pun~" + text + "~" + str(r1) + "\n"
                f_write.write(new_line)
                if r1 == 0 or r1 == 1:
                    f_write_polarized.write(new_line)
                elif(r1 == -1):
                    f_write_neutrals.write(new_line)
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


if __name__ == "__main__":
    create_ml_r1_file_read_line()
