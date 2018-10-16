import traceback
import random
import logging as logger
import utils
import globals
import pandas as pd


# This class performs custom data analysis operations, aggregations, by mainly using dictionary objects. It is a good option to work on exported files rather than writing complex MongoDB queries

# 0 remain
# 1 leave


def r1_stats():
    df = utils.read_file("F:/tmp/full_features.csvl_out.csv", "~", names=['ID', 'user_id', 'datetime', 'text', 'r1'])
    logger.info("started")
    try:
        print(df['r1'].value_counts())
        grouped = df.groupby(['datetime','r1'])['datetime']
        grouped.count().to_csv('F:/tmp/tt.txt')

    except Exception as e:
        logger.error(str(e))


#def rule_based_user_stance():
#    grouped = pandas_users_total_topic_counts()
#    users_total_topic_counts = utils.convert_grouped_user_and_stances_to_dict(grouped)
#    users_stances = extract_users_stances(users_total_topic_counts)
#    counter_neutral_user = 0
#    counter_remain_user = 0
#    counter_leave_user = 0
#    for key, value in users_stances.items():
#        if value == 0:
#            counter_neutral_user += 1
#        elif value == 1:
#            counter_remain_user += 1
#        elif value == 2:
#            counter_leave_user += 1
#    logger.info("n,r,s:" + str(counter_neutral_user) + "," + str(counter_remain_user) + "," + str(counter_leave_user))
#    #dict = extract_daily_involvement_of_prev_calculated_user_stances(users_stances)


def temp_convert_train_set():
    file = "F:/tmp/full_features.csv"
    file_remain = "F:/tmp/remain-train.txt"
    file_leave = "F:/tmp/leave-train.txt"

    list_remain = []
    list_leave = []

    dict_remain = {}
    dict_leave = {}
    with open(file_remain, "r", encoding="utf-8", errors='ignore') as ins:
        for line in ins:
            list_remain.append(line.rstrip('\n'))

    with open(file_leave, "r", encoding="utf-8", errors='ignore') as ins:
        for line in ins:
            list_leave.append(line.rstrip('\n'))

    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        users_total_topic_counts = {}

        for line in ins:
            try:
                count_tweet += 1
                if count_tweet % 100000 == 0:
                    logger.info("count_tweet:" + str(count_tweet))

                fields = line.split("~")
                id = fields[0]
                text = fields[3]

                if id in list_remain:
                    dict_remain[id] = text.rstrip('\n')
                elif id in list_leave:
                    dict_leave[id] = text.rstrip('\n')
            except Exception as ex:
                logger.error(ex)
    f_write_remain = open("F:/tmp/remain_train_out.txt", "w", encoding="utf-8")

    for key, value in dict_remain.items():
        try:
            f_write_remain.write(key + "~" + value + "~" + "0" + "\n")
        except Exception as ex:
            logger.error(ex)
    f_write_remain.flush()

    f_write_leave = open("F:/tmp/leave_train_out.txt", "w", encoding="utf-8")
    for key, value in dict_leave.items():
        try:
            f_write_leave.write(key + "~" + value + "~" + "1" + "\n")
        except Exception as ex:
            logger.error(ex)

    f_write_leave.flush()

    print("ok")


def pandas_users_stances(file):
    try:
        df = utils.read_file(file, "~", ['ID', 'user_id', 'datetime', 'text', 'r1'])

        dict = {}
        logger.info(str(df.shape))

        df_grouped = df[['user_id', 'r1']].groupby(['user_id', 'r1']).agg({'r1':["count"]}).reset_index().groupby('user_id')
        user_id = -1
        counter_remain_users = 0
        counter_leave_users = 0
        logger.info("number of users: " + str(len(df_grouped)))

        #TODO, this part could be simplified with idmax function of pandas dataframe
        for user_id, values_per_stance in df_grouped:
            count_remain = 0
            count_leave = 0
            for value in values_per_stance.values:
                if value[1] == 0:
                    count_remain = value[2]
                elif value[1] == 1:
                    count_leave = value[2]

            user_stance = utils.calculate_user_stance(count_remain, count_leave)
            if user_stance != -1:
                if user_stance == 0:
                    counter_remain_users += 1
                elif user_stance == 1:
                    counter_leave_users += 1
                dict[str(user_id)] = user_stance

        logger.info("number of users: [remain, leave]: " + str(counter_remain_users)+","+str(counter_leave_users))
        return dict

    except Exception as ex:
        logger.error(ex)


#def extract_users_total_topic_counts():
#    file = globals.INPUT_FILE_FULL_FEATURES
#    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
#        count_tweet = 0
#        users_total_topic_counts = {}
#
#        counter_en = 0
#        counter_und = 0
#        for line in ins:
#            try:
#                count_tweet += 1
#                if count_tweet % 100000 == 0:
#                    logger.info("count_tweet:" + str(count_tweet))
#
#                fields = line.split("~")
#
#                first_col = fields[0][0:2]
#                language = fields[len(fields) - 3]
#                tweet_text = fields[3]
#                user_id = fields[2]
#                tweet_id = fields[1]
#                p1 = fields[len(fields) - 1].rstrip("\r\n")
#
#                if not is_read_line_eligible(first_col, language, tweet_text, p1):
#                    counter_und += 1
#                    continue
#                counter_en += 1
#
#                if p1 == '0' or p1 == '1' or p1 == '2':
#
#                    if user_id in users_total_topic_counts.keys():
#                        if p1 == '0':
#                            users_total_topic_counts[user_id][0] += 1
#                        elif p1 == '1':
#                            users_total_topic_counts[user_id][1] += 1
#                        elif p1 == '2':
#                            users_total_topic_counts[user_id][2] += 1
#
#                    else:
#                        if p1 == '0':
#                            value = [1, 0, 0]
#                        elif p1 == '1':
#                            value = [0, 1, 0]
#                        elif p1 == '2':
#                            value = [0, 0, 1]
#                        users_total_topic_counts[user_id] = value
#
#            except Exception as ex:
#                logger.info(ex)
#                logger.info(traceback.format_exc())
#
#        logger.info("counter_und:" + str(counter_und))
#        logger.info("completed. number of english tweets: " + str(counter_en) + " and size of dict " + str(
#            len(users_total_topic_counts.items())))
#        return users_total_topic_counts


 #stance_score = (p0_count - p1_count) / (p0_count + p1_count)
 #       stance = -1
 #       if stance_score < -0.2:
 #           counter_stance_users_leave += 1
 #           stance = 2
 #       elif stance_score > 0.2:
 #           counter_stance_users_remain += 1
 #           stance = 1
 #       else:
 #           counter_stance_users_neutral += 1
 #           stance = 0
 #       users_stances[key] = stance#

def extract_users_stances(users_total_topic_counts):
    users_stances = {}
    # 1 remain
    # 2 leave
    # 0 neutral

    counter_stance_users_remain = 0
    counter_stance_users_leave = 0
    logger.info(" number of total users: " + str(len(users_total_topic_counts.items())))
    counter = 0
    for key, value in users_total_topic_counts.items():
        if counter % 1000 == 0:
            logger.info("***** counter:" + str(counter))
        p0_count = int(value[0])
        p1_count = int(value[1])
        stance = -1
        if p0_count >= p1_count:
            stance = 0
        else:
            stance = 1

        users_stances[key] = stance

        counter += 1

    logger.info("counter_stance_users_remain,counter_stance_users_leave,counter_stance_users_neutral: " + str(
        counter_stance_users_remain) + "," + str(counter_stance_users_leave) + "," + str(counter_stance_users_neutral))

    return users_stances


def calculate_r1():
    try:
        df = utils.read_file("F:/tmp/full_features.csv_out.csv", "~", ['ID', 'text', 'tw_hashtags', 'r1'])
        print(df["r1"].value_counts())
        print("ok")
    except Exception as e:
        logger.error(str(e))


def create_ml_r1_file_read_line():
    #calculate_r1()
    f = open(globals.INPUT_FILE_FULL_FEATURES, "r", encoding="utf-8", errors="ignore")
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


def extract_daily_average_retweet_likes(file):
    dict = {}
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
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


def extract_monthly_tweets_volume(file):
    dict = {}
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        counter = 0
        for line in ins:
            counter += 1
            try:
                fields = line.split("~")
                datetime = fields[4]
                if datetime[0:2] != '20' or len(datetime) != 19:
                    continue
                yearmonth = datetime[0:7]

                if not yearmonth in dict.keys():
                    dict[yearmonth] = 1
                else:
                    dict[yearmonth] += 1
            except Exception as ex:
                logger.error(str(ex))

        logger.info("counter: " + str(counter))
        logger.info("len of dict: " + str(len(dict)))
    return dict


def extract_users_total_topic_counts(file):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        users_total_topic_counts = {}

        for line in ins:
            try:
                count_tweet += 1
                if count_tweet % 100000 == 0:
                    logger.info("count_tweet:" + str(count_tweet))

                fields = line.split("~")

                user_id = fields[1]
                r1 = fields[4]

                if r1 == '0' or r1 == '1':

                    if user_id in users_total_topic_counts.keys():
                        if r1 == '0':
                            users_total_topic_counts[user_id][0] += 1
                        elif r1 == '1':
                            users_total_topic_counts[user_id][1] += 1

                    else:
                        if r1 == '0':
                            value = [1, 0]
                        elif r1 == '1':
                            value = [0, 1]
                        users_total_topic_counts[user_id] = value

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        logger.info("completed. number tweets: " + str(count_tweet) + " and size of dict " + str(len(users_total_topic_counts.items())))
    return users_total_topic_counts


def extract_daily_involvement_of_prev_calculated_user_stances(file, users_stance):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        unique_days = {}
        counter_en = 0
        counter_und = 0
        list_non_existing_users = []
        logger.info("dict size: " + str(len(users_stance)))
        counter_not_compatible = 0
        for line in ins:
            try:
                if line=='\n':
                    continue;
                count_tweet += 1
                if count_tweet % 10000 == 0:
                    logger.info("count_tweet:" + str(count_tweet))

                fields = line.split("~")
                if(len(fields)!=5):
                    counter_not_compatible += 1
                    logger.error("counter not compatible: " + str(counter_not_compatible) + " for line: " + str(count_tweet))
                    logger.error(line)
                    continue;
                user_id = fields[1]
                datetime = fields[2]

                if user_id not in users_stance:
                    counter_und += 1
                    if user_id not in list_non_existing_users:
                        list_non_existing_users.append(user_id)
                    logger.info("counter_und:" + str(counter_und))
                    logger.error("major problem. user id not existing in user stance for user id: " + str(user_id))
                    continue;
                stance_of_user = str(users_stance[user_id])

                key = datetime
                if key in unique_days:
                    if stance_of_user == '0':
                        unique_days[key][0] += 1
                    elif stance_of_user == '1':
                        unique_days[key][1] += 1

                else:
                    if stance_of_user == '0':
                        value = [1, 0]
                    elif stance_of_user == '1':
                        value = [0, 1]

                    unique_days[key] = value

            except Exception as ex:
                logger.error(ex)
                logger.error(line)
                logger.error(traceback.format_exc())

        logger.info("counter_und:" + str(counter_und))
        logger.info("len missing users:"+str(len(list_non_existing_users)))
        logger.info(
            "completed. number of tweets: " + str(count_tweet) + " and size of dict " + str(len(unique_days.items())))
        return unique_days


def extract_daily_polarized_tweets(file):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        dict = {}
        counter_en = 0
        for line in ins:
            try:
                count_tweet += 1
                if (count_tweet == 45138):
                    logger.info("ok")
                fields = line.split("~")
                language = fields[len(fields) - 2]
                if language == 'en':
                    counter_en += 1
                else:
                    continue

                p1 = fields[len(fields) - 3]
                if p1 == '1' or p1 == '2':
                    datetime = fields[0][0:10]
                    if (datetime[0:2] != '20'):
                        continue
                    logger.info(str(count_tweet) + "," + str(datetime))
                    key = datetime
                    if key in dict:
                        if p1 == '1':
                            dict[key][0] += 1
                        elif p1 == '2':
                            dict[key][1] += 1
                    else:
                        if p1 == '1':
                            value = [1, 0]
                        elif p1 == '2':
                            value = [0, 1]
                        dict[key] = value


            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        logger.info(
            "completed. number of english tweets: " + str(counter_en) + " and size of dict " + str(len(dict.items())))
        return dict


def extract_tweet_text(file):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
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


def count_total_topic_labels(file):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        counter_en = 0
        counter_remain = 0
        counter_leave = 0
        counter_neutral = 0
        counter_mixed = 0
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
                tweet_text = fields[3]

                if tweet_text == "" or tweet_text == "undefined":
                    counter_und += 1
                    continue

                topic_label = fields[len(fields) - 3]
                if topic_label == '0':
                    counter_neutral += 1
                elif topic_label == '1':
                    counter_remain += 1
                elif topic_label == '2':
                    counter_leave += 1
                elif topic_label == '3':
                    counter_mixed += 1

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        logger.info("count_tweets:" + str(count_tweet))
        logger.info("count english tweets:" + str(counter_en))
        logger.info("count_tweets:" + str(count_tweet))
        logger.info("count neutral tweets:" + str(counter_neutral))
        logger.info("count remain tweets:" + str(counter_remain))
        logger.info("count leave tweets:" + str(counter_leave))
        logger.info("count mixed tweets:" + str(counter_mixed))


def group_users_by_posts(file):
    users = {}
    counter = 0
    try:
        with open(file, "r", encoding="utf-8", errors='ignore') as ins:
            for line in ins:
                counter += 1
                if counter < 4:
                    continue
                username = str(line.rstrip("\r\n"))
                if username is None or username == "":
                    print("whatsap")
                if username in users:
                    users[username] += 1
                else:
                    users[username] = 1
    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())
    logger.info(str(len(users)))
    return users


def extract_post_frequency(users):
    frequency = {}
    for key, value in users.items():
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    return frequency


def extract_number_of_tweet_ml_labels_topics(file):
    counter = 0
    counter_neutral = 0
    counter_remain = 0
    counter_leave = 0
    counter_unknown = 0
    counter_undefined = 0
    ml_labeled_tweet = {}
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        users = {}
        count_tweet = 0
        dict = {}
        for line in ins:
            try:
                counter += 1
                fields = line.split("~")
                opinion = fields[len(fields) - 1]
                tweet_id = fields[1]
                if opinion == None:
                    logger.info("opinion is none for line: " + line)
                else:
                    opinion = opinion.rstrip("\r\n")
                    if opinion == "0.0":
                        counter_neutral += 1
                        ml_labeled_tweet[tweet_id] = str(int(float(opinion)))
                    elif opinion == "1.0":
                        counter_remain += 1
                        ml_labeled_tweet[tweet_id] = str(int(float(opinion)))
                    elif opinion == "2.0":
                        counter_leave += 1
                        ml_labeled_tweet[tweet_id] = str(int(float(opinion)))
                    elif opinion == "undefined":
                        counter_undefined += 1
                    else:
                        counter_unknown += 1
                        logger.info("unknown line : " + line)
            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        logger.info("counter: " + str(counter))
        logger.info("counter_neutral: " + str(counter_neutral))
        logger.info("counter_remain: " + str(counter_remain))
        logger.info("counter_leave: " + str(counter_leave))
        logger.info("counter_unknown: " + str(counter_unknown))
        return ml_labeled_tweet


def get_user_id_stances(file):
    dict = {}
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        for line in ins:
            fields = line.split("~")
            dict[fields[0]] = fields[1].rstrip("\r\n")

    return dict


def get_user_names_ids(file):
    dict = {}
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        for line in ins:
            fields = line.split("~")
            if len(fields) == 2:
                name = fields[1].rstrip("\r\n").lower()
                if name in dict:
                    continue
                id = fields[0]
                dict[name] = id

    return dict


def add_stance_to_last_column_for_bots2(file, filename_write, dict_user_names_ids, dict_user_stances):
    file_write = open(filename_write, "w", encoding='utf-8')

    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
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

                    if counter == 1:
                        counter += 1
                        continue
                    field = field.rstrip("\r\n")
                    output_line += str(field)
                    output_line += "~"
                    counter += 1

                output_line += str(stance)
                file_write.write(output_line)
                file_write.write("\n")
            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())
        print(str(counter_non_exist))


def extract_desired_field_distinct_user(file, index):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        users = {}
        count_tweet = 0
        dict = {}
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")

                user_id = fields[2]
                target = fields[len(fields) - index]
                if user_id not in users:
                    users[user_id] = target

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        logger.info("count_tweets:" + str(count_tweet))
        logger.info("count_user:" + str(len(users.items())))

        for key, value in users.items():
            if value in dict:
                dict[value] = dict[value] + 1
            else:
                dict[value] = 1

        for key, value in dict.items():
            logger.info(str(key) + ":" + str(dict[key]))


def extract_tweet_text_discover_neutrals(file_train, file_all):
    texts_0_train = {}

    with open(file_train, "r", encoding="utf-8", errors='ignore') as ins:
        logger.info("started to load train neutrals")
        count_tweet_train = 0
        for line in ins:
            count_tweet_train += 1
            fields = line.split("~")
            tweet_id = fields[0]
            texts_0_train[tweet_id] = "1"
    logger.info("completed loading neutrals: records count: " + str(count_tweet_train))

    with open(file_all, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        texts_0 = {}
        count_skip_train = 0

        logger.info("started to select texts randomly")
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")

                tweet_id = fields[0]
                if tweet_id in texts_0_train.keys():
                    count_skip_train += 1
                    logger.info("skipping line.. counter skipped" + str(count_skip_train))
                    continue

                nbr_retweet = fields[1]
                nbr_favorite = fields[2]
                nbr_reply = fields[3]
                datetime = fields[4]
                tw_full = fields[5]
                tw_lang = fields[6]
                p1 = fields[7]
                user_favourites_count = fields[8]
                user_followers_count = fields[9]
                user_friends_count = fields[10]
                user_statuses_count = fields[11]
                api_res = fields[12]
                if (datetime[0:2] != '20'):
                    continue
                if tw_lang != 'en':
                    continue

                if tw_full == 'undefined':
                    continue

                # "ID", "nbr_retweet", "nbr_favorite", "nbr_reply", "datetime", "tw_full", "tw_lang", "p1",
                # "user_favourites_count", "user_followers_count", "user_friends_count", "user_statuses_count",
                # "api_res"

                if p1 != '0':
                    continue

                text = nbr_retweet + "~" + nbr_favorite + "~" + nbr_reply + "~" + datetime + "~" + tw_full + "~" + tw_lang + "~" + p1 + "~" + user_favourites_count + "~" + user_followers_count + "~" + user_friends_count + "~" + user_statuses_count + "~" + api_res

                texts_0[tweet_id] = text

            except Exception as ex:
                logger.info(line)
                logger.info(ex)
                logger.info(traceback.format_exc())

        return texts_0


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


def extract_fields_by_r1(file, r1):
    df = utils.read_file(file, "~", names=['ID', 'user_id', 'datetime', 'text', 'r1'])
    df = df[df['r1'] == r1]
    logger.info(df.head())
    if r1 == 1:
        df['r1']= 0
    elif r1 == 2:
        df['r1'] = 1
    logger.info(df.head())
    df.to_csv(file + "_" + str(r1) + "_out.csv", index=False, columns=['ID', 'user_id', 'datetime', 'text', 'r1'], header=None, sep="~")


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


def extract_tweet_text_by_topic_label_random_n_records(file, requested_amount):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        texts_1 = {}
        texts_2 = {}
        logger.info("started to read file")
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")

                tweet_id = fields[0]
                nbr_retweet = fields[1]
                nbr_favorite = fields[2]
                nbr_reply = fields[3]
                datetime = fields[4]
                tw_full = fields[5]
                tw_lang = fields[6]
                p1 = fields[7]
                user_favourites_count = fields[8]
                user_followers_count = fields[9]
                user_friends_count = fields[10]
                user_statuses_count = fields[11]
                api_res = fields[12]
                if (datetime[0:2] != '20'):
                    continue
                if tw_lang != 'en':
                    continue

                if tw_full == 'undefined':
                    continue

                # "ID", "nbr_retweet", "nbr_favorite", "nbr_reply", "datetime", "tw_full", "tw_lang", "p1",
                # "user_favourites_count", "user_followers_count", "user_friends_count", "user_statuses_count",
                # "api_res"

                if p1 != '1' and p1 != '2':
                    continue

                text = nbr_retweet + "~" + nbr_favorite + "~" + nbr_reply + "~" + datetime + "~" + tw_full + "~" + tw_lang + "~" + p1 + "~" + user_favourites_count + "~" + user_followers_count + "~" + user_friends_count + "~" + user_statuses_count + "~" + api_res

                if p1 == '1':
                    texts_1[tweet_id] = text
                elif p1 == '2':
                    texts_2[tweet_id] = text

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())
        logger.info("started to select texts randomly")

        randomly_selected_texts_1 = get_random_texts(requested_amount, texts_1)
        randomly_selected_texts_2 = get_random_texts(requested_amount, texts_2)

        return randomly_selected_texts_1, randomly_selected_texts_2


def extract_write_tweet_text_by_topic_label(file, filename_write, topic_label):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        texts_0 = {}
        file_write = open(filename_write, "w", encoding='utf-8')

        logger.info("started to select topic label: " + str(topic_label))
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")
                if (fields[0][0:2] != '20'):
                    continue
                language = fields[len(fields) - 2]
                if language != 'en':
                    continue

                tweet_id = fields[1]
                text = fields[len(fields) - 10]
                if text == 'undefined':
                    continue

                p1 = fields[len(fields) - 3]
                text += '~' + str(p1)

                if p1 != topic_label:
                    continue

                output_line = ""
                counter = 0
                for field in fields:
                    output_line += str(field)
                    if counter != len(fields):
                        output_line += "~"
                    counter += 1
                file_write.write(output_line)
                file_write.write("\n")

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        return texts_0


def get_random_texts(requested_amount, texts):
    count_random = 0
    randomly_selected_texts = {}
    logger.info("started to select random numbers")
    while count_random < requested_amount:
        random_index = random.randint(0, len(texts.items()) - 1)
        logger.info("random index:" + str(random_index))

        random_text_key = list(texts.keys())[random_index]
        random_text_value = list(texts.values())[random_index]
        if random_text_key not in randomly_selected_texts.keys() and random_text_value != 'undefined':
            randomly_selected_texts[random_text_key] = random_text_value
            count_random += 1
    logger.info("completed selecting random numbers")
    return randomly_selected_texts


def extract_convert_lda_input(file):
    logger.info("started LDA related operations")
    df = utils.read_file(file, "~", names=['ID', 'datetime', 'text'])
    df_new = utils.preprocess_text_for_topic_discovery(df)
    df_new.to_csv(file+ "_out.csv", index=False)


def extract_hashtag_usage(file):
    hashtag_usage = {}
    counter_en = 0

    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
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


def write_list_to_file(filename_write, texts):
    logger.info("started to write")
    file_write = open(filename_write, "w", encoding='utf-8')
    for row in texts:
        file_write.write(str(row[0]) + "," + str(row[1]))
        file_write.write("\n")
    logger.info("completed writing")


def write_dict_to_file(filename_write, texts):
    logger.info("started to write")
    counter = 0
    file_write = open(filename_write, "w", encoding='utf-8')
    for key, value in texts.items():
        counter += 1
        if counter % 1000 == 0:
            logger.info(str(counter))
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


def print_pro_remain(dict):
    for key, value in dict.items():
        if str(value) == '1':
            logger.info(str(key))


def main():
    logger.basicConfig(level="INFO", filename="F:/tmp/custom.log", format="%(asctime)s %(message)s")

    try:
        print("ok")
        extract_convert_lda_input("F:/tmp/test.txt")
        #extract_fields_by_r1("F:/tmp/full_en3.v.csv",1)
        extract_hashtag_usage("F:/tmp/full_en3.csv")
        #create_ml_r1_file_read_line()
        users_stance = utils.read_file_to_dict("F:/tmp/merged_users.csv","~")
        #unique_days = extract_daily_involvement_of_prev_calculated_user_stances("F:/tmp/merged_tweets.csv", users_stance)
        #write_dict_to_file("F:/tmp/unique_days.csv",unique_days)
        #filename = "merged_RB_MLMA_out.csv"
        #filename = "F:/tmp/test.txt"
        #users_stances = pandas_users_stances("F:/tmp/full_en3.csv_out.csv")
        #write_dict_to_file("F:/tmp/full_en3_rule_based_out.csv", users_stances)
        #dict = extract_daily_involvement_of_prev_calculated_user_stances(filename, users_stances)
        #write_text_list_to_file(filename+"_out.csv", dict)
        #create_ml_r1_file_read_line()
        # analyze_group_by_influence("F:/tmp/impact.csv")
        # analyze_duplicate_tweets("F:/tmp/full_features.csv")
        # extract_fields_by_r1(globals.INPUT_FILE_NAME_RB, 2)
        # pandas_extract_tweet_text_by_topic_label_random_n_records("F:/tmp/full_features.csv",5000, 1)
        # pandas_extract_tweet_text_by_topic_label_random_n_records("F:/tmp/full_features.csv",5000, 2)
        # extract_neutrals("F:/tmp/full_features.csvl_out.csv")
        # r1_stats()
        # rule_based_user_stance()
        # create_ml_r1_file_read_line()
        # filename = "C:/mongo/bin/mongo_export_latest_best.csv"
        # filename = "F:tmp/test-1-out.txt"
        #filename_user_id_names = "C:/mongo/bin/user_id_name.csv"
        #filename_stance = "F:tmp/ml_stance.txt"
        ## filename = "F:tmp/bots-378k.csv"
        ## filename = "F:tmp/full_fields.csv"
        #filen#ame = "F:/tmp/user_screen_names.csv"
        #filename_test = "F:tmp/ml_test.txt"
        #filename_ml = "F:tmp/pred_data.csv"
        #filename_write_bot = "F:tmp/bot-378k-wuserid"
        #filename_write = "F:/tmp/user_screen_names_out_2.csv"
        #filename_write_ml = "F:tmp/test-1-out.txt"
        #test = "C:/mongo/bin/tt2.csv"
        # dict = extract_daily_average_retweet_likes(filename)
        # users = group_users_by_posts(filename)
        # dict = extract_post_frequency(users)
        # write_text_list_to_file(filename_write, dict)

        # dict_user_names_ids = get_user_names_ids(filename_user_id_names)
        # dict_user_stances = get_user_id_stances(filename_stance)

        # add_stance_to_last_column_for_bots2(filename, filename_write, dict_user_names_ids, dict_user_stances)
        # create_ml_p1_file(filename, filename_ml, filename_write_ml)
        # dict = extract_tweet_text(filename)
        # count_total_topic_labels(filename)
        # extract_desired_field_distinct_user(filename, 1)
        # texts_1,texts_2 = extract_tweet_text_by_topic_label_random_n_records(filename, 1000)
        # dict = extract_daily_polarized_tweets(test)
        # extract_number_of_tweet_ml_labels_topics(filename)
        # update_mongo_with_ml_tweet_labels(filename_ml)
        #users_total_topic_counts = extract_users_total_topic_counts("F:/tmp/test.txt")
        print("ok")
        # users_stances = extract_users_stances(users_total_topic_counts, False)
        # extract_write_tweet_text_by_topic_label(filename, filename_write, "0")
        # logger.info_pro_remain(users_stances)
        # dict = extract_tweet_text_discover_neutrals("C:/mongo/bin/neutral.csv", "C:/mongo/bin/full_features.csv")
        # dict = extract_daily_involvement_of_prev_calculated_user_stances(filename, users_stances)
        # write_text_list_to_file(filename_write,dict)
        # sorted_hashtags = extract_hashtag_usage(filename)
        # write_list_to_file(filename_write, sorted_hashtags)
        # write_dict_to_file(filename_write, dict)
        # filename_write_1 = filename_write + '1'
        # filename_write_2 = filename_write + '2'
        # write_dict_to_file(filename_write_0, texts_0)
        # write_dict_to_file(filename_write_1, texts_1)
        # write_dict_to_file(filename_write_2, texts_2)



    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()

