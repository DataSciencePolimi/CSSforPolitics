import traceback
import logging as logger
import random
import sys

#1 remain
#2 leave
#0 neutral
#3 mixed

def extract_users_total_topic_counts(file):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        users_total_topic_counts = {}

        counter_en = 0
        counter_und = 0
        for line in ins:
            try:
                count_tweet += 1
                #if count_tweet % 10000 == 0:
                #    break
                fields = line.split("~")
                if (fields[0][0:2] != '20'):
                    continue
                language = fields[len(fields)-2]
                if language == 'en':
                    counter_en += 1
                else:
                    continue
                tweet_text = fields[3]
                if tweet_text == "" or tweet_text == "undefined":
                    counter_und += 1
                    continue

                p1 = fields[len(fields) - 3]
                if p1 == '0' or p1 == '1' or p1 == '2':
                    user_id = fields[2]
                    if user_id in users_total_topic_counts:
                        if p1 == '0':
                            users_total_topic_counts[user_id][0] += 1
                        elif p1 == '1':
                            users_total_topic_counts[user_id][1] += 1
                        elif p1 == '2':
                            users_total_topic_counts[user_id][2] += 1
                    else:
                        if p1 == '0':
                            value = [1, 0, 0]
                        elif p1 == '1':
                            value = [0, 1, 0]
                        elif p1 == '2':
                            value = [0, 0, 1]
                        users_total_topic_counts[user_id] = value


            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        print("counter_und:" + str(counter_und))
        print("completed. number of english tweets: " + str(counter_en) + " and size of dict " + str(len(users_total_topic_counts.items())))
        return users_total_topic_counts


def extract_users_stances(users_total_topic_counts):
    users_stances = {}
    # 1 remain
    # 2 leave
    # 0 neutral
    # 3 mixed
    counter_stance_users_remain = 0
    counter_stance_users_leave = 0
    counter_stance_users_neutral = 0
    for key, value in users_total_topic_counts.items():
        p0_count = int(value[0])
        p1_count = int(value[1])
        p2_count = int(value[2])
        if(key=='610369662'):
            print("wait")

        stance_score = (p1_count - p2_count) / (p0_count + p1_count + p2_count)
        stance = -1
        if stance_score < -0.2:
            counter_stance_users_leave += 1
            stance = 2
        elif stance_score > 0.2:
            counter_stance_users_remain += 1
            stance = 1
        else:
            counter_stance_users_neutral += 1
            stance = 0
        users_stances[key] = stance
    print("counter_stance_users_remain,counter_stance_users_leave,counter_stance_users_neutral: " + str(counter_stance_users_remain) + "," + str(counter_stance_users_leave) + "," + str(counter_stance_users_neutral))

    return users_stances


def extract_daily_involvement_of_prev_calculated_user_stances(file, users_stance):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        dict = {}
        counter_en = 0
        counter_und = 0
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")
                if (fields[0][0:2] != '20'):
                    continue
                language = fields[len(fields)-2]
                if language == 'en':
                    counter_en += 1
                else:
                    continue
                tweet_text = fields[3]
                if tweet_text == "" or tweet_text == "undefined":
                    counter_und += 1
                    continue

                datetime = fields[0][0:10]
                user_id = fields[2]
                stance_of_user = users_stance[user_id]
                stance_of_user = str(stance_of_user)
                key = datetime
                if key in dict:
                    if stance_of_user == '0':
                        dict[key][0] += 1
                    elif stance_of_user == '1':
                        dict[key][1] += 1
                    elif stance_of_user == '2':
                        dict[key][2] += 1
                else:
                    if stance_of_user == '0':
                        value = [1, 0, 0]
                    elif stance_of_user == '1':
                        value = [0, 1, 0]
                    elif stance_of_user == '2':
                        value = [0, 0, 1]
                    dict[key] = value


            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        print("counter_und:" + str(counter_und))
        print("completed. number of english tweets: " + str(counter_en) + " and size of dict " + str(len(dict.items())))
        return dict

def extract_daily_polarized_tweets(file):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        dict = {}
        counter_en = 0
        for line in ins:
            try:
                count_tweet += 1
                if(count_tweet == 45138):
                    print("ok")
                fields = line.split("~")
                language = fields[len(fields)-2]
                if language == 'en':
                    counter_en += 1
                else:
                    continue

                p1 = fields[len(fields) - 3]
                if p1 == '1' or p1 == '2':
                    datetime = fields[0][0:10]
                    if(datetime[0:2]!='20'):
                        continue
                    print(str(count_tweet) + "," + str(datetime))
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

        print("completed. number of english tweets: " + str(counter_en) + " and size of dict " + str(len(dict.items())))
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
                language = fields[len(fields)-2]
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


        print("counter undefined:" + str(counter_und))

        print("count_tweets:" + str(count_tweet))
        print("count english tweets:" + str(counter_en))
        print("count_tweets:" + str(count_tweet))
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
                language = fields[len(fields)-2]
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

        print("count_tweets:" + str(count_tweet))
        print("count english tweets:" + str(counter_en))
        print("count_tweets:" + str(count_tweet))
        print("count neutral tweets:" + str(counter_neutral))
        print("count remain tweets:" + str(counter_remain))
        print("count leave tweets:" + str(counter_leave))
        print("count mixed tweets:" + str(counter_mixed))


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

        print("count_tweets:" + str(count_tweet))
        print("count_user:" + str(len(users.items())))

        for key, value in users.items():
            if value in dict:
                dict[value] = dict[value] + 1
            else:
                dict[value] = 1

        for key, value in dict.items():
            print(str(key) + ":" + str(dict[key]))


def extract_tweet_text_by_topic_label_random_n_records(file, topic_label, requested_amount):
    with open(file, "r", encoding="utf-8", errors='ignore') as ins:
        count_tweet = 0
        texts = {}
        randomly_selected_texts = {}
        print("started to select texts randomly for topic label: " + str(topic_label))
        for line in ins:
            try:
                count_tweet += 1
                fields = line.split("~")
                tweet_id = fields[1]
                p1 = fields[len(fields) - 3]
                if p1 == topic_label:
                    text = fields[len(fields) - 10]
                    texts[tweet_id]=text

            except Exception as ex:
                logger.info(ex)
                logger.info(traceback.format_exc())

        count_random = 0
        if len(texts.items()) < requested_amount:
            print("error, requested amount should be more or equal than file size")
            sys.exit(-1)
        while count_random < requested_amount:
            random_index = random.randint(0, len(texts.items())-1)
            print("random index:" + str(random_index))

            random_text_key = list(texts.keys())[random_index]
            random_text_value = list(texts.values())[random_index]
            if random_text_key not in randomly_selected_texts and random_text_value!='undefined':
                randomly_selected_texts[random_text_key]=random_text_value
                count_random += 1

        return randomly_selected_texts


def write_texts_to_file(filename_write, texts):
    print("started to write")
    file_write = open(filename_write, "w", encoding='utf-8')
    for key,value in texts.items():
        file_write.write(str(key)+"~"+str(value))
        file_write.write("\n")
    print("completed writing")


def write_text_list_to_file(filename_write, texts):
    print("started to write")
    file_write = open(filename_write, "w", encoding='utf-8')
    for key,value in texts.items():


        #file_write.write(str(key) + "~" + str(value))

        #if len(value) == 2:
        #    file_write.write(str(key)+","+str(value[0]) + "," + str(value[1]))
        if len(value) == 3:
            file_write.write(str(key)+","+str(value[0]) + "," + str(value[1]) + "," + str(value[2]) )

        file_write.write("\n")
    print("completed writing")


def print_pro_remain(dict):
    for key, value in dict.items():
        if str(value)=='1':
            print(str(key))


def main():
    logger.basicConfig(level="INFO", filename="F:/tmp/bot.log", format="%(asctime)s %(message)s")

    filename = "C:/mongo/bin/mongo_export_latest_best.csv"
    filename_write = "C:/mongo/bin/out11.csv"
    test = "C:/mongo/bin/tt2.csv"
    try:
        #dict = extract_tweet_text(filename)
        count_total_topic_labels(filename)
        #extract_desired_field_distinct_user(filename, 1)
        #dict = extract_tweet_text_by_topic_label_random_n_records(test, '1', 2)
        #dict = extract_daily_polarized_tweets(test)
        #users_total_topic_counts = extract_users_total_topic_counts(filename)
        #users_stances = extract_users_stances(users_total_topic_counts)
        #print_pro_remain(users_stances)

        #dict = extract_daily_involvement_of_prev_calculated_user_stances(test, users_stances)
        #write_text_list_to_file(filename_write,dict)

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()
