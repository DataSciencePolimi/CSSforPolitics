import traceback
import logging as logger
from util import text_utils, globals, ml_utils, utils
import collections
from datetime import datetime
import pandas as pd



def r1_stats():
    df = utils.read_file("F:/tmp/full_features.csvl_out.csv", "~", names=['ID', 'user_id', 'datetime', 'text', 'r1'])
    logger.info("started")
    try:
        print(df['r1'].value_counts())
        grouped = df.groupby(['datetime','r1'])['datetime']
        grouped.count().to_csv('F:/tmp/tt.txt')

    except Exception as e:
        logger.error(str(e))


def extract_stance_changes_of_users_before_after_ref(file):
    #note: input file contains the prediction results of tweets on the last column
    try:
        df = utils.read_file(file,"~", globals.STANCE_FILE_COLUMNS)
        dict_users_before_ref = {}
        dict_users_after_ref = {}
        datetime_ref = datetime.strptime("2016-06-24", '%Y-%m-%d')
        # at first, we count the user-centric stance posts for the period before and after ref
        for index, row in df.iterrows():
            try:
                user_id = row['user_id']
                datetime_object = datetime.strptime(row['datetime'], '%Y-%m-%d')
                r1 = int(row['r1'])
                if(datetime_object < datetime_ref):
                    if not user_id in dict_users_before_ref.keys():
                        if (r1 == 0):
                            value = (1, 0)
                        elif (r1 == 1):
                            value = (0, 1)

                    else:
                        (remain,leave) = dict_users_before_ref[user_id]
                        if (r1 == 0):
                            remain += 1
                        elif (r1 == 1):
                            leave +=1
                        value = (remain, leave)
                    dict_users_before_ref[user_id] = value

                elif (datetime_object > datetime_ref):
                    if not user_id in dict_users_after_ref.keys():
                        if (r1 == 0):
                            value = (1, 0)
                        elif (r1 == 1):
                            value = (0, 1)

                    else:
                        (remain,leave) = dict_users_after_ref[user_id]
                        if (r1 == 0):
                            remain += 1
                        elif (r1 == 1):
                            leave +=1
                        value = (remain, leave)
                    dict_users_after_ref[user_id] = value

            except Exception as ex:
                logger.error(row)
                logger.error(str(ex))
                logger.info(traceback.format_exc())
        logger.info("number of people-stance couple before ref: " + str(len(dict_users_before_ref)))
        logger.info("number of people-stance couple after ref: " + str(len(dict_users_after_ref)))
        keys_to_be_deleted_from_dict_before = []

        logger.info("discarding operation for the people who don't have tweets in after ref time periods. current size: " + str(len(dict_users_before_ref)))

        for key in dict_users_before_ref.keys():
            has_found = False
            for key2 in dict_users_after_ref.keys():
                if(key == key2):
                    has_found = True
                    break
            if not has_found:
                keys_to_be_deleted_from_dict_before.append(key)

        for item in keys_to_be_deleted_from_dict_before:
            del dict_users_before_ref[item]
        logger.info("discarded " + str(len(keys_to_be_deleted_from_dict_before)) + " of people who don't have tweets in after ref time periods. final size: " + str(len(dict_users_before_ref)))

        ########################################
        keys_to_be_deleted_from_dict_after = []
        logger.info("discarding operation for the people who don't have tweets in after ref time periods. current size: " + str(len(dict_users_after_ref)))

        for key in dict_users_after_ref.keys():
            has_found = False
            for key2 in dict_users_before_ref.keys():
                if(key == key2):
                    has_found = True
                    break
            if not has_found:
                keys_to_be_deleted_from_dict_after.append(key)

        for item in keys_to_be_deleted_from_dict_after:
            del dict_users_after_ref[item]
        logger.info("discarded " + str(len(keys_to_be_deleted_from_dict_after)) + " of people who don't have tweets in after ref time periods. final size: " + str(len(dict_users_after_ref)))


        # now calculating a single value for each user
        dframe = pd.DataFrame(data=None, columns=['before','after'], index=dict_users_before_ref.keys())
        for key, value in dict_users_before_ref.items():
            (remain,leave) = value
            if(remain>=leave):
                dframe.at[key,'before']=0
            else:
                dframe.at[key,'before']=1

        for key, value in dict_users_after_ref.items():
            (remain,leave) = value

            if(remain>=leave):
                dframe.at[key,'after']=0
            else:
                dframe.at[key,'after']=1
        dframe.to_csv(file+"out_stance_change.csv")

    except Exception as ex:
            logger.error(str(ex))
            logger.info(traceback.format_exc())


def extract_stance_changes_of_users_with_only_two_tweets(file):
    #note: this method is no longer used
    try:
        df = utils.read_file(file,"~", globals.STANCE_FILE_COLUMNS)
        dict_users={}

        for index, row in df.iterrows():
            try:
                user_id = row['user_id']
                datetime_object = datetime.strptime(row['datetime'], '%Y-%m-%d')
                r1 = row['r1']
                if not user_id in dict_users.keys():
                    dict_users[user_id] = {1:(datetime_object, str(r1))}
                else:
                    records = dict_users[user_id]
                    if(len(records)==1):
                        old_datetime_object, old_r1 = records[1]
                        if old_datetime_object > datetime_object:
                            records[1] = (datetime_object,str(r1))
                            records[2] = (old_datetime_object,str(old_r1))
                        elif old_datetime_object < datetime_object:
                            records[2] = (datetime_object, str(r1))
                    else:
                        old_datetime_object, old_r1 = records[1]
                        if old_datetime_object > datetime_object:
                            records[1] = (datetime_object,str(r1))
                        elif old_datetime_object < datetime_object:
                            old_datetime_object, old_r1 = records[2]
                            if old_datetime_object < datetime_object:
                                records[2] = (datetime_object, str(r1))
            except Exception as ex:
                logger.error(row)
                logger.error(str(ex))
                logger.info(traceback.format_exc())

        file_write = open(file+"_out.csv","w")
        counter = 0
        for key, value in dict_users.items():
            output = ""
            values_dict = dict_users[key]
            if(len(values_dict) != 2):
                continue;
            output += str(key)+","
            ordered_values = collections.OrderedDict(sorted(values_dict.items()))
            for key2, value2 in ordered_values.items():
                (date, stance) = value2
                output += str(stance) + ","
            output = output[0:len(output)-1]
            counter += 1
            file_write.write(output)
            file_write.write("\n")
            if(counter % 1000 == 0):
                file_write.flush()

        print("ok")
    except Exception as ex:
        logger.error(str(ex))
        logger.info(traceback.format_exc())


def extract_daily_polarized_tweets(file):
    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
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


def calculate_stance_transitions(first, second):
    if(len(first)!=len(second)):
        logger.error("fatal error, list sizes are not equal")
        return None

    zero_to_zero = 0
    one_to_one = 0
    zero_to_one = 0
    one_to_zero = 0
    for i in range(0, len(first)):
        if(first[i]==0 and second[i]==0):
            zero_to_zero += 1
        elif(first[i] == 1 and second[i] == 1):
            one_to_one += 1
        elif (first[i] == 0 and second[i] == 1):
            zero_to_one += 1
        elif (first[i] == 1 and second[i] == 0):
            one_to_zero += 1
        else:
            logger.error("fatal error, list sizes are not equal")
            return None
    zero_to_zero_res = zero_to_zero / (zero_to_zero + one_to_one + zero_to_one + one_to_zero)
    one_to_one_res = one_to_one / (zero_to_zero + one_to_one + zero_to_one + one_to_zero)
    zero_to_one_res = zero_to_one / (zero_to_zero + one_to_one + zero_to_one + one_to_zero)
    one_to_zero_res = one_to_zero / (zero_to_zero + one_to_one + zero_to_one + one_to_zero)
    val = 10
    return zero_to_zero_res*val, one_to_one_res*val, zero_to_one_res*val, one_to_zero_res*val

    print("good")


def plot_stance_transition():
    try:
        scaling_enabled = False
        col_label_list = ['2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12','2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12','2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09']
        col_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33']

        df = utils.read_file("F:/tmp/datt_out.csv", names=col_list)
        rowsize,colsize = df.shape
        plt.interactive(False)
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111)

        plt.yticks([0,1])

        counter = 0
        for i in range(0,colsize):
            if(i==colsize-1):
                break
            counter += 1
            if(counter % 1000 == 0):
                logger.info(str(counter) + " out of " + str(colsize) + " is completed")
            first = df.iloc[:,i].tolist()
            second = df.iloc[:,i+1].tolist()
            zero_to_zero, one_to_one, zero_to_one, one_to_zero = calculate_combinated_weights(first, second)
            logger.info("zero_to_zero, one_to_one, zero_to_one, one_to_zero:"+str(zero_to_zero) +str(one_to_one) +str(zero_to_one) +str(one_to_zero))
            if scaling_enabled:
                if zero_to_zero!=0.0:
                    zero_to_zero=np.log(zero_to_zero)
                if one_to_one != 0.0:
                    one_to_one=np.log(one_to_one)
                if zero_to_one != 0.0:
                    zero_to_one=np.log(zero_to_one)
                if one_to_zero != 0.0:
                    one_to_zero=np.log(one_to_zero)

            plt.plot([i, i+1],[0, 0],linewidth=zero_to_zero, c="black", solid_capstyle="round")
            plt.plot([i, i+1],[1, 1], linewidth=one_to_one,  c="black", solid_capstyle="round")
            plt.plot([i, i+1],[0, 1], linewidth=zero_to_one, c="black", solid_capstyle="round")
            plt.plot([i, i+1],[1, 0], linewidth=one_to_zero, c="black", solid_capstyle="round")

            i+=1
        ax.set_xticklabels(col_label_list, rotation=45)
        plt.savefig("F:/tmp/pplot")
        print("ok")
    except Exception as ex:
        logger.info(str(ex))

def extract_stance_changes_of_users_old(file):
    try:
        df = utils.read_file(file,"~", globals.STANCE_FILE_COLUMNS)
        df_records = df[['user_id', 'r1', 'datetime']].groupby(['user_id', 'r1', 'datetime']).agg({'r1': ["count"]}).reset_index()
        converted = utils.convert_consolidate_monthly_stances(df_records)
        df_final = utils.convert_nested_dict_to_line_chart_input_and_write(converted)
        df_final.to_csv(file+"_out.csv")
        print("ok")
    except Exception as ex:
        logger.error(str(ex))
        logger.info(traceback.format_exc())


def extract_daily_involvement_of_prev_calculated_user_stances(file, users_stance):
    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
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


def extract_users_total_topic_counts(file):
    with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
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



def main():
    if (globals.os == "windows"):
        log_path = "F:/tmp/custom.log"
    else:
        log_path = "custom.log"
    logger.basicConfig(level="INFO", filename=log_path, format="%(asctime)s %(message)s")

    try:
        print("ok")
        extract_stance_changes_of_users_before_after_ref("F:/tmp/test.txt")

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())

    if __name__ == "__main__":
        main()


def group_users_by_posts(file):
    #prepares the input for the method extract_post_frequency. note: these two methods could be simplified by using pandas groupby function.
    users = {}
    counter = 0
    try:
        with open(file, "r", encoding="utf-8", errors='ignore', newline='\n') as ins:
            for line in ins:
                counter += 1
                if counter < 4:
                    continue
                username = str(line.rstrip("\r\n"))
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
    #using the output of method group_users_by_posts
    frequency = {}
    for key, value in users.items():
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    return frequency
