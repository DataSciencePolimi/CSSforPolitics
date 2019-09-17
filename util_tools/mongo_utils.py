
import json
from util_tools import utils

import pandas as pd


def convert_mongo_user_json_to_csv():
    for i in range(1,2):
        input_file = None

        if i<10:
            letter = "0" + str(i)
        else:
            letter = str(i)

        filename = '/Users/emrecalisir/git/cortico/discovery/users_6sep.csv'

        input_file = open(filename)
        #input_file = open ('/Users/emrecalisir/git/cortico/discovery/test.csv')
        json_array = json.load(input_file)
        store_list = []

        counter = 0
        with open(filename + '.out',"w") as f:
            for res in json_array:
                try:
                    if(counter%100==0):
                        print("current status : " + str(counter))

                    if "user_id" in res:
                        user_id = res["user_id"]
                    else:
                        user_id = ""

                    if "user_screen_name" in res:
                        user_screen_name = res["user_screen_name"]
                    else:
                        user_screen_name = ""

                    if "bot_sc" in res:
                        bot_sc = res["bot_sc"]
                    else:
                        bot_sc = ""

                    if "t_eth" in res:
                        t_eth = res["t_eth"]
                    else:
                        t_eth = ""

                    if "t_age" in res:
                        t_age = res["t_age"]
                    else:
                        t_age = ""

                    if "t_gender" in res:
                        t_gender = res["t_gender"]
                    else:
                        t_gender = ""

                    if "api_time" in res:
                        api_time = res["api_time"]
                    else:
                        api_time = ""

                    if "bot_err" in res:
                        bot_err = res["bot_err"]
                    else:
                        bot_err = ""


                    write_line = str(user_id)+"~"+str(user_screen_name)+"~"+str(bot_sc)+"~"+str(api_time)+"~"+str(bot_err)+"~"+str(t_eth)+"~"+str(t_age)+"~"+str(t_gender)
                    counter += 1
                    f.write(write_line + "\n")
                except Exception as ex:
                    print(ex)

        print(letter + " completed at : " + str(counter))




def convert_mongo_tweet_json_to_csv():
    for i in range(1,11):
        input_file = None

        if i<10:
            letter = "0" + str(i)
        else:
            letter = str(i)

        input_file = open ('/Users/emrecalisir/git/cortico/discovery/tws_5sep_2018_'+letter+'.csv')
        #input_file = open ('/Users/emrecalisir/git/cortico/discovery/test.csv')
        json_array = json.load(input_file)
        store_list = []

        counter = 0
        with open('/Users/emrecalisir/git/cortico/discovery/tws_5sep_2018_'+letter+'.out',"w") as f:
            for res in json_array:
                try:
                    if(counter%100==0):
                        print("current status : " + str(counter))

                    if "usernameTweet" in res:
                        usernameTweet = res["usernameTweet"]
                    else:
                        usernameTweet = ""

                    if "ID" in res:
                        ID = res["ID"]
                    else:
                        ID = ""

                    if "text" in res:
                        text = res["text"]
                    else:
                        text = ""

                    if "tw_lang" in res:
                        tw_lang = res["tw_lang"]
                    else:
                        tw_lang = ""

                    if "url" in res:
                        url = res["url"]
                    else:
                        url = ""

                    if "nbr_retweet" in res:
                        nbr_retweet = res["nbr_retweet"]
                    else:
                        nbr_retweet = ""

                    if "nbr_favorite" in res:
                        nbr_favorite = res["nbr_favorite"]
                    else:
                        nbr_favorite = ""

                    if "nbr_reply" in res:
                        nbr_reply = res["nbr_reply"]
                    else:
                        nbr_reply = ""

                    if "datetime" in res:
                        datetime = res["datetime"]
                    else:
                        datetime = ""

                    if "is_reply" in res:
                        is_reply = res["is_reply"]
                    else:
                        is_reply = ""

                    if "is_retweet" in res:
                        is_retweet = res["is_retweet"]
                    else:
                        is_retweet = ""

                    if "user_id" in res:
                        user_id = res["user_id"]
                    else:
                        user_id = ""

                    if "tw_coordinates" in res:
                        tw_coordinates = res["tw_coordinates"]
                    else:
                        tw_coordinates = ""

                    if "tw_favorite_count" in res:
                        tw_favorite_count = res["tw_favorite_count"]
                    else:
                        tw_favorite_count = ""

                    if "tw_favorited" in res:
                        tw_favorited = res["tw_favorited"]
                    else:
                        tw_favorited = ""

                    if "tw_geo" in res:
                        tw_geo = res["tw_geo"]
                    else:
                        tw_geo = ""

                    if "tw_hashtags" in res:
                        tw_hashtags = res["tw_hashtags"]
                    else:
                        tw_hashtags = ""

                    if "tw_loc_country" in res:
                        tw_loc_country = res["tw_loc_country"]
                    else:
                        tw_loc_country = ""

                    if "tw_loc_fullname" in res:
                        tw_loc_fullname = res["tw_loc_fullname"]
                    else:
                        tw_loc_fullname = ""

                    if "tw_loc_name" in res:
                        tw_loc_name = res["tw_loc_name"]
                    else:
                        tw_loc_name = ""

                    if "tw_loc_type" in res:
                        tw_loc_type = res["tw_loc_type"]
                    else:
                        tw_loc_type = ""

                    if "tw_retweet_count" in res:
                        tw_retweet_count = res["tw_retweet_count"]
                    else:
                        tw_retweet_count = ""

                    if "tw_source" in res:
                        tw_source = res["tw_source"]
                    else:
                        tw_source = ""

                    if "user_created_at" in res:
                        user_created_at = res["user_created_at"]
                    else:
                        user_created_at = ""

                    if "user_default_profile" in res:
                        user_default_profile = res["user_default_profile"]
                    else:
                        user_default_profile = ""

                    if "user_description" in res:
                        user_description = res["user_description"]
                    else:
                        user_description = ""

                    if "user_favourites_count" in res:
                        user_favourites_count = res["user_favourites_count"]
                    else:
                        user_favourites_count = ""

                    if "user_followers_count" in res:
                        user_followers_count = res["user_followers_count"]
                    else:
                        user_followers_count = ""

                    if "user_friends_count" in res:
                        user_friends_count = res["user_friends_count"]
                    else:
                        user_friends_count = ""

                    if "user_geo_enabled" in res:
                        user_geo_enabled = res["user_geo_enabled"]
                    else:
                        user_geo_enabled = ""

                    if "user_lang" in res:
                        user_lang = res["user_lang"]
                    else:
                        user_lang = ""

                    if "user_listed_count" in res:
                        user_listed_count = res["user_listed_count"]
                    else:
                        user_listed_count = ""

                    if "user_location" in res:
                        user_location = res["user_location"]
                    else:
                        user_location = ""

                    if "user_name" in res:
                        user_name = res["user_name"]
                    else:
                        user_name = ""

                    if "user_profile_image_url" in res:
                        user_profile_image_url = res["user_profile_image_url"]
                    else:
                        user_profile_image_url = ""

                    if "user_screen_name" in res:
                        user_screen_name = res["user_screen_name"]
                    else:
                        user_screen_name = ""

                    if "user_statuses_count" in res:
                        user_statuses_count = res["user_statuses_count"]
                    else:
                        user_statuses_count = ""

                    if "user_timezone" in res:
                        user_timezone = res["user_timezone"]
                    else:
                        user_timezone = ""

                    if "user_url" in res:
                        user_url = res["user_url"]
                    else:
                        user_url = ""

                    if "user_utc_offset" in res:
                        user_utc_offset = res["user_utc_offset"]
                    else:
                        user_utc_offset = ""

                    if "user_verified" in res:
                        user_verified = res["user_verified"]
                    else:
                        user_verified = ""

                    if "t_age" in res:
                        t_age = res["t_age"]
                    else:
                        t_age = ""

                    if "t_eth" in res:
                        t_eth = res["t_eth"]
                    else:
                        t_eth = ""

                    if "t_gender" in res:
                        t_gender = res["t_gender"]
                    else:
                        t_gender = ""

                    if "tw_full" in res:
                        tw_full = res["tw_full"]
                    else:
                        tw_full = ""

                    if "\n" in text:
                        text.replace("\n","")

                    if "\n" in text:
                        text = text.replace("\n","")

                    if "\n" in tw_full:
                        tw_full = tw_full.replace("\n","")

                    if "\n" in user_description:
                        user_description = user_description.replace("\n","")

                    if "\n" in user_location:
                        user_location = user_location.replace("\n","")

                    write_line = str(usernameTweet)+"~"+str(ID)+"~"+str(text)+"~"+str(tw_lang)+"~"+str(url)+"~"+str(nbr_retweet)+"~"+str(nbr_favorite)+"~"+str(nbr_reply)+"~"+str(datetime)+"~"+str(is_reply)+"~"+str(is_retweet)+"~"+str(user_id)+"~"+str(tw_coordinates)+"~"+str(tw_favorite_count)+"~"+str(tw_favorited)+"~"+str(tw_geo)+"~"+str(tw_hashtags)+"~"+str(tw_loc_country)+"~"+str(tw_loc_fullname)+"~"+str(tw_loc_name)+"~"+str(tw_loc_type)+"~"+str(tw_retweet_count)+"~"+str(tw_source)+"~"+str(user_created_at)+"~"+str(user_default_profile)+"~"+str(user_description)+"~"+str(user_favourites_count)+"~"+str(user_followers_count)+"~"+str(user_friends_count)+"~"+str(user_geo_enabled)+"~"+str(user_lang)+"~"+str(user_listed_count)+"~"+str(user_location)+"~"+str(user_name)+"~"+str(user_profile_image_url)+"~"+str(user_screen_name)+"~"+str(user_statuses_count)+"~"+str(user_timezone)+"~"+str(user_url)+"~"+str(user_utc_offset)+"~"+str(user_verified)+"~"+str(t_age)+"~"+str(t_eth)+"~"+str(t_gender)+"~"+str(tw_full)
                    counter += 1
                    f.write(write_line + "\n")
                except Exception as ex:
                    print(ex)

        print(letter + " completed at : " + str(counter))



def read_file(filename, delimiter=None, names=None, dtype=None, lineterminator='\n'):
    if dtype is None:

        #df = pd.read_csv(filename, delimiter=delimiter, error_bad_lines=False,
        #                 names=names, index_col=False, engine='python')
        df = pd.read_csv(filename, delimiter=delimiter, encoding="ISO-8859-1", error_bad_lines=False,
                         names=names, index_col=False, engine='python')
    else:
        df = pd.read_csv(filename, delimiter=delimiter, encoding="ISO-8859-1", error_bad_lines=False,
                     names=names,lineterminator=lineterminator, dtype=dtype, index_col=False)
    return df


if __name__ == "__main__":

        # convert_mongo_user_json_to_csv()
        df = read_file("/Users/emrecalisir/git/cortico/discovery/users_6sep.csv_all_users.out","~", ["user_id","user_screen_name","bot_sc","api_time","bot_err","t_eth","t_age","t_gender"],
                             dtype={'user_id':str, 'user_screen_name':object, 'bot_sc':object, 'api_time':object, 'bot_err':object, 't_eth':object, 't_age':object, 't_gender':object})

        df_tw_users = read_file("/Users/emrecalisir/git/cortico/discovery/user_list_having_tweets_for_15months.csv", None, ["user_id"],dtype={'user_id':str})
        print(df.head())
        print(df.shape)

        print(df_tw_users.head())
        print(df_tw_users.shape)

        #s1 = pd.merge(df, df_tw_users, how='inner', on=['user_id'])


        s1 = pd.merge(left=df, right=df_tw_users, how='left', left_on='user_id', right_on='user_id')
        print(type(s1))
        print(s1.head())
        print(s1.shape)

        print(s1.dtypes)
        s1['user_id'] = s1['user_id'].astype(str)
        print(s1.dtypes)

        s1.to_csv("/Users/emrecalisir/git/cortico/discovery/user_list_having_tweets_for_15months_with_right_join_having_bot_scores.csv", sep="~",index=None)


        print("Operation completed")
