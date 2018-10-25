import traceback
from util import globals
from pymongo import MongoClient
import logging as logger
import tweepy
from pymongo import MongoClient
import time
import sys
import logging as logger
import traceback


def get_followers_of_user(screen_name, api):
    #this method is used to make an API request to Twitter to collect follower information of given user
    ids = []
    counter = 0
    for page in tweepy.Cursor(api.followers_ids, screen_name=screen_name).pages():
        ids.extend(page)
        logger.info(str(counter) + "st call made. now sleeping")
        time.sleep(60)
        counter += 1
        logger.info("counter'" + str(
            counter) + " query for screen_name: " + screen_name + " len of fetched follower count: " + str(
            len(ids)))
    logger.info("followers of " + screen_name + " are: " + str(ids))
    return ids


def enrich_mongo_with_followers_info():
    # this method is used to add the follower information as an additional attribute to the existing MongoDB record.
    consumer_key = sys.argv[1]
    consumer_secret = sys.argv[2]
    access_token = sys.argv[3]
    access_token_secret = sys.argv[4]
    log_path = sys.argv[5]
    month_start = int(sys.argv[6])
    month_end = int(sys.argv[7])
    day_start = int(sys.argv[8])
    day_end = int(sys.argv[9])

    logger.basicConfig(level="INFO", filename=log_path, format="%(asctime)s %(message)s")
    logger.info("started to tweepy enrichment operations on MongoDB")

    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        # mongo listens in port 2017 by default
        client = MongoClient('localhost:27017')
        db = client.TweetScraper

        counter_new = 0
        counter_already_enriched = 0

        # Important note: There are nested loops here because the code is generating the value of datetime field which is indexed in MongoDB.
        for month in range(month_start, month_end):
            mymonth = str(month).rjust(2, '0')
            for day in range(day_start, day_end):
                myday = str(day).rjust(2, '0')
                for hour in range(0, 24):
                    myhour = str(hour).rjust(2, '0')
                    for min in range(0, 60):
                        mymin = str(min).rjust(2, '0')
                        for sec in range(0, 60):
                            mysec = str(sec).rjust(2, '0')
                            filterdate = "2018-" + str(mymonth) + "-" + str(myday) + " " + str(myhour) + ":" + str(
                                mymin) + ":" + str(mysec)

                            for res in db.tweet.find({"datetime": filterdate}):
                                try:

                                    if res is None:
                                        logger.info("end of tweet cursor")
                                        break

                                    if "api_res" in res:
                                        continue;

                                    if "user_err" in res:
                                        logger.info("tweet ID: " + res[
                                            "ID"] + " has already user error. skipping. no need to enrich.")
                                        continue;

                                    if not "user_id" in res:
                                        logger.info(res["ID"] + " has not user id. skipping record.")
                                        continue;
                                    existing_user = False

                                    for user_mongo in db.user.find({"user_id": res["user_id"]}):
                                        counter_already_enriched += 1
                                        logger.info(user_mongo[
                                                        "user_id"] + " already enriched. total counter already enriched: " + str(
                                            counter_already_enriched))
                                        existing_user = True

                                    if not existing_user:
                                        followers = get_followers_of_user(res["user_screen_name"], api)
                                        len_followers = len(followers)
                                        logger.info(
                                            res["user_screen_name"] + ": nb of followers" + str(len_followers))
                                        followers_str = [str(follower) for follower in followers]
                                        counter_new += 1
                                        db.user.insert(
                                            {"user_id": res["user_id"], "user_screen_name": res["user_screen_name"],
                                             "f": followers_str, "f_cnt": len_followers})

                                        logger.info(
                                            "new enrichment count:" + str(counter_new) + " screen_name: " + res[
                                                "user_id"] + " date: " + filterdate)


                                except Exception as e:
                                    logger.info('Skipping tweet, not known error, for ID:', res["ID"], ',', e)
                                    logger.error("Something bad happened: %s", e)
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"user_err": '1'}})

    except Exception as e:
        logger.error("Something bad happened: %s", e)


def extract_followers_from_mongo():
    #this method extracts the records from database by converting into a readable format
    try:
        client = MongoClient('localhost:27017')
        db = client.TweetScraper
        filename_write = "F:\tmp\graph_input_edges.txt"
        file_write = open(filename_write, "w", encoding='utf-8')

        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")
        for res in db.user.find():
            user_id = res["user_id"]
            followers = res["f"]
            for follower in followers:
                file_write.write(user_id + "," + follower)
                file_write.write("\n")
                file_write.flush()


    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    extract_followers_from_mongo()