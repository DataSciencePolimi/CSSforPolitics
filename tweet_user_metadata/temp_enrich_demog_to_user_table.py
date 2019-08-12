import tweepy
from pymongo import MongoClient
import logging as logger
from util import globals


def get_all_followers(screen_name, api):

    ids = []
    counter = 0
    for page in tweepy.Cursor(api.followers_ids, screen_name=screen_name).pages():
        ids.extend(page)
        logger.info("len")
        #time.sleep(60)
        counter += 1
        logger.info("counter'"+str(counter) + " query for screen_name: " + screen_name + " len of fetched follower count: " + str(len(ids)))
    logger.info("followers of " + screen_name + " are: " + str(ids))
    return ids


def main():

    logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")
    logger.info("started to tweepy enrichment operations on MongoDB")

    try:
        # mongo listens in port 2017 by default
        client = MongoClient('localhost:27017')
        db = client.TweetScraper

        counter_new = 0
        counter_existing_user_new = 0
        counter_already_enriched = 0

        # Important note: There are nested loops here because the code is generating the value of datetime field which is indexed in MongoDB.
        for month in range(1, 13):
            mymonth = str(month).rjust(2, '0')
            for day in range(1, 32):
                myday = str(day).rjust(2, '0')
                for hour in range(0, 24):
                    myhour = str(hour).rjust(2, '0')
                    for min in range(0, 60):
                        mymin = str(min).rjust(2, '0')
                        for sec in range(0, 60):
                            mysec = str(sec).rjust(2, '0')
                            filterdate = "2016-" + str(mymonth) + "-" + str(myday) + " " + str(myhour) + ":" + str(
                                mymin) + ":" + str(mysec)

                            for res in db.tweet.find({"datetime": filterdate}):
                                try:

                                    if res is None:
                                        logger.info("end of tweet cursor")
                                        break

                                    if not "user_id" in res:
                                        logger.info(res["ID"] + " has not user id. skipping record.")
                                        continue;

                                    if not "t_age" in res:
                                        logger.info(res["user_id"] + " has not demographics info. skipping record.")
                                        continue;

                                    #if "user_err" in res:
                                        #    logger.info("tweet ID: " + res["ID"] + " has already user error. skipping. no need to enrich.")
                                    #continue;
                                    existing = False
                                    for res_user in db.user.find({"user_id":res["user_id"]}):
                                        existing = True
                                        if "t_age" in res_user:
                                            counter_already_enriched += 1
                                            logger.info(res_user["user_id"] + " already enriched. total counter already enriched: " + str(
                                                counter_already_enriched))
                                        else:
                                            db.user.update({"user_id": res["user_id"]}, {"$set": {"t_age": res["t_age"], "t_eth": res["t_eth"], "t_gender": res["t_gender"]}})
                                            counter_existing_user_new += 1
                                            logger.info(res_user["user_id"] + " counter_existing_user_new: " + str(counter_existing_user_new))
                                        break;

                                    # here the code runs because the user is not existing in users table
                                    if not existing:
                                        db.user.insert({"user_id": res["user_id"],"t_age": res["t_age"], "t_eth": res["t_eth"], "t_gender": res["t_gender"]})
                                        counter_new += 1
                                        logger.info("new enrichment count:" + str(counter_new) + " screen_name: " + res[
                                            "user_id"] + " date: " + filterdate)


                                except Exception as e:
                                    logger.error("Something bad happened: res:" + str(res), e)

    except Exception as e:
        logger.error("Something bad happened: %s", e)


if __name__ == "__main__":
    main()
