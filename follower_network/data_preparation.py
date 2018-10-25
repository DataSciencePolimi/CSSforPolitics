import traceback
from util import globals
from pymongo import MongoClient
import logging as logger


def extract_followers_from_mongo():
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