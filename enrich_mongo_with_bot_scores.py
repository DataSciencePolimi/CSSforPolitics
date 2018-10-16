import logging as logger
from BotService import *


def main():

    logger.basicConfig(level="INFO", filename="temp_bot.log", format="%(asctime)s %(message)s")
    logger.info("started to bot score related enrichment operations on MongoDB")

    try:

        # mongo listens in port 2017 by default

        db = utils.get_mongo_client_db()
        counter_already_enriched = 0
        counter_new = 0
        bot_api = BotService().get_api()
        user_screen_name = None
        for res in db.user.find():
            try:
                if res is None:
                    logger.info("end of tweet cursor")
                    break

                if "bot_sc" in res:
                    counter_already_enriched += 1
                    if (counter_already_enriched % 1000 == 0):
                        logger.info(res["user_id"] + " already enriched with bot score. skipping. counter_already_enriched:" + str(counter_already_enriched))
                    continue

                if not "user_screen_name" in res:
                    logger.error("fatal error, does not contain screen name: " + res["user_id"])
                    continue

                counter_new += 1
                user_screen_name = res["user_screen_name"]

                logger.info("this user needs to be enriched with his bot score:" + user_screen_name)

                account = utils.remove_ampercant_first_char_if_exists(user_screen_name)
                bot_res = bot_api.check_account(account)
                score = str(bot_res['scores']['universal'])
                score = score[0:4]
                db.user.update({"user_screen_name": user_screen_name}, {"$set": {"bot_sc": score}})

                logger.info("new enrichment count:" + str(counter_new) + " user id:" + user_screen_name)

            except Exception as e:
                logger.info('Skipping tweet, not known error, for ID:', user_screen_name, ',', e)
                logger.error("Something bad happened: %s", e)
                db.user.update({"user_screen_name": user_screen_name}, {"$set": {"bot_err": '1'}})

    except Exception as e:
        logger.error("Something bad happened: %s", e)


if __name__ == "__main__":
    main()
