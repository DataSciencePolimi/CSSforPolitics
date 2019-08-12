import botometer
import traceback
import logging as logger
from util import globals, utils, text_utils
from bot_service import BotService

#This class performs api calls using Botometer API, and creates a new file with bot scores from a given input file.


def request_bot_scores(api, accounts, file_write):
    logger.info("nb of accounts to be requested:" + str(len(accounts)))
    accounts_w_bbs = {}
    counter = 0
    for account in accounts:
        try:
            counter += 1
            account = text_utils.remove_ampercant_first_char_if_exists(account)
            res = api.check_account(account)
            logger.info("counter api call: " + str(counter))
            accounts_w_bbs[account] = res['scores']['universal']
            file_write.write(account+","+str(res['scores']['universal']))
            file_write.write("\n")
            file_write.flush()
        except Exception as ex:
            logger.info(ex)
            logger.info(traceback.format_exc())

    logger.info("nb of accounts successfully requested:" + str(len(accounts_w_bbs)))
    return accounts_w_bbs


def enrich_mongo_with_bot_scores():
    try:

        # mongo listens in port 2017 by default

        db = utils.get_mongo_client_db()
        counter_already_enriched = 0
        counter_new = 0
        bot_api = bot_service.get_api()
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



def main():
    try:
        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")
        filename_read = "F:/tmp/bot_input_influencers.txt"
        filename_write = filename_read + "_outtest"
        file_write = open(filename_write, "w", encoding='utf-8')

        logger.info("Started requesting bot scores")

        bot_api = BotService().get_api()

        input_list = [line.rstrip('\n') for line in open(filename_read)]

        request_bot_scores(bot_api, input_list,file_write)

        logger.info("Completed all")

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()