import botometer
import traceback
import logging as logger
import utils
import globals

#This class performs api calls using Botometer API, and creates a new file with bot scores from a given input file.


def request_bot_scores(api, accounts, file_write):
    logger.info("nb of accounts to be requested:" + str(len(accounts)))
    accounts_w_bbs = {}
    counter = 0
    for account in accounts:
        try:
            counter += 1
            account = utils.remove_ampercant_first_char_if_exists(account)
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


def main():
    try:
        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")
        filename_read = "F:/tmp/bot_input_influencers.txt"
        filename_write = filename_read + "_outtest"
        file_write = open(filename_write, "w", encoding='utf-8')

        logger.info("Started requesting bot scores")

        api = botometer.Botometer(wait_on_ratelimit=True,
                                  mashape_key=globals.mashape_key,
                                  **globals.twitter_app_auth)

        input_list = [line.rstrip('\n') for line in open(filename_read)]

        request_bot_scores(api, input_list,file_write)

        logger.info("Completed all")

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()