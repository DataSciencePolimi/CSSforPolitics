import botometer
import traceback
import logging as logger
import utils
import globals

#This class performs api calls using Botometer API, and creates a new file with bot scores based on an input file.

def get_bot_scores(api, accounts):
    accounts_w_bbs = {}
    for account in accounts:
        account = utils.remove_ampercant_first_char_if_exists(account)
        res = api.check_account(account)
        accounts_w_bbs[account] = res['scores']['universal']
    return accounts_w_bbs


def main():
    try:
        logger.basicConfig(level="INFO", filename=globals.WINDOWS_LOG_PATH, format="%(asctime)s %(message)s")
        filename_read = "F:/tmp/bot_input_influencers.txt"
        filename_write = filename_read + "_outtest"
        file_write = open(filename_write, "w", encoding='utf-8')

        logger.info("started to bot detection")

        twitter_app_auth = {
            'consumer_key': '5qCReXUD50uljLWk6swCkizTw',
            'consumer_secret': 'II1fSNwsLdB8GQvv8klC1FyHlEslobOrjybEIrl62dvQLweUvX',
            'access_token': '1559646294-IyP62RoCVwt6gM9mXFdJs2ZiyazLZEMtsFejX9I',
            'access_token_secret': 'sC5Gy0DgWaTfE6h6w3bQxCJ37w58lGyCbdj44qEzVwFRo',
        }
        # bon = botornot.BotOrNot(**twitter_app_auth)
        mashape_key = "mrI6bB5qjrmshTVQOHh20ZQwINjqp1JTYHdjsnFKeOCMEIgM36"
        mashape_key2 = "szsWGLORHqmsh7hLrvOi8sS2AU1Pp1Q33ZWjsnGRKb7jC30wt6"

        api = botometer.Botometer(wait_on_ratelimit=True,
                                  mashape_key=mashape_key,
                                  **twitter_app_auth)

        input_list = [line.rstrip('\n') for line in open(filename_read)]

        accounts_w_bbs = get_bot_scores(api, input_list)
        utils.write_dict_to_file(file_write, accounts_w_bbs)

        logger.info("completed all")

    except Exception as ex:
        logger.info(ex)
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main()