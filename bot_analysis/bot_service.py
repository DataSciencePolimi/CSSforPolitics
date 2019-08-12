import botometer
from util import globals


class BotService:
    api = None

    def __init__(self):
        self.api = botometer.Botometer(wait_on_ratelimit=True,
                                       mashape_key=globals.MASHAPE_KEY,
                                       **globals.TWITTER_APP_AUTH)

    def get_api(self):
        return self.api
