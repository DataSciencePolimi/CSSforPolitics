remain_hashtag_list = ["#strongerin", "#voteremain", "#intogether", "#labourinforbritain", "#moreincommon",
                       "#greenerin",
                       "#catsagainstbrexit", "#bremain", "#betteroffin", "#leadnotleave", "#remain", "#stay", "#ukineu",
                       "#votein", "#voteyes", "#yes2eu", "#yestoeu", "#sayyes2europe"]

leave_hashtag_list = ["#independenceDay", "#leaveeuofficial", "#leaveeu", "leave", "#labourleave", "#votetoleave",
                      "#voteleave", "#takebackcontrol", "#ivotedleave", "beleave", "#betteroffout", "#britainout",
                      "#nottip", "#takecontrol", "#voteno", "#voteout", "#voteleaveeu"]

WINDOWS_LOG_PATH = "F:/tmp/predictor.log"
UNIX_LOG_PATH = "predictor.log"

ORIGINAL_TEXT_COLUMN = "tweet_text"
PROCESSED_TEXT_COLUMN = "processed_text"

FILE_COLUMNS = ["ID", "nbr_retweet", "nbr_favorite", "nbr_reply", "datetime", "tw_full", "tw_lang", "new_p1",
                        "user_favourites_count", "user_followers_count", "user_friends_count", "user_statuses_count",
                        "api_res","eye_p1"]

DATAFRAME_COLUMNS_INT = ['nbr_retweet', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'new_p1',
                 'hashtag_count', 'mention_count', 'contains_link']

DATAFRAME_COLUMNS = ['tw_full', 'nbr_retweet', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'new_p1',
                 'hashtag_count', 'mention_count', 'contains_link']

TARGET_COLUMN = 'eye_p1'

twitter_app_auth = {
    'consumer_key': '5qCReXUD50uljLWk6swCkizTw',
    'consumer_secret': 'II1fSNwsLdB8GQvv8klC1FyHlEslobOrjybEIrl62dvQLweUvX',
    'access_token': '1559646294-IyP62RoCVwt6gM9mXFdJs2ZiyazLZEMtsFejX9I',
    'access_token_secret': 'sC5Gy0DgWaTfE6h6w3bQxCJ37w58lGyCbdj44qEzVwFRo',
}

mashape_key = "mrI6bB5qjrmshTVQOHh20ZQwINjqp1JTYHdjsnFKeOCMEIgM36"

# twitter_app_auth = {
#    'consumer_key': 'YOUR_KEY',
#    'consumer_secret': 'YOUR_SECRET',
#    'access_token': 'YOUR_TOKEN',
#    'access_token_secret': 'YOUR_TOKEN_SECRET',
# }

# mashape_key = "YOUR BOTOMETER API KEY"
