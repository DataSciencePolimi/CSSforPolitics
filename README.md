# TweetAnalyser
This project contains separate python classes that performs data enrichment for the tweets. 

1 ############ enrich_mongo_with_twitter_api.py ############

This class uses Tweepy framework to make Twitter api calls. It enriches the records in MongoDb by using tweetId.

Tips:

Tweepy has by default the sleep mechanism to respect the api call quotas of Twitter.
If you restart the program, you have the ability to resume where you left off in the database.
Before running the class, the datetime column should be indexed in MongoDB for faster execution


2 ############ enrich_mongo_with_yandex_api.py ############

It calls Yandex api based on 3 criteria: A. Tweet may contain coordinates information. B. Tweet may contain location name (city, country..). C. Tweet contains only and only user location info(city, country, user generated info..)

In the case of A, reverse geocode api is called. In B and C, geocode api is called.

Tips:

Yandex permits 25K api call per day.
If you restart the program, you have the ability to resume where you left off in the database.
Before running the class, the datetime column should be indexed in MongoDB for faster execution

This program updates tweet records with a new country column (the last column: tw_enr_country). 

Sample output :
{
        "_id" : ObjectId("5a36f5ecb824b200ace853a2"),
        "usernameTweet" : "adam_steinert",
        "ID" : "703490712730468353",
        "text" : "I don't remember the Greeks playing hardball with a referendum working out terribly well, #brexit",
        "url" : "/adam_steinert/status/703490712730468353",
        "nbr_retweet" : 0,
        "nbr_favorite" : 0,
        "nbr_reply" : 0,
        "datetime" : "2016-02-27 09:03:50",
        "is_reply" : false,
        "is_retweet" : false,
        "user_id" : "2579491",
        "tw_coordinates" : null,
        "tw_favorite_count" : 0,
        "tw_favorited" : false,
        "tw_geo" : null,
        "tw_hashtags" : "brexit",
        "tw_lang" : "en",
        "tw_loc_country" : null,
        "tw_loc_fullname" : null,
        "tw_loc_name" : null,
        "tw_loc_type" : null,
        "tw_retweet_count" : 0,
        "tw_source" : "Twitter Web Client",
        "user_created_at" : "2007-03-28 00:18",
        "user_default_profile" : false,
        "user_description" : "Le Brexit, c'est l'horreur - time to stop it and #stay. Expert legal translator, Italian to English. Solicitor (NP).",
        "user_favourites_count" : 5022,
        "user_followers_count" : 213,
        "user_friends_count" : 594,
        "user_geo_enabled" : true,
        "user_lang" : "en",
        "user_listed_count" : 6,
        "user_location" : "Bad Wildbad, Deutschland",
        "user_name" : "adam steinert ????",
        "user_profile_image_url" : "http://pbs.twimg.com/profile_images/902989334969819137/krCTwb8l_normal.jpg",
        "user_screen_name" : "adam_steinert",
        "user_statuses_count" : 8966,
        "user_timezone" : "London",
        "user_url" : null,
        "user_utc_offset" : 0,
        "user_verified" : false,
        "tw_enr_country" : "Germany"
}






