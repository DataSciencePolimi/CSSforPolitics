# TweetAnalyser
this project contains separate python classes that performs data enrichment for the tweets. 

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
