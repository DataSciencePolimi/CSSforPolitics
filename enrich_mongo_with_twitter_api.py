import tweepy
from tweepy import User
from tweepy import TweepError
from pymongo import MongoClient
import pprint
import datetime
import time
import sys

def main():
    #for arg in sys.argv[1:]:
    #    print(arg)

    #sys.exit()

    consumer_key = 'your consumer key'
    consumer_secret = 'your consumer secret'
    access_token ='your access token'
    access_token_secret = 'your access token secret'
    try:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

		#mongo listens in port 2017 by default
        client = MongoClient('localhost:27017')
        db = client.TweetScraper

        counter_general = 0
        counter_new = 0

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
                            filterdate = "2016-" + str(mymonth) + "-" + str(myday) + " " + str(myhour) + ":" + str(mymin) + ":" + str(mysec)

                            try:
                                for res in db.tweet.find({"datetime": filterdate}):
                                    if res is None:
                                        break

                                    counter_general = counter_general + 1
                                    if "tw_coordinates" in res.keys():
                                        if counter_general % 1000 == 0 :
                                            print("counter general:" + str(counter_general) + " date: " + filterdate + " " + str(res))
                                        continue
                                    if "api_res" in res.keys():
                                        if counter_general % 1000 == 0 :
                                            print("counter general:" + str(counter_general) + " date: " + filterdate + " " + str(res))
                                        continue

                                    counter_new = counter_new + 1
                                    if counter_new % 100 == 0:
                                        print("new enrichment count:" + str(counter_new) + " date: " + filterdate + " " + str(res))

                                    t = api.get_status(res["ID"])
                                    tw_retweet_count= t.retweet_count
                                    tw_favorite_count=t.favorite_count
                                    tw_favorited=t.favorited
                                    tw_hashtags=""
                                    hashtags = list(t.entities.values())
                                    for hashtag_entity in hashtags:
                                        for hashtag_dict in hashtag_entity:
                                            tag = hashtag_dict.get('text')
                                            if(tag != None):
                                                tw_hashtags += tag
                                                tw_hashtags += ';'
                                    tw_hashtags = tw_hashtags[:-1]
                                    tw_source=t.source
                                    tw_geo=t.geo
                                    tw_coordinates=t.coordinates
                                    tw_loc_country = None
                                    tw_loc_fullname = None
                                    tw_loc_name = None
                                    tw_loc_type = None
                                    if(t.place != None):
                                        tw_loc_country = t.place.country
                                        tw_loc_fullname = t.place.full_name
                                        tw_loc_name = t.place.name
                                        tw_loc_type = t.place.place_type
                                    tw_lang=t.lang
                                    user_location=t.author.location
                                    user_profile_image_url=t.author.profile_image_url
                                    user_lang=t.author.lang
                                    user_description=t.author.description
                                    user_url=t.author.url
                                    user_friends_count=t.author.friends_count
                                    user_followers_count=t.author.followers_count
                                    user_name=t.author.name
                                    user_screen_name=t.author.screen_name
                                    user_listed_count=t.author.listed_count
                                    user_favourites_count=t.author.favourites_count
                                    user_statuses_count=t.author.statuses_count
                                    user_created_at=t.author.created_at.strftime('%Y-%m-%d %H:%M')
                                    user_verified=t.author.verified
                                    user_utc_offset=t.author.utc_offset
                                    user_timezone=t.author.time_zone
                                    user_geo_enabled=t.author.geo_enabled
                                    user_default_profile=t.author.default_profile
                                    db.tweet.update({"ID": res["ID"]}, {
                                        "$set": {"tw_retweet_count": tw_retweet_count, "tw_favorite_count": tw_favorite_count,
                                                 "tw_favorited": tw_favorited, "tw_hashtags": tw_hashtags, "tw_source": tw_source, "tw_geo": tw_geo,
                                                 "tw_coordinates": tw_coordinates, "tw_loc_country": tw_loc_country, "tw_loc_fullname": tw_loc_fullname, "tw_loc_name": tw_loc_name, "tw_loc_type": tw_loc_type, "tw_lang": tw_lang,
                                                 "user_location": user_location, "user_profile_image_url": user_profile_image_url,
                                                 "user_lang": user_lang, "user_description": user_description, "user_url": user_url,
                                                 "user_friends_count": user_friends_count, "user_followers_count": user_followers_count,
                                                 "user_name": user_name, "user_screen_name": user_screen_name,
                                                 "user_listed_count": user_listed_count, "user_favourites_count": user_favourites_count,
                                                 "user_statuses_count": user_statuses_count, "user_created_at": user_created_at,
                                                 "user_verified": user_verified, "user_utc_offset": user_utc_offset,
                                                 "user_timezone": user_timezone, "user_geo_enabled": user_geo_enabled,
                                                 "user_default_profile": user_default_profile}})

                            except TweepError as e:
                                if e.api_code == 63:
                                    print('Skipping tweet, account is suspended for ID:', res["ID"] , ',',  e)
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"api_res": '1'}})
                                elif e.api_code == 144:
                                    print('Skipping tweet, no status related to this account for ID:', res["ID"] , ',',  e)
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"api_res": '2'}})
                                else:
                                    print('Skipping tweet, not known error, for ID:', res["ID"], ',', e)
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"api_res": '3'}})
                            except Exception as exception:
                                print('Skipping tweet, not known error, for ID:', res["ID"], ',', e)
                                db.tweet.update({"ID": res["ID"]}, {"$set": {"api_res": '4'}})

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)


if __name__ == "__main__":
    main()