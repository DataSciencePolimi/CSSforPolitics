from geopy.exc import GeocoderTimedOut
from pymongo import MongoClient
from time import sleep
try:

    counter_general = 0
    counter_result_remain = 0
    counter_result_neutral = 0
    counter_result_leave = 0
    counter_result_mixed = 0
    counter_unknown = 0
    counter_not_english = 0
    counter_already_enriched = 0
    counter_already_tweepy_api_error = 0
    client = MongoClient('localhost:27017')
    db = client.TweetScraper

    remain_hashtag_list=["strongerin", "voteremain", "intogether", "labourinforbritain","moreincommon","greenerin","catsagainstbrexit","bremain","betteroffin","leadnotleave","remain","stay","ukineu","votein","voteyes","yes2eu","yestoeu","sayyes2europe"]
    leave_hashtag_list = ["independenceDay","leaveeuofficial","leaveeu","leave", "labourleave","votetoleave","voteleave","takebackcontrol","ivotedleave","beleave","betteroffout","britainout","nottip","takecontrol","voteno","voteout", "voteleaveeu"]
    other_neutral_hashtag_list=["euref", "eureferendum"]

    for month in range(17, 13):
        mymonth = str(month).rjust(2, '0')
        for day in range(1, 32):
            myday = str(day).rjust(2, '0')
            for hour in range(0, 25):
                myhour = str(hour).rjust(2, '0')
                for min in range(0, 60):
                    mymin = str(min).rjust(2, '0')
                    for sec in range(0, 60):
                        mysec = str(sec).rjust(2, '0')
                        filterdate = "2017-" + str(mymonth) + "-" + str(myday) + " " + str(myhour) + ":" + str(
                            mymin) + ":" + str(mysec)
                        # res = db.tweet.find_one({"ID":"815355320516153344"})

                        for res in db.tweet.find({"datetime": filterdate}):
                            try:

                                if res is None:
                                    break

                                counter_general = counter_general + 1
                                if counter_general % 10000 == 0:
                                    print("current date:" + filterdate)

                                if "p1" in res.keys():
                                    counter_already_enriched = counter_already_enriched + 1
                                    if counter_already_enriched % 100 == 0:
                                        print("already has a prediction result. counter_already_enriched:" + str(counter_already_enriched))
                                    continue;

                                elif "api_res" in res.keys():
                                    counter_already_tweepy_api_error = counter_already_tweepy_api_error + 1
                                    if counter_already_tweepy_api_error % 1000 == 0:
                                        print("already has tweepy error. counter_already_tweepy_api_error:" + str(counter_already_tweepy_api_error))
                                    continue;
                                elif "tw_hashtags" not in res.keys():
                                    print("major error")
                                elif res["tw_lang"] != "en":
                                    counter_not_english = counter_not_english + 1
                                    if counter_not_english % 1000 == 0:
                                        print("counter_not_english:" + str(counter_not_english))
                                    continue;

                                #it is an expected tweet
                                tw_hashtags =  res["tw_hashtags"].lower()
                                # check if it is neutral
                                if tw_hashtags == "brexit":
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"p1": '0'}})
                                    counter_result_neutral = counter_result_neutral + 1
                                    if counter_result_neutral % 10 == 0:
                                        print("counter_result_neutral: " + str(counter_result_neutral))
                                    continue;

                                #contains multiple hashtags

                                hashtags = tw_hashtags.split(";")
                                count_of_hashtags = len(hashtags)

                                has_other_neutral = False
                                has_leave_hashtag = False
                                has_remain_hashtag = False

                                for hashtag in hashtags:
                                    if hashtag in remain_hashtag_list:
                                        has_remain_hashtag = True
                                    elif hashtag in leave_hashtag_list:
                                        has_leave_hashtag = True
                                    elif hashtag in other_neutral_hashtag_list:
                                        has_other_neutral = True

                                # 0 neutral, 1 remain, 2 leave, 3 mixed

                                if has_remain_hashtag and not has_leave_hashtag:
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"p1": '1'}})
                                    counter_result_remain = counter_result_remain + 1
                                    if counter_result_remain % 10 == 0:
                                        print("counter_result_remain: " + str(counter_result_remain))
                                elif has_leave_hashtag and not has_remain_hashtag:
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"p1": '2'}})
                                    counter_result_leave = counter_result_leave + 1
                                    if counter_result_leave % 10 == 0:
                                        print("counter_result_leave: " + str(counter_result_leave))
                                elif has_other_neutral and not has_remain_hashtag and not has_leave_hashtag:
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"p1": '0'}})
                                    counter_result_neutral = counter_result_neutral + 1
                                    if counter_result_neutral % 10 == 0:
                                        print("counter_result_neutral: " + str(counter_result_neutral))
                                elif has_remain_hashtag and has_leave_hashtag:
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"p1": '3'}})
                                    counter_result_mixed = counter_result_mixed + 1
                                    if counter_result_mixed % 10 == 0:
                                        print("counter_result_mixed: " + str(counter_result_mixed))
                                else:
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"p1": '0'}})
                                    counter_result_neutral = counter_result_neutral + 1
                                    if counter_result_neutral % 10 == 0:
                                        print("counter_result_neutral: " + str(counter_result_neutral))

                            except Exception as exception:
                                #db.tweet.update({"ID": res["ID"]}, {"$set": {"loc_res": '6'}})
                                print('Oops!  An error occurred in loop.  Try again... res: ' + str(res), exception)

except Exception as exception:
    print('Oops!  An error occurred.  Try again...', exception)

