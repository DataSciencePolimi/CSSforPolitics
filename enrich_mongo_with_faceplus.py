from pymongo import MongoClient
import requests
from time import sleep

client = MongoClient('localhost:27017')
db = client.TweetScraper
faceplusplusurldetect = "https://api-us.faceplusplus.com/facepp/v3/detect?api_key=your_api_key&api_secret=your_api_secret"
faceplusplusurlanalyse = "https://api-us.faceplusplus.com/facepp/v3/face/analyze?api_key=your_api_key&api_secret=your_api_secret&return_attributes=gender,age,ethnicity"


def isTweetEligibleForFaceOperation(row):
    isEligible = False

    try:
        keys = row.keys()
        if "t_gender" in keys or "t_f_api" in keys or "api_res" in keys:
            return isEligible

        if "user_profile_image_url" not in row:
            return isEligible

        if row["user_profile_image_url"] is None:
            #this user has not a profile image.. we add this info to db to prevent repetitive controls
            db.tweet.update({"ID": row["ID"]}, {"$set": {"t_f_api": '2'}})
            return isEligible

        isEligible = True
        return isEligible
    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... line: ' + str(row), exception)


def addIfNotExistsTotalTweetCountOfUser(row):
    try:

        user_id = str(row["user_id"])
        if "t_cnt" not in row.keys():
            t_cnt = db.tweet.find({"user_id": user_id}).count()
            db.tweet.update({"ID": row["ID"]}, {"$set": {"t_cnt": t_cnt}})

    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... line: ' + str(row), exception)


def imageUrlToSend(row):
    img = ""
    try:
        img = row["user_profile_image_url"]
        if "default_profile" in img:
            print("this profile has default egg picture")
            db.tweet.update({"ID": row["ID"]}, {"$set": {"t_f_api": 0}})
        elif "_normal." in img:
            img = img.replace("_normal.", ".")
        else:
            print("unexpected photo url pattern: " + img + " for " + str(row))

    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... line: ' + str(row), exception)

    return img


def updateMongoWithImageDemographics(row):
    try:
        img = imageUrlToSend(row)
        if img is None or img == "":
            return

        newurldetect = faceplusplusurldetect + "&image_url=" + img;
        responsedetect = requests.post(newurldetect)
        responsejsondetect = responsedetect.json()
        if "faces" not in responsejsondetect:
            print("this profile's picture is not convenient for faceplusplus. img: " + str(img))
            db.tweet.update({"ID": row["ID"]}, {"$set": {"t_f_api": 3}})
            return

        if len(responsejsondetect["faces"]) == 1:
            # when the object type is dict, you should use that kind of form
            faceobjects = responsejsondetect["faces"]
            faceobject = faceobjects[0]
            facetoken = faceobject["face_token"]
            newurlanalyse = faceplusplusurlanalyse + "&face_tokens=" + facetoken;
            responseanalyse = requests.post(newurlanalyse)
            responsejsonanalyse = responseanalyse.json()
            if len(responsejsonanalyse["faces"]) == 1:
                facesanalyse = responsejsonanalyse["faces"]
                gender = facesanalyse[0]["attributes"]["gender"]["value"]
                gender = gender.lower()
                genderint = -1
                if gender == "female":
                    genderint = 1
                elif gender == "male":
                    genderint = 2

                age = facesanalyse[0]["attributes"]["age"]["value"]
                ethnicity = facesanalyse[0]["attributes"]["ethnicity"]["value"]
                db.tweet.update({"ID": row["ID"]}, {"$set": {"t_gender": genderint, "t_age": age, "t_eth": ethnicity}})

                # now also adding these demographic info to other tweets of that user..
                user_id = str(row["user_id"])
                for other_tweet in db.tweet.find({"user_id": user_id}):
                    try:
                        if not isTweetEligibleForFaceOperation(other_tweet):
                            continue
                        else:
                            other_tweet_id = other_tweet["ID"]
                            db.tweet.update({"ID": other_tweet_id}, {"$set": {"t_gender": genderint, "t_age": age, "t_eth": ethnicity}})


                    except Exception as exception:
                        print('Oops!  An error occurred in inner loop.  Try again... line: ' + str(row), exception)

        else:
            db.tweet.update({"ID": row["ID"]}, {"$set": {"t_f_api": 1}})
            return


    except Exception as exception:
        print('Oops!  An error occurred in loop.  Try again... line: ' + str(row), exception)


def main():
    try:

        counter_general = 0
        counter_new_user = 0
        for month in range(6, 13):
            mymonth = str(month).rjust(2, '0')
            for day in range(25, 32):
                myday = str(day).rjust(2, '0')
                for hour in range(0, 25):
                    myhour = str(hour).rjust(2, '0')
                    for min in range(0, 60):
                        mymin = str(min).rjust(2, '0')
                        for sec in range(0, 60):
                            mysec = str(sec).rjust(2, '0')
                            filterdate = "2016-" + str(mymonth) + "-" + str(myday) + " " + str(myhour) + ":" + str(
                                mymin) + ":" + str(mysec)
                            # res = db.tweet.find_one({"ID":"815355320516153344"})

                            for res in db.tweet.find({"datetime": filterdate}):
                                try:
                                    if res is None:
                                        break

                                    counter_general = counter_general + 1
                                    if counter_general % 100 == 0:
                                        print(filterdate + " counter_general:" + str(counter_general))

                                    if not isTweetEligibleForFaceOperation(res):
                                        continue

                                    counter_new_user = counter_new_user + 1
                                    if counter_new_user % 50 == 0:
                                        print(filterdate + " counter_new_user:" + str(counter_new_user))

                                    addIfNotExistsTotalTweetCountOfUser(res)

                                    updateMongoWithImageDemographics(res)

                                except Exception as exception:
                                    print('Oops!  An error occurred in loop.  Try again... line: ' + str(res), exception)

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)

if __name__ == "__main__":
    main()