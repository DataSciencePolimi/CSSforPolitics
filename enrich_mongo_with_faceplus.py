from pymongo import MongoClient
import requests
from time import sleep

try:

    counter_general = 0
    counter_already_enriched = 0
    counter_already_face_api_error = 0
    counter_face_operation_completed = 0
    counter_already_tweepy_api_error = 0

    client = MongoClient('localhost:27017')
    db = client.TweetScraper
    faceplusplusurldetect = "https://api-us.faceplusplus.com/facepp/v3/detect?api_key=PXEVys_bftBdHnmE84fyrf3h-xTNRkyw&api_secret=lhz5uJskIlko-j5YEETs3lULeFtUoY6a"
    faceplusplusurlanalyse = "https://api-us.faceplusplus.com/facepp/v3/face/analyze?api_key=PXEVys_bftBdHnmE84fyrf3h-xTNRkyw&api_secret=lhz5uJskIlko-j5YEETs3lULeFtUoY6a&return_attributes=gender,age,ethnicity"

    for month in range(1, 12):
        mymonth = str(month).rjust(2, '0')
        for day in range(1, 32):
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
                                if "t_gender" in res.keys():
                                    counter_already_enriched = counter_already_enriched + 1
                                    if counter_already_enriched % 100 == 0:
                                        print("already enriched with face plus. counter_already_enriched:" + str(counter_already_enriched))
                                    continue

                                elif "t_f_api" in res.keys():
                                    counter_already_face_api_error = counter_already_face_api_error + 1
                                    if counter_already_face_api_error % 100 == 0:
                                        print("already has face error. counter_already_face_api_error:" + str(counter_already_face_api_error))
                                    continue

                                elif "api_res" in res.keys():
                                    counter_already_tweepy_api_error = counter_already_tweepy_api_error + 1
                                    if counter_already_tweepy_api_error % 100 == 0:
                                        print("already has tweepy error. counter_already_tweepy_api_error:" + str(counter_already_tweepy_api_error) + " date:" + filterdate)
                                    continue


                                if "user_profile_image_url" not in res:
                                    print("unexpected situation:" + str(res))
                                    continue;

                                if res["user_profile_image_url"] is None:
                                    counter_null_column = counter_null_column + 1
                                    if counter_null_column % 10 == 0:
                                        print("does not have photo. counter_null_column:" + str(counter_null_column))
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"t_f_api": '2'}})
                                    continue;

                                user_id = str(res["user_id"])
                                for res_of_user in db.tweet.find({"user_id": user_id}):
                                    if "t_gender" in res_of_user.keys():
                                        print("we already found photo of this user in his another tweet")
                                        t_gender =  str(res["t_gender"])
                                        t_age = str(res["t_age"])
                                        t_eth = str(res["t_eth"])
                                        db.tweet.update({"ID": res["ID"]}, {"$set": {"t_gender": t_gender, "t_age": t_age, "t_eth": t_eth}})
                                        break;


                                img = res["user_profile_image_url"]
                                if "default_profile" in img:
                                    print("this profile has default egg picture")
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"t_f_api": 0}})
                                    continue;

                                elif "_normal." in img:
                                    img = img.replace("_normal.", ".")
                                else:
                                    print("unexpected photo url pattern: " + img + " for " + str(res))

                                # we are doing api calls here
                                newurldetect = faceplusplusurldetect + "&image_url=" + img;
                                responsedetect = requests.post(newurldetect)
                                responsejsondetect = responsedetect.json()
                                if "faces" not in responsejsondetect:
                                    print("this profile's picture is not convenient for faceplusplus. img: " + str(img))
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"t_f_api": 3}})
                                    continue;

                                if len(responsejsondetect["faces"])==1:
                                    #when the object type is dict, you should use that kind of form
                                    faceobjects = responsejsondetect["faces"]
                                    faceobject = faceobjects[0]
                                    facetoken = faceobject["face_token"]
                                    newurlanalyse = faceplusplusurlanalyse + "&face_tokens=" + facetoken;
                                    responseanalyse = requests.post(newurlanalyse)
                                    responsejsonanalyse = responseanalyse.json()
                                    if len(responsejsonanalyse["faces"])==1:
                                        facesanalyse = responsejsonanalyse["faces"]
                                        gender = facesanalyse[0]["attributes"]["gender"]["value"]
                                        gender = gender.lower()
                                        genderint = -1
                                        if gender=="female":
                                            genderint = 1
                                        elif gender=="male":
                                            genderint = 2

                                        age= facesanalyse[0]["attributes"]["age"]["value"]
                                        ethnicity = facesanalyse[0]["attributes"]["ethnicity"]["value"]
                                        db.tweet.update({"ID": res["ID"]}, {"$set": {"t_gender": genderint, "t_age": age, "t_eth": ethnicity}})

                                        counter_face_operation_completed = counter_face_operation_completed + 1
                                        if counter_face_operation_completed % 5 == 0:
                                            print("counter_face_operation_completed for:" + str(counter_face_operation_completed) + " records" + " last res: " + str(res) + " found attributes about its face: gender=" + str(genderint) + " age= " + str(age) + " ethnicity=" + str(ethnicity) )

                                        # sleep 1 second for preventing api quota errors
                                        sleep(1)
                                else:
                                    db.tweet.update({"ID": res["ID"]}, {"$set": {"t_f_api": 1}})
                                    continue;

                            except Exception as exception:
                                print('Oops!  An error occurred in loop.  Try again... line: ' + str(res), exception)

except Exception as exception:
    print('Oops!  An error occurred.  Try again...', exception)

#print("counter_general, counter_already_enriched,  counter_null_column,  counter_already_face_api_error,  counter_having_zero_face,  counter_having_one_face, counter_having_more_than_one_faces,  counter_error,  counter_already_tweepy_api_error: " + str(counter_general) + " , " + str(counter_already_enriched) + " , " +  str(counter_null_column) + " , " +  str(counter_already_face_api_error) + " , " +  str(counter_having_zero_face) + " , " +  str(counter_having_one_face) + " , " + str(counter_having_more_than_one_faces) + " , " +  str(counter_error) + " , " +  str(counter_already_tweepy_api_error))

# Geocoding an address

