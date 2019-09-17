import pandas as pd
from util_tools import utils

try:
    df = utils.read_file("/Users/emrecalisir/git/cortico/discovery/tws_5sep_2018_12.out", "~", ['usernameTweet','ID','text','tw_lang','url','nbr_retweet',
                                                                                                'nbr_favorite','nbr_reply','datetime','is_reply','is_retweet',
                                                                                                'user_id','tw_coordinates','tw_favorite_count','tw_favorited','tw_geo','tw_hashtags','tw_loc_country','tw_loc_fullname','tw_loc_name','tw_loc_type','tw_retweet_count','tw_source','user_created_at','user_default_profile','user_description','user_favourites_count','user_followers_count','user_friends_count','user_geo_enabled','user_lang','user_listed_count','user_location','user_name','user_profile_image_url','user_screen_name','user_statuses_count','user_timezone','user_url','user_utc_offset','user_verified','t_age','t_eth','t_gender','tw_full'],
                               dtype={'usernameTweet':object,'ID':object,'text':'U','tw_lang':object,'url':object,'nbr_retweet':object,
                                      'nbr_favorite':object,'nbr_reply':object,'datetime':object,'is_reply':object,'is_retweet':object,
                                      'user_id':object,'tw_coordinates':object,'tw_favorite_count':object,'tw_favorited':object,'tw_geo':object,
                                      'tw_hashtags':object,'tw_loc_country':object,'tw_loc_fullname':object,'tw_loc_name':object,'tw_loc_type':object,
                                      'tw_retweet_count':object,'tw_source':object,'user_created_at':object,'user_default_profile':object,
                                      'user_description':object,'user_favourites_count':object,'user_followers_count':object,'user_friends_count':object,
                                      'user_geo_enabled':object,'user_lang':object,'user_listed_count':object,'user_location':object,'user_name':object,
                                      'user_profile_image_url':object,'user_screen_name':object,'user_statuses_count':object,'user_timezone':object,
                                      'user_url':object,'user_utc_offset':object,'user_verified':object,'t_age':object,'t_eth':object,'t_gender':object,'tw_full':'U'
                               })
    print(df.shape)
    print(df.head())
except Exception as ex:
    print(ex)