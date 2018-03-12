import numpy as np
from nltk.corpus import stopwords
from nltk import download
from gensim.models import Word2Vec
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale


def remove_extra_chars_from_word(word):
    #this method is related with Word2Vec
    word = word.replace('?', '')
    word = word.replace('.', '')
    word = word.replace('!', '')
    word = word.replace('-', ' ')
    word = word.replace('(', '')
    word = word.replace(')', '')
    word = word.replace(':', '')
    word = word.replace('&', '')
    word = word.replace('√', '')
    word = word.replace('®', '')

    return word


def is_weblink(word):
    #this method is related with Word2Vec
    res = False
    if 'http' in word or 'www' in word:
        res = True
    return res


def get_stop_words():
    #this method is related with Word2Vec
    download('stopwords')  # stopwords dictionary, run once
    stop_words_it = stopwords.words('italian')
    stop_words_en = stopwords.words('english')
    stop_words_en.extend(stop_words_it)
    return stop_words_en


stop_words_voc = get_stop_words()


def is_stopword(word):
    #this method is related with Word2Vec
    res = False
    if stop_words_voc is None:
        exit(-1)
    if word in stop_words_voc:
        res = True
    return res


def get_trained_word2vec_model(dimension):
    #this method is related with Word2Vec

    if dimension == 50:
        model = Word2Vec.load('C:/Users/emre2/Desktop/Museums/latest_data/model_voc_02_03_vocab50')
    elif dimension == 100:
        model = Word2Vec.load('C:/Users/emre2/Desktop/Museums/latest_data/model_voc_01_03_vocab100')
    elif dimension == 300:
        model = Word2Vec.load('C:/Users/emre2/Desktop/Museums/latest_data/model_voc_01_03_vocab300')
    return model


def get_mean_vector_value_of_text(text, dimension):
    #this method is related with Word2Vec
    word2VecModel = get_trained_word2vec_model(dimension)

    splitted = text.split(" ")
    current_word2vec = []
    try:

        for word in splitted:
            word = remove_extra_chars_from_word(word)

            if is_weblink(word):
                continue
            elif is_stopword(word):
                continue
            else:
                if word in word2VecModel.wv.vocab:
                    vec_word = word2VecModel[word]
                    current_word2vec.append(vec_word)
                else:
                    print("not existing in model: " + word)

        if len(current_word2vec) == 0:
            if dimension == 50:
                zeros = [0] * 50
            elif dimension == 100:
                zeros = [0] * 100
            elif dimension == 300:
                zeros = [0] * 300
            current_word2vec.append(zeros)

        averaged_word2vec = list(np.array(current_word2vec).mean(axis=0))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)

    return averaged_word2vec


def Train_Word2Vec_CSV(dimensions):
    # with open('C:/Users/emre2/Desktop/Museums/step-1.csv', newline='', encoding='utf-8') as f:

    with open('C:/Users/emre2/Desktop/Museums/latest_data/vocab_build_all_after_fulltext_0203.csv', newline='', encoding='utf-8') as f:
        try:
            reader = csv.reader(f)
            whole_tweet_contents = []
            counter_vocabulary_word = 0
            whole_words = []
            unique_words = []
            counter = 0
            for row in reader:
                try:
                    counter += 1

                    tweet_content = row[2]
                    if tweet_content == 'text':
                        # this is for skipping header row
                        continue
                    tweet_content_words = tweet_content.split(" ")
                    tweet_content_list = []

                    for word in tweet_content_words:
                        word = remove_extra_chars_from_word(word)
                        if is_weblink(word):
                            continue
                        elif is_stopword(word):
                            continue
                        else:
                            whole_words.append(word)
                            if word not in unique_words:
                                unique_words.append(word)
                            counter_vocabulary_word += 1
                            tweet_content_list.append(word)

                    whole_tweet_contents.append(tweet_content_list)

                except Exception as exception:
                    print('Oops!  An error occurred.  Try again...', exception)

            print("not unique all words count" + str(len(whole_words)))
            print("unique all words count" + str(len(unique_words)))

            if dimensions == 50:
                model = Word2Vec(whole_tweet_contents, size=50, window=5, min_count=1, workers=4)
            elif dimensions == 100:
                model = Word2Vec(whole_tweet_contents, size=100, window=5, min_count=1, workers=4)
            elif dimensions == 300:
                model = Word2Vec(whole_tweet_contents, size=300, window=5, min_count=1, workers=4)

            model.save('C:/Users/emre2/Desktop/Museums/latest_data/model_voc_02_03_vocab50')

        except Exception as exception:
            print('Oops!  An error occurred.  Try again...', exception)


def tp_ratio(y_true, y_pred, pct=0.1):
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]
    n = int(round(len(y_true) * pct))
    t = np.argsort(y_pred)
    idx = np.argsort(y_pred)[-n:]
    return y_true[idx].sum() / float(n)


def Build_Data_Set_Feature_Word2Vec(begin_index, end_index, dimension,filename, isYes1):
    #This is based on single feature, which contains the average Vector value for each tweet.
    try:
        data_df = pd.DataFrame.from_csv(filename)

        data_df = data_df[begin_index:end_index]
        textvalues = np.array(data_df["text"])
        vect_means=[]

        for textvalue in textvalues:
            vect_mean = get_mean_vector_value_of_text(textvalue, dimension)
            vect_means.append(vect_mean)

        X = vect_means
        if isYes1:
            y = (np.array(data_df["check"]
             .replace("N",0)
             .replace("Y",1)))
        else:
            y = (np.array(data_df["check"]
                          .replace("N", 1)
                          .replace("Y", 0)))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)
    return X,y


def evaluate_performance_predictor(test_labels, predicted_labels):
	# test precision
	precision = precision_score(test_labels, predicted_labels, average=None)
	print("Precision score: " + str(precision[1]))
	print('########################################################################')
	# test recall
	recall = recall_score(test_labels, predicted_labels, average=None)
	print("Recall score: " + str(recall[1]))
	print('########################################################################')
	# test F1 score
	f_measure = f1_score(test_labels, predicted_labels, average=None)
	print("F1 score score: " + str(f_measure[1]))



def evaluate_probability_based_model(clf_prob, X_train, y_train, X_test, y_test):
    clf_prob.fit(X_train, y_train)
    print("Natural TP rate=" + str(sum(y_test) / len(y_test)))
    y_pred_prob = clf_prob.predict_proba(X_test)[:, 1]
    a = clf_prob.predict_proba(X_test)
    print('Made predictions for test')
    print('ROC AUC:', roc_auc_score(y_test, y_pred_prob))
    print('True positive ratio at top 10%%: %0.2f%%' % (tp_ratio(y_test, y_pred_prob, pct=0.1) * 100))
    print('True positive ratio at top 20%%: %0.2f%%' % (tp_ratio(y_test, y_pred_prob, pct=0.2) * 100))
    print('True positive ratio at top 50%%: %0.2f%%' % (tp_ratio(y_test, y_pred_prob, pct=0.5) * 100))
    print('True positive ratio at top 75%%: %0.2f%%' % (tp_ratio(y_test, y_pred_prob, pct=0.75) * 100))


def convert_text_to_word2vec(data, dimension):
    vect_means = []
    for textvalue in data:
        vect_mean = get_mean_vector_value_of_text(textvalue, dimension)
        vect_means.append(vect_mean)

    np_vect_means = np.asarray(vect_means)
    return np_vect_means


def create_features_train_and_test(begin_index_train, end_index_train, begin_index_test, end_index_test, dimension, filename, isyes1):
    try:
        data_df = pd.DataFrame.from_csv(filename)

        data_df_train = data_df[begin_index_train:end_index_train]
        data_df_test = data_df[begin_index_test:end_index_test]

        print("distribution of train output classes: " + str(data_df_train["check"].value_counts()/data_df_train["check"].count()))
        print("distribution of test output classes: " + str(data_df_test["check"].value_counts()/data_df_test["check"].count()))

        npvectmeans_train= convert_text_to_word2vec(np.array(data_df_train["text"]), dimension)
        npvectmeans_test= convert_text_to_word2vec(np.array(data_df_test["text"]), dimension)

        #X_train = np.concatenate((npvectmeans_train, data_df["tw_retweet_count"]), axis=1)
        #X_test = np.concatenate((npvectmeans_test, data_df["tw_retweet_count"]), axis=1)

        # Define which columns should be encoded vs scaled
        columns_to_scale = ['tw_retweet_count','tw_favorite_count','user_friends_count','user_followers_count','user_listed_count','user_favourites_count','user_statuses_count']
        columns_categorical = ['tw_source','tw_lang','user_screen_name','user_verified','user_geo_enabled','user_default_profile']
        X_train = data_df[begin_index_train:end_index_train]
        X_test = data_df[begin_index_test:end_index_test]

        #currently sentiment is not included in feature list
        #X_train['sentiment'] = X_train['sentiment'].fillna('missing')
        #X_test['sentiment'] = X_test['sentiment'].fillna('missing')

        # Instantiate encoder/scaler
        scaler = StandardScaler()
        enc = OneHotEncoder(sparse=False)
        min_max = MinMaxScaler()
        le = LabelEncoder()

        for col in X_test.columns.values:
            # Encoding only categorical variables
            try:
                if col not in columns_categorical:
                    continue
                print("type:", str(X_test[col].dtypes))
                if X_test[col].dtypes == 'object':
                    # Using whole data to form an exhaustive list of levels
                    data = X_train[col].append(X_test[col])
                    try:
                        le.fit(data.values)
                    except Exception as exception:
                        print('Oops!  An error occurred.  Try again...', exception)
                        continue
                    X_train[col] = le.transform(X_train[col])
                    X_test[col] = le.transform(X_test[col])
            except Exception as exception:
                print('Oops!  An error occurred.  Try again...', exception)
                continue

        #scaled_columns = min_max.fit_transform(data_df[columns_to_scale])
        #scaled_columns = scaler.fit_transform(data_df[columns_to_scale])
        #encoded_columns = ohe.fit_transform(data_df[columns_to_encode])

        #scaled_columns_train = scale(data_df_train[columns_to_scale])
        #scaled_columns_test = scale(data_df_test[columns_to_scale])

        #one hot encoding ready?
        #X_train = np.concatenate((npvectmeans_train, scaled_columns_train), axis=1)
        #X_test = np.concatenate((npvectmeans_test, scaled_columns_test), axis=1)

        X_train_1 = X_train
        X_test_1 = X_test

        columns_categorical_finale = []
        for col in columns_categorical:
            # creating an exhaustive list of all possible categorical values
            data = X_train[[col]].append(X_test[[col]])
            try:
                enc.fit(data)
            except Exception as exception:
                print('Oops!  An error occurred.  Try again...', exception)
                continue

            # Fitting One Hot Encoding on train data
            temp = enc.transform(X_train[[col]])
            # Changing the encoded features into a data frame with new column names
            temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col]
                                .value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the X_train data frame
            temp = temp.set_index(X_train.index.values)
            # adding the new One Hot Encoded varibales to the train data frame
            X_train_1 = pd.concat([X_train_1, temp], axis=1)
            # fitting One Hot Encoding on test data
            temp = enc.transform(X_test[[col]])
            # changing it into data frame and adding column names
            temp = pd.DataFrame(temp, columns=[(col + "_" + str(i)) for i in data[col]
                                .value_counts().index])
            # Setting the index for proper concatenation
            print(list(temp.columns.values))
            for new_col in temp.columns.values:
                columns_categorical_finale.append(new_col)
            temp = temp.set_index(X_test.index.values)
            # adding the new One Hot Encoded varibales to test data frame
            X_test_1 = pd.concat([X_test_1, temp], axis=1)

        #todo scale
        #X_train_scale = scale(X_train_1)
        #X_test_scale = scale(X_test_1)

        X_train = X_train_1
        X_test = X_test_1

        #scaled_columns_train = scale(X_train[columns_to_scale])
        #scaled_columns_test = scale(X_test[columns_to_scale])

        scaled_columns_train = min_max.fit_transform(X_train[columns_to_scale])
        scaled_columns_test = min_max.fit_transform(X_test[columns_to_scale])

        one_hot_res_train = X_train_1[columns_categorical_finale]
        one_hot_res_test = X_test_1[columns_categorical_finale]

        X_train = np.concatenate((npvectmeans_train, scaled_columns_train, one_hot_res_train), axis=1)
        X_test = np.concatenate((npvectmeans_test, scaled_columns_test, one_hot_res_test), axis=1)

        if isyes1:
            y_train = (np.array(data_df_train["check"].replace("N",0).replace("Y",1)))
        else:
            y_train = (np.array(data_df_train["check"].replace("N", 1).replace("Y", 0)))


        if isyes1:
            y_test = (np.array(data_df_test["check"].replace("Y",1).replace("N",0)))
        else:
            y_test = (np.array(data_df_test["check"].replace("N", 1).replace("Y", 0)))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)
    return X_train,y_train, X_test, y_test


def create_features(begin_index, end_index, dimension,filename, isyes1):
    try:
        data_df = pd.DataFrame.from_csv(filename)

        data_df = data_df[begin_index:end_index]
        textvalues = np.array(data_df["text"])

        print("distribution of output classes: " + str(data_df["check"].value_counts()/data_df["check"].count()))

        #data_df[data_df.dtypes[(data_df.dtypes == "float64") | (data_df.dtypes == "int64")]
         #   .index.values].hist(figsize=[11, 11])

        vect_means=[]

        for textvalue in textvalues:
            vect_mean = get_mean_vector_value_of_text(textvalue, dimension)
            vect_means.append(vect_mean)

        npvectmeans = np.asarray(vect_means)

        # Define which columns should be encoded vs scaled
        columns_to_scale = ['tw_retweet_count','tw_favorite_count','user_friends_count','user_followers_count','user_listed_count','user_favourites_count','user_statuses_count']
        columns_to_encode = ['tw_source','tw_lang','user_screen_name','user_verified','user_geo_enabled','user_default_profile','sentiment']

        # Instantiate encoder/scaler
        scaler = StandardScaler()
        ohe = OneHotEncoder(sparse=False)
        min_max = MinMaxScaler()
        le = LabelEncoder()

        #scaled_columns = min_max.fit_transform(data_df[columns_to_scale])

        #scaled_columns = scaler.fit_transform(data_df[columns_to_scale])

        #encoded_columns = ohe.fit_transform(data_df[columns_to_encode])

        scaled_columns = scale(data_df[columns_to_scale])

        #X = np.concatenate((npvectmeans, tw_retweet_count[:, None], tw_favorite_count[:, None], user_friends_count[:, None], user_followers_count[:, None], user_listed_count[:, None], user_favourites_count[:, None], user_statuses_count[:, None]), axis=1)
        X = np.concatenate((npvectmeans, scaled_columns), axis=1)
        if isyes1:
            y = (np.array(data_df["check"]
             .replace("N",0)
             .replace("Y",1)))
        else:
            y = (np.array(data_df["check"]
                          .replace("N", 1)
                          .replace("Y", 0)))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)
    return X,y


def train_1(begin_index, end_index, dimension,filename, isYes1):
    #This is based on single feature, which contains the average Vector value for each tweet.
    try:
        data_df = pd.DataFrame.from_csv(filename)

        X_train, y_train = get_train_test(data_df, begin_index, end_index, dimension)
        X_test, y_true = get_train_test(data_df, 21, 25, dimension)

        classifier = svm.LinearSVC()

        model = classifier.fit(X_train, y_train)
        print("ok")
        y_predicted = model.predict(X_test)

        print(sklearn.metrics.accuracy_score(y_true, y_predicted))
        print("good so far")

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)


def get_yes_no_count(set):
    actual_yes = 0
    actual_no = 0
    for res in set:
        if res == 1:
            actual_yes += 1
        elif res == 0:
            actual_no +=1
    return actual_yes, actual_no


def get_splitted_data(feature_type, train_start_index, train_end_index, test_start_index, test_end_index, vocab_dimension, filename, is_yes_1):
    ############################################################################################
    # feature type 1: one single feature, mean vector value of Word2Vec#########################
    # feature type 2: two features, like count and share count##################################
    # feature type 3: two features, like count and sentiment####################################
    # feature type 4: latest code: combine Word2Vec with other features#########################
    # feature type 5: many features including mean vector. scaling and one hot encoding enabled#
    ############################################################################################
    if feature_type == 1:
        print("feature type 1: one single feature, mean vector value of Word2Vec")
        if is_yes_1:
            X_train, y_train = Build_Data_Set_Feature_Word2Vec(train_start_index, train_end_index, vocab_dimension, filename, True)
            X_test, y_test = Build_Data_Set_Feature_Word2Vec(test_start_index, test_end_index, vocab_dimension, filename, True)
        else:
            X_train, y_train = Build_Data_Set_Feature_Word2Vec(train_start_index, train_end_index, vocab_dimension,
                                                               filename, False)
            X_test, y_test = Build_Data_Set_Feature_Word2Vec(test_start_index, test_end_index, vocab_dimension,
                                                             filename, False)
    elif feature_type == 2:
        print("feature type 2: two features, like count and share count")
        X_train, y_train = Build_Data_Set_Features_Likes_Share(train_start_index, train_end_index)
        X_test, y_test = Build_Data_Set_Features_Likes_Share(test_start_index, test_end_index)
    elif feature_type == 3:
        print("feature type 3: two features, like count and sentiment")
        X_train, y_train = Build_Data_Set_Features_Likes_Sentiment(train_start_index, train_end_index)
        X_test, y_test = Build_Data_Set_Features_Likes_Sentiment(test_start_index, test_end_index)
    elif feature_type == 4:
        print("feature type 4: many features including mean vector value of Word2Vec")
        if is_yes_1:
            X_train, y_train= create_features(train_start_index, train_end_index, vocab_dimension, filename, True)
            X_test, y_test = create_features(test_start_index, test_end_index, vocab_dimension, filename, True)
        else:
            X_train, y_train = create_features(train_start_index, train_end_index, vocab_dimension, filename, False)
            X_test, y_test = create_features(test_start_index, test_end_index, vocab_dimension, filename, False)
    elif feature_type == 5:
        print("feature type 5: many features including mean vector. one hot encoding enabled")
        if is_yes_1:
            X_train, y_train,  X_test, y_test = create_features_train_and_test(train_start_index, train_end_index, test_start_index, test_end_index, vocab_dimension, filename, True)
        else:
            X_train, y_train,  X_test, y_test = create_features_train_and_test(train_start_index, train_end_index, test_start_index, test_end_index, vocab_dimension, filename, False)

    else:
        return
    return X_train, y_train, X_test, y_test


def main():
    try:

        ###############################################################
        # build Word2Vec vocabulary from scratch or load trained model#
        # possible vector dimensions : 50, 100 and 300#################
        ###############################################################

        vocab_dimension = 50
        train_vocab_enabled = False
        if train_vocab_enabled:
            print("started building Word2Vec vocabulary from scratch")
            Train_Word2Vec_CSV(vocab_dimension)
            print("completed building Word2Vec vocabulary from scratch")
            return

        ##############
        ##input file##
        ##############
        file_id = 4
        if file_id == 0:
            filename = "C:/Users/emre2/Desktop/Museums/latest_data/test.csv"
        elif file_id == 1:
            filename = "C:/Users/emre2/Desktop/Museums/latest_data/scala_sample_fulltext_utf_header.csv"
        elif file_id == 2:
            filename = "C:/Users/emre2/Desktop/Museums/latest_data/pompei_sample_fulltext_utf_header.csv"
        elif file_id == 3:
            filename = "C:/Users/emre2/Desktop/Museums/latest_data/colosseo_sample_fulltext_utf_header.csv"
        elif file_id == 4:
            filename = "C:/Users/emre2/Desktop/Museums/latest_data/pompei_colosseo_scala_sample_fulltext_utf_header_rnd.csv"
        elif file_id == 5:
            filename = "C:/Users/emre2/Desktop/Museums/latest_data/merged_with_sentiment.csv"

        num_lines = sum(1 for line in open(filename,newline='', encoding='utf-8'))

        ####################################
        # model type 1: SVM Linear Kernel###
        # model type 2: SVM RBF Kernel######
        # model type 3: Random Forest#######
        # model type 4: Logistic Regression#
        ####################################

        model_type = 1
        C = 1  # SVM regularization parameter
        if model_type == 1:
            print("model type: SVM Linear Kernel")
            clf = svm.SVC(kernel="linear", C=C)
        elif model_type == 2:
            print("model type: SVM RBF Kernel")
            clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        elif model_type == 3:
            print("model type: Random Forest")
            clf = RandomForestClassifier(n_estimators=100)
        elif model_type == 4:
            print("model type: Logistic Regression")
            clf = LogisticRegression()
        elif model_type == 5:
            print("model type: KNeighborsClassifier")
            clf = KNeighborsClassifier(n_neighbors=5)
            return

        #####################
        ##train - test split#
        #####################
        train_start_index = 0
        train_end_index = int(num_lines * 0.70)
        test_start_index = train_end_index
        test_end_index = num_lines

        #for testttt purpose, I create a sample dataset.
        is_test = False

        if is_test:
            train_start_index = 0
            train_end_index = 17
            test_start_index = 17
            test_end_index = 26

        ########################################################################
        # feature type 1: one single feature, mean vector value of Word2Vec#####
        # feature type 2: two features, like count and share count##############
        # feature type 3: two features, like count and sentiment################
        # feature type 4: many features icnluding mean vector value of Word2Vec#
        ########################################################################
        feature_type = 5
        X_train, y_train, X_test, y_test = get_splitted_data(feature_type, train_start_index, train_end_index, test_start_index, test_end_index, vocab_dimension,filename, True)

        ##################
        # build ML model##
        ##################
        print("started model fitting for train and test")
        try:
            clf.fit(X_train, y_train)
        except Exception as exception:
            print('Oops!  An error occurred.  Try again...', exception)

        print("train size: " + str(len(X_train)))
        print("test size: " + str(len(X_test)))

        #######################
        # evaluation of model##
        #######################
        y_pred = clf.predict(X_test)
        print("predicted test results: " + str(y_pred))
        print("expected test results  :" + str(y_test))
        print("score:" + str(clf.score(X_test, y_test)))
        print(precision_recall_fscore_support(y_test, y_pred,average=None))

        evaluate_performance_predictor(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        actual_yes, actual_no = get_yes_no_count(y_test)
        pred_yes_cnt, pred_no_cnt = get_yes_no_count(y_pred)

        print("output of confusion matrix: tn:" + str(tn) + " fp:" + str(fp) + " fn: " + str(fn) + " tp:" + str(tp))
        print("manual accuracy: " + str((tp+tn)/len(y_pred)))
        print("manual misclassification rate: " + str((fp + fn) / len(y_pred)))
        recall = tp / actual_yes
        print("manual tp rate (sensitivity-recall): " + str(recall))
        print("manual fp rate: " + str((fp) / actual_no))
        print("manual specificity: " + str((tn) / actual_no))
        precision = tp / pred_yes_cnt
        print("manual precision: " + str(precision))
        print("manual prevalence: " + str(actual_yes / len(y_pred)))
        print("manual f1 score: " + str(2 * recall * precision / (recall + precision)))

        ##################
        #cross validation#
        ##################
        print("started cross validation calculation")
        X_all, y_all = Build_Data_Set_Feature_Word2Vec(feature_type, num_lines, vocab_dimension, filename, True)
        clf.fit(X_all, y_all)
        print("cross val score: " + str(cross_val_score(clf, X_all, y_all, cv=10).mean()))

        #################################
        #probability enabled prediction##
        #################################
        if model_type == 1:
            print("probability enabled prediction started. model type: SVM Linear Kernel")
            clf_prob = svm.SVC(kernel="linear", C=C, probability=True)
        elif model_type == 2:
            print("probability enabled prediction started. model type: SVM RBF Kernel")
            clf_prob = svm.SVC(kernel='rbf', gamma=0.7, C=C, probability=True)
        else:
            print("the probability based percentile scores are valid only for SVM models")
            return

        evaluate_probability_based_model(clf_prob,X_train, y_train, X_test, y_test)

        is_yes_1 = False
        print("Y=0, N=1")
        X_train, y_train, X_test, y_test = get_splitted_data(feature_type, train_start_index, train_end_index,test_start_index, test_end_index, vocab_dimension, filename, is_yes_1)
        evaluate_probability_based_model(clf_prob, X_train, y_train, X_test, y_test)

        print("completed")
    except Exception as exception:
            print('Oops!  An error occurred.  Try again...', exception)


if __name__ == "__main__":
    main()