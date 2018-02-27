import numpy as np
from nltk.corpus import stopwords
from nltk import download
from gensim.models import Word2Vec
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import csv


def make_meshgrid(x, y, h=.02):
    #this method is related with plotting
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    #this method is related with plotting
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plott(clf, X, y):
    #this method is related with plotting. It works when there are 2 features.
    fig, sub = plt.subplots(1, 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(plt.axes(), clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    plt.axes().scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    plt.axes().set_xlim(xx.min(), xx.max())
    plt.axes().set_ylim(yy.min(), yy.max())
    plt.axes().set_xlabel('Likes')
    plt.axes().set_ylabel('Shares')
    plt.axes().set_xticks(())
    plt.axes().set_yticks(())
    plt.axes().set_title('rbf linear')

    plt.show()


def remove_extra_chars_from_word(word):
    #this method is related with Word2Vec
    word = word.replace('?', '')
    word = word.replace('.', '')
    word = word.replace('!', '')
    word = word.replace('-', ' ')
    word = word.replace('(', '')
    word = word.replace(')', '')
    word = word.replace(':', '')
    word = word.replace('#', '')
    word = word.replace('&', '')
    word = word.replace('√', '')
    word = word.replace('®', '')

    return word


def is_weblink_or_mention(word):
    #this method is related with Word2Vec
    res = False
    if 'http' in word or 'www' in word or '@' in word:
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


def get_trained_model():
    #this method is related with Word2Vec
    model = Word2Vec.load('C:/Users/emre2/Desktop/Museums/step-2-model-2')
    return model

model = get_trained_model()

def get_mean_vector_value_of_text(text):
    #this method is related with Word2Vec

    if model is None:
        word2VecModel = get_trained_model()

    splitted = text.split(" ")
    current_word2vec = []
    try:

        for word in splitted:
            word = remove_extra_chars_from_word(word)

            if is_weblink_or_mention(word):
                continue
            elif is_stopword(word):
                continue
            else:
                if word in model.wv.vocab:
                    vec_word = model[word]
                    current_word2vec.append(vec_word)
                else:
                    print("not existing in model: " + word)
        if len(current_word2vec) == 0:
            zeros = [0] * 100
            current_word2vec.append(zeros)

        averaged_word2vec = list(np.array(current_word2vec).mean(axis=0))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)

    return averaged_word2vec


def Train_Word2Vec():
    with open('C:/Users/emre2/Desktop/Museums/step-1.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        whole_tweet_contents = []
        counter_vocabulary_word = 0
        for row in reader:
            try:
                tweet_content = row[6]
                    if tweet_content == 'text':
                        # this is for skipping header row
                        continue
                    tweet_content_words = tweet_content.split(" ")
                    tweet_content_list = []

                    for word in tweet_content_words:
                        word = remove_extra_chars_from_word(word)
                        if is_weblink_or_mention(word):
                            continue
                        elif is_stopword(word):
                            continue
                        else:
                            counter_vocabulary_word += 1
                            tweet_content_list.append(word)

                    whole_tweet_contents.append(tweet_content_list)

                except Exception as exception:
                    print('Oops!  An error occurred.  Try again...', exception)

        model = Word2Vec(min_count=1)
        model.build_vocab(whole_tweet_contents)
        model.train(whole_tweet_contents, total_examples=model.corpus_count, epochs=model.iter)

        model.save('C:/Users/emre2/Desktop/step-2-model-2')

        print("counter_vocabulary_word:" + str(whole_tweet_contents))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)


def Build_Data_Set_Feature_Word2Vec(begin_index, end_index):
    #This is based on single feature, which contains the average Vector value for each tweet.
    try:
        data_df = pd.DataFrame.from_csv("C:/Users/emre2/Desktop/Museums/step-1.csv")

        data_df = data_df[begin_index:end_index]
        textvalues = np.array(data_df["text"])
        vect_means=[]

        for textvalue in textvalues:
            vect_mean = get_mean_vector_value_of_text(textvalue)
            vect_means.append(vect_mean)

        X = vect_means
        y = (np.array(data_df["check"]
             .replace("N",0)
             .replace("Y",1)))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)
    return X,y


def Build_Data_Set_Features_Likes_Share(begin_index, end_index):
    #This is based on 2 features, that are the count of likes and count of shares...
    try:
        data_df = pd.DataFrame.from_csv("C:/Users/emre2/Desktop/Museums/step-1.csv")

        data_df = data_df[begin_index:end_index]
        likes = np.array(data_df["likes"])
        likes = np.nan_to_num(likes)
        share = np.array(data_df["share"])
        share = np.nan_to_num(share)
        likes_share = np.array([likes, share])
        X = likes_share.T

        y = (np.array(data_df["check"]
             .replace("N",0)
             .replace("Y",1)))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)
    return X,y


def Build_Data_Set_Features_Likes_Sentiment(begin_index, end_index):
    #This is based on 2 features, that are the count of likes and count of shares...
    try:
        data_df = pd.DataFrame.from_csv("C:/Users/emre2/Desktop/Museums/step-1.csv")

        data_df = data_df[begin_index:end_index]
        likes = np.array(data_df["likes"])
        likes = np.nan_to_num(likes)

        #print(data_df["sentiment"].value_counts())
        #data_df["sentiment"] = data_df["sentiment"].astype('category')
        #print(data_df.dtypes)
        sentiment = (np.array(data_df["sentiment"].replace("negative", 0).replace("neutral", 1).replace("positive", 2)))
        sentiment = np.nan_to_num(sentiment)

        likes_sentiment = np.array([likes, sentiment])
        X = likes_sentiment.T

        y = (np.array(data_df["check"]
             .replace("N",0)
             .replace("Y",1)))

    except Exception as exception:
        print('Oops!  An error occurred.  Try again...', exception)
    return X,y


def main():
    try:
        print("started")

        is_plot_enabled = False
        train_start_index = 0
        train_end_index = 1200
        test_start_index = 1200
        test_end_index = 1343

        #feature type 1: one single feature, mean vector value of Word2Vec
        #feature type 2: two features, like count and share count
        #feature type 3: two features, like count and sentiment
        feature_type = 2

        #kernel type 1: linear
        #kernel type 2: rbf
        kernel_type = 1

        if feature_type == 1:
            X_train, y_train = Build_Data_Set_Feature_Word2Vec(train_start_index, train_end_index)
            X_test, y_test = Build_Data_Set_Feature_Word2Vec(test_start_index, test_end_index)
        elif feature_type == 2:
            X_train, y_train = Build_Data_Set_Features_Likes_Share(train_start_index, train_end_index)
            X_test, y_test = Build_Data_Set_Features_Likes_Share(test_start_index, test_end_index)
        elif feature_type == 3:
            X_train, y_train = Build_Data_Set_Features_Likes_Sentiment(train_start_index, train_end_index)
            X_test, y_test = Build_Data_Set_Features_Likes_Sentiment(test_start_index, test_end_index)
        else:
            return


        # data since we want to plot the support vectors
        C = 1  # SVM regularization parameter
        if kernel_type == 1:
            clf = svm.SVC(kernel="linear", C=C)
        elif kernel_type == 2:
            clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
        else:
            return

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("predicted test results: " + str(y_pred))
        print("expected test results: " + str(y_test))

        print("score:" + str(clf.score(X_test, y_test)))

        print(precision_recall_fscore_support(y_test, y_pred))

        if is_plot_enabled:
            plott(clf, X_train, y_train)

        print("completed")
    except Exception as exception:
            print('Oops!  An error occurred.  Try again...', exception)


if __name__ == "__main__":
    main()