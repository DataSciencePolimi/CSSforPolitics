import json
import io
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
import tweepy
import ijson
from nltk.tokenize import TweetTokenizer
import re, string, unicodedata
import nltk
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd
from scipy.stats.stats import pearsonr
import numpy as np
import sys
import utils as utils
import logging as logger
import traceback


def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


def get_twitter_api():
    consumer_key = 'gLYplQnuS4ru2ohCFPML8KG1u'
    consumer_secret = 'PY3vqvEgAblhbNwWNen5u3CsJIUGV4zV1EvKPLyhw4XqnpDC3z'
    access_token = '1559646294-rKshkZBgfIAFVmReDPEL3Qdp1ImOovpGQOg5yDI'
    access_token_secret = 'PlfuYnIcIMvhoyApEWqEy8DTTL972Cx1so9HIMGvUGrok'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    return api



def read_tweets_in_json_format(filename, is_for_test):
    tweets = []
    if is_for_test:
        max_rows_count = 10
    counter = 0
    for line in open(filename, 'r', encoding='cp866'):
        if is_for_test:
            if counter == max_rows_count:
                break
        tweets.append(json.loads(line))
        counter += 1

    return np.array(tweets)


def remove_extra_chars_from_word(word):
    # this method is related with Word2Vec
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
    word = word.replace(',', '')
    word = word.replace('#', '')
    return word


def get_trained_word2vec_model(filename):
    # this method is related with Word2Vec
    try:
        # global word2vec_model
        # if word2vec_model is not None:
        #    return word2vec_model

        new_model = Word2Vec.load(filename)
        word2vec_model = new_model

    except Exception as ex:
        logger.error(traceback.format_exc())


    return word2vec_model


def is_weblink(word):
    # this method is related with Word2Vec
    res = False
    if 'http' in word or 'www' in word:
        res = True
    return res


def get_stop_words():
    # this method is related with Word2Vec
    # download('stopwords')  # stopwords dictionary, run once
    stop_words_it = stopwords.words('italian')
    stop_words_en = stopwords.words('english')
    stop_words_en.extend(stop_words_it)
    return stop_words_en


stop_words_voc = get_stop_words()


def is_less_than_character_count(word):
    max_limit = 3
    res = False
    list_characters = list(word)
    if len(list_characters) < max_limit:
        res = True
    return res


def is_stopword(word):
    # this method is related with Word2Vec
    res = False
    # print(str(stop_words_voc))
    if stop_words_voc is None:
        exit(-1)
    if word in stop_words_voc:
        res = True
    return res


def get_mean_vector_value_of_text(text, dimension, model):
    # this method is related with Word2Vec

    splitted = text.split(" ")
    current_word2vec = []
    try:

        for word in splitted:
            word = remove_extra_chars_from_word(word)

            if is_weblink(word):
                continue
            elif is_less_than_character_count(word):
                continue
            elif is_stopword(word):
                continue
            else:
                word = word.lower()
                if word in model.wv.vocab:
                    vec_word = model[word]
                    current_word2vec.append(vec_word)
                # else:
                # print("not existing in model: " + word)

        if len(current_word2vec) == 0:
            zeros = [0] * dimension
            current_word2vec.append(zeros)

        averaged_word2vec = list(np.array(current_word2vec).mean(axis=0))

    except Exception as exception:
        logger.error(traceback.format_exc())

    return averaged_word2vec


def train_and_predict_neutrals(clf_prob, X_train, y_train, X_test):
    clf_prob.fit(X_train, y_train)
    y_pred_prob = clf_prob.predict_proba(X_test)[:, 1]
    ids_scores = np.column_stack((y_pred_prob, X_test))
    ordered = ids_scores[np.lexsort(np.fliplr(ids_scores).T)]
    f = io.open('C:/Users/emre2/Desktop/neutrals_01_07_2016', 'w', encoding='utf-8')
    for row in ordered:
        f.write(str(row[0]) + "," + str(row[1]) + "\n")


def train_evaluate_probability_based_train_test_model(clf_prob, X_train, y_train, X_test, y_test):
    clf_prob.fit(X_train, y_train)
    print("Natural TP rate=" + str(sum(y_test) / len(y_test)))
    y_pred_prob = clf_prob.predict_proba(X_test)[:, 1]
    y_sorted = np.sort(y_pred_prob)[::-1]
    print("test started")
    # print('Top 100 first' + str(len(y_test)) + ' records in test dataset) -> ' + str(
    #    ratio(y_test, y_pred_prob, 1)))

    print("test ended")
    print('ROC AUC:', roc_auc_score(y_test, y_pred_prob))
    for i in [x * 0.1 for x in range(1, 6)]:
        i = round(i, 1)
        print('Top' + str(int(i * 100)) + 'percentile = (first ' + str(
            int(i * len(y_test))) + ' records in test dataset) -> ' + str(
            ratio(y_test, y_pred_prob, pct=i)))

    is_roc_plot_enabled = True
    if is_roc_plot_enabled:
        plot_roc(y_test, y_pred_prob)


def ratio(y_true, y_pred, pct):
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]
    n = int(round(len(y_true) * pct))
    idx = np.argsort(y_pred)[-n:]
    prob_min = y_pred[idx[0]]
    y_true_sum = y_true[idx].sum()
    y_emre = []
    for id in idx:
        y_emre.append(str(y_pred[id]) + ";" + str(y_true[id]))
        # print(y_emre)
    ratio_float = (y_true_sum / float(n))
    ratio_val = "{0:.2f}%".format(ratio_float * 100)

    res = "tp_ratio: " + str(ratio_val) + " , lowest probability score: " + str(round(prob_min, 2))
    return res


def plot_roc(y_test, preds):
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def remove_polarized_hashtags_urls(tweet):
    eliminated_text = ""
    splits = tweet.split(" ")
    for word in splits:
        if word == "":
            continue
        lowerr = word.lower()
        if lowerr in remain_hashtag_list or lowerr in leave_hashtag_list:
            continue
        if "http" in word:
            continue
        if "www" in word:
            continue
        eliminated_text += " " + word
    return eliminated_text


def remove_hashtags_urls(tweet):
    eliminated_text = ""
    splits = tweet.split(" ")
    for word in splits:
        if word == "":
            continue
        if word[0] == "#":
            continue
        if "http" in word:
            continue
        if "www" in word:
            continue
        eliminated_text += " " + word
    return eliminated_text


def convert_text_to_word2vec(data, dimension, model):
    vect_means = []
    for textvalue in data:
        vect_mean = get_mean_vector_value_of_text(textvalue, dimension, model)
        vect_means.append(vect_mean)

    np_vect_means = np.asarray(vect_means)
    return np_vect_means


def evaluate_train_test(clf, y_test, y_pred):
    print("expected test results  :" + str(y_test))
    print("predicted test results: " + str(y_pred))

    print("accuracy score:" + str(accuracy_score(y_test, y_pred)))
    # print(precision_recall_fscore_support(y_test, y_pred, average=None))

    # test precision
    # precision = precision_score(y_test, y_pred, average=None)
    # print("Precision: " + str(precision[1]))
    # test recall
    # recall = recall_score(y_test, y_pred, average=None)
    # print("Recall: " + str(recall[1]))
    # test F1 score
    # f_measure = f1_score(y_test, y_pred, average=None)
    # print("F1 score score: " + str(f_measure[1]))

    print(metrics.classification_report(y_test, y_pred, target_names=None))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # actual_yes, actual_no = get_yes_no_count(y_test)
    # pred_yes_cnt, pred_no_cnt = get_yes_no_count(y_pred)

    print("output of confusion matrix: tn:" + str(tn) + " fp:" + str(fp) + " fn: " + str(fn) + " tp:" + str(tp))
    # print("manual accuracy: " + str((tp + tn) / len(y_pred)))
    # print("manual misclassification rate: " + str((fp + fn) / len(y_pred)))
    # recall = tp / actual_yes
    # print("manual tp rate (sensitivity-recall): " + str(recall))
    # print("manual fp rate: " + str((fp) / actual_no))
    # print("manual specificity: " + str((tn) / actual_no))
    # precision = tp / pred_yes_cnt
    # print("manual precision: " + str(precision))
    # print("manual prevalence: " + str(actual_yes / len(y_pred)))
    # print("manual f1 score: " + str(2 * recall * precision / (recall + precision)))


def get_ml_model(model_id, prob_enabled):
    C = 1  # SVM regularization parameter
    if model_id == 1:
        print("model type: SVM Linear Kernel")
        clf = svm.SVC(kernel="linear", C=1, probability=prob_enabled)
    elif model_id == 2:
        print("model type: SVM RBF Kernel")
        clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    elif model_id == 3:
        print("model type: Random Forest")
        clf = RandomForestClassifier(n_estimators=100)
    elif model_id == 4:
        print("model type: Logistic Regression")
        clf = LogisticRegression()
    elif model_id == 5:
        print("model type: KNeighborsClassifier")
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_id == 6:
        print("model type: unigram, bigram and trigrams with Tfidf")
        clf = Pipeline([('vect', CountVectorizer(min_df=3, ngram_range=(1, 3), analyzer='word')),
                        ('tfidf', TfidfTransformer()),
                        ('clf', svm.SVC(kernel='linear', probability=prob_enabled)),
                        ])
    elif model_id == 7:
        print("model type: unigram with Tfidf")
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), analyzer='word')),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LogisticRegression()),
                        ])
    return clf


def text_preprocessing_for_tweet_texts(tweets):

    processed_tweets = []
    for tweet in tweets:
        words = tokenize_tweet(tweet)
        words = normalize_for_political_stance(words)
        processed = untokenize(words)
        processed_tweets.append(processed)
    logger.info("tokenization steps completed")
    logger.info("normalizing steps completed")

    return processed_tweets


def json_to_csv_converter(filename_read, filename_write, preprocessing=True):
    # print("filename: " + str(filename_read))
    tweets = []
    tweet_ids = []
    tweet_dates = []
    df_tweets = []
    cols = ["ID", "datetime", "tweet"]
    file = open(filename_write, "w", encoding='utf-8')

    with open(filename_read, encoding='utf-8') as f:
        for obj in ijson.items(f, 'item'):
            try:
                tweet = obj['tw_full']
                datetime = obj['datetime']
                ID = obj['ID']
                tweets.append([ID, datetime, tweet])
            except Exception as ex:
                logger.error(traceback.format_exc())

    df1 = pd.DataFrame(tweets, columns=cols)
    counter = 0

    if preprocessing:
        for index, row in df1.iterrows():
            try:
                tweet = row["tweet"]
                id = row["ID"]
                datetime = row["datetime"]
                counter += 1
                if counter % 1000 == 0:
                    print("file writing operation #" + str(counter))

                words = tokenize_tweet(tweet)
                words = normalize(words)
                converted_tweet = untokenize(words)
                file.write(str(id) + "," + str(datetime) + "," + converted_tweet)
                file.write("\n")

            except Exception as ex:
                logger.error(traceback.format_exc())

    else:
        for tweet in tweets:
            counter += 1
            file.write(tweet)
            file.write("\n")
    logger.info("tokenization and normalization steps of preprocessing completed")

    print("write completed. total tweet count: " + str(counter))


def tokenize_tweet(tweet):

    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokenized_tweet = tknzr.tokenize(tweet)

    return tokenized_tweet


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers_with_string(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def discard_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if not word.isdigit():
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def is_weblink(word):
    # this method is related with Word2Vec
    res = False
    if 'http' in word or 'www' in word:
        res = True
    return res


def remove_weblinks(words):
    new_words = []
    for word in words:
        if not is_weblink(word):
            new_words.append(word)
    return new_words


def stem_words(words):
    logger.info("stemming")
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    logger.info("lemmatizing")
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize_general(words):

    words = remove_weblinks(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers_with_string(words)
    words = remove_stopwords(words)

    return words


def normalize_for_political_stance(words):

    words = remove_weblinks(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = discard_numbers(words)
    #words = remove_stopwords(words)

    return words


def main():
    print("started")
    directory = "C:/_Documents/POLIMI/Research/Brexit/Periods/"
    text_files = ["p2_before_ref_retweet_5_50.json", "p3_ref_retweet_5_50.json", "p4_2016_after_ref_retweet_5_50.json",
                  "p5_2017_first_half_retweet_5_50.json", "p6_2017_second_half_retweet_5_50.json",
                  "p7_2018_first_three_retweet_5_50.json", "p1_full_retweet_5_50.json"]
    for file in text_files:
        file = directory + file
        print("started to " + file)
        file_out = file.replace("json", "csv")
        json_to_csv_converter(file, file_out)
        print("completed processing " + file)
    print("completed")


def calc():
    print("5	" + str(np.exp(-1 * -8.337156107991888)))

    print("10	" + str(np.exp(-1 * -8.496581625109977)))

    print("15	" + str(np.exp(-1 * -8.571572196003235)))

    print("20	" + str(np.exp(-1 * -8.664658426883669)))

    print("25	" + str(np.exp(-1 * -8.728475032233765)))

    print("30	" + str(np.exp(-1 * -8.80057700241029)))

    print("35	" + str(np.exp(-1 * -8.855940319680213)))

    print("40	" + str(np.exp(-1 * -8.908294493441964)))

    print("45	" + str(np.exp(-1 * -8.919373673881234)))

    print("50	" + str(np.exp(-1 * -8.952970892679456)))


def pearson():
    list1 = [241, 69, 72, 143, 128, 68, 126, 82, 126, 108, 68, 90, 81, 60, 72, 93, 80, 97, 65, 74, 71]
    list2 = [621711, 190310, 204282, 319612, 367879, 200600, 329108, 226406, 399833, 253989, 233108, 301069, 257548,
             206579, 255322, 268418, 279106, 304694, 216643, 236923, 254406]
    print("len list1: ", len(list1))
    print("len list2: ", len(list2))

    print(pearsonr(list1, list2))
    print(str(np.corrcoef(list1, list2)))


if __name__ == "__main__":
    list1 = [241, 69, 72, 143, 128, 68, 126, 82, 126, 108, 68, 90, 81, 60, 72, 93, 80, 97, 65, 74, 710]
    list2 = [621711, 190310, 204282, 319612, 367879, 200600, 329108, 226406, 399833, 253989, 233108, 301069, 257548,
             206579, 255322, 268418, 279106, 304694, 216643, 236923, 254406]

    if len(list1) != len(list2):
        print("error, two series should contain same size of elements")
        sys.exit

    # scipy library
    print("scipy result: ", pearsonr(list1, list2))

    # numpy library
    print("numpy result: ", str(np.corrcoef(list1, list2)))
