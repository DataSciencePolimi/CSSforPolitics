import nltk;
import ijson
from itertools import islice

nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from pprint import pprint
import sys, traceback
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import json
from pandas.io.json import json_normalize

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging as logger

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def remove_unnecessary_characters(data):

    before = data[:1]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data = [re.sub("\"", "", sent) for sent in data]
    data = [re.sub("  ", "", sent) for sent in data]

    #data = [re.sub("brexit", "", sent.lower()) for sent in data]
    data = [re.sub("#", "", sent) for sent in data]
    data = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', sent, flags=re.MULTILINE) for sent in data]
    after = data[:1]
    if(before!=after):
        print("before: " + str(before))
        print("after: " + str(after))
    return data

def is_weblink(word):
    # this method is related with Word2Vec
    res = False
    if 'http' in word or 'www' or '.com' in word:
        res = True
    return res

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and "brexit" not in word.lower()] for doc in texts]
    #return [[word for word in simple_preprocess(str(doc)) if word not in stop_words and not is_weblink(word)] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def main():
    try:
        print("ok")
        logger.basicConfig(level="INFO", filename="C:/mongo/bin/lda.log", format="%(asctime)s %(message)s")
        logger.info("started to read document")
        try:
            texts = []
            ids = []
            datetimes = []
            counter = 0
            with open('C:/mongo/bin/tweet.json', encoding='utf-8') as f:
                for obj in ijson.items(f,'item'):
                    counter += 1
                    texts.append(obj['tw_full'])
                    ids.append(obj['ID'])
                    datetimes.append(obj['datetime'])

                print("ok" + str(counter))
                print("texts count: " + str(len(texts)))
                print("ids count: " + str(len(ids)))
                print("datetimes count: " + str(len(datetimes)))

        except Exception as ex:
            logger.info("Something bad happened: %s", ex)
            print(ex)

        logger.info("completed reading document")
        #df = pd.read_csv("C:/mongo/bin/test_input.csv", encoding='utf-8', error_bad_lines=False, sep='delimiter')
        # print(df.head())

        #data = df.tweet.values.tolist()
        logger.info("started LDA related operations")

        data_clean = remove_unnecessary_characters(texts)
        data_words = list(sent_to_words(data_clean))
        print("tokenized")
        print(data_words[:1])

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        print("trigram example")
        print(trigram_mod[bigram_mod[data_words[0]]])

        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
        print(data_words_bigrams[:1])

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        #nlp = spacy.load('C:/Users/emre2/Anaconda3/envs/tensorflow/lib/site-packages/spacy/lang/en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        #data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        #print(data_lemmatized[:1])

        # Create Dictionary
        logger.info("creating corpora")

        id2word = corpora.Dictionary(data_words_bigrams)

        # Create Corpus
        texts = data_words_bigrams

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        #df["bow"]=corpus


        # View
        print(corpus[:1])

        print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

        logger.info("building LDA model")

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=20,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        lda_model.save("model_tweet_topic_20")

        logger.info("model saved. now enriching json to create an input file for visualization")

        counter = 0
        #df_topics = []
        df_topic = pd.DataFrame()
        list_topic_tweets = []
        topic_ids = []
        topic_counts = []
        topic_max_probs = []
        #topic_words = []
        for bow in corpus:
            topics = lda_model.get_document_topics(bow)
            topic_counter = 0
            max_prob = 0
            max_prob_topic = None
            for topic in topics:
                prob = topic[1]
                if max_prob < prob:
                    max_prob = prob
                    max_prob_topic = topic
                else:
                    break

            max_prob_topic_top10words = lda_model.show_topic(max_prob_topic[0])
            #list_topic_tweet = {"topic_count": len(topics), "topic_id": topic[0], "max_topic_prob": max_prob, "topic_words":max_prob_topic_top10words}
            #list_topic_tweets.append(list_topic_tweet)
            #df_topics.append(tweet_topics)
            topic_ids.append(topic[0])
            topic_max_probs.append(max_prob)
            topic_counts.append(len(topics))

        #df["topics"] = pd.Series(list_topic_tweets, index=df.index)
        if len(ids)!= len(topic_ids):
            print("FATAL ERROR: len other cols: " + len(ids) + " len new topic cols:" + len(topic_ids))
            exit(-1)

        newdf = pd.DataFrame()
        newdf["ID"] = ids

        newdf["datetime"] = datetimes
        newdf["t_count"] = topic_counts
        newdf["t_id"] = topic_ids
        newdf["t_max_prob"] = topic_max_probs

        logger.info("completed enrichment. now saving into a file")

        newdf.to_json("tweet_topic.json", orient='records')

        logger.info("saved succesfully into a file")

        # Print the Keyword in the 10 topics
        pprint(lda_model.print_topics())
        logger.info("topics: %s", lda_model.print_topics())

        # Compute Perplexity
        print('\nPerplexity: ',
              lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
        logger.info("Perplexity: %s", lda_model.log_perplexity(corpus))

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_bigrams, dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        logger.info("Coherence Scor: %s", coherence_lda)

        # Visualize the topics
        #pyLDAvis.enable_notebook()
        visual_enabled = True
        if visual_enabled:
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
            pyLDAvis.save_html(vis, 'LDA_Visualization.html')


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  limit=2, file=sys.stdout)
        print(ex)
        logger.info("Something bad happened: %s", ex)

    logger.info("Completed everything. Program is being terminated")


if __name__ == "__main__":
    main()
