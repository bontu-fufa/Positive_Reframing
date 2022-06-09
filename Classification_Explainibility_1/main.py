import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer, TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from accuracy import calculate_accuracy
data = pd.read_csv('./Data/wholetrain_clean.tsv', sep='\t')

# COUNT VECTORIZER


def feature_gen_with_BOW():
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words="english",
                         ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(data['reframed_text'])
    return train_test_split(text_counts, data['strategy'], test_size=0.3, random_state=1212)


# TF_IDF
def feature_gen_with_TFIDF():
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(data['reframed_text'])
    #  x_train, x_test, y_train, y_test
    return train_test_split(text_tf, data['strategy'], test_size=0.3, random_state=1212)


# PREDICTION

# SVM_IDTDF_accuracy, SVM_IDTDF_prediction = calculate_accuracy(
#     *feature_gen_with_TFIDF(), 'svm')


# [ Explainability ]

# Task 1
# CLASSIFICATION REPORT
"""
NOT WORKING
print(classification_report(
    y_test, MNB_BOW_prediction, target_names=data['strategy']))
"""


def report_BOW_MNB(model='ALL'):
    _, _, _, y_test = feature_gen_with_BOW()

    if model == "MNB" or model == "ALL":
        MNB_BOW_accuracy, MNB_BOW_prediction = calculate_accuracy(
            *feature_gen_with_BOW())
        print(" Classification Report Using Count Vectorizer with MNB")
        print(classification_report(
            y_test, MNB_BOW_prediction))
    if model == "SVM" or model == "ALL":
        MNB_BOW_accuracy, MNB_BOW_prediction = calculate_accuracy(
            *feature_gen_with_BOW(), 'SVM')
        print(" Classification Report Using Count Vectorizer with SVM")
        print(classification_report(
            y_test, MNB_BOW_prediction))


def report_TDFIDF_MNB(model='ALL'):
    _, _, _, y_test = feature_gen_with_TFIDF()
    if model == "MNB" or model == "ALL":
        MNB_BOW_accuracy, MNB_BOW_prediction = calculate_accuracy(
            *feature_gen_with_TFIDF())
        print(" Classification Report Using TFTDF with MNB")
        print(classification_report(
            y_test, MNB_BOW_prediction))
    if model == "SVM" or model == "ALL":
        MNB_BOW_accuracy, MNB_BOW_prediction = calculate_accuracy(
            *feature_gen_with_TFIDF(), 'SVM')
        print(" Classification Report Using TFTDF with SVM")
        print(classification_report(
            y_test, MNB_BOW_prediction))


report_BOW_MNB()
report_TDFIDF_MNB()
# ----------- TODO : Which classes are easy to predict? Which are hard?


# Task 2
def print_words(words, top=10):

    res = []
    for word, count in words:
        if len(res) == top:
            break
        if len(word) > 2:
            # print(word,count)
            res.append((word, count))

    # print(res)
    return res


# with series
def get_top_words_associated_with_class_with_occurance_of_word_in_a_class(cls, n_tops=10):

    # print(data['strategy'].value_counts())
    # load
    doc = data[data['strategy'].isin([cls])]

    count = len(doc)  # total sum of class

    # DTM
    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    token = TweetTokenizer()

    #  token_pattern=r"?u)\b\w\w+\b|!|\?|\"|\'"
    # tokenizer=token.tokenize
    cv = CountVectorizer(lowercase=True, stop_words="english",
                         ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(doc['reframed_text'])
    # print(text_counts)

    sum_of_words = text_counts.sum(axis=0)  # returns [[]]

    print((sum_of_words[0].sum(axis=1)))
    words_freq = [(word, sum_of_words[0, idx]/np.array(sum_of_words[0].sum(axis=1))[0][0])
                  for idx, word in enumerate(cv.get_feature_names_out())]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # print(words_freq[:n_tops])
    # print()
    res = print_words(words_freq, n_tops)
    # print(res)
    return res


# with series converted to array
def get_top_words_associated_with_class(cls, n_tops):
    # print(n_tops)
    # load
    doc = data[data['strategy'].isin([cls])]

    count = len(doc)  # total sum of class

    # DTM
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')

    #  token_pattern=r"(?u)\b\w\w+\b|!|\?|\"|\'"
    cv = CountVectorizer(lowercase=True, stop_words="english",
                         ngram_range=(1, 1), tokenizer=token.tokenize)

    corpus = doc['reframed_text'].array
    vec = cv.fit(corpus)
    bow = vec.transform(corpus)

    # print(text_counts)
    sum_of_words = bow.sum(axis=0)  # returns [[]]
    words_freq = [(word, sum_of_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    # words_freq = [(word, sum_of_words[0, idx])
    #               for idx, word in enumerate(cv.get_feature_names_out())]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # print(words_freq[:n_tops])
    print_words(words_freq, n_tops)


def get_top_words_of_all_classes():
    print('----------------------')
    print("self_affirmation")
    print(get_top_words_associated_with_class('self_affirmation', 15))
    print('----------------------------------------------------------------------')
    print("thankfulness")
    print(get_top_words_associated_with_class('thankfulness', 15))
    print('----------------------------------------------------------------------')
    print("impermanence")
    print(get_top_words_associated_with_class('impermanence', 15))
    print('----------------------------------------------------------------------')
    print("growth")
    print(get_top_words_associated_with_class('growth', 15))
    print('----------------------------------------------------------------------')
    print("neutralizing")
    print(get_top_words_associated_with_class('neutralizing', 15))
    print('----------------------------------------------------------------------')
    print("optimism")
    print(get_top_words_associated_with_class('optimism', 15))
    print('----------------------------------------------------------------------')


def get_intersection_words(strategies):
    print("Intersections")
    print(*strategies)
    intersections = set(
        [word for word, _ in get_top_words_associated_with_class(strategies[0])])
    for i in range(1, len(strategies)):
        intersections.union(
            set([word for word, _ in get_top_words_associated_with_class(strategies[i])]))
    print(intersections)


# get_top_words_of_all_classes()


# get_intersection_words(['self_affirmation', 'thankfulness'])
