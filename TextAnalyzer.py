import pandas as pd

import re

import gensim
from gensim.models import Phrases

from collections import Counter

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class TextAnalyzer:

    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.text_data = None

    def clean_data(self):

        # read in text data
        self.text_data = pd.read_csv(self.data_filename)

        # delete  empty messages
        self.text_data = self.text_data.dropna(subset=['Message'])
        # delete redundant first column
        self.text_data.drop(['Unnamed: 0'], axis = 1, inplace = True)
        # delete error messages
        self.text_data = self.text_data[self.text_data.length != 'Err:511']
        self.text_data = self.text_data[self.text_data.length != 'Err:510']
        self.text_data = self.text_data[self.text_data.length != 'Err:509']
        self.text_data = self.text_data[self.text_data.length != 'Err:508']

        # convert length data from string to integer
        self.text_data.length = pd.to_numeric(self.text_data.length)

        print("*** finished cleaning ***")

        # for testing
        # print('*** Cleaned pd table ***')
        # print(self.text_data.info())
        # print(self.text_data.Message)

    # *** helper functions for noise removal *** #

    def remove_punctuation(self, text):
        return re.sub(r',|\.|\:|;|-|/|&|!|\?|\(|\)|\+|@|<|>|#|~|=|\$|\*|[|]|{|}', ' ', text)

    def remove_apostrophe(self, text):
        return re.sub(r"'", ' ',text)

    def remove_crossmark(self, text):
        return re.sub(r"┾", ' ', text)

    def remove_extra_whitespace(self, text):
        return re.sub(r'\s+', ' ', text)

    # stopwords: words to filter out
    def remove_stopwords(self):

        # load common english stopwords
        eng_stopwords = set(stopwords.words('english'))

        # add column for tokenized, stopword-filtered messages
        self.text_data['Message_tokenized_nostop'] = [
            list()
            for i in range(len(self.text_data['Message_tokenized']))
            ]

        # filter out stopwords
        for i in range(len(self.text_data['Message_tokenized'])):
            text = self.text_data.Message_tokenized.iloc[i]
            for word in text:
                if not word in eng_stopwords:
                    self.text_data.Message_tokenized_nostop.iloc[i].append(word)

    # *** *** #

    def remove_noise(self):

        # remove decimals
        self.text_data.Message = self.text_data.Message.replace('<DECIMAL>', ' ', regex = False)

        # remove all other extra symbols, whitespace
        self.text_data.Message = self.text_data.Message.apply(lambda x: self.remove_apostrophe(x))
        self.text_data.Message = self.text_data.Message.apply(lambda x: self.remove_crossmark(x))
        self.text_data.Message = self.text_data.Message.apply(lambda x: self.remove_punctuation(x))
        self.text_data.Message = self.text_data.Message.apply(lambda x: self.remove_extra_whitespace(x))

    # to help lemmatizer
    def get_part_of_speech(self, word):
        probable_part_of_speech = wordnet.synsets(word)
        pos_counts = Counter()
        pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos()=="n"])
        pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos()=="v"])
        pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos()=="a"])
        pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos()=="r"])
        most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
        return most_likely_part_of_speech

    def lemmatize(self):

        # import lemmatizer
        lemmatizer = WordNetLemmatizer()

        # create additional column
        self.text_data['Msg_token_nostop_lemmmed'] = [list() for i in range(len(self.text_data['Message_tokenized_nostop']))]

        # lemmatize each token of each message
        for i in range(len(self.text_data['Message_tokenized_nostop'])):
            text = self.text_data.Message_tokenized_nostop.iloc[i]
            for word in text:
                lemmatized = lemmatizer.lemmatize(word, self.get_part_of_speech(word))
                self.text_data.Msg_token_nostop_lemmmed.iloc[i].append(lemmatized)


    def preprocess_text(self):

        self.remove_noise()

        # convert everything to lowercase
        self.text_data.Message = self.text_data.Message.apply(lambda x: x.lower())

        # tokenize using NLTK (split each message into individual words) and add new column
        self.text_data['Message_tokenized'] = self.text_data.Message.apply(lambda text_message: word_tokenize(text_message))

        self.remove_stopwords()

        self.lemmatize()

        print("*** finished pre-processing ***")

        # for testing
        # print("*** Pre-processed table ***")
        # print(self.text_data.Message)
        # print(self.text_data.Message_tokenized_nostop)
        # print(self.text_data.Msg_token_nostop_lemmed)

    # term frequency analysis: "what are common words/topics for senders from xx country?"
    def tf_analysis(self, country):

        # get text data from specified country
        country_text_data = self.text_data[self.text_data.country == country]

        # create master corpus of all lemmatized words
        corpus = []
        for i in range(len(country_text_data['Msg_token_nostop_lemmmed'])):
            text = country_text_data.Msg_token_nostop_lemmmed.iloc[i]
            for word in text:
                corpus.append(word)


        count_of_words = Counter(corpus)
        word_counts = pd.DataFrame.from_dict(count_of_words, orient = 'index').reset_index()
        word_counts.rename(columns = {'index': 'word', 0: 'count'}, inplace = True)
        word_counts.sort_values(by = ['count'], ascending = False, inplace = True)

        # print 15 most commonly used words
        print("*** 15 most commonly used words used by senders from {country} ***".format(country = country))
        print(word_counts.nlargest(15, 'count'))

    # term frequency inverse document frequency analysis:
    # "what are common words/topics for senders from xx country compared to other countries?"
    def tf_idf_analysis(self, country):

        # get text data from specified country
        country_text_data = self.text_data[self.text_data.country == country]

        # master corpus of text messages in lemmatized strings
        corpus = []
        for i in range(len(country_text_data.Msg_token_nostop_lemmmed)):
            this_list = country_text_data.Msg_token_nostop_lemmmed.iloc[i]
            this_str = ', '.join(this_list)
            corpus.append(this_str)

        # vectorize corpus
        vectorizer = TfidfVectorizer(use_idf = True)
        tf_idf_vectors = vectorizer.fit_transform(corpus)

        # filter out TF-IDF scores below 0
        for i in range(len(country_text_data.Msg_token_nostop_lemmmed)):
            tf_idf_vector = tf_idf_vectors[i]
            tf_idf_results = pd.DataFrame(tf_idf_vector.T.todense(), index = vectorizer.get_feature_names(), columns = ["TF-IDF Score"])
            nonzero_words = tf_idf_results[tf_idf_results['TF-IDF Score'] > 0.0]
            if i == 0:
                top_tf_idf_scores = nonzero_words
            else:
                top_tf_idf_scores = top_tf_idf_scores.append(nonzero_words)

        top_tf_idf_scores = top_tf_idf_scores.index[top_tf_idf_scores['TF-IDF Score'] > 0.0].to_list()

        # counting top 15 most occuring of these "important" words
        countof_top_tf_idf_scores = Counter(top_tf_idf_scores)
        top_tf_idf_word_counts = pd.DataFrame.from_dict(countof_top_tf_idf_scores, orient='index').reset_index()
        top_tf_idf_word_counts.rename(columns = {'index': 'word', 0: 'count'}, inplace = True)
        top_tf_idf_word_counts.sort_values(by=["count"], ascending = False, inplace = True)

        print("*** 15 most commonly used important words used by senders from {country} ***".format(country = country))
        print(top_tf_idf_word_counts.nlargest(15,'count'))

    # predicting country of origin of messages
    def predict_country_origin(self):

        # only getting texts from countries with more than 100 data points
        over100_countries = ['Singapore','India','United States','Sri Lanka','Malaysia','Pakistan','unknown','Canada', 'Bangladesh','China']
        text_data_over100 =  self.text_data[self.text_data.country.isin(over100_countries)]

        # convert message and country columns to lists
        master_text_list = text_data_over100.Message.to_list()
        master_country_list = text_data_over100.country.to_list()

        # split data for training and testing
        train_data, test_data, train_labels, test_labels = train_test_split(master_text_list, master_country_list, train_size = 0.85, test_size = 0.15, random_state = 42)

        # convert training and test data into words counts
        count_vector = CountVectorizer()
        count_vector.fit(train_data)
        train_counts = count_vector.transform(train_data)
        test_counts = count_vector.transform(test_data)

        # multinomial naive Bayes
        mnb_classifier = MultinomialNB()
        mnb_classifier.fit(train_counts, train_labels)
        mnb_predictions = mnb_classifier.predict(test_counts)

        # Determine the model's accuracy and print
        print('MNB classifier accuracy: ' + str(round(accuracy_score(test_labels, mnb_predictions),2)))
        print('MNB Classification Report:')
        print(classification_report(test_labels, mnb_predictions, labels = over100_countries, target_names = over100_countries, zero_division = 0.0))
