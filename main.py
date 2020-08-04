import gc
import os
import time
import re
from multiprocessing import Pool
import numpy as np
import gensim
from autocorrect import Speller
import spellchecker
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
TERMCOLOR = True
if TERMCOLOR:
    from termcolor import colored

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)


CONTRACTION_MAP = {
    "ain`t": "is not",
    "aren`t": "are not",
    "can`t": "cannot",
    "can`t`ve": "cannot have",
    "`cause": "because",
    "could`ve": "could have",
    "couldn`t": "could not",
    "couldn`t`ve": "could not have",
    "didn`t": "did not",
    "doesn`t": "does not",
    "don`t": "do not",
    "hadn`t": "had not",
    "hadn`t`ve": "had not have",
    "hasn`t": "has not",
    "haven`t": "have not",
    "he`d": "he would",
    "he`d`ve": "he would have",
    "he`ll": "he will",
    "he`ll`ve": "he he will have",
    "he`s": "he is",
    "how`d": "how did",
    "how`d`y": "how do you",
    "how`ll": "how will",
    "how`s": "how is",
    "I`d": "I would",
    "I`d`ve": "I would have",
    "I`ll": "I will",
    "I`ll`ve": "I will have",
    "I`m": "I am",
    "It`s": "It is",
    "I`ve": "I have",
    "i`d": "i would",
    "i`d`ve": "i would have",
    "i`ll": "i will",
    "i`ll`ve": "i will have",
    "i`m": "i am",
    "i`ve": "i have",
    "isn`t": "is not",
    "it`d": "it would",
    "it`d`ve": "it would have",
    "it`ll": "it will",
    "it`ll`ve": "it will have",
    "it`s": "it is",
    "let`s": "let us",
    "ma`am": "madam",
    "mayn`t": "may not",
    "might`ve": "might have",
    "mightn`t": "might not",
    "mightn`t`ve": "might not have",
    "must`ve": "must have",
    "mustn`t": "must not",
    "mustn`t`ve": "must not have",
    "needn`t": "need not",
    "needn`t`ve": "need not have",
    "o`clock": "of the clock",
    "oughtn`t": "ought not",
    "oughtn`t`ve": "ought not have",
    "shan`t": "shall not",
    "sha`n`t": "shall not",
    "shan`t`ve": "shall not have",
    "she`d": "she would",
    "she`d`ve": "she would have",
    "she`ll": "she will",
    "she`ll`ve": "she will have",
    "she`s": "she is",
    "should`ve": "should have",
    "shouldn`t": "should not",
    "shouldn`t`ve": "should not have",
    "so`ve": "so have",
    "so`s": "so as",
    "that`d": "that would",
    "that`d`ve": "that would have",
    "that`s": "that is",
    "there`d": "there would",
    "there`d`ve": "there would have",
    "there`s": "there is",
    "they`d": "they would",
    "they`d`ve": "they would have",
    "they`ll": "they will",
    "they`ll`ve": "they will have",
    "they`re": "they are",
    "they`ve": "they have",
    "to`ve": "to have",
    "wasn`t": "was not",
    "we`d": "we would",
    "we`d`ve": "we would have",
    "we`ll": "we will",
    "we`ll`ve": "we will have",
    "we`re": "we are",
    "we`ve": "we have",
    "weren`t": "were not",
    "what`ll": "what will",
    "what`ll`ve": "what will have",
    "what`re": "what are",
    "what`s": "what is",
    "what`ve": "what have",
    "when`s": "when is",
    "when`ve": "when have",
    "where`d": "where did",
    "where`s": "where is",
    "where`ve": "where have",
    "who`ll": "who will",
    "who`ll`ve": "who will have",
    "who`s": "who is",
    "who`ve": "who have",
    "why`s": "why is",
    "why`ve": "why have",
    "will`ve": "will have",
    "won`t": "will not",
    "won`t`ve": "will not have",
    "would`ve": "would have",
    "wouldn`t": "would not",
    "wouldn`t`ve": "would not have",
    "y`all": "you all",
    "y`all`d": "you all would",
    "y`all`d`ve": "you all would have",
    "y`all`re": "you all are",
    "y`all`ve": "you all have",
    "you`d": "you would",
    "you`d`ve": "you would have",
    "you`ll": "you will",
    "you`ll`ve": "you will have",
    "you`re": "you are",
    "you`ve": "you have",
}


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()

    return df


def printm(text, color='white', switch=False):
    # wrapper for standard print function
    if switch:
        print(colored(text, color))
    else:
        print(text)
    return


def build_dfs(data_types, paths):
    dfs = []
    for i in range(len(data_types)):
        if data_types[i] == 'csv':
            dfs.append(pd.read_csv(paths[i]))
    return dfs


def lemmatize_text(text):
    # lemmatize words
    w_tokenizer = WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w)+' ' for w in w_tokenizer.tokenize(text.lower())]
    text = ''.join(words)

    return text


def expand_contraction(text):
    # replace contracted word with expanded form in CONTRACTION_MAP
    tk = WhitespaceTokenizer().tokenize(text)
    contracted = [word for word in tk if "`" in word]
    for word in contracted:
        try:
            text = text.replace(word, CONTRACTION_MAP[word])
        except:
            pass

    return text


def spelling_correction(text):
    # take numbers out of the string if possible
    try:
        text = re.sub("\d+", ' ', text).lower()
    except (AttributeError, TypeError) as error:
        print(type(text))

    # convert string to lowercase and take out some punctuation that is difficult to map to sentiment
    try:
        text = text.replace(',', ' ').replace('.', ' ')
    except (AttributeError, TypeError) as error:
        print(type(text))

    # correct spelling
    spell = spellchecker.SpellChecker()
    misspelled = spell.unknown(word_tokenize(text))
    corrections = [(word, spell.correction(word)) for word in list(misspelled)]
    for correction in corrections:
        text = text.replace(correction[0], correction[1])

    return text


def basic_preprocess_raw_text(corpus):
    # it makes the most sense to apply the corrections in this sequence
    print('correcting spelling')
    corpus = corpus.apply(spelling_correction)
    print('expanding contractions')
    corpus = corpus.apply(expand_contraction)
    print('lemmatizing')
    corpus = corpus.apply(lemmatize_text)
    # print(corpus)

    return corpus


def get_top_n_words(corpus, n=None, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def eda(train, test, *args):
    printm('Performing exploratory data analysis ... ', color='green', switch=TERMCOLOR)
    train = train.sample(frac=0.01, replace=True, random_state=42)
    # printm(train.describe())
    # print(test.describe())
    print(train.head(50))
    train = train.dropna()
    # print(train.isna().sum())
    # print(test.head())
    # print(args[0].head())

    # print(len(set(train.textID.values).union(set(test.textID.values))))
    # printm('Is test.textID subset of train.textID : ',set(test.textID.values).issubset(set(train.textID.values)))
    # printm('How many textIDs are common : ',len(set(test.textID.values).intersection(set(train.textID.values))))

    # basic text preprocessing in parallel
    a=time.time()
    train['text_processed'] = parallelize_dataframe(train['text'], basic_preprocess_raw_text)
    # print(time.time()-a,' seconds', (time.time()-a)*100/60, ' estimated')
    print(time.time() - a, ' seconds')
    print(train.head())

    var_text = 'text_processed'

    # univariate visualizations
    gb = train.groupby(['sentiment']).count().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].set_title('Sentiment Distribution - Train')
    axes[0] = sns.barplot(x="sentiment", y="text", hue='sentiment', palette=['red','green', 'gray'], data=gb, dodge=False, ax=axes[0]).legend_.remove()
    gb = test.groupby(['sentiment']).count().reset_index()
    axes[1].set_title('Sentiment Distribution - Test')
    axes[1] = sns.barplot(x="sentiment", y="text", hue='sentiment', palette=['red','green', 'gray'], data=gb, dodge=False, ax=axes[0]).legend_.remove()
    del gb
    gc.collect()

    # distribution of top unigrams
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    common_words = get_top_n_words(train[train['sentiment'] == 'negative'][var_text].fillna('NaN'), 20, ngram_range=(1, 1))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[0].set_title('Top 20 Unigrams with (-) Sentiment - Train')
    axes[0] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Reds", ax=axes[0]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'neutral'][var_text].fillna('NaN'), 20, ngram_range=(1, 1))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[1].set_title('Top 20 Unigrams with (+-) Sentiment - Train')
    axes[1] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Greys", ax=axes[1]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'positive'][var_text].fillna('NaN'), 20, ngram_range=(1, 1))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[2].set_title('Top 20 Unigrams with (+) Sentiment - Train')
    axes[2] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="BuGn",ax=axes[2]).legend_.remove()
    del df2, df3
    gc.collect()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)

    # distribution of top bigrams
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    common_words = get_top_n_words(train[train['sentiment'] == 'negative'][var_text].fillna('NaN'), 20, ngram_range=(2, 2))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[0].set_title('Top 20 Bigrams with (-) Sentiment - Train')
    axes[0] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Reds", ax=axes[0]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'neutral'][var_text].fillna('NaN'), 20, ngram_range=(2, 2))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[1].set_title('Top 20 Bigrams with (+-) Sentiment - Train')
    axes[1] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Greys", ax=axes[1]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'positive'][var_text].fillna('NaN'), 20, ngram_range=(2, 2))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[2].set_title('Top 20 Bigrams with (+) Sentiment - Train')
    axes[2] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="BuGn", ax=axes[2]).legend_.remove()
    del df2, df3
    gc.collect()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)

    # distribution of top Trigrams
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    common_words = get_top_n_words(train[train['sentiment'] == 'negative'][var_text].fillna('NaN'), 20, ngram_range=(3, 3))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[0].set_title('Top 20 Trigrams with (-) Sentiment - Train')
    axes[0] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Reds", ax=axes[0]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'neutral'][var_text].fillna('NaN'), 20, ngram_range=(3, 3))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[1].set_title('Top 20 Trigrams with (+-) Sentiment - Train')
    axes[1] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Greys", ax=axes[1]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'positive'][var_text].fillna('NaN'), 20, ngram_range=(3, 3))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[2].set_title('Top 20 Trigrams with (+) Sentiment - Train')
    axes[2] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="BuGn", ax=axes[2]).legend_.remove()
    del df2, df3
    gc.collect()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    plt.show()

    print(train.shape, train.textID.nunique())





    return


if __name__ == '__main__':
    # build dataframes
    start_time = time.time()
    printm('Importing data ... ', 'green', TERMCOLOR)

    DATA_FOLDER = 'data/'

    files_train = [os.path.join(DATA_FOLDER, 'train.csv'),
                   os.path.join(DATA_FOLDER, 'test.csv'),
                   os.path.join(DATA_FOLDER, 'sample_submission.csv')]

    train, test, sample = build_dfs(['csv' for file in files_train], [file for file in files_train])

    eda(train, test, sample)



