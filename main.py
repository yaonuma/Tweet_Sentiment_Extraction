import gc
import os
import time
import re
import tokenizers
import numpy as np
import spellchecker
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from transformers import *
import matplotlib.pyplot as plt
from multiprocessing import Pool
from nltk import jaccard_distance
import tensorflow.keras.backend as K
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
TERMCOLOR = True
if TERMCOLOR:
    from termcolor import colored
print('TF version', tf.__version__)
print('NumPy version', np.__version__)
print('Pandas version', pd.__version__)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)
pd.set_option('display.width', 10000)



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
            df = pd.read_csv(paths[i])
            try:
                df['text'] = df['text'].astype(str)
            except KeyError:
                pass
            try:
                df['selected_text'] = df['selected_text'].astype(str)
            except KeyError:
                pass
            dfs.append(df)
    return dfs


def lemmatize_text(text):
    # lemmatize words
    w_tokenizer = WhitespaceTokenizer()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w)+' ' for w in w_tokenizer.tokenize(text.lower())]
    text = ''.join(words)

    return text


def expand_contraction(text):
    CONTRACTION_MAP = {
        "ain`t": "is not",
        "aren`t": "are not",
        "are`t": "are not",
        "can`t": "cannot",
        "can`t`ve": "cannot have",
        "`cause": "because",
        "could`ve": "could have",
        "couldn`t": "could not",
        "could`t": "could not",
        "couldn`t`ve": "could not have",
        "didn`t": "did not",
        "did`t": "did not",
        "doesn`t": "does not",
        "does`t": "does not",
        "don`t": "do not",
        "don`": "do not",
        "hadn`t": "had not",
        "had`t": "had not",
        "hadn`t`ve": "had not have",
        "hasn`t": "has not",
        "has`t": "has not",
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
        "in`t": "is not",
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
        "may`t": "may not",
        "might`ve": "might have",
        "mightn`t": "might not",
        "might`t": "might not",
        "mightn`t`ve": "might not have",
        "must`ve": "must have",
        "mustn`t": "must not",
        "must`t": "must not",
        "mustn`t`ve": "must not have",
        "`nd": "and",
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
        "should`t": "should not",
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
        "`til": "until",
        "to`ve": "to have",
        "`twas": "it was",
        "`tis": "it is",
        "u`d": "you would",
        "u`d": "you would",
        "u`d`ve": "you would have",
        "u`ll": "you will",
        "u`ll`ve": "you will have",
        "u`re": "you are",
        "u`ve": "you have",
        "wasn`t": "was not",
        "was`t": "was not",
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
        "would`t": "would not",
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
        "ya`ll": "you all"
    }
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


def remove_pattern(input_txt, pattern="@[\w]*"):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(re.escape(i), ' ', input_txt)

    return input_txt


def basic_preprocess_raw_text(corpus):
    a = time.time()
    # it makes the most sense to apply the corrections in this sequence

    # # vectorize the apply function
    # print('removing handles')
    # corpus = pd.Series(np.vectorize(remove_pattern)(corpus, "@[\w]*"))
    # print('correcting spelling')
    # corpus = pd.Series(np.vectorize(spelling_correction)(corpus))
    # # corpus = corpus.apply(spelling_correction)
    # print('expanding contractions')
    # # corpus = corpus.apply(expand_contraction)
    # corpus = pd.Series(np.vectorize(expand_contraction)(corpus))
    # print('removing punctuation')
    # corpus = pd.Series(np.vectorize(remove_pattern)(corpus, "[`][a-zA-Z]*"))
    # corpus = pd.Series(np.vectorize(remove_pattern)(corpus, "[^a-zA-Z^\s]*"))
    # print('lemmatizing')
    # # corpus = corpus.apply(lemmatize_text)
    # corpus = pd.Series(np.vectorize(lemmatize_text)(corpus))
    # # print(corpus)


    corpus.dropna(inplace=True)
    # remove URLs
    # corpus = corpus.apply(remove_pattern, pattern='http\S+')
    corpus = corpus.apply(lambda x: re.compile(r'http\S+').sub(' ', x))
    # remove non-ascii characters
    corpus = corpus.apply(lambda x: x.encode('ascii', 'ignore').decode())
    # remove handles
    # corpus = corpus.apply(remove_pattern, pattern="@[\w]*")
    corpus = corpus.apply(lambda x: re.compile(r'@[\w]*').sub(' ', x))
    # spell check
    corpus = corpus.apply(spelling_correction)
    # expand contracted words
    corpus = corpus.apply(expand_contraction)
    # remove remaining apostophe pieces
    # corpus = corpus.apply(remove_pattern, pattern="[`][a-z]*")
    corpus = corpus.apply(lambda x: re.compile(r'[`][a-z]*').sub(' ', x))
    # lemmatize
    corpus = corpus.apply(lemmatize_text)
    # remove 1 or two letter words
    corpus = corpus.apply(lambda x: re.compile(r'\W*\b\w{1,3}\b').sub(' ', x))
    print('\t.')

    return corpus


def get_top_n_words(corpus, n=None, ngram_range=(1, 1)):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def eda(train, test, filepath, *args):
    printm('Performing exploratory data analysis ... ', color='green', switch=TERMCOLOR)
    # train = train.sample(frac=0.01, replace=True, random_state=42)
    train = train.sample(frac=0.01, replace=True)
    # printm(train.describe())
    # print(test.describe())
    # print(train.head(50))
    train = train.dropna()
    # print(train.isna().sum())
    # print(test.head())
    # print(args[0].head())
    print(train.head())
    print(test.head())

    # print(len(set(train.textID.values).union(set(text_dataID.values))))
    # printm('Is text_dataID subset of train.textID : ',set(text_dataID.values).issubset(set(train.textID.values)))
    # printm('How many textIDs are common : ',len(set(text_dataID.values).intersection(set(train.textID.values))))

    # basic text preprocessing in parallel
    train['text_processed'] = parallelize_dataframe(train['text'], basic_preprocess_raw_text)
    train['selected_text_processed'] = parallelize_dataframe(train['selected_text'], basic_preprocess_raw_text)
    # train['text_processed'] = basic_preprocess_raw_text(train['text'])


    var_text = 'text'

    # univariate visualizations
    gb = train.groupby(['sentiment']).count().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].set_title('Sentiment Distribution - Train')
    axes[0] = sns.barplot(x="sentiment", y="text", hue='sentiment', palette=['red','green', 'gray'],
                          data=gb, dodge=False, ax=axes[0]).legend_.remove()
    gb = test.groupby(['sentiment']).count().reset_index()
    axes[1].set_title('Sentiment Distribution - Test')
    axes[1] = sns.barplot(x="sentiment", y="text", hue='sentiment', palette=['red','green', 'gray'],
                          data=gb, dodge=False, ax=axes[0]).legend_.remove()
    plt.savefig(filepath+'sentiment_distribution.png', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait',  format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
             metadata=None)
    del gb
    gc.collect()

    # distribution of top unigrams
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    common_words = get_top_n_words(train[train['sentiment'] == 'negative'][var_text].fillna('NaN'), 20, ngram_range=(1, 1))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[0].set_title('Top 20 Unigrams with (-) Sentiment - Train')
    axes[0] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False, palette="Reds",
                          ax=axes[0]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'neutral'][var_text].fillna('NaN'), 20, ngram_range=(1, 1))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[1].set_title('Top 20 Unigrams with (+-) Sentiment - Train')
    axes[1] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="Greys", ax=axes[1]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'positive'][var_text].fillna('NaN'), 20, ngram_range=(1, 1))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[2].set_title('Top 20 Unigrams with (+) Sentiment - Train')
    axes[2] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="BuGn",ax=axes[2]).legend_.remove()
    del df2, df3
    gc.collect()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    plt.savefig(filepath + 'top_20_unigrams.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait',  format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                 metadata=None)

    # distribution of top bigrams
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    common_words = get_top_n_words(train[train['sentiment'] == 'negative'][var_text].fillna('NaN'), 20, ngram_range=(2, 2))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[0].set_title('Top 20 Bigrams with (-) Sentiment - Train')
    axes[0] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="Reds", ax=axes[0]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'neutral'][var_text].fillna('NaN'), 20, ngram_range=(2, 2))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[1].set_title('Top 20 Bigrams with (+-) Sentiment - Train')
    axes[1] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="Greys", ax=axes[1]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'positive'][var_text].fillna('NaN'), 20, ngram_range=(2, 2))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[2].set_title('Top 20 Bigrams with (+) Sentiment - Train')
    axes[2] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="BuGn", ax=axes[2]).legend_.remove()
    del df2, df3
    gc.collect()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    plt.savefig(filepath + 'top_20_bigrams.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait',  format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                 metadata=None)

    # distribution of top Trigrams
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), tight_layout=True)
    common_words = get_top_n_words(train[train['sentiment'] == 'negative'][var_text].fillna('NaN'), 20, ngram_range=(3, 3))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[0].set_title('Top 20 Trigrams with (-) Sentiment - Train')
    axes[0] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="Reds", ax=axes[0]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'neutral'][var_text].fillna('NaN'), 20, ngram_range=(3, 3))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[1].set_title('Top 20 Trigrams with (+-) Sentiment - Train')
    axes[1] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="Greys", ax=axes[1]).legend_.remove()
    del df2, df3
    gc.collect()
    common_words = get_top_n_words(train[train['sentiment'] == 'positive'][var_text].fillna('NaN'), 20, ngram_range=(3, 3))
    df2 = pd.DataFrame(common_words, columns=['Text', 'count'])
    df3 = df2.groupby('Text').sum()['count'].sort_values(ascending=False).reset_index()
    axes[2].set_title('Top 20 Trigrams with (+) Sentiment - Train')
    axes[2] = sns.barplot(x="Text", y="count", hue="count", data=df3, dodge=False,
                          palette="BuGn", ax=axes[2]).legend_.remove()
    del df2, df3
    gc.collect()
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
    plt.savefig(filepath + 'top_20_trigrams.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait',  format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                 metadata=None)

    # jaccard similarity dataframe
    train['jaccard_raw'] = train.apply(lambda x: jaccard_distance(set(x[1]), set(x[2])), axis=1)
    train['jaccard_processed'] = train.apply(lambda x: jaccard_distance(set(x[4]), set(x[5])), axis=1)
    train['raw_text_count_diff'] = train.apply(lambda x: len(word_tokenize(x[1]))-len(word_tokenize(x[2])), axis=1)
    train['processed_text_count_diff'] = train.apply(lambda x: len(word_tokenize(x[4])) - len(word_tokenize(x[5])), axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), tight_layout=True)
    axes[0, 0].set_title('Raw Text Jaccard Distribution')
    axes[0, 0] = sns.distplot(train['jaccard_raw'], bins=20, kde=False, ax=axes[0, 0])
    axes[0, 1].set_title('Raw Text Word Count Difference')
    axes[0, 1] = sns.distplot(train['raw_text_count_diff'], bins=20, kde=False, ax=axes[0, 1])
    axes[1, 0].set_title('Processed Text Jaccard Distribution')
    axes[1, 0] = sns.distplot(train['jaccard_raw'], bins=20, kde=False, ax=axes[1, 0])
    axes[1, 1].set_title('Processed Text Word Count Difference')
    axes[1, 1] = sns.distplot(train['processed_text_count_diff'], bins=20, kde=False, ax=axes[1, 1])
    plt.savefig(filepath + 'text_selected_text_similarity.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait',  format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                 metadata=None)


    # plt.show()


    print(train.head())
    print(test.head())
    return


def roBERTa(train):
    # max tweet length
    MAX_LEN = 128
    PATH = 'data/roberta_base/'
    printm('Building models ... ', 'green', TERMCOLOR)
    tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file=PATH+'roberta-base-vocab.json',
                                                 merges_file=PATH+'roberta-base-merges.txt',
                                                 lowercase=True,
                                                 add_prefix_space=True)

    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    train = pd.read_csv('data/train.csv').fillna('')
    # train = train.sample(frac=0.01, replace=True)

    ct = train.shape[0]
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
    start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(train.shape[0]):

        # FIND OVERLAP
        text1 = " " + " ".join(train.loc[k, 'text'].split())
        text2 = " ".join(train.loc[k, 'selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ': chars[idx - 1] = 1
        enc = tokenizer.encode(text1)


        # ID_OFFSETS
        offsets = []
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        # START END TOKENS
        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm > 0: toks.append(i)
        s_tok = sentiment_id[train.loc[k, 'sentiment']]
        input_ids[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask[k, :len(enc.ids) + 5] = 1
        if len(toks) > 0:
            start_tokens[k, toks[0] + 1] = 1
            end_tokens[k, toks[-1] + 1] = 1



    test = pd.read_csv('data/test.csv').fillna('')

    ct = test.shape[0]
    input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(test.shape[0]):
        # INPUT_IDS
        text1 = " " + " ".join(test.loc[k, 'text'].split())
        enc = tokenizer.encode(text1)
        s_tok = sentiment_id[test.loc[k, 'sentiment']]
        input_ids_t[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask_t[k, :len(enc.ids) + 5] = 1

    def build_model():
        ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

        config = RobertaConfig.from_pretrained(PATH + 'roberta-base-config.json')
        bert_model = TFRobertaModel.from_pretrained(PATH + 'roberta-base-tf_model.h5', config=config)
        x = bert_model(ids, attention_mask=att, token_type_ids=tok)

        x1 = tf.keras.layers.Dropout(0.1)(x[0])
        x1 = tf.keras.layers.Conv1D(1, 1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0])
        x2 = tf.keras.layers.Conv1D(1, 1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        return model

    def jaccard(str1, str2):
        a = set(str1.lower().split())
        b = set(str2.lower().split())
        if (len(a) == 0) & (len(b) == 0): return 0.5
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    jac = []
    VER = 'v0'
    DISPLAY = 1  # USE display=1 FOR INTERACTIVE
    oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    for fold, (idxT, idxV) in tqdm(enumerate(skf.split(input_ids, train.sentiment.values))):

        print('\n')
        print('#' * 25)
        print('### FOLD %i' % (fold + 1))
        print('#' * 25)

        K.clear_session()
        model = build_model()

        sv = tf.keras.callbacks.ModelCheckpoint(
            '%s-roberta-%i.h5' % (VER, fold), monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')

        model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]],
                  [start_tokens[idxT,], end_tokens[idxT,]],
                  epochs=3, batch_size=32, verbose=DISPLAY, callbacks=[sv],
                  validation_data=([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]],
                                   [start_tokens[idxV,], end_tokens[idxV,]]))

        print('Loading model...')
        model.load_weights('%s-roberta-%i.h5' % (VER, fold))

        print('Predicting OOF...')
        oof_start[idxV,], oof_end[idxV,] = model.predict(
            [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY)

        print('Predicting Test...')
        preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
        preds_start += preds[0] / skf.n_splits
        preds_end += preds[1] / skf.n_splits

        # DISPLAY FOLD JACCARD
        all = []
        for k in idxV:
            a = np.argmax(oof_start[k,])
            b = np.argmax(oof_end[k,])
            if a > b:
                st = train.loc[k, 'text']  # IMPROVE CV/LB with better choice here
            else:
                text1 = " " + " ".join(train.loc[k, 'text'].split())
                enc = tokenizer.encode(text1)
                st = tokenizer.decode(enc.ids[a - 1:b])
            all.append(jaccard(st, train.loc[k, 'selected_text']))
        jac.append(np.mean(all))
        print('>>>> FOLD %i Jaccard =' % (fold + 1), np.mean(all))
        print()

    all = []
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a > b:
            st = test.loc[k, 'text']
        else:
            text1 = " " + " ".join(test.loc[k, 'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a - 1:b])
        all.append(st)

    test['selected_text'] = all
    test[['textID', 'selected_text']].to_csv('submission.csv', index=False)
    pd.set_option('max_colwidth', 60)
    print(test.sample(25))

    return


def preprocess(train, test):

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Tokenize our training data
    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
                          split=' ', char_level=False, oov_token='<UNK>', document_count=0)

    tokenizer.fit_on_texts(train.text)

    # Get our training data word index
    word_index = tokenizer.word_index
    nunique_train = np.max([int(v) for k,v in word_index.items()])
    print('\nTotal number of unique words in train.text is', nunique_train)

    # Encode training data sentences into sequences
    train_sequences = np.array(tokenizer.texts_to_sequences(train.text))

    # Get max training sequence length
    maxlen = max([len(x) for x in train_sequences])

    # Pad the training sequences
    pad_type = 'post'
    trunc_type = 'post'
    train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen) / nunique_train

    # Output the results of our work
    print("\nPadded training sequences:\n", train_padded)
    print("\nPadded training shape:", train_padded.shape)
    print("Training sequences data type:", type(train_sequences))
    print("Padded Training sequences data type:", type(train_padded))

    test_sequences = np.array(tokenizer.texts_to_sequences(test.text))
    test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen) / nunique_train

    print("\nPadded testing sequences:\n", test_padded)
    print("\nPadded testing shape:", test_padded.shape)
    print("Test sequences data type:", type(train_sequences))
    print("Padded Test sequences data type:", type(train_padded))

    def f(x):
        if x == 'neutral':
            return 0
        if x == 'negative':
            return 0
        if x == 'positive':
            return 1

    Y_train = np.array(train.sentiment.apply(f))
    Y_train = Y_train.reshape(1, len(Y_train))
    X_train = train_padded.T

    Y_test = np.array(test.sentiment.apply(f))
    Y_test = Y_test.reshape(1, len(Y_test))
    X_test = test_padded.T

    print('\n')
    print('X_train, Y_train shape is ', np.shape(X_train), np.shape(Y_train))
    print('X_test, Y_test shape is ', np.shape(X_test), np.shape(Y_test))
    print('\n')

    return X_train, Y_train, X_test, Y_test


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def tanh(Z):

    """
    Compute the tanh of z

    Arguments:
    :param z -- A scalar or numpy array of any size.

    Return:
    :param A -- tanh(z). Output of Z, same shape
    :param cache -- just a copy of Z with a new name to indicate its function, for efficient backpropagation
    """

    A = np.tanh(Z)
    cache = Z

    return A, cache


def tanh_backward(dA, cache):
    """
        Implement the backward propagation for a single tanh unit


        Arguments:
        :param dA -- post-activation gradient any size.
        :param cache -- cache of Z, a scalar or numpy array of any size. Was stored for efficient computation of
                    backward propagation

        Return:
        :param dZ -- Gradient of the cost function with respect to Z
        """

    Z = cache
    dZ = dA * (1 - np.tanh(Z)**2)

    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid(Z):

    """
    Compute the sigmoid of z

    Arguments:
    :param z -- A scalar or numpy array of any size.

    Return:
    :param A -- sigmoid(z). Output of Z, same shape
    :param cache -- just a copy of Z with a new name to indicate its function, for efficient backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):

    """
    Implement the backward propagation for a single sigmoid unit


    Arguments:
    :param dA -- post-activation gradient any size.
    :param cache -- cache of Z, a scalar or numpy array of any size. Was stored for efficient computation of
                backward propagation

    Return:
    :param dZ -- Gradient of the cost function with respect to Z
    """

    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert(dZ.shape == Z.shape)

    return dZ


def relu(Z):

    """
        Compute the ReLu of z

        Arguments:
        :param Z -- A scalar or numpy array of any size. Output of the linear layer

        Return:
        :param A -- post-activation parameter, of the same shape as Z
        :param cache -- a dictionary containing "A", stored for efficient computation of backprop pass
    """

    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)

    cache = Z

    return A, cache


def relu_backward(dA, cache):

    """
    Implements the backward propagation for a single ReLu unit

    Arguments:
    :param dA -- post-activation gradient any size.
    :param cache -- cache of Z, a scalar or numpy array of any size. Was stored for efficient computation of
                backward propagation

    Return:
    :param dZ -- Gradient of the cost function with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    assert(dZ.shape == Z.shape)

    return dZ


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    ### START CODE HERE ### (≈ 1 line of code)
    Z = np.dot(W, A) + b
    ### END CODE HERE ###

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
        ### END CODE HERE ###

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###

    elif activation == "tanh":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
        ### END CODE HERE ###

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    logprobs = 1 / Y.shape[1] * (np.multiply(np.log(AL), Y) + np.multiply(np.log(1 - AL), 1 - Y))
    cost = - np.sum(logprobs)
    ### END CODE HERE ###

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    return cost


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    ### END CODE HERE ###

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    elif activation == "tanh":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # derivative of cost with respect to AL
    ### END CODE HERE ###

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")
    ### END CODE HERE ###

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.075, num_iterations = 3000, print_cost=False):

    '''
    The general architecture is as follows:

    1. Initialize parameters / Define hyperparameters
    2. Loop for num_iterations:
        a. Forward propagation
        b. Compute cost function
        c. Backward propagation
        d. Update parameters (using parameters, and grads from backprop)
    4  . Use trained parameters to predict labels

    Programmatically, this function implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (max tweet length, number of examples)
    Y -- true "label" vector (containing 0 if tweet sentiment is negative or neutral, 1 if positive), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    '''

    ### CONSTANTS ###
    layers_dims = [X.shape[0], 20, 7, 5, 1]  # 4-layer model

    np.random.seed(1)
    costs = []  # keep track of cost

    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(X, y, parameters):
    """
    This function is used to predict the results of the L-layer neural network

    Arguments:
    :param X -- dataset of examples you would like to label
    :param parameters -- parameters of the trained model

    Returns:
    :param p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1, m))

    #forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == y)/m)))

    return p


if __name__ == '__main__':
    # build dataframes
    start_time = time.time()
    printm('Importing data ... ', 'green', TERMCOLOR)

    DATA_FOLDER = 'data/'

    files_train = [os.path.join(DATA_FOLDER, 'train.csv'),
                   os.path.join(DATA_FOLDER, 'test.csv'),
                   os.path.join(DATA_FOLDER, 'sample_submission.csv')]

    df_train, df_test, sample = build_dfs(['csv' for file in files_train], [file for file in files_train])

    # eda(train, test, DATA_FOLDER, sample)

    # roBERTa(train)

    X_train, Y_train, X_test, Y_test = preprocess(df_train, df_test)
    layers_dims = [X_train.shape[0], 10, 5, 1]  # 4-layer model
    parameters = L_layer_model(X_train, Y_train, layers_dims, num_iterations=1000, print_cost=True)

    pred_train = predict(X_train, Y_train, parameters)
    pred_test  = predict(X_test, Y_test, parameters)


