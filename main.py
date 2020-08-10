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


def eda(train, test, *args):
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

    # print(len(set(train.textID.values).union(set(test.textID.values))))
    # printm('Is test.textID subset of train.textID : ',set(test.textID.values).issubset(set(train.textID.values)))
    # printm('How many textIDs are common : ',len(set(test.textID.values).intersection(set(train.textID.values))))

    # basic text preprocessing in parallel
    train['text_processed'] = parallelize_dataframe(train['text'], basic_preprocess_raw_text)
    train['selected_text_processed'] = parallelize_dataframe(train['selected_text'], basic_preprocess_raw_text)
    # train['text_processed'] = basic_preprocess_raw_text(train['text'])


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

    # plt.show()
    # print(trlain.shape, train.textID.nunique())
    # print(train.head(100))
    # txt = train['text_processed'].values
    # for item in txt:
    #     if "`" in item:
    #         print(item)

    # jaccard similarity dataframe
    train['jaccard_raw'] = train.apply(lambda x: jaccard_distance(set(x[1]), set(x[2])), axis=1)
    train['jaccard_processed'] = train.apply(lambda x: jaccard_distance(set(x[4]), set(x[5])), axis=1)
    train['raw_text_count_diff'] = train.apply(lambda x: len(word_tokenize(x[1]))-len(word_tokenize(x[2])), axis=1)
    train['processed_text_count_diff'] = train.apply(lambda x: len(word_tokenize(x[4])) - len(word_tokenize(x[5])), axis=1)
    print(train.head(100))



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


if __name__ == '__main__':
    # build dataframes
    start_time = time.time()
    printm('Importing data ... ', 'green', TERMCOLOR)

    DATA_FOLDER = 'data/'

    files_train = [os.path.join(DATA_FOLDER, 'train.csv'),
                   os.path.join(DATA_FOLDER, 'test.csv'),
                   os.path.join(DATA_FOLDER, 'sample_submission.csv')]

    train, test, sample = build_dfs(['csv' for file in files_train], [file for file in files_train])

    # eda(train, test, sample)

    roBERTa(train)



