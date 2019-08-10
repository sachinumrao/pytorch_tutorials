# This file take dataframe containing reviews and corresponding sentiments.
# This will output indexed train and test dataframe, embedding matrix with GloVe ..
# embeddings and token2idx dictionary saved as pickle file.

# import dependencies
import time
import pickle as pkl
import itertools
import collections
import pandas as pd
import numpy as np

# nlp specific libraries
import contractions
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet


# find stop words
nlp = spacy.load('en')
stops = nlp.Defaults.stop_words
retain_words = ['always', 'nobody', 'cannot', 'none', 'never', 'no', 'not']

for j in retain_words:
    stops.discard(j)
    
remove_chars = ['br', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '~', '`', '!', '@', '#', '$', '%', '^',
               '&', '*', '(', ')', '-', '_', '+', '=', '[', ']', '{', '}', '|', ':', ';', '<', '>', ',',
               '.', '?', ",", '"']

for j in remove_chars:
    stops.add(j)

def get_wordnet_pos(word):
    "Map pos tags to first character lemmatize function accepts"
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J" : wordnet.ADJ,
                "N" : wordnet.NOUN,
                "V" : wordnet.VERB,
                "R" : wordnet.ADV
                }
    
    return tag_dict.get(tag, wordnet.NOUN)

def regex_tokenizer(text, stops):
    
    # fix contractions
    print(type(text))
    print("Debug Mode On")
    text2 = contractions.fix(text)
    
    # tokennzer
    tokenizer = RegexpTokenizer(r'\w+')
    words1 = tokenizer.tokenize(text2)
    
    # remove numbers
    #words2 = [x for x in words1 if x.digit() == False]
    
    # convert to lowercase
    words3 = [x.lower() for x in words1]
    
    # remove stopwords
    words4 = [w for w in words3 if w not in stops]
    
    # use lemmatizer
    words5 = [wordnet_lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words4]
    
    return words5


def text_preprocessing_fn(df, x_col, max_seq_len=128, max_vocab_size=20000):
    
    data = df[[x_col]]
    print("Total Samples : ", df.shape[0])

    # parse and tokenize text data
    
    data['parse_text'] = data.apply(lambda x : regex_tokenizer(x, stops))
    print("Tokenization Completed ...")

    # build dictionary
    seq_list = data['parse_text'].tolist()
    big_list = list(itertools.chain.from_iterable(seq_list))

    # apply max_vocab_size
    # make collection of big_list with duplicate token entries
    coll = collections.Counter(big_list)

    # find most common key,value pairs
    ms_coll = coll.most_common(max_vocab_size)

    # convert it to dictionary
    token2idx = dict(ms_coll)

    # add support for padding and unknown tokens
    token2idx['<pad>'] = max(token2idx.values())+1
    token2idx['<unk>'] = max(token2idx.values())+1

    token2idx = token2idx
    print("Dictionary Completed ...")

    # cut long sentences short
    data['parse_text_short'] = data['parse_text'].apply(
        lambda x : x[:min(max_seq_len, len(x))]
        )
    print("Sentence Normalization Completed ...")

    # convert tokens to indicies
    data['tokenized'] = data['parse_text_short'].apply(

        lambda x: [token2idx.get(j, token2idx['<unk>']) for j in x] 

        # another but less efficient way to do the same.
        # lambda x : [token2idx[j] if j in token2idx.keys()
        #                             else token2idx['<unk>'] for j in x]
        )
    print("Index Conversion Completed ...")

    # add padding to make all samples of equal length
    data['tok_pad'] = data['tokenized'].apply(
        lambda x : x + [token2idx['<pad>']]*(max_seq_len - len(x))
        )
    print("Padding Completed ...")

    
    return data, token2idx

def glove_embedding_generator(embedding_file_path, token2idx, vocab_size, embed_dim=300):
    seed = 99
    np.random.seed(seed)
    embed_mat = np.random.rand(vocab_size, embed_dim)

    words_found = 0
    vocab = token2idx.keys()

    t3 = time.time()
    with open(embedding_file_path, 'rb') as embed_file:
        for line in embed_file:
            l = line
            l = l.decode().split()
            word = l[0]
            vec = np.array(l[1:]).astype(np.float)
            
            # check if word is in vocab
            if word in vocab:
                embed_mat[token2idx['word']] = vec
                words_found += 1
                
    print("Words found : ", words_found)
    t4 = time.time()
    print("Time Taken in Embedding Generation : ", t4-t3)
    return embed_mat





if __name__ == "__main__":

    # load IMDB movie reviews data from disk
    df = pd.read_csv('~/Data/IMDB_Dataset.csv', nrows=1)

    # convert positive/negative review into categorical variable
    df['y'] = df['sentiment'].apply(lambda x : 0 if x == 'negative' else 1)

    # pre-process the text
    MAX_SEQ_LEN = 128
    MAX_VOCAB_SIZE  = 20000
    wordnet_lemmatizer = WordNetLemmatizer()
    indexed_data, token2idx = text_preprocessing_fn(df, x_col='review', max_seq_len = MAX_SEQ_LEN, 
                                                    max_vocab_size = MAX_VOCAB_SIZE)

    print("Length of Dictionary : ", len(token2idx))

    # load word embeddings
    vocab_size = len(token2idx)
    EMBED_DIM = 300
    embedding_file_path = '~/Data/glove.6B/glove.6B.300d.txt'
    embed_mat = glove_embedding_generator(embedding_file_path, token2idx, vocab_size=vocab_size,
                                            embed_dim=EMBED_DIM)


    # train test split the dataset
    TRAIN_SIZE = 0.8
    mask = np.random.rand(len(df)) < TRAIN_SIZE
    train_df = indexed_data[mask]
    test_df = indexed_data[~mask]

    # save the preprocessed data into pickles

    # save embedding matrix
    path = '~/Data/IMDB/'
    filename1 = path + 'IMDB_Embed'
    fileObject1 = open(filename1, 'wb')
    pkl.dump(embed_mat, fileObject1)
    fileObject1.close()

    # save token dictionary
    filename2 = path + 'IMDB_TOKEN'
    fileObject2 = open(filename2, 'wb')
    pkl.dump(token2idx, fileObject2)
    fileObject2.close()

    # save train_data_frame dictionary
    filename3 = path + 'IMDB_TRAIN'
    fileObject3 = open(filename3, 'wb')
    pkl.dump(train_df, fileObject3)
    fileObject3.close()

    # save test_data_frame dictionary
    filename4 = path + 'IMDB_TEST'
    fileObject4 = open(filename4, 'wb')
    pkl.dump(test_df, fileObject4)
    fileObject4.close()


    # code for loading pickle
    # load = True
    # if load:
    #     fileObject2 = open(fileName, 'wb')
    #     embed_mat = pkl.load(fileObject2)
    #     fileObject2.close()