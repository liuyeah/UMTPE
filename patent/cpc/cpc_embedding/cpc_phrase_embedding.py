import numpy as np
import gensim
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import ipdb
import pickle
import logging
logging.basicConfig(level=logging.DEBUG)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def cpc_phrase_embedding(word_embedding_file, cpc_phrase_file, output_file):
    non_find_number = 0
    phrase_embedding = {}
    # ipdb.set_trace()
    word_embedding = Word2Vec.load(word_embedding_file)
    # ipdb.set_trace()
    with open(cpc_phrase_file, 'r', encoding='utf-8') as f_in:
        for item in f_in:
            item = item.strip('\n')
            temp_phrase = item.lower()
            temp_phrase_list = temp_phrase.split(' ')
            if temp_phrase != '' and temp_phrase not in phrase_embedding:
                temp_phrase_embedding = []
                for temp_phrase_item in temp_phrase_list:
                    if temp_phrase_item not in word_embedding.wv:
                        print('count : ' + str(non_find_number) + '\tThere is an error, cannot find: ' + temp_phrase_item)
                        non_find_number = non_find_number + 1
                        # ipdb.set_trace()
                        continue
                    temp_phrase_item_embedding = word_embedding.wv[temp_phrase_item]
                    temp_phrase_embedding.append(temp_phrase_item_embedding)
                if temp_phrase_embedding == []:
                    final_phrase_embedding = np.zeros(500, dtype='float32')
                else:
                    final_phrase_embedding = np.array(np.mean(temp_phrase_embedding, axis=0))
                phrase_embedding[temp_phrase] = final_phrase_embedding
    save_obj(phrase_embedding, output_file)


if __name__ == '__main__':
    word_embedding_file = 'word_embedding/cpc_title_abstract_claim_word_embedding.bin'
    cpc_phrase_file = 'example_data/example_cpc/cpc_phrase.txt'
    output_file = 'patent/cpc/cpc_embedding/cpc_phrase_embedding.pkl'
    cpc_phrase_embedding(word_embedding_file, cpc_phrase_file, output_file)
