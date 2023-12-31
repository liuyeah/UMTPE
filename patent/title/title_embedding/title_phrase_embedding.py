import numpy as np
import gensim
import json
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import ipdb
import pickle
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.DEBUG)
import os
TOTAL_NUMBER = int(os.environ.get('TOTAL_NUMBER'))
EMBEDDING_SIZE = int(os.environ.get('EMBEDDING_SIZE'))


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def phrase_embedding(cpc_embedding_file, word_embedding_file, title_phrase_file, output_file):
    my_non_count = 0
    phrase_embedding = load_obj(cpc_embedding_file)
    word_embedding = Word2Vec.load(word_embedding_file)
    with open(title_phrase_file, 'r', encoding='utf-8') as f_in:
        title_phrase = json.load(f_in)
    for item in tqdm(title_phrase, total=TOTAL_NUMBER):
        for phrase_item in item['superspan']:
            temp_phrase = phrase_item.lower()
            temp_phrase_list = temp_phrase.split(' ')
            if temp_phrase != '' and temp_phrase not in phrase_embedding:
                temp_phrase_embedding = []
                for temp_phrase_item in temp_phrase_list:
                    if temp_phrase_item not in word_embedding.wv:
                        my_non_count = my_non_count + 1
                        # print('cannot find word: ' + temp_phrase_item)
                        # print('count: ' + str(my_non_count))
                        # ipdb.set_trace()
                        continue
                    temp_phrase_item_embedding = word_embedding.wv[temp_phrase_item]
                    temp_phrase_embedding.append(temp_phrase_item_embedding)
                if temp_phrase_embedding == []:
                    final_phrase_embedding = np.zeros(EMBEDDING_SIZE, dtype='float32')
                else:
                    final_phrase_embedding = np.array(np.mean(temp_phrase_embedding, axis=0))
                phrase_embedding[temp_phrase] = final_phrase_embedding
    save_obj(phrase_embedding, output_file)


if __name__ == '__main__':
    cpc_embedding_file = 'patent/cpc/cpc_embedding/cpc_phrase_embedding.pkl'
    word_embedding_file = 'word_embedding/cpc_title_abstract_claim_word_embedding.bin'
    title_phrase_file = 'patent/title/title_candidate/title_candidate_synthesis.json'
    output_file = 'patent/title/title_embedding/cpc_title_phrase_embedding.pkl'
    phrase_embedding(cpc_embedding_file, word_embedding_file, title_phrase_file, output_file)