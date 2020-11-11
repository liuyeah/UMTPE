import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
logging.basicConfig(level=logging.DEBUG)

def word_propress(cpc_path, title_path, abstract_path, claim_path, output_path):
    document = []
    with open(cpc_path, "r", encoding='utf-8') as data_read_1:
        for i in data_read_1:
            str_i = i.strip('\n')
            output_i = str_i.lower()
            document.append(output_i)

    with open(title_path, "r", encoding='utf-8') as data_read_2:
        for i in data_read_2:
            str_i = i.strip('\n')
            output_i = str_i.lower()
            document.append(output_i)

    with open(abstract_path, "r", encoding='utf-8') as data_read_3:
        for i in data_read_3:
            str_i = i.strip('\n')
            output_i = str_i.lower()
            document.append(output_i)

    with open(claim_path, "r", encoding='utf-8') as data_read_4:
        for i in data_read_4:
            str_i = i.strip('\n')
            output_i = str_i.lower()
            document.append(output_i)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for output_item in document:
            f_out.write(output_item)
            f_out.write('\n')



def my_embedding(text_path, model_path):
    sentences = LineSentence(text_path)
    # these parameter is set for small example data, if you need to run on large datasets, you need set 'min_count=5'
    skip_gram = gensim.models.Word2Vec(sentences, min_count=1, sg=1, hs=1, size=500, iter=10)
    skip_gram.save(model_path)

def my_load_embedding():
    my_word = Word2Vec.load('embedding.bin')

if __name__ == '__main__':
    cpc_path = 'example_data/example_cpc/cpc.txt'
    title_path = 'example_data/example_title/title.txt'
    abstract_path = 'example_data/example_abstract/abstract.txt'
    claim_path = 'example_data/example_claim/claim.txt'
    output_path = 'word_embedding/data2embedding.txt'
    text_path = 'word_embedding/data2embedding.txt'
    model_path = 'word_embedding/cpc_title_abstract_claim_word_embedding.bin'
    word_propress(cpc_path, title_path, abstract_path, claim_path, output_path)
    my_embedding(text_path, model_path)
