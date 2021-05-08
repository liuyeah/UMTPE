import json
from tqdm import tqdm
from gensim import corpora
import pickle
from gensim.models import LsiModel
from gensim import models
from gensim import similarities
import numpy as np
# import logging
# logging.basicConfig(level=logging.DEBUG)


def topk_score(query_document_similarity, id, topk):
    # 如果在topk中，则score为1，否则为0
    count = 0
    topk_list = []
    score_list = []
    for document_number, score in sorted(enumerate(query_document_similarity), key=lambda x: x[1], reverse=True):
        count = count + 1
        if count <= topk:
            topk_list.append(document_number)
            score_list.append(score)
        else:
            break
    if id in topk_list:
        if (np.array(score_list) == 0.0).all():
            return 0
        else:
            return 1
    else:
        return 0


def lsi_score(document_list, query_list, topk):
    result = []

    dictionary = corpora.Dictionary(document_list)
    bow_corpus = [dictionary.doc2bow(text) for text in document_list]

    lsi = LsiModel(corpus=bow_corpus, id2word=dictionary, num_topics=500)

    index = similarities.MatrixSimilarity(lsi[bow_corpus])

    for idx in range(len(query_list)):
        temp_result = []
        # 表示对于每一条文档的词分别处理
        if query_list[idx] == []:
            continue
        for word_list in query_list[idx]:
            query_bow = dictionary.doc2bow(word_list)
            sims = index[lsi[query_bow]]
            temp_result.append(topk_score(sims, idx, topk=topk))
        result.append(temp_result)

    return result

def ire_score(predicted_list, reference_list_file, labeled_100_sample_file, other_900_sample_file):
    document_list = []
    query_list = []
    with open(reference_list_file, 'r', encoding='utf-8') as f_reference:
        reference_list = json.load(f_reference)
    with open(labeled_100_sample_file, 'r', encoding='utf-8') as f_label:
        for item in f_label:
            item = item.strip('\n')
            item_list = item.split(' ')
            document_list.append(item_list)
    with open(other_900_sample_file, 'r', encoding='utf-8') as f_unlabel:
        for item in f_unlabel:
            item = item.strip('\n')
            item_list = item.split(' ')
            document_list.append(item_list)
    for item in predicted_list:
        temp_query = []
        for key in item:
            key_list = key.split(' ')
            temp_query.append(key_list)
        query_list.append(temp_query)
        
    result = lsi_score(document_list, query_list, topk=10)
    final_result = []
    original_result = []
    for idx in range(len(result)):
        temp_score = np.sum(result[idx])/len(result[idx])
        temp_reference_len = len(reference_list[idx])
        if len(result[idx]) >= temp_reference_len:
            temp_bp = 1
        else:
            temp_bp = np.exp(1-temp_reference_len/len(result[idx]))
        penalty_score = temp_bp * temp_score
        original_result.append(temp_score)
        final_result.append(penalty_score)
    return np.mean(original_result), np.mean(final_result)


if __name__ == '__main__':
    predicted_file = 'result/sample_output.json'
    referenece_title_file = 'evaluation/H/title/title_stemmed_annotation.json'
    referenece_abstract_file = 'evaluation/H/abstract/abstract_stemmed_annotation.json'
    referenece_claim_file = 'evaluation/H/claim/claim_stemmed_annotation.json'
    title_100_sample_file = 'evaluation/H/title/title_sample.txt'
    title_900_sample_file = 'evaluation/H/title/title_900_sample.txt'
    abstract_100_sample_file = 'evaluation/H/abstract/abstract_sample.txt'
    abstract_900_sample_file = 'evaluation/H/abstract/abstract_900_sample.txt'
    claim_100_sample_file = 'evaluation/H/claim/claim_sample.txt'
    claim_900_sample_file = 'evaluation/H/claim/claim_900_sample.txt'
    output_file = 'evaluation/H/IRE_result.json'

    # 需要对预测得到的 phrase 进行 stem 处理
    with open(predicted_file, 'r', encoding='utf-8') as f_predicted:
        predicted_list = json.load(f_predicted)
    predicted_title = []
    predicted_abstract = []
    predicted_claim = []
    for item in predicted_list:
        predicted_title.append(item['title'])
        predicted_abstract.append(item['abstract'])
        predicted_claim.append(item['claim'])
    final_ire_score = {'title': {}, 'abstract': {}, 'claim': {}}
    final_ire_score['title']['original_ire'], final_ire_score['title']['ire'] = ire_score(predicted_title, referenece_title_file, title_100_sample_file, title_900_sample_file)
    final_ire_score['abstract']['original_ire'], final_ire_score['abstract']['ire'] = ire_score(predicted_abstract, referenece_abstract_file, abstract_100_sample_file, abstract_900_sample_file)
    final_ire_score['claim']['original_ire'], final_ire_score['claim']['ire'] = ire_score(predicted_claim, referenece_claim_file, claim_100_sample_file, claim_900_sample_file)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(final_ire_score, f_out)
    print(final_ire_score)
    