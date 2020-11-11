import pickle
import json
import hdbscan
from tqdm import tqdm
import ipdb
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import re
import logging
logging.basicConfig(level=logging.DEBUG)


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def title_centroid(title_list_path, cpc_title_phrase_embedding_file, title_path, output_file):
    # ranked_phrase = load_obj(ranked_phrase_file)
    phrase_embedding = load_obj(cpc_title_phrase_embedding_file)

    title_sen = []
    title_result = []
    count = 0
    with open(title_path, 'r', encoding='utf-8') as f_title:
        for item in f_title:
            item = item.strip('\n')
            temp_len = len(sent_tokenize(item))
            title_sen.append(temp_len)
    with open(title_list_path, 'r', encoding='utf-8') as f_title:
        for item in f_title:
            item = item.strip('\n')
            item_dic = json.loads(item)
            temp_result = []
            for k in item_dic:
                temp_result.append(k)
            temp_result = temp_result[:2*title_sen[count]]
            title_result.append(temp_result)
            count = count + 1


    process_data = []
    output_data = {}
    centroids = []
    zero_count = 0
    for item in tqdm(title_result, total=20):
        # item 表示一个字典
        if len(item) == 0:
            continue
        for key_phrase in item:
            if key_phrase not in phrase_embedding:
                print('There is an error: cannot find: ' + key_phrase)
                ipdb.set_trace()
            else:
                if not (phrase_embedding[key_phrase] == 0.0).all():
                    process_data.append(phrase_embedding[key_phrase])
                else:
                    print('count: ' + str(zero_count) + 'zero embedding phrase: ' + key_phrase)
                    zero_count = zero_count + 1
            break

    cluster_er = hdbscan.HDBSCAN(min_cluster_size=100)
    cluster_labels = cluster_er.fit_predict(np.array(process_data))
    for item in range(len(cluster_labels)):
        if cluster_labels[item] not in output_data:
            output_data[cluster_labels[item]] = []
        output_data[cluster_labels[item]].append(process_data[item])
    for key in output_data:
        centroids.append(np.mean(output_data[key], axis=0))
    # ipdb.set_trace()
    print('number of centroids: ' + str(len(centroids)))
    save_obj(centroids, output_file)


if __name__ == '__main__':
    title_list_path = 'patent/title/title_rank/ranked_title_influence_phrase_score_text.json'
    cpc_title_phrase_embedding_file = 'patent/title/title_embedding/cpc_title_phrase_embedding.pkl'
    output_file = 'patent/title/title_clustering/title_influence_phrase_centroids.pkl'
    title_path = 'example_data/example_title/title.txt'
    title_centroid(title_list_path, cpc_title_phrase_embedding_file, title_path, output_file)
