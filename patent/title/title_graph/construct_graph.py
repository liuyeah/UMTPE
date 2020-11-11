import numpy as np
import pickle
import json
from tqdm import tqdm
from scipy.spatial.distance import cosine, pdist, squareform


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def build_graph(super_sapn_file, phrase_embedding_file, output_graph_file, output_graph_embedding_file):
    graph_list = []
    graph_embedding_list = []
    with open(super_sapn_file, 'r', encoding='utf-8') as f_in:
        super_span = json.load(f_in)
    phrase_embedding = load_obj(phrase_embedding_file)
    for spans in tqdm(super_span, total=20):
        # construct the basic graph
        graph = {}
        graph_embedding = []
        graph_number = 0
        temp_list = spans['superspan']
        count = 0
        for j in temp_list:
            temp_node = j.lower()
            if temp_node != '' and temp_node not in graph:
                # 构建出来graph，key为phrase，value为序号（从0开始）
                # 构建graph时候，希望筛选掉那些embedding为全0的phrase
                if temp_node in phrase_embedding:
                    if not (phrase_embedding[temp_node] == 0.0).all():
                        graph[temp_node] = count
                        count = count + 1
                        graph_embedding.append(phrase_embedding[temp_node])
                    else:
                        print("zero embedding phrase: " + temp_node)
                else:
                    # 打印找不到embedding的节点
                    print('cannot find graph : ' + str(graph_number) + ', node: ' + temp_node)

            graph_number = graph_number + 1

        graph_list.append(graph)
        graph_embedding_list.append(graph_embedding)

    save_obj(graph_list, output_graph_file)
    save_obj(graph_embedding_list, output_graph_embedding_file)

    # temp_similarity = pdist(graph_embedding, metric='cosine')
    # similarity_matrix = squareform(temp_similarity)





if __name__ == '__main__':
    super_sapn_file = 'patent/title/title_candidate/title_candidate_synthesis.json'
    phrase_embedding_file = 'patent/title/title_embedding/cpc_title_phrase_embedding.pkl'
    output_graph_file = 'patent/title/title_graph/title_graph_list.pkl'
    output_graph_embedding_file = 'patent/title/title_graph/title_graph_embedding_list.pkl'
    build_graph(super_sapn_file, phrase_embedding_file, output_graph_file, output_graph_embedding_file)















