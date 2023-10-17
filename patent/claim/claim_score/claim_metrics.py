import numpy as np
from tqdm import tqdm
import pickle
import ipdb
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import pdist, squareform, cosine


BASIC_THRESHOLD = 0.5


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def most_similar(x, phrase_embedding_idx,  phrase_embedding_distance_matrix, topN=None):
    if x not in phrase_embedding_idx:
        # 如果没有发现，则令距离都是1
        print('cannot find word: ' + str(x))
        ipdb.set_trace()
        if topN:
            return [1]*topN
        else:
            return []
    if len(phrase_embedding_idx) == 1:
        if topN:
            return [1]*topN
        else:
            return []
    idx = phrase_embedding_idx[x]
    output_list = []
    # 正序排列, 从小到大
    distance_list = sorted(phrase_embedding_distance_matrix[idx], reverse=False)
    if topN:
        if topN >= len(distance_list):
            print('There is an error in most similar function!!!')
            ipdb.set_trace()
        for i in range(1, topN+1):
            output_list.append(distance_list[i])
    else:
        for i in range(1, len(distance_list)):
            output_list.append(distance_list[i])
    # 返回的是topN相似的词的分数列表
    return output_list


def scalability(node, phrase_embedding_idx, phrase_embedding_distance_matrix):
    # 扩展性，返回距离小于阈值的个数/节点总个数
    # neighbor_word2sim 表示的距离最近的topN个词的距离的list
    if len(phrase_embedding_idx) == 1:
        return 0
    neighbor_word2sim = [sim for sim in
                         most_similar(node, phrase_embedding_idx,  phrase_embedding_distance_matrix)
                         if sim <= BASIC_THRESHOLD]
    return float(len(neighbor_word2sim))/float(len(phrase_embedding_idx)-1)


def independence(node, phrase_embedding_idx, phrase_embedding_distance_matrix):
    # 独立性
    if len(phrase_embedding_idx) == 1:
        return 1
    neighbor_word2sim = most_similar(node, phrase_embedding_idx,  phrase_embedding_distance_matrix, topN=1)
    # 当前graph中最近的一个phrase的distance
    return neighbor_word2sim[0]


def self_score(node):
    # 长度是2-4给1分，5给0.5分，其余给0分
    if len(node.split(' ')) == 2 or len(node.split(' ')) == 3 or len(node.split(' ')) == 4:
        return 1.0
    elif len(node.split(' ')) == 5:
        return 0.5
    else:
        return 0.0


def occurance(node, sentence_text):
    # node是一个节点的文本，document是原始的文本
    # 计算在这个document中出现的次数
    count = 0
    document_list = sentence_text.split(' ')
    temp_node = node.split(' ')
    # 一些补充规则
    if len(document_list) < len(temp_node):
        # print('cannot find node: ' + node + '\t in:')
        # print(sentence_text)
        return 0
    elif len(document_list) == len(temp_node) and document_list[0] != temp_node[0]:
        # print('cannot find node: ' + node + '\t in:')
        # print(sentence_text)
        return 0

    for i in range(len(document_list)-len(temp_node)+1):
        # len(document_list) = 5, len(temp_node) = 2
        signal = 0
        for j in range(len(temp_node)):
            if document_list[i+j] != temp_node[j]:
                # 如果signal==1，代表不匹配
                signal = 1
                break
        if signal == 0:
            count = count + 1
    # if count == 0:
    #     print('cannot find node: ' + node + '\t in:')
    #     print(sentence_text)
    return count


def influence(node, sentence_text):
    influence_count = 0
    sentence_list = sent_tokenize(sentence_text)
    for item in sentence_list:
        if occurance(node, item) == 0:
            continue
        else:
            influence_count = influence_count + 1
    if influence_count == 0:
        print('cannot find node: ' + node + '\t in:')
        print(sentence_text)
        # ipdb.set_trace()
    return influence_count



def centroid_score(node_embedding, centroid_list):
    similarity_list = []
    zero_list = np.zeros(500, dtype='float32')
    for centroid in centroid_list:
        if (centroid == node_embedding).all():
            similarity_list.append(1.0)
        elif (centroid == zero_list).all() or (node_embedding == zero_list).all():
            similarity_list.append(0.0)
        else:
            similarity_list.append(1 - cosine(node_embedding, centroid))
    return np.max(similarity_list)



def node_score(node, phrase_embedding_idx, phrase_embedding_distance_matrix,
               lower_sentence_text, node_embedding, centroid_list):

    final_score = [independence(node, phrase_embedding_idx, phrase_embedding_distance_matrix),
                   scalability(node, phrase_embedding_idx, phrase_embedding_distance_matrix),
                   self_score(node),
                   influence(node, lower_sentence_text),
                   centroid_score(node_embedding, centroid_list)]
    return final_score


def calculate_score(graph_list_file, phrase_embedding_matrix_file, lower_text_file,
                    phrase_embedding_file, centroid_file, output_score_file):
    lower_text =[]
    graph_list = load_obj(graph_list_file)
    phrase_embedding_matrix = load_obj(phrase_embedding_matrix_file)
    with open(lower_text_file, 'r', encoding='utf-8') as f_in:
        for text_i in f_in:
            text_i = text_i.strip('\n')
            lower_text.append(text_i)
    phrase_embedding = load_obj(phrase_embedding_file)
    centroid_list = load_obj(centroid_file)
    output_score_list = []
    for item in tqdm(range(20)):
        score = {}
        # ipdb.set_trace()
        if len(graph_list[item]) == 0:
            output_score_list.append(score)
            continue
        temp_matrix = pdist(phrase_embedding_matrix[item], metric='cosine')
        phrase_embedding_distance_matrix = squareform(temp_matrix)
        for node in graph_list[item]:
            temp_score = node_score(node, graph_list[item], phrase_embedding_distance_matrix,
                                    lower_text[item], phrase_embedding[node], centroid_list)
            score[node] = temp_score
        output_score_list.append(score)
    save_obj(output_score_list, output_score_file)


if __name__ == '__main__':
    graph_list_file = 'patent/claim/claim_graph/claim_graph_list.pkl'
    phrase_embedding_matrix_file = 'patent/claim/claim_graph/claim_graph_embedding_list.pkl'
    lower_text_file = 'example_data/example_claim/claim.txt'
    output_score_file = 'patent/claim/claim_score/claim_influence_phrase_score.pkl'
    phrase_embedding_file = 'patent/claim/claim_embedding/cpc_title_abstract_claim_phrase_embedding.pkl'
    centroid_file = 'patent/abstract/abstract_clustering/abstract_influence_phrase_centroids.pkl'
    calculate_score(graph_list_file, phrase_embedding_matrix_file, lower_text_file,
                    phrase_embedding_file, centroid_file, output_score_file)
