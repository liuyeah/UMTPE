import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
from tqdm import tqdm
import pickle


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class NeGraph(object):
    def __init__(self, graph, value, similarity_matrix):
        self.graph = graph
        self.value = value
        self.similarity_matrix = similarity_matrix
        self.pr = []
        for item in range(len(self.graph)):
            self.pr.append(1/len(self.graph))

    def calculate_pr(self, d):
        modify_sum = []
        for modify_idx in range(len(self.similarity_matrix)):
            modify_sum.append(np.sum(self.similarity_matrix[modify_idx]))
        temp_pr = self.pr.copy()
        for item in self.graph:
            # 确定节点序号
            item_idx = self.graph[item]
            # 首先计算一个偏置值
            item_bias = (1 - d) * self.value[item_idx]
            modify_value = 0.0
            for temp_idx in range(len(self.similarity_matrix[item_idx])):
                if item_idx == temp_idx or modify_sum[temp_idx] == 0.0:
                    # 不计算自己
                    continue
                modify_value = modify_value + self.similarity_matrix[item_idx][temp_idx]/modify_sum[temp_idx] * \
                               temp_pr[temp_idx]
            if modify_value == 0.0:
                self.pr[item_idx] = item_bias
                continue
            self.pr[item_idx] = item_bias + d * self.value[item_idx] * modify_value

    def calculate_pr_times(self, d, times):
        for i in range(times):
            self.calculate_pr(d)

    def calculate_pr_converge(self, d, threshold):
        if len(self.graph) >= 2:
            count_number = 0
            changes = np.array([100.0] * len(self.graph))
            while (changes > threshold).any() and count_number <= 100:
                old_pr = self.pr.copy()
                self.calculate_pr(d)
                for i in range(len(changes)):
                    changes[i] = abs(self.pr[i] - old_pr[i])
                count_number = count_number + 1

    def get_final_pr_score_dic(self):
        temp_pr_score = {}
        final_pr_score = {}
        for item in self.graph:
            item_idx = self.graph[item]
            temp_pr_score[item] = self.pr[item_idx]
        final_pr_score_list = sorted(temp_pr_score.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        for i in final_pr_score_list:
            final_pr_score[i[0]] = i[1]
        return final_pr_score


def test(graph_list_file, graph_embedding_list_file, graph_value_list_file, output_final_pr_file):
    graph_list = load_obj(graph_list_file)
    graph_embedding_list = load_obj(graph_embedding_list_file)
    graph_value_list = load_obj(graph_value_list_file)
    final_phrase = []
    for idx in tqdm(range(len(graph_list))):
        if len(graph_list[idx]) == 0:
            final_phrase.append({})
            continue
        temp_similarity = pdist(graph_embedding_list[idx], metric='cosine')
        temp_similarity_matrix = squareform(temp_similarity)
        # 算相似度矩阵
        temp_ones = np.ones((len(graph_list[idx]), len(graph_list[idx])), dtype='float64')
        used_similarity_matrix = temp_ones - temp_similarity_matrix
        graph_score = NeGraph(graph_list[idx], graph_value_list[idx], used_similarity_matrix)
        graph_score.calculate_pr_converge(0.85, 0.0001)
        final_phrase.append(graph_score.get_final_pr_score_dic())
    save_obj(final_phrase, output_final_pr_file)


if __name__ == '__main__':
    graph_list_file = 'patent/abstract/abstract_graph/abstract_graph_list.pkl'
    graph_embedding_list_file = 'patent/abstract/abstract_graph/abstract_graph_embedding_list.pkl'
    graph_value_list_file = 'patent/abstract/abstract_score/abstract_influence_phrase_list_normalized_score.pkl'
    test_file = 'patent/abstract/abstract_rank/ranked_abstract_influence_phrase_score.pkl'
    test(graph_list_file, graph_embedding_list_file, graph_value_list_file, test_file)
