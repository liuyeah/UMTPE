import pickle
import numpy as np
from tqdm import tqdm


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def score_normalize(original_score_file, output_score_list_file):
    original_score = load_obj(original_score_file)
    normalized_score = []
    for graph_item in tqdm(original_score, total=20):
        temp_normalized_score = []
        final_normalized_score = []
        for node in graph_item:
            # 首先求五项分数之和
            temp_normalized_score.append(np.sum(graph_item[node]))
        if temp_normalized_score != []:
            temp_max = np.max(temp_normalized_score)
            temp_min = np.min(temp_normalized_score)
            score_bottom = temp_max - temp_min
            for score_item in temp_normalized_score:
                score_top = score_item - temp_min
                if score_bottom == 0:
                    final_normalized_score.append(1.0)
                else:
                    final_normalized_score.append(score_top / score_bottom)

        normalized_score.append(final_normalized_score)
    save_obj(normalized_score, output_score_list_file)


if __name__ == '__main__':
    original_score_file = 'patent/title/title_score/title_influence_phrase_score.pkl'
    output_score_list_file = 'patent/title/title_score/title_influence_phrase_list_normalized_score.pkl'
    score_normalize(original_score_file, output_score_list_file)