import json
from tqdm import tqdm
from pattern.en import lemma
import numpy as np

def phrase_lemma(phrase):
    word_list = phrase.split(' ')
    output_word_list = []
    for item in word_list:
        output_word_list.append(lemma(item))
    return ' '.join(output_word_list)


def precision_recall_f1(reference_list, predicted_list):
    result = {'precision': [], 'recall': [], 'F1-score': []}
    for idx in range(len(predicted_list)):
        reference_len = len(reference_list[idx])
        predicted_len = len(predicted_list[idx])
        match_count = 0
        for reference_item in reference_list[idx]:
            select_number = 0
            for predict_item in predicted_list[idx]:
                select_number = select_number + 1
                if reference_item == predict_item:
                    match_count = match_count + 1
                    break

        if match_count == 0:
            result['precision'].append(0)
            result['recall'].append(0)
            result['F1-score'].append(0)
        else:
            temp_precision = match_count / predicted_len
            temp_recall = match_count / reference_len
            temp_f1_score = 2 * temp_precision * temp_recall / (temp_precision + temp_recall)
            result['precision'].append(temp_precision)
            result['recall'].append(temp_recall)
            result['F1-score'].append(temp_f1_score)

    return np.mean(result['precision']), np.mean(result['recall']), np.mean(result['F1-score'])





referenece_title_file = 'evaluation/F/title/title_stemmed_annotation.json'
referenece_abstract_file = 'evaluation/F/abstract/abstract_stemmed_annotation.json'
referenece_claim_file = 'evaluation/F/claim/claim_stemmed_annotation.json'

predicted_file = 'result/sample_output.json'

output_file = 'evaluation/F/prf_result.json'

predicted_title = []
predicted_abstract = []
predicted_claim = []

with open(referenece_title_file, 'r', encoding='utf-8') as f_title:
    reference_title = json.load(f_title)
with open(referenece_abstract_file, 'r', encoding='utf-8') as f_abstract:
    reference_abstract = json.load(f_abstract)
with open(referenece_claim_file, 'r', encoding='utf-8') as f_claim:
    reference_claim = json.load(f_claim)

# 需要对预测得到的 phrase 进行 stem 处理
with open(predicted_file, 'r', encoding='utf-8') as f_predicted:
    temp_output = json.load(f_predicted)
for item in temp_output:
    temp_title = []
    temp_abstract = []
    temp_claim = []
    for phrase in item['title']:
        temp_title.append(phrase_lemma(phrase))
    for phrase in item['abstract']:
        temp_abstract.append(phrase_lemma(phrase))
    for phrase in item['claim']:
        temp_claim.append(phrase_lemma(phrase))
    
    predicted_title.append(temp_title)
    predicted_abstract.append(temp_abstract)
    predicted_claim.append(temp_claim)


whole_reference = []
whole_predicted = []
for idx in range(len(reference_title)):
    temp_reference = []
    # title
    for item in reference_title[idx]:
        if item not in temp_reference:
            temp_reference.append(item)
    # abstract
    for item in reference_abstract[idx]:
        if item not in temp_reference:
            temp_reference.append(item)
    # claim
    for item in reference_claim[idx]:
        if item not in temp_reference:
            temp_reference.append(item)
    whole_reference.append(temp_reference)

    temp_predicted = []
    # title
    for item in predicted_title[idx]:
        if item not in temp_predicted:
            temp_predicted.append(item)
    # abstract
    for item in predicted_abstract[idx]:
        if item not in temp_predicted:
            temp_predicted.append(item)
    # claim
    for item in predicted_claim[idx]:
        if item not in temp_predicted:
            temp_predicted.append(item)
    whole_predicted.append(temp_predicted)

final_result = {}
# final_result['title']['precision'], final_result['title']['recall'], final_result['title']['f1-score'] = precision_recall_f1(reference_title, predicted_title)
# final_result['abstract']['precision'], final_result['abstract']['recall'], final_result['abstract']['f1-score'] = precision_recall_f1(reference_abstract, predicted_abstract)
# final_result['claim']['precision'], final_result['claim']['recall'], final_result['claim']['f1-score'] = precision_recall_f1(reference_claim, predicted_claim)
# with open(output_file, 'w', encoding='utf-8') as f_out:
#     json.dump(final_result, f_out)
final_result['precision'], final_result['recall'], final_result['f1-score'] = precision_recall_f1(whole_reference, whole_predicted)
print(final_result)