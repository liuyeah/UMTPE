import json
from tqdm import tqdm
from pattern.en import lemma

f_title_path = 'evaluation/F/title/title_annotation.txt'
text_path = 'evaluation/F/title/title_sample.txt'
output_path = 'evaluation/F/title/title_stemmed_annotation.json'

title_word = []
title_score = []

with open(f_title_path, 'r', encoding='utf-8') as f_title:
    count = 0
    for item in tqdm(f_title, total=100):
        count = count + 1
        word = []
        score = []
        item = item.strip('\n')
        if item != '':
            temp = item.split('; ')
            for temp_item in temp:
                temp_word_score = temp_item.split(', ')
                if temp_word_score[0].lower() not in word and len(temp_word_score[0].split(' ')) > 1:
                    word.append(temp_word_score[0].lower())
                    score.append(temp_word_score[1])
        title_word.append(word)
        title_score.append(score)
        print(str(count))

texts = []
with open(text_path, 'r', encoding='utf-8') as f_text:
    for item_text in f_text:
        item_text = item_text.strip('\n')
        item_text = item_text.split(' ')
        temp_item_text = []
        for i in item_text:
            temp_item_text.append(i)
        texts.append(temp_item_text)

# 检测是否全部都在原来的文本中
for i in range(len(title_word)):
    # output_i = []
    for j in range(len(title_word[i])):
        k = title_word[i][j].split(' ')
        length_k = len(k)
        non_error = 0
        for m in range(len(texts[i]) - length_k + 1):
            signal = 0
            for temp_k in range(length_k):
                if texts[i][m + temp_k] != k[temp_k]:
                    # signal = 1 表示不匹配
                    signal = 1
                    break
            if not signal:
                # temp_output = {'st': m, 'ed': m + length_k, 'text': words[i][j], 'score': scores[i][j]}
                # output_i.append(temp_output)
                non_error = 1
        if non_error == 0:
            print(str(i) + ': ' + title_word[i][j])

output_list = []
for line_word_list in title_word:
    line_word_list_new = []
    for word_item in line_word_list:
        word_split = word_item.split(' ')
        word_split_new = []
        for temp_item in word_split:
            word_split_new.append(lemma(temp_item))
        line_word_list_new.append(' '.join(word_split_new))
    output_list.append(line_word_list_new)

with open(output_path, 'w', encoding='utf-8') as f_out:
    json.dump(output_list, f_out, indent=True)






f_title_path = 'evaluation/F/abstract/abstract_annotation.txt'
text_path = 'evaluation/F/abstract/abstract_sample.txt'
output_path = 'evaluation/F/abstract/abstract_stemmed_annotation.json'

title_word = []
title_score = []

with open(f_title_path, 'r', encoding='utf-8') as f_title:
    count = 0
    for item in tqdm(f_title, total=100):
        count = count + 1
        word = []
        score = []
        item = item.strip('\n')
        if item != '':
            temp = item.split('; ')
            for temp_item in temp:
                temp_word_score = temp_item.split(', ')
                if temp_word_score[0].lower() not in word and len(temp_word_score[0].split(' ')) > 1:
                    word.append(temp_word_score[0].lower())
                    score.append(temp_word_score[1])
        title_word.append(word)
        title_score.append(score)
        print(str(count))

texts = []
with open(text_path, 'r', encoding='utf-8') as f_text:
    for item_text in f_text:
        item_text = item_text.strip('\n')
        item_text = item_text.split(' ')
        temp_item_text = []
        for i in item_text:
            temp_item_text.append(i)
        texts.append(temp_item_text)

# 检测是否全部都在原来的文本中
for i in range(len(title_word)):
    # output_i = []
    for j in range(len(title_word[i])):
        k = title_word[i][j].split(' ')
        length_k = len(k)
        non_error = 0
        for m in range(len(texts[i]) - length_k + 1):
            signal = 0
            for temp_k in range(length_k):
                if texts[i][m + temp_k] != k[temp_k]:
                    # signal = 1 表示不匹配
                    signal = 1
                    break
            if not signal:
                # temp_output = {'st': m, 'ed': m + length_k, 'text': words[i][j], 'score': scores[i][j]}
                # output_i.append(temp_output)
                non_error = 1
        if non_error == 0:
            print(str(i) + ': ' + title_word[i][j])

output_list = []
for line_word_list in title_word:
    line_word_list_new = []
    for word_item in line_word_list:
        word_split = word_item.split(' ')
        word_split_new = []
        for temp_item in word_split:
            word_split_new.append(lemma(temp_item))
        line_word_list_new.append(' '.join(word_split_new))
    output_list.append(line_word_list_new)

with open(output_path, 'w', encoding='utf-8') as f_out:
    json.dump(output_list, f_out, indent=True)











f_title_path = 'evaluation/F/claim/claim_annotation.txt'
text_path = 'evaluation/F/claim/claim_sample.txt'
output_path = 'evaluation/F/claim/claim_stemmed_annotation.json'

title_word = []
title_score = []

with open(f_title_path, 'r', encoding='utf-8') as f_title:
    count = 0
    for item in tqdm(f_title, total=100):
        count = count + 1
        word = []
        score = []
        item = item.strip('\n')
        if item != '':
            temp = item.split('; ')
            for temp_item in temp:
                temp_word_score = temp_item.split(', ')
                if temp_word_score[0].lower() not in word and len(temp_word_score[0].split(' ')) > 1:
                    word.append(temp_word_score[0].lower())
                    score.append(temp_word_score[1])
        title_word.append(word)
        title_score.append(score)
        print(str(count))

texts = []
with open(text_path, 'r', encoding='utf-8') as f_text:
    for item_text in f_text:
        item_text = item_text.strip('\n')
        item_text = item_text.split(' ')
        temp_item_text = []
        for i in item_text:
            temp_item_text.append(i)
        texts.append(temp_item_text)

# 检测是否全部都在原来的文本中
for i in range(len(title_word)):
    # output_i = []
    for j in range(len(title_word[i])):
        k = title_word[i][j].split(' ')
        length_k = len(k)
        non_error = 0
        for m in range(len(texts[i]) - length_k + 1):
            signal = 0
            for temp_k in range(length_k):
                if texts[i][m + temp_k] != k[temp_k]:
                    # signal = 1 表示不匹配
                    signal = 1
                    break
            if not signal:
                # temp_output = {'st': m, 'ed': m + length_k, 'text': words[i][j], 'score': scores[i][j]}
                # output_i.append(temp_output)
                non_error = 1
        if non_error == 0:
            print(str(i) + ': ' + title_word[i][j])

output_list = []
for line_word_list in title_word:
    line_word_list_new = []
    for word_item in line_word_list:
        word_split = word_item.split(' ')
        word_split_new = []
        for temp_item in word_split:
            word_split_new.append(lemma(temp_item))
        line_word_list_new.append(' '.join(word_split_new))
    output_list.append(line_word_list_new)

with open(output_path, 'w', encoding='utf-8') as f_out:
    json.dump(output_list, f_out, indent=True)