import os
from openai import OpenAI
from small_tools import get_files, read_text, write_text, remove_head_tail_space
import tiktoken
import spacy
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stopwords = read_text(r'UGIR_stopwords.txt')
stopwords = stopwords.split('\n')


def comb_rank(dict1, dict2, left_ratio):
    if left_ratio == 0:
        return dict2
    elif left_ratio == 1:
        return dict1
    else:
        ratio1 = left_ratio
        ratio2 = 1 - ratio1
        new_dict = {}
        new_k = list(dict1.keys()) + list(dict2.keys())
        for k3 in new_k:
            if k3 in dict1 and k3 in dict2:
                new_dict[k3] = dict1[k3] * ratio1 + dict2[k3] * ratio2
            elif k3 not in dict1:
                new_dict[k3] = dict2[k3]
            else:
                new_dict[k3] = dict1[k3]
        return new_dict


big_window_list = \
[
    [
        [13],
        [30]
    ],
    [
        [13],
        [30]
    ],
    [
        [13],
        [30]
    ]
]
root_path = r""

date = '20250603_may_appear'
for model_index, model in enumerate(['gpt3']):  # , 'llama3', 'gpt4o'
    print(model)
    for data_index, data_set in enumerate(['Nguyen2007','SemEval2010']):
        print(data_set)

        key_path = root_path + r'data/' + data_set + '/stemmed_keys_no_punc/'

        total_print = ''
        for window in big_window_list[model_index][data_index]:

            total_print += 'window='+str(window)+'\n'

            for top_k in range(3):
                total_print += '\n'
                top_k += 1
                top_k = top_k*5

                # load text

                data_path1 = root_path + r'data\data_'+date+'/'+data_set+'/'+model+'_round_1\stem_extracted_keyphrase_processed_no_acronym_list/'
                data_path2 = data_path1.replace('stem_extracted_keyphrase_processed', 'stem_pagerank_rank') + 'window_'+str(window) +'/'

                files = get_files(data_path1)

                for left_ratio in range(0, 11, 1):
                    left_ratio = left_ratio / 10

                    if data_set == 'SemEval2010':
                        if left_ratio != 0.5:
                            continue
                    if data_set == 'Nguyen2007':
                        if left_ratio != 0.5:
                            continue
                    f_total = 0
                    for file in files[:]:

                        text1 = read_text(data_path1 + file)
                        candidates0 = [line.lower() for line in text1.split('\n') if line]
                        # print(len(list(set(candidates0))))
                        # a = []
                        # for item in candidates0:
                        #     if item not in a:
                        #         a.append(item)
                        #     else:
                        #         print(item)

                        candidates1 = []
                        [candidates1.append(item) for item in candidates0 if item not in candidates1]
                        candi_dict1 = dict(zip(candidates1, list(range(1, len(candidates1) + 1))))
                        # print(candi_dict1)

                        text2 = read_text(data_path2 + file.replace('.txt', '.csv'))
                        candidates2 = [line.replace('"', '').split('],')[0]+']' for line in text2.split('\n') if line]
                        candi_dict2 = dict(zip(candidates2, list(range(1, len(candidates2) + 1))))
                        # print(candi_dict2)

                        candi_dict3 = comb_rank(candi_dict1, candi_dict2, left_ratio)
                        candidates = []
                        for k, v in sorted(candi_dict3.items(), key=lambda x: x[1], reverse=False):
                            # print(k, v)
                            candidates.append(k)

                        # candidates = [item for item in candidates if item not in stopwords]
                        candidates = candidates[:top_k]  # some line has number but has a number only
                        candidates = [line.replace('[', '').replace(']', '').split(', ') for line in candidates]  # convert str into list

                        # stemmed
                        # candidates = [ps.stem(item) for item in candidates]

                        keys = read_text(key_path + file.replace('.abstr', '').replace('.txt', '.key'))
                        keys = [line.lower() for line in keys.split('\n') if line]
                        # print(keys)

                        # note the situation like "electro-chemical deposition (ECD)"
                        score = 0
                        recall_list = []
                        for candidate_list in candidates:
                            local_score = 0
                            for candidate in candidate_list:
                                if candidate in keys:
                                    local_score += 1
                                    recall_list.append(candidate)
                            if local_score > 0:
                                score += 1

                        # f, p, r
                        if score == 0:
                            f = 0
                        else:
                            p = score / min(top_k, len(candidates))
                            r = len(set(recall_list))/len(keys)  # r = score/len(keys)
                            # if p != 0 and r != 0:
                            f = 2 * p * r / (p + r)
                        # else:
                        #     f = 0
                        # print(file, f)
                        f_total += f

                    f_avg = f_total / len(files)
                    # print('left_ratio', left_ratio, 'f1@' + str(top_k), f_avg)
                    total_print += ' '.join(['left_ratio', str(left_ratio), 'f1@' + str(top_k), str(f_avg), '\n'])
        print(total_print)

# LongDocRank+: may_appear
"""------------------ gpt3 ------------------ 

[[Nguyen2007 window=13]]

left_ratio 0.5 f1@5 0.2442863906397764 

left_ratio 0.5 f1@10 0.2520684251612055 

left_ratio 0.5 f1@15 0.22657144929411285 

[[SemEval2010 window=30]]

left_ratio 0.5 f1@5 0.18660134126529612 

left_ratio 0.5 f1@10 0.2226262779295869 

left_ratio 0.5 f1@15 0.220371709265139 

------------------ llama3 ------------------ 

[[Nguyen2007 window=13]]

left_ratio 0.5 f1@5 0.2956428896860893 

left_ratio 0.5 f1@10 0.29685540050621884 

left_ratio 0.5 f1@15 0.26313641218135647 

[[SemEval2010 window=30]]

left_ratio 0.5 f1@5 0.22030658422196928 

left_ratio 0.5 f1@10 0.2519004963202477 

left_ratio 0.5 f1@15 0.24646701510274646 

------------------ gpt4o ------------------ 

[[Nguyen2007 window=13]]

left_ratio 0.5 f1@5 0.2412760561297817 

left_ratio 0.5 f1@10 0.23389019712686512 

left_ratio 0.5 f1@15 0.21145303223412448 

[[SemEval2010 window=30]]

left_ratio 0.5 f1@5 0.2186050918040925 

left_ratio 0.5 f1@10 0.2630663902219115 

left_ratio 0.5 f1@15 0.26032462601556694 

"""