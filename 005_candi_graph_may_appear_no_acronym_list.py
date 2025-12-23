from small_tools import remove_head_tail_space, read_text, get_files, read_csv, make_dir, csv_writer

# load text
root_path = r""

date = '20250603_may_appear'

for model in ['gpt3']:  # , 'llama3', 'gpt4o'
    print(model)
    for data_set in ['SemEval2010', 'Nguyen2007']:
        print(data_set)

        sent_path = root_path + r'data/' + data_set + '/stemmed_docsutf8_by_sent_no_punc/'

        max_window = 405

        for window in ['dynamic'] + list(range(2, 21, 1)) + list(range(25, max_window, 5)):  # :
            print('window =', window)

            candi_path = root_path + r'data\data_' + date + '/' + data_set + '/' + model + \
                         '_round_1\stem_extracted_keyphrase_processed_no_acronym_list/'
            save_path = candi_path.replace('extracted_keyphrase_processed', 'edges_with_weights') + '/window_' + str(
                window) + '/'

            make_dir(save_path)

            files = get_files(sent_path)

            for f, file in enumerate(files[:]):

                # candidates
                text = read_text(candi_path + file.replace('.csv', '.txt'))
                candidates = [line.lower() for line in text.split('\n') if
                              line]  # ['[oper transform, ot, oper transform ot]' ]
                candidates = [line for line in candidates if line]  # some line has number but has a number only

                candidates = [line.replace('[', '').replace(']', '').split(', ') for line in
                              candidates]  # convert str into list

                # sentences
                sentences_token_num_pair = read_csv(sent_path + file)
                sentences = [item[0] for item in sentences_token_num_pair]

                if window == 'dynamic':
                    window = len(sentences)

                sent_dict = dict(zip(sentences, list(range(len(sentences)))))

                occur_dict = {}
                for k_list in candidates:
                    find_pos = []
                    for k in k_list:
                        if not k:
                            continue
                        for k1, v1 in sent_dict.items():
                            local_k1 = ' ' + k1 + ' '
                            local_k = ' ' + k + ' '
                            if local_k1.lower().find(local_k.lower()) != -1:
                                find_pos.append(v1)
                    node_name = '[' + ', '.join(k_list) + ']'
                    occur_dict[node_name] = find_pos

                w = csv_writer(save_path + file, 'w')
                occur_dict2 = occur_dict.copy()
                adj_dict = {}
                for k2, v2 in occur_dict.items():
                    adj_list = []
                    for item in v2:
                        for k3, v3 in occur_dict2.items():
                            for item2 in v3:
                                if abs(item - item2) < window and k3 != k2:
                                    adj_list.append([k3, abs(item - item2)])

                    adj_list2 = {}
                    for item in adj_list:

                        # consider position and frequency
                        if item[0] not in adj_list2:
                            adj_list2[item[0]] = window - item[1]
                        else:
                            adj_list2[item[0]] = (adj_list2[item[0]]) + (window - item[1])

                    adj_dict[k2] = adj_list2

                    for k3, v3 in adj_list2.items():
                        w.writerow([k2, k3, v3])

