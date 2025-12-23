from small_tools import f1, cut_off_percent, csv_writer, make_dir
from small_tools import get_files, read_text, write_text, read_csv, remove_head_tail_space
import networkx as nx

# path
root_path = r""

date = '20250603_may_appear'
for model in ['gpt3']:  # , 'llama3', 'gpt4o']:
    print(model)
    for data_set in ['SemEval2010', 'Nguyen2007']:
        print(data_set)

        for window in ['dynamic']:
            edge_path_1 = root_path + r'data\data_' + date + '/' + data_set + '/' \
                          + model + '_round_1/stem_edges_with_weights_no_acronym_list/window_' + str(window) + '/'

            save_path = edge_path_1.replace('edges_with_weights', 'pagerank_rank')

            make_dir(save_path)

            files = get_files(edge_path_1)

            for file in files[:]:
                w = csv_writer(save_path + file, 'w')
                # load candi edges
                local_edge_list = []
                total_node = []
                for row in read_csv(edge_path_1 + file):
                    node1, node2 = row[0], row[1]
                    weight = float(row[2])

                    # convert our graph to the pageRank friendly feeding data
                    local_edge_list.append((node1, node2, weight))
                    for node_x in [node1, node2]:
                        if node_x in total_node:
                            continue
                        else:
                            total_node.append(node_x)

                # build graph
                FG = nx.Graph()
                FG.add_weighted_edges_from(local_edge_list)
                ratio = 0.85
                pr = nx.pagerank(FG, ratio)  # the second para is alpha

                rank = 0
                predictions = []
                for k, v in sorted(pr.items(), key=lambda item: item[1], reverse=True):
                    if not k:
                        continue

                    rank += 1
                    w.writerow([k, rank])
                    predictions.append(
                        k)  # in case the prediction is less than 10, we calculate acc use min(len(pred), 10)
                # print(file, 'total nodes:', len(total_node))
