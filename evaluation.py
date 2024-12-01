# -*- coding:utf-8 -*-
import numpy as np
import pickle
import itertools
import random
import math


def find_neighbors(adj_matrix, node):
    neighbors = []
    for f_i, connected in enumerate(adj_matrix[node]):
        if connected == 1:
            neighbors.append(f_i)
    return neighbors


def find_rank_descending(number, num_list):
    num_list.append(number)
    sorted_list = sorted(num_list, reverse=True)
    rank = sorted_list.index(number) + 1
    return rank


num_molecule = 2382
mol_interact_relations = pickle.load(open('./Data/mol_interact_relations.pkl', 'rb'))
mol_index = pickle.load(open('./Data/mol_index.pkl', 'rb'))
mol_syno = pickle.load(open('./Data/mol_syno.pkl', 'rb'))
can_mol_index = pickle.load(open('./Data/can_mol_index.pkl', 'rb'))
mol_fps = pickle.load(open('Data/FCFP4/FCFP4.pkl', 'rb'))

substr_weight = 1
mol_input_fps = mol_fps
mol_interact = mol_interact_relations

substructure_matrix = np.array(mol_input_fps, dtype=np.float64)
substructure_matrix = substructure_matrix[:, np.sum(substructure_matrix, axis=0) != 0]
mol_num, substructure_num = substructure_matrix.shape
substructure_links = []
for mol in range(mol_num):
    for i in range(substructure_num):
        if substructure_matrix[mol, i] == 1:
            substructure_links.append([mol, mol_num + i])

substructure_links = [item + [substr_weight] for item in substructure_links]
mol_interact_relations = [item + [1/substr_weight] for item in mol_interact_relations]
links = mol_interact_relations + substructure_links

mat_nodes = list(itertools.chain.from_iterable(links))
mat_nodes = set(mat_nodes)
mat_nodes = {np.int32(node) for node in mat_nodes}
mat_size = np.int32(max(mat_nodes) + 1)

network = np.zeros((mat_size, mat_size))
for item in links:
    network[np.int32(item[0]), np.int32(item[1])] = item[2]
network = network + network.T
sub_network = network[:num_molecule, :num_molecule]

# count_ones_per_row_list 
count_ones_per_row = np.sum(network == 1, axis=1)
count_ones_per_row_list = count_ones_per_row.tolist()

sub_row = np.sum(sub_network == 1, axis=1)
sub_list = sub_row.tolist()

ok_edges = [edge for edge in mol_interact]
ok_edges_swapped = [[edge[1], edge[0]] for edge in ok_edges]
ok_edges = ok_edges + ok_edges_swapped

AUC_scores = []
RS_scores = []
num_runs = 1

Alpha = 0.4
Beta = 0.4
Gama = 0.1

Emm = 1 - Beta
Esm = Beta

for run in range(num_runs):

    T_number = 0
    F_number = 0
    D_number = 0
    RS_all = []

    for k_i in range(len(ok_edges)):
        item = ok_edges[k_i]
        net_changed = network.copy()
        target_node, divide_node = item
        net_changed[target_node][divide_node] = 0
        net_changed[divide_node][target_node] = 0


        # 子结构必须给资源，否则无法计算NCE
        # if sub_list[target_node] <= 1:  # 如果是关键边，直接跳过，因为如果只有1个分子邻居，断开以后没法分配资源（因为有的惩罚是对）
        #     continue

        scores = [0] * len(network)
        neighbors_of_target_node = find_neighbors(net_changed, target_node)
        for i_1 in neighbors_of_target_node:
            scores[i_1] = (1 - Alpha)
        s_neighbors_of_target_node = [num for num in neighbors_of_target_node if num > len(mol_index)]  # 找出子结构节点和分子节点做区分
        for sj in s_neighbors_of_target_node:
            scores[sj] = Alpha

        ## K=1
        nodes_value1 = []
        for i_2 in range(len(scores)):
            if scores[i_2] != 0:
                nodes_value1.append(i_2)
        for k1 in nodes_value1:
            neighbors_k1 = find_neighbors(net_changed, k1)
            score_k1 = scores[k1]
            if k1 >= num_molecule:
                give_score1 = score_k1 / len(neighbors_k1)
                scores[k1] = 0
                for n1 in neighbors_k1:
                    scores[n1] += give_score1
            else:
                Neigh_S1 = 0
                for count_NS1 in neighbors_k1:
                    if count_NS1 < num_molecule:
                        Neigh_S1 += 1

                Neigh_M1 = len(neighbors_k1) - Neigh_S1
                attend_M1 = score_k1 * (Emm / (Emm * Neigh_M1 + Esm * Neigh_S1))
                attend_S1 = score_k1 * (Esm / (Emm * Neigh_M1 + Esm * Neigh_S1))
                scores[k1] = 0
                for n1 in neighbors_k1:
                    if n1 < num_molecule:
                        scores[n1] += attend_M1
                    else:
                        scores[n1] += attend_S1

        ## K=2
        new_hub2 = []
        for i_3 in range(len(scores)):
            if scores[i_3] != 0:
                new_hub2.append(i_3)
        for k2 in new_hub2:
            neighbors_k2 = find_neighbors(net_changed, k2)
            score_k2 = scores[k2]
            if k2 >= num_molecule:
                give_score2 = score_k2 / len(neighbors_k2)
                scores[k2] = 0
                for n2 in neighbors_k2:
                    scores[n2] += give_score2

            else:
                Neigh_S2 = 0
                for count_NS2 in neighbors_k2:
                    if count_NS2 < num_molecule:
                        Neigh_S2 += 1

                Neigh_M2 = len(neighbors_k2) - Neigh_S2
                attend_M2 = score_k2 * (Emm / (Emm * Neigh_M2 + Esm * Neigh_S2))
                attend_S2 = score_k2 * (Esm / (Emm * Neigh_M2 + Esm * Neigh_S2))
                scores[k2] = 0
                for n2 in neighbors_k2:
                    if n2 < num_molecule:
                        scores[n2] += attend_M2
                    else:
                        scores[n2] += attend_S2

        degree_list = count_ones_per_row_list.copy()
        degree_list = degree_list[:num_molecule]
        degree_list[divide_node] -= 1
        degree_list = [x ** Gama for x in degree_list]

        for f in neighbors_of_target_node:
            scores[f] = 0
        scores = scores[:num_molecule]
        result = [s_elem / y_elem for s_elem, y_elem in zip(scores, degree_list)]



        score_random_node = result[divide_node]
        not_real_edges_scores = [result[i] for i in range(len(result)) if i not in neighbors_of_target_node]
        if not not_real_edges_scores:
            continue

        rank = find_rank_descending(score_random_node, not_real_edges_scores)
        RS_i = rank / len(not_real_edges_scores)
        RS_all.append(RS_i)

        for epoch in range(1000):
            random_score = random.choice(not_real_edges_scores)
            if score_random_node > random_score:
                T_number += 1
            elif score_random_node == random_score:
                D_number += 1
            else:
                F_number += 1

        if (k_i) % 10 == 0:
            print('完成第', k_i, '次计算', "第", run+1, "轮")

    average_RS_i = np.mean(RS_all)
    AUC = ((T_number + 0.5 * D_number) / (T_number + D_number + F_number))
    AUC_scores.append(AUC)
    RS_scores.append(average_RS_i)

average_AUC = np.mean(AUC_scores)
print('平均 AUC=', average_AUC)
print("所有次数的AUC：", AUC_scores)

average_RS = np.mean(RS_scores)
print('平均 RS=', average_RS)
print("所有次数的RS：", RS_scores)

with open('results1.txt', 'w') as f:
    f.write(f'平均 AUC= {average_AUC}\n')
    f.write(f'所有次数的AUC： {AUC_scores}\n')
    f.write(f'平均 RS= {average_RS}\n')
    f.write(f'所有次数的RS： {RS_scores}\n')