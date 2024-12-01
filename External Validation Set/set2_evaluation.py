# -*- coding:utf-8 -*-
import numpy as np
import pickle
import itertools
import random
import math


def find_neighbors(adj_matrix, node):  # 找邻居节点
    neighbors = []
    for f_i, connected in enumerate(adj_matrix[node]):
        if connected == 1:
            neighbors.append(f_i)
    return neighbors


def find_rank_descending(number, num_list):
    num_list.append(number)  # 将数字添加到列表中
    sorted_list = sorted(num_list, reverse=True)
    rank = sorted_list.index(number) + 1
    return rank


# 分子节点2382 共晶组合4881
num_molecule = 2382
mol_interact_relations = pickle.load(open('./Data/文件1.pkl', 'rb'))
mol_index = pickle.load(open('./Data/文件2.pkl', 'rb'))
mol_syno = pickle.load(open('./Data/文件3.pkl', 'rb'))
can_mol_index = pickle.load(open('./Data/文件4.pkl', 'rb'))
mol_fps = pickle.load(open('Data/KR.pkl', 'rb'))

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
sub_network = network[:num_molecule, :num_molecule]   # 得到一个只有分子相互作用的网络

# count_ones_per_row_list 为所有节点(分子+子结构)的邻居数量列表
count_ones_per_row = np.sum(network == 1, axis=1)
count_ones_per_row_list = count_ones_per_row.tolist()  # 其中最小值为1
# lg1=0,0不能作为被除数，所以全部加1。degree_list_all是count_ones_per_row_list基础上加1


# sub_list是只有分子部分的数据
sub_row = np.sum(sub_network == 1, axis=1)
sub_list = sub_row.tolist()   # 构建关于分子之间的度关系，从而剔除断开后导致分子间相互作用网络不联通的连接

# ok_edges实际上就是所有存在的边，正反取了两遍
ok_edges = [[1500,8],[1501,20],[1502,188],[2437,66],[1503,767],[1504,985],[1505,2098],[1505,11],[1505,185],[1506,341],[1506,296],[1506,303],[1506,977],[1507,387],[1508,321],[1508,40],[1508,8],[1508,296],[1508,253],[1508,1499],[1508,992],[1509,754],[1510,455],[1510,8],[1511,57],[1511,247],[1511,755],[1511,11],[1511,15],[1511,341],[1513,146],[1506,165],[1506,12],[1514,286],[1515,672],[1516,29],[1516,211],[1517,1499],[1518,633],[1519,944],[1519,341],[1515,25],[1515,1612],[1515,23],[1520,1485],[1521,754],[1522,809],[1503,783],[1503,593],[1503,128],[1503,75],[1523,1353],[1516,165],[1516,1115],[1516,13],[1524,130],[1524,165],[1524,1077],[1524,228],[1524,152],[1506,633],[1506,286],[1525,0],[1526,0],[1527,0],[1528,296],[1529,493],[1530,330],[1531,13],[1532,14],[1533,14],[1532,75],[1534,624],[1531,2034],[1535,977],[1535,12],[1536,10],[1537,34],[1538,810],[1539,29],[1540,1133],[1541,1133],[1542,1133],[1543,644],[1544,14],[1545,66],[1545,65],[1546,325],[1547,303],[1548,87],[1549,0],[1549,202],[1550,261],[1551,40],[1552,75]]


AUC_scores = []
RS_scores = []
num_runs = 1
rank_all = []
rank_quan = []
Alpha = 0.3  # Alpha越大，子结构初始分配的资源越多，Alpha=0.5时，两类节点等同
Beta = 0.4
Gama = 0.5

Emm = 1 - Beta
Esm = Beta
# 重复num_runs次测试，一次1000次抽样，抽样后比较1000次
for run in range(num_runs):

    T_number = 0
    F_number = 0
    D_number = 0
    RS_all = []

    # 针对ok_edges进行NBI算法的检验
    for k_i in range(len(ok_edges)):
        item = ok_edges[k_i]  # 从可断开的列表中随机断开一条边
        net_changed = network.copy()
        target_node, divide_node = item
        net_changed[target_node][divide_node] = 0
        net_changed[divide_node][target_node] = 0           # 生成断开边的新网络


        # 子结构必须给资源，否则无法计算NCE
        # if sub_list[target_node] <= 1:  # 如果是关键边，直接跳过，因为如果只有1个分子邻居，断开以后没法分配资源（因为有的惩罚是对）
        #     continue

        ### 开始NBI算法

        ## 初始化分数列表，全为0
        scores = [0] * len(network)
        # 分配资源，找到目标节点的邻居，分配资源，重新给子结构分配资源
        neighbors_of_target_node = find_neighbors(net_changed, target_node)  # 找到目标节点的邻居
        for i_1 in neighbors_of_target_node:  # 进行赋值，全部赋值为1
            scores[i_1] = (1 - Alpha)
        s_neighbors_of_target_node = [num for num in neighbors_of_target_node if num > len(mol_index)]  # 找出子结构节点和分子节点做区分
        for sj in s_neighbors_of_target_node:
            scores[sj] = Alpha

        ## K=1
        nodes_value1 = []
        # 收集有分节点：遍历所有节点，将有资源的节点的索引收集到列表 nodes_value1 中
        for i_2 in range(len(scores)):
            if scores[i_2] != 0:
                nodes_value1.append(i_2)
        # 分配资源：遍历所有有分节点，将其资源全部平分给其邻居，更新 scores 列表
        for k1 in nodes_value1:
            neighbors_k1 = find_neighbors(net_changed, k1)  # 找到非0节点的邻居
            score_k1 = scores[k1]                           # 此有分节点的值
            if k1 >= num_molecule:
                give_score1 = score_k1 / len(neighbors_k1)  # 此节点应该平均分发的得分
                scores[k1] = 0  # 此节点得分清0
                for n1 in neighbors_k1:
                    scores[n1] += give_score1
            else:
                Neigh_S1 = 0
                for count_NS1 in neighbors_k1:
                    if count_NS1 < num_molecule:
                        Neigh_S1 += 1

                Neigh_M1 = len(neighbors_k1) - Neigh_S1
                # 计算两种节点应该分到的资源
                attend_M1 = score_k1 * (Emm / (Emm * Neigh_M1 + Esm * Neigh_S1))
                attend_S1 = score_k1 * (Esm / (Emm * Neigh_M1 + Esm * Neigh_S1))
                # 此节点得分清0
                scores[k1] = 0
                for n1 in neighbors_k1:
                    if n1 < num_molecule:
                        scores[n1] += attend_M1
                    else:
                        scores[n1] += attend_S1

        ## K=2，操作同前
        new_hub2 = []
        for i_3 in range(len(scores)):
            if scores[i_3] != 0:
                new_hub2.append(i_3)
        for k2 in new_hub2:
            neighbors_k2 = find_neighbors(net_changed, k2)
            score_k2 = scores[k2]
            if k2 >= num_molecule:
                give_score2 = score_k2 / len(neighbors_k2)  # 此节点应该平均分发的得分
                scores[k2] = 0  # 此节点得分清0
                for n2 in neighbors_k2:
                    scores[n2] += give_score2

            else:
                # 统计两种节点的个数
                Neigh_S2 = 0
                for count_NS2 in neighbors_k2:
                    if count_NS2 < num_molecule:
                        Neigh_S2 += 1

                Neigh_M2 = len(neighbors_k2) - Neigh_S2
                # 计算两种节点应该分到的资源
                attend_M2 = score_k2 * (Emm / (Emm * Neigh_M2 + Esm * Neigh_S2))
                attend_S2 = score_k2 * (Esm / (Emm * Neigh_M2 + Esm * Neigh_S2))
                # 此节点得分清0
                scores[k2] = 0
                for n2 in neighbors_k2:
                    if n2 < num_molecule:
                        scores[n2] += attend_M2
                    else:
                        scores[n2] += attend_S2

        ### 给得分适当的惩罚
        degree_list = count_ones_per_row_list.copy()
        degree_list = degree_list[:num_molecule]         # 只计算分子部分，舍弃子结构段
        degree_list[divide_node] -= 1                    # 断开了目标节点的一个边，所以其度需要-1
        degree_list = [x ** Gama for x in degree_list]

        for f in neighbors_of_target_node:  # 目标邻居节点不参与比较，全部设置为0，最后结果也就会是0
            scores[f] = 0
        scores = scores[:num_molecule]
        # 惩罚后的结果为 result
        result = [s_elem / y_elem for s_elem, y_elem in zip(scores, degree_list)]  # 进行大度节点的惩罚，防止大度节点每次都出现


        ### 完成NBI，进行比较
        # 被断开的节点得分
        score_random_node = result[divide_node]
        # 目标节点的非邻居节点的得分
        not_real_edges_scores = [result[i] for i in range(len(result)) if i not in neighbors_of_target_node]
        not_real_edges_scores = [x for x in not_real_edges_scores if x != 0]
        # 防止列表为空，但只要给子结构也分配资源了，就不可能为空
        if not not_real_edges_scores:
            continue

        ### 计算排名及单次RS值
        rank = find_rank_descending(score_random_node, not_real_edges_scores)
        print(rank)
        rank_quan.append(rank)
        if rank <= 100:
            rank_all.append(rank)
        RS_i = rank / len(not_real_edges_scores)
        RS_all.append(RS_i)

        ### 进行AUC的比较
        for epoch in range(1000):
            random_score = random.choice(not_real_edges_scores)
            if score_random_node > random_score:
                T_number += 1
            elif score_random_node == random_score:
                D_number += 1
            else:
                F_number += 1

        # 每计算100次进行一次打印提示
        if (k_i) % 10 == 0:
            print('完成第', k_i, '次计算', "第", run+1, "轮")

    ### 完成一轮的AUC和RS计算
    average_RS_i = np.mean(RS_all)
    AUC = ((T_number + 0.5 * D_number) / (T_number + D_number + F_number))
    AUC_scores.append(AUC)
    RS_scores.append(average_RS_i)

### 完成N轮的AUC和RS计算，打印结果
average_AUC = np.mean(AUC_scores)
print('平均 AUC=', average_AUC)
print("所有次数的AUC：", AUC_scores)

average_RS = np.mean(RS_scores)
print('平均 RS=', average_RS)
print("所有次数的RS：", RS_scores)
print(rank_quan)
print(np.sum(rank_quan))
print(np.sum(rank_all))
print(len(rank_all))