from Initialization import *
from SeedSelection import *

data_setting = 1
sc_option = 1
cm = 1
prod_setting = 1
wd = 1

model_seq = ['mmioaMepw', 'mdag1Mepw', 'mdag2Mepw',
             'mmioaepw', 'mmioarepw', 'mdag1epw', 'mdag1repw', 'mdag2epw', 'mdag2repw',
             'mmioa', 'mmioapw', 'mmioar', 'mmioarpw',
             'mdag1', 'mdag1pw', 'mdag1r', 'mdag1rpw',
             'mdag2', 'mdag2pw', 'mdag2r', 'mdag2rpw',
             'mng', 'mngpw', 'mngr', 'mngrpw',
             'mbcs', 'mhd', 'mr']
num_product = 3
epw_flag = True


dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
               'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                   'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)

ini = Initialization(dataset_name, product_name, wallet_distribution_type)
seed_cost_dict = ini.constructSeedCostDict(seed_cost_option)
wallet_dict = ini.constructWalletDict()
num_node = len(wallet_dict)
graph_dict = ini.constructGraphDict(cascade_model)
product_list, product_weight_list = ini.constructProductList()

ssmioa_model = SeedSelectionMIOA(graph_dict, seed_cost_dict, product_list, product_weight_list)
seed_mioa_dict = [{} for _ in range(num_product)]
mioa_dict = ssmioa_model.generateMIOA()
if epw_flag:
    mioa_dict = ssmioa_model.updateMIOAEPW(mioa_dict)
celf_heap = [(round((sum(mioa_dict[k][i][j][0] for j in mioa_dict[k][i]) * product_list[k][0]) * (1.0 if epw_flag else product_weight_list[k]), 4), k, i, 0)
             for k in range(num_product) for i in mioa_dict[k]]
heap.heapify_max(celf_heap)

# celf_heap = [(safe_div(celf_item[0], seed_cost_dict[celf_item[1]][celf_item[2]]), celf_item[1], celf_item[2], 0) for celf_item in celf_heap]
# heap.heapify_max(celf_heap)
title = 'k\ti\twallet\tcost\theap_index'
for model_name in model_seq:
    title = title + '\t' + model_name
r_list = [title]
for k in range(num_product):
    for i in seed_cost_dict[k]:
        celf_item = [celf_item for celf_item in celf_heap if celf_item[1] == k and celf_item[2] == i]
        celf_item = celf_item[0] if celf_item else -1
        if celf_item != -1:
            r = str(k) + '\t' + i + '\t' + str(wallet_dict[i]) + '\t' + str(seed_cost_dict[k][i]) + '\t' + str(celf_heap.index(celf_item))
        else:
            r = str(k) + '\t' + i + '\t' + str(wallet_dict[i]) + '\t' + str(seed_cost_dict[k][i]) + '\t-1'
        # print(r)
        r_list.append(r)

for model_name in model_seq:
    path0 = 'result/' + new_dataset_name + '_' + cascade_model + '_' + seed_cost_option
    path = path0 + '/' + wallet_distribution_type + '_' + new_product_name + '_bi8'
    result_name = path + '/' + model_name + '.txt'
    model_dict = [{i: '\t' for i in seed_cost_dict[k]} for k in range(num_product)]

    try:
        with open(result_name) as f:
            for lnum, line in enumerate(f):
                if lnum < 13:
                    continue
                else:
                    (l) = line.split()
                    k_prod = l[0]
                    k_prod = k_prod.replace('(', '')
                    k_prod = k_prod.replace(',', '')
                    k_prod = int(k_prod)
                    i_node = l[1]
                    i_node = i_node.replace('\'', '')
                    i_node = i_node.replace(')', '')
                    model_dict[k_prod][i_node] = '\t' + str(lnum - 12)
        for k in range(num_product):
            for i in seed_cost_dict[k]:
                r_list[1 + k * num_node + int(i)] += model_dict[k][i]
    except FileNotFoundError:
        for k in range(num_product):
            for i in seed_cost_dict[k]:
                r_list[1 + k * num_node + int(i)] += '\t'
        continue

for r in r_list:
    print(r)