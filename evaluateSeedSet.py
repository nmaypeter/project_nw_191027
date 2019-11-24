from Evaluation import *

dataset_seq = [1, 2, 3, 4]
sc_option_seq = [1, 2, 3]
ds_option_seq = [1, 2]
cm_seq = [1, 2]
prod_seq = [1, 2]
model_seq = ['mbcs', 'mbcsd', 'mbcsM', 'mbcsMd',
             'mmioa', 'mmioad', 'mmioar', 'mmioard',
             'mdag1', 'mdag1d', 'mdag1r', 'mdag1rd',
             'mdag2', 'mdag2d', 'mdag2r', 'mdag2rd',
             'mng', 'mngd', 'mngr', 'mngrd', 'mhd', 'mr']

for data_setting in dataset_seq:
    dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                   'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
    new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                       'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
    for sc_option in sc_option_seq:
        seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for ds_option in ds_option_seq:
                diff_seed_option = False if ds_option == 1 else True
                for prod_setting in prod_seq:
                    product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                    new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                    for model_name in model_seq:
                        for bi in range(10, 6, -1):

                            path0 = 'result/' + new_dataset_name + '_' + cascade_model + '_' + seed_cost_option
                            path = path0 + '/' + model_name + '_ds' * diff_seed_option
                            result_name = path + '/m50e25_' + new_product_name + '_bi' + str(bi) + '.txt'

                            try:
                                ss_time = 0.0
                                s_set = []
                                with open(result_name) as f:
                                    for lnum, line in enumerate(f):
                                        if lnum == 5:
                                            (l) = line.split()
                                            ss_time = float(l[-1])
                                            # print(ss_time)
                                        if lnum == 12:
                                            (l) = line.split()
                                            # print(l)
                                            for l_item in l[14:]:
                                                l_item = l_item.replace(',', '')
                                                l_item = l_item.replace('\'', '')
                                                l_item = l_item.replace('}', '')
                                                l_item = l_item.replace('[', '')
                                                l_item = l_item.replace(']', '')
                                                # print(l_item)
                                                if 'set()' in l_item:
                                                    s_set.append(set())
                                                elif '{' in l_item:
                                                    l_item = l_item.replace('{', '')
                                                    s_set.append(set())
                                                    s_set[-1].add(l_item)
                                                else:
                                                    s_set[-1].add(l_item)
                                print(s_set)
                                evaM = EvaluationM(model_name, dataset_name, product_name, seed_cost_option, diff_seed_option, cascade_model)
                                evaM.evaluate(bi, 'm66e34', s_set, ss_time)
                            except FileNotFoundError:
                                continue