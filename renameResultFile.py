import os

dataset_seq = [1, 2, 3, 4]
sc_option_seq = [1, 2, 3]
ds_option_seq = [1, 2]
cm_seq = [1, 2]
prod_seq = [1, 2]
wallet_distribution_seq = [1, 2]
model_seq = ['mmioaepw', 'mmioadepw', 'mmioarepw', 'mmioardepw',
             'mdag1epw', 'mdag1depw', 'mdag1repw', 'mdag1rdepw',
             'mdag2epw', 'mdag2depw', 'mdag2repw', 'mdag2rdepw',
             'mbcs', 'mbcsd', 'mbcsM', 'mbcsMd',
             'mmioa', 'mmioad', 'mmioapw', 'mmioadpw',
             'mmioar', 'mmioard', 'mmioarpw', 'mmioardpw',
             'mdag1', 'mdag1d', 'mdag1pw', 'mdag1dpw',
             'mdag1r', 'mdag1rd', 'mdag1rpw', 'mdag1rdpw',
             'mdag2', 'mdag2d', 'mdag2pw', 'mdag2dpw',
             'mdag2r', 'mdag2rd', 'mdag2rpw', 'mdag2rdpw',
             'mng', 'mngd', 'mngpw', 'mngdpw',
             'mngr', 'mngrd', 'mngrpw', 'mngrdpw',
             'mhd', 'mr']

for data_setting in dataset_seq:
    dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
                   'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
    new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                       'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
    for sc_option in sc_option_seq:
        seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            for bi in range(10, 5, -1):
                for ds_option in ds_option_seq:
                    diff_seed_option = False if ds_option == 1 else True
                    for prod_setting in prod_seq:
                        product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                        new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                        for wallet_distribution in wallet_distribution_seq:
                            wallet_distribution_type = 'm50e25' * (wallet_distribution == 1) + 'm99e96' * (wallet_distribution == 2)

                            # r = dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                            #     wallet_distribution_type + '\t' + product_name + '\t' + str(diff_seed_option) + '\t' + str(bi)
                            r = new_dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                                wallet_distribution_type + '\t' + new_product_name + '\t' + str(diff_seed_option) + '\t' + str(bi)
                            print(r)
                            for model_name in model_seq:
                                try:
                                    result_name = 'result/' + \
                                                  model_name + '_' + wallet_distribution_type + '/' + \
                                                  dataset_name + '_' + cascade_model + '_' + product_name + '_' + seed_cost_option + \
                                                  '_ds' * diff_seed_option + '_bi' + str(bi) + '.txt'
                                    new_path0 = 'result/' + new_dataset_name + '_' + cascade_model + '_' + seed_cost_option
                                    if not os.path.isdir(new_path0):
                                        os.mkdir(new_path0)
                                    new_path = new_path0 + '/' + model_name + '_ds' * diff_seed_option
                                    if not os.path.isdir(new_path):
                                        os.mkdir(new_path)
                                    new_result_name = new_path + '/' + wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '.txt'
                                    os.rename(result_name, new_result_name)

                                except FileNotFoundError:
                                    continue