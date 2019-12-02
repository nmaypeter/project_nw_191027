import xlwings as xw

dataset_seq = [1, 2, 3, 4]
sc_option_seq = [1, 2, 3]
ds_option_seq = [1, 2]
cm_seq = [1, 2]
prod_seq = [1, 2]
wd_seq = [1, 2, 3]
model_seq = ['mmioaepw', 'mmioadepw', 'mmioarepw', 'mmioardepw',
             'mdag1epw', 'mdag1depw', 'mdag1repw', 'mdag1rdepw',
             'mdag2epw', 'mdag2depw', 'mdag2repw', 'mdag2rdepw',
             'mmioa', 'mmioad', 'mmioapw', 'mmioadpw',
             'mmioar', 'mmioard', 'mmioarpw', 'mmioardpw',
             'mdag1', 'mdag1d', 'mdag1pw', 'mdag1dpw',
             'mdag1r', 'mdag1rd', 'mdag1rpw', 'mdag1rdpw',
             'mdag2', 'mdag2d', 'mdag2pw', 'mdag2dpw',
             'mdag2r', 'mdag2rd', 'mdag2rpw', 'mdag2rdpw',
             'mng', 'mngd', 'mngpw', 'mngdpw',
             'mngr', 'mngrd', 'mngrpw', 'mngrdpw',
             'mbcs', 'mbcsd', 'mhd', 'mr']

for data_setting in dataset_seq:
    # dataset_name = 'email' * (data_setting == 1) + 'dnc_email' * (data_setting == 2) + \
    #                'email_Eu_core' * (data_setting == 3) + 'NetHEPT' * (data_setting == 4)
    new_dataset_name = 'email' * (data_setting == 1) + 'dnc' * (data_setting == 2) + \
                    'Eu' * (data_setting == 3) + 'Net' * (data_setting == 4)
    for sc_option in sc_option_seq:
        seed_cost_option = 'dp' * (sc_option == 1) + 'd' * (sc_option == 2) + 'p' * (sc_option == 3)
        for cm in cm_seq:
            cascade_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
            profit_list = []
            for bi in range(10, 5, -1):
                for ds_option in ds_option_seq:
                    diff_seed_option = False if ds_option == 1 else True
                    for prod_setting in prod_seq:
                        # product_name = 'item_lphc' * (prod_setting == 1) + 'item_hplc' * (prod_setting == 2)
                        new_product_name = 'lphc' * (prod_setting == 1) + 'hplc' * (prod_setting == 2)
                        for wd in wd_seq:
                            wallet_distribution_type = 'm50e25' * (wd == 1) + 'm99e96' * (wd == 2) + 'm66e34' * (wd == 3)

                            profit = []
                            # r = dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                            #     wallet_distribution_type + '\t' + product_name + '\t' + str(diff_seed_option) + '\t' + str(bi)
                            r = new_dataset_name + '\t' + seed_cost_option + '\t' + cascade_model + '\t' + \
                                wallet_distribution_type + '\t' + new_product_name + '\t' + str(diff_seed_option) + '\t' + str(bi)
                            print(r)
                            for model_name in model_seq:
                                try:
                                    result_name = 'result/' + \
                                                  new_dataset_name + '_' + cascade_model + '_' + seed_cost_option + '/' + \
                                                  model_name + '_ds' * diff_seed_option + '/' + \
                                                  wallet_distribution_type + '_' + new_product_name + '_bi' + str(bi) + '.txt'

                                    with open(result_name) as f:
                                        p = 0.0
                                        for lnum, line in enumerate(f):
                                            if lnum < 4:
                                                continue
                                            elif lnum == 4:
                                                (l) = line.split()
                                                p = float(l[-1])
                                            elif lnum == 5:
                                                (l) = line.split()
                                                c = float(l[-1])
                                                profit.append(str(round(p - c, 4)))
                                            else:
                                                break
                                except FileNotFoundError:
                                    profit.append('')
                            profit_list.append(profit)
                    profit_list.append(['' for _ in range(len(model_seq))])

            result_path = 'result/result_' + new_dataset_name + '.xlsx'
            wb = xw.Book(result_path)
            sheet_name = 'profit_' + seed_cost_option + ' (' + cascade_model + ')'
            sheet = wb.sheets[sheet_name]
            sheet.cells(8, "D").value = profit_list