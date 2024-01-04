# pKa predictor
# Ziling Luo, Jerome Genzling, Ben Weiser, Nicolas Moitessier
# Department of Chemistry, McGill University
# Montreal, QC, Canada

# To visualize SMILES: http://www.cheminfo.org/Chemistry/Cheminformatics/Smiles/index.html
# To train the model:
#  python main.py --mode train --n_graph_layers 3 --train_data Complete_Set.csv --test1_data test_pKa.csv
#  --test2_data sampl6.csv --test3_data sampl7.csv --verbose 2 --output train20230104 --n_random_smiles 1
# To infer with the model:
#  python main.py --mode_train infer --n_random_smiles 0 --model_name model.pkl --n_graph_layers 3 --input set.csv
import pandas as pd
import time

from argParser import argsParser


if __name__ == '__main__':

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| %s                                                                                                        |' % s)
    
    args = argsParser()

    # opening the training/testing set in csv format
    filename = args.data_path + args.train_data
    dataset = pd.read_csv(filename, sep=',')

    smiles = []
    smiles_revised = []
    train = False
    test = False
    fail_train_mol_num = []
    fail_train_obs = []
    fail_train_pred = []
    fail_train_smiles = []
    fail_train_fail_num = []
    fail_train_center = []

    fail_test_mol_num = []
    fail_test_obs = []
    fail_test_pred = []
    fail_test_smiles = []
    fail_test_fail_num = []
    fail_test_center = []
    with open(args.input) as f:
        lines = f.readlines()
        for line in lines:
            word = line.split()
            if len(word) > 2:
                if word[1] == 'original:':
                    smiles.append(word[2])
                if word[1] == 'revised:':
                    smiles_revised.append(word[2])
                if word[1] == 'Training':
                    train = True
                    continue
                if word[1] == 'Testing':
                    train = False
                    test = True
                    continue

            if len(word) > 11 and train is True and word[1] != 'Training':
                if word[11] not in fail_train_smiles:
                    fail_train_mol_num.append(word[1])
                    fail_train_obs.append(float(word[3]))
                    fail_train_pred.append(float(word[5]))
                    fail_train_smiles.append(word[11])
                    fail_train_fail_num.append(1)
                    row = dataset[dataset['Smiles'] == word[11]].index
                    center = dataset['Index'][row[0]]

                    fail_train_center.append(center)

            elif len(word) > 11 and test is True and word[1] != 'Testing':
                if word[11] not in fail_test_smiles:
                    fail_test_mol_num.append(word[1])
                    fail_test_obs.append(float(word[3]))
                    fail_test_pred.append(float(word[5]))
                    fail_test_smiles.append(word[11])
                    fail_test_fail_num.append(1)
                    row = dataset[dataset['Smiles'] == word[11]].index
                    center = dataset['Index'][row[0]]
                    fail_test_center.append(center)

    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| mol # |  obs. | pred. | center | smiles (original and revised)                                                             |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    for i in range(len(fail_train_mol_num)):
        print('| %5s | %5.2f | %5.2f | %6s | %-89s |' % (fail_train_mol_num[i], fail_train_obs[i],
                                                         fail_train_pred[i]/fail_train_fail_num[i],
                                                         fail_train_center[i], fail_train_smiles[i]))
        print('|       |       |       |        | %-89s |' % smiles_revised[smiles.index(fail_train_smiles[i])])
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    for i in range(len(fail_test_mol_num)):
        print('| %5s | %5.2f | %5.2f | %6s | %-89s |' % (fail_test_mol_num[i], fail_test_obs[i],
                                                         fail_test_pred[i]/fail_test_fail_num[i],
                                                         fail_test_center[i], fail_test_smiles[i]))
        print('|       |       |       |        | %-89s |' % smiles_revised[smiles.index(fail_test_smiles[i])])
    print('|----------------------------------------------------------------------------------------------------------------------------|')

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| Job finished at %s                                                                                        |' % s)
    print('|----------------------------------------------------------------------------------------------------------------------------|')
