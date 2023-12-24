import itertools
from sklearn import preprocessing
import warnings
# plot libraries
import matplotlib.pyplot as plt
import seaborn as sns
# calculate parameters
# from numba import cuda
import tensorflow as tf
import random
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
# import OS libraries
import glob
import os
import joblib
# data preparing libraries
import numpy as np
import pandas as pd
# import keras libraries
from keras.utils import np_utils
# calculate parameters
from sklearn.model_selection import train_test_split
# importing time library to avoid the script stuck
import time

warnings.filterwarnings('ignore')


def drop_duplicates(df, column_name):
    """ this function is going to remove the duplicates (it removes all of them, it will not keep
    any of them)"""
    # get a set of the duplicated sequences
    duplicated_set = set(list(df[df.duplicated(subset=[column_name])][column_name]))
    if duplicated_set:
        for seq in duplicated_set:
            # drop the rows with the duplicate sequence
            df.drop(df.index[df[column_name] == seq], inplace=True)
    # reset the index and drop the index column after resetting the index
    df.reset_index(inplace=True, drop=True)


def min_max_mean(df, threshold):
    """
    min_max_ mean function:
    input arguments:
    df: a data frame of the sequences with the df/f values
    threshold: if the differences between max and min are higher than the threshold, that wavelength will be kept
    outputs:
    df_selected: a data frame of the sequences with the df/for values that satisfies the threshold
    df_minmax: a data frame that contains min, max, mean, median, std and std of each wavelength
    df_minmax_selected: a data frame that contains min, max, mean, median, std and std of each selected
                        wavelength
    wavelength_value: a list contains the wavelength (column name of the dataframes)
    wavelength_value_selected: a list contains the selected wavelength (column name of the dataframes)
    """
    df_minmax = df.iloc[:, 1:].agg(['min', 'max', 'mean', 'median', 'std'])
    df_minmax.loc['diff'] = df_minmax.loc['max'] - df_minmax.loc['min']
    wavelength_value = [float(i) for i in list(df_minmax.columns)]
    selected_wavelength = []
    for diff_index, diff_value in enumerate(list(df_minmax.loc['diff'] > threshold)):
        if diff_value:
            selected_wavelength.append(list(df_minmax.columns)[diff_index])
    df_minmax_selected = df_minmax[selected_wavelength]
    wavelength_value_selected = [float(i) for i in list(df_minmax_selected.columns)]
    df_selected = df[['sequence'] + selected_wavelength]
    return df_selected, df_minmax, df_minmax_selected, wavelength_value, wavelength_value_selected


def sequence_removing(df_full, df_predict):
    for sequence in list(df_predict['sequence']):
        df_full.drop(df_full.index[df_full['sequence'] == sequence], inplace=True)
        df_full.reset_index(inplace=True, drop=True)


def threshold_sequence_removing(df, wavelength, increase_mean_med=0, tolerance=0.05, accept_percent=0.8):
    """
    This function removes sequences according to their DFF value if placed out of the threshold.
    e.g.: if the threshold is >0.9 and <0.7, this function will remove the sequences if the DFF
    value is between 0.7 and 0.9
    input arguments:
    df : dataframe which contains all or selected wavelength
    wavelength: the wavelength that threshold will apply on it
    increase_mean_med: changing the value of the mean and median
    tolerance: the threshold range
    accept_percent: Percentage of remaining sequences compare to the initial. If the percentage
    is below this value, that wavelength will be ignored
    outputs:
    df_in_threshold : a data frame which contains sequences inside the threshold
    df_out_threshold : a data frame which contains sequences outside the threshold
    """
    average_mean_median = (df[wavelength].median() + df[wavelength].mean()) / 2
    average_mean_median += increase_mean_med  # move the value upper or lower
    df_tmp = df[['sequence', wavelength]]
    n_acceptable_sequence = int(len(df_tmp) * accept_percent)
    # find the sequence indices that are out of the threshold
    indexNames = df_tmp[(df_tmp[wavelength] < average_mean_median + tolerance)
                        & (df_tmp[wavelength] > average_mean_median - tolerance)].index
    # make a class column for classification models. class 1 -> high dff value and in threshold
    # class 0 -> low dff value and in threshold
    df_tmp.loc[(df_tmp[wavelength] > average_mean_median + tolerance), 'class'] = 1
    df_tmp.loc[(df_tmp[wavelength] < average_mean_median - tolerance), 'class'] = 0
    df_tmp['class'] = df_tmp.loc[:, 'class'].astype('category')
    df_out_threshold = df_tmp.loc[df_tmp.index[indexNames]]
    df_in_threshold = df_tmp.drop(indexNames)
    # if enough sequences remained will return the data frames
    if len(df_in_threshold) >= n_acceptable_sequence:
        return df_in_threshold, df_out_threshold
    else:
        return None


def genComb(combination_length, letter_list):
    """
    function for finding all the combinations of given letters and with a defined length
    e.g.letters --> [A,B,C]
        length --> 2
        result --> ['AA', 'AC', 'AB', 'CA', 'CC', 'CB', 'BA', 'BC', 'BB']
    input_argument:
    combination_length: the length of combinations
    letter_list: it can be nucleotide or amino acids
    output:
    all_combinations: It is a dictionary in which the keys are all combinations, and the values are set to zero
    """
    all_combinations = {}
    keywords = [''.join(ga) for ga in itertools.product(letter_list, repeat=combination_length)]
    for gb in keywords:
        all_combinations[gb] = 0
    return all_combinations


def sequence_to_numeric(df_in_threshold, df_out_threshold, n_gram, wavelength, path_wavelength, method,
                        nucleotide_list=('A', 'C', 'G', 'T')):
    """
    This function will convert the dataframe to NumPy arrays. It will not return any variable and only save the
    dataframes and the converted array

    input_argument:
    df_in_threshold: a dataframe which contains sequences in threshold to convert it into NumPy arrays
    df_out_threshold: a dataframe which contains sequences out of threshold and only save it
    n_gram: the length of the combination of nucleotides
    wavelength: the wavelength which the sequences are converted and saved
    path_wavelength: the path for saving the arrays
    method: if it is regression or classification
    nucleotide_list: the nucleotides which is going to be encoded. default: is a tuple contains ('A', 'C', 'G', 'T')
    output:
    It doesn't return anything
    saved_items:
    X__: contains sequences converted to NumPy arrays
    y__: contains DFF values if it is regression problem of class 1 or 0 if it is a classification problem
    df_in_threshold: save the sequences in the threshold dataframe
    df_out_threshold: save the sequences out of the threshold dataframe
    """
    os.makedirs(path_wavelength)
    result = []
    for seq in df_in_threshold['sequence']:
        tmp = []
        for ls in range(0, len(seq)):
            if len(seq[ls:ls + n_gram]) == n_gram:
                tmp.append(seq[ls:ls + n_gram])
        result.append(tmp)
        all_combinations = genComb(n_gram, nucleotide_list)
        # encode all the combinations
        encoded_list = []
        for lst in result:
            tmp1 = []
            for i in lst:
                tmp1.append(list(all_combinations).index(i))
            encoded_list.append(tmp1)
            # convert the list to array
        encoded_array = np.array(encoded_list)
        # encode the sequence to 0 and 1 and put in two
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder.fit(np.array(range(len(all_combinations))).reshape(-1, 1))
        encoded_onehot = onehot_encoder.transform(encoded_array.reshape(-1, 1)).reshape(-1,
                                                                                        len(result[0]) * len(
                                                                                            all_combinations))
        X__ = encoded_onehot
        # convert to numpy array
        if method == "regression":
            y__ = df_in_threshold[wavelength].values
        elif method == "classification":
            Y = df_in_threshold['class'].values
            y__ = np_utils.to_categorical(Y)
        with open(path_wavelength + '/pX_' + wavelength + '_' + str(n_gram) + 'g' + '.npy', 'wb') as f:
            np.save(f, X__)
        with open(path_wavelength + '/py_' + wavelength + '_' + str(n_gram) + 'g' + '.npy', 'wb') as f:
            np.save(f, y__)
        df_in_threshold.to_csv(path_wavelength + '/' + wavelength + '_' + str(n_gram) + 'g_in_threshold' + '.csv',
                               index=False)
        df_out_threshold.to_csv(path_wavelength + '/' + wavelength + '_' + str(n_gram) + 'g_out_threshold' + '.csv',
                                index=False)


def estimation_mean(df_actual, df_predict, sequence):
    """
    This function calculates the average difference between the actual value of DDF and the predicted value
    Negative value means over estimation. Positive value means under estimation
    df_actual: a dataframe contains the experimental DFF value for each sequence
    df_predict: a dataframe contains the predicted DFF value for each sequence
    sequence: sequence for checking for the estimation mean
    output:
    difference: the estimation mean over selected wavelength
    """
    actual = df_actual.loc[df_actual.index[df_actual['sequence'] == sequence]]
    predict = df_predict.loc[df_predict.index[df_predict['sequence'] == sequence]]
    difference = float(pd.concat([predict, actual]).set_index('sequence').diff().dropna().mean(axis=1))
    return difference


def machine_learning_regression(model, name, numpy_array_path, model_numbers, n_gram, r2_score_limit, metric_csv_name,
                                validate_sequence, df_validate, nucleic_list, max_time_hour):
    metric_parameter = {'model_name': [], 'random_state': [], 'ngram': [], 'c_cl0_train': [], 'c_cl1_train': [],
                        'c_cl0_test': [], 'c_cl1_test': [], 'r2_score': [], 'mean_squared_error': [],
                        'mean_absolute_error': [], 'max_error': [], 'explained_variance_score': []}
    numpy_arrays_list = glob.glob(numpy_array_path)
    for wavelength_dir in numpy_arrays_list:
        x_numpy_file = glob.glob(wavelength_dir + "/pX*.npy")[0]
        y_numpy_file = glob.glob(wavelength_dir + "/py*.npy")[0]
        for ng in n_gram:
            count_models = 1
            X = np.load(x_numpy_file)
            y = np.load(y_numpy_file)
            random_list = []
            timeout = time.time() + 3600*max_time_hour
            while count_models <= model_numbers and time.time() <= timeout:
                sub_random_dir_name = wavelength_dir + '/' + str(count_models).zfill(2)
                random_seed = random.randint(0, 2**32-1)
                while random_seed in random_list and time.time() <= timeout:
                    random_seed = random.randint(0, 2**32-1)
                random_list.append(random_seed)
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed)
                count_train_class_0 = 0
                count_train_class_1 = 0
                count_test_class_0 = 0
                count_test_class_1 = 0

                for ctr in range(len(y_train)):
                    if int(y_train[ctr]) == 1:
                        count_train_class_0 = count_train_class_0 + 1
                    else:
                        count_train_class_1 = count_train_class_1 + 1

                for cte in range(len(y_test)):
                    if int(y_test[cte]) == 1:
                        count_test_class_0 = count_test_class_0 + 1
                    else:
                        count_test_class_1 = count_test_class_1 + 1
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                m_r2_score = metrics.r2_score(y_test, y_predict)
                if m_r2_score >= r2_score_limit:
                    os.makedirs(sub_random_dir_name)
                    with open(sub_random_dir_name + '/X_train_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                        np.save(f, X_train)
                    with open(sub_random_dir_name + '/X_test_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                        np.save(f, X_test)
                    with open(sub_random_dir_name + '/y_train_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                        np.save(f, y_train)
                    with open(sub_random_dir_name + '/y_test_' + str(ng) + 'gram' + '.npy', 'wb') as f:
                        np.save(f, y_test)

                    dict_test = {'test_Num': [], 'Pred_R': [], 'Exp_R': []}
                    for test_sequence in range(len(X_test)):
                        y_reg = model.predict(X_test[test_sequence].reshape(1, -1))
                        dict_test['test_Num'].append(test_sequence)
                        dict_test['Pred_R'].append(y_reg[0])
                        dict_test['Exp_R'].append(y_test[test_sequence])

                    # save the predicted to the pandas dataframe
                    df_predicted = pd.DataFrame.from_dict(dict_test)
                    df_predicted.to_csv(sub_random_dir_name + '/pred_xtest.csv', index=False)
                    x_fig_plt = df_predicted['Pred_R'].values
                    y_fig_plt = df_predicted['Exp_R'].values
                    fig_predict_plt = plt.figure()
                    plt.scatter(x_fig_plt, y_fig_plt)
                    plt.xlabel('Pred_R')
                    plt.ylabel('Exp_R')
                    plt.plot([min(x_fig_plt), max(y_fig_plt)], [min(x_fig_plt), max(y_fig_plt)], '--', c='red')
                    fig_predict_plt.savefig(sub_random_dir_name + '/pred_xtest.png', dpi=600)
                    # calculate parameters
                    y_predict = model.predict(X_test)
                    joblib.dump(model, sub_random_dir_name + '/' + str(ng) + 'gram' + '_' + str(random_seed) + 'rnds' +
                                '.sav')
                    if validate_sequence:
                        dict_validate = {'sequence': [], str(count_models): []}
                        X_validate = df_validate['sequence'].values
                        X_validate_ = np.array([list(_) for _ in X_validate])
                        le = preprocessing.LabelEncoder().fit(nucleic_list)
                        X_validate_ = le.transform(X_validate_.reshape(-1)).reshape(-1, len(X_validate[0]))
                        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
                        onehot_encoder.fit(np.array(list(range(len(nucleic_list)))).reshape(-1, 1))
                        X_validate__ = onehot_encoder.transform(X_validate_.reshape(-1, 1)).reshape(-1,
                                                                                                    len(nucleic_list) *
                                                                                                    len(X_validate[0]))
                        for test_val in range(len(X_validate__)):
                            y_reg_validate = model.predict(X_validate__[test_val].reshape(1, -1))
                            dict_validate['sequence'].append(X_validate[test_val])
                            dict_validate[str(count_models)].append(y_reg_validate[0])
                        df_validate_result = pd.DataFrame.from_dict(dict_validate)
                        df_validate_result.to_csv(sub_random_dir_name + '/prediction_validate.csv', index=False)

                    m_r2_score = metrics.r2_score(y_test, y_predict)
                    m_mean_squared_error = metrics.mean_squared_error(y_test, y_predict)
                    m_mean_absolute_error = metrics.mean_absolute_error(y_test, y_predict)
                    m_max_error = metrics.max_error(y_test, y_predict)
                    m_explained_variance_score = metrics.explained_variance_score(y_test, y_predict)
                    metric_parameter['model_name'].append(str(name))
                    metric_parameter['ngram'].append(ng)
                    metric_parameter['random_state'].append(random_seed)
                    metric_parameter['c_cl0_train'].append(count_train_class_0)
                    metric_parameter['c_cl1_train'].append(count_train_class_1)
                    metric_parameter['c_cl0_test'].append(count_test_class_0)
                    metric_parameter['c_cl1_test'].append(count_test_class_1)
                    metric_parameter['r2_score'].append(m_r2_score)
                    metric_parameter['mean_squared_error'].append(m_mean_squared_error)
                    metric_parameter['mean_absolute_error'].append(m_mean_absolute_error)
                    metric_parameter['max_error'].append(m_max_error)
                    metric_parameter['explained_variance_score'].append(m_explained_variance_score)
                    plt.close('all')
                    count_models += 1
    df_para = pd.DataFrame.from_dict(metric_parameter)
    df_para.to_csv(metric_csv_name, index=False)


def machine_learning_classification():
    pass
