import argparse
import time
import pandas as pd
import re
import pickle
import seaborn as sns
import numpy as np
from sklearn import metrics
from itertools import groupby, chain

from tqdm import tqdm

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import OneHotEncoder as ohe

def make_one_hot_vectors(y):
    unique_labels = np.unique(y)
    #print(unique_labels)
    y_ohe = []
    for j in y:
        oh_j = 1.0 * (unique_labels == j)  # onehot for one sample
        y_ohe.append(oh_j)
    y_ohe = np.array(y_ohe)
    #print(len(y_ohe))
    return y_ohe

def most_frequent_label(Y_):
    #Y = tuple(Y)
    print(Y_)
    Y=[]
    for y in Y_:
        Y.append(tuple(y))
    Hash = dict() #all labels with frequences
    for i, y in enumerate(Y):
    #for i in range(len(y)):
        if y in Hash:
            Hash[y] += 1
        else:
            Hash[y] = 1
    # find the max frequency
    print('HHHHHHHHHHHHHHH', Hash)
    max_count = 0
    res = -1
    for y in Hash:
        if (max_count < Hash[y]):
            res = y
            max_count = Hash[y]
    return res


def get_dataset_sample (pkl_file, number):
    with open(pkl_file, 'rb') as file:
        all_data = pickle.load(file)
    action = all_data[number]
    array = action['data'] #shape == (n_frames, n_keypoints, n_coordinates)
    label = action['action_id']
    return array, label

def get_X_y_list(pkl_file):
    with open(pkl_file, 'rb') as file:
        all_data = pickle.load(file)
    X_list = []
    labels_list = []
    for i in range(len(all_data)):
    #for i in range(20):
        x_sample = get_dataset_sample(pkl_file, i)[0]  #shape == (n_frames, n_keypoints, n_coordinates)
        label = get_dataset_sample(pkl_file, i)[1]
        #print('AAAAAAA', x_sample.shape)
        X_list.append(x_sample)
        labels_list.append(label)
    return X_list, labels_list


def get_X_y_extended(pkl_file, n_frames = 30, stride = 20, with_derivatives = False): #for each data sample makes fixed frames split and saves it as another sample with the same label
    X, Y = get_X_y_list(pkl_file)
    big_list = []
    labels = []
    #print(len(X))
    for _, x in enumerate(X):
        #print(x.shape)
        #print(Y[_])
        #print('---------------------------------------------')
        for iter in range(0, x.shape[0], stride): #iter over number of frames
                n_left = x.shape[0] - iter
                #print(n_left)
                if n_left >= n_frames:
                    if with_derivatives:
                        x_chunk = x[iter:iter + n_frames, :, :]
                        x_chunk_d1 = np.diff(x_chunk, n=1, axis=0)
                        x_chunk_d2 = np.diff(x_chunk, n=2, axis=0)
                        x_all = np.concatenate((x_chunk, x_chunk_d1, x_chunk_d2), axis=0)
                    else:
                        x_all = x[iter:iter + n_frames, :, :]
                    big_list.append(x_all.reshape(-1, 1))
                    labels.append(Y[_])
    X_extended = np.squeeze(np.stack(big_list))
    y_extended = np.stack(labels)
    return X_extended, y_extended
'''

def get_X_y_extended(pkl_file, n_frames = 30, stride = 20, with_derivatives = False, scaling = False): #for each data sample makes fixed frames split and saves it as another sample with the same label
    X, Y = get_X_y_list(pkl_file)
    big_list = []
    only_x = []
    only_features = []
    labels = []
    print(len(X))
    for _, x in enumerate(X):
        print(x.shape)
        print(Y[_])
        print('---------------------------------------------')
        for iter in range(0, x.shape[0], stride): #iter over number of frames
                n_left = x.shape[0] - iter
                #print(n_left)
                if n_left >= n_frames:
                    if with_derivatives:
                        x_chunk = x[iter:iter + n_frames, :, :]
                        x_chunk_d1 = np.diff(x_chunk, n=1, axis=0)
                        x_chunk_d2 = np.diff(x_chunk, n=2, axis=0)
                        x_features = np.concatenate((x_chunk_d1, x_chunk_d2), axis=0)
                        #x_all = np.concatenate((x_chunk, x_chunk_d1, x_chunk_d2), axis=0)
                        only_x.append(x_chunk)#.reshape(-1,1))
                        only_features.append(x_features) #.reshape(-1, 1))
                    else:
                        x_chunk = x[iter:iter + n_frames, :, :]
                        only_x.append(x_chunk) #.reshape(-1, 1))
                        #big_list.append(x_all.reshape(-1, 1))
                    labels.append(Y[_])
    only_features = np.array(only_features)
    print(only_features.shape, 'dsvld;bvkdlbk;')
    only_x = np.array(only_x)
    if scaling:
        only_features = StandardScaler().fit_transform(only_features)
    if with_derivatives:
        X_extended = only_x.reshape(len(X), -1)
    else:
        X_extended = np.concatenate((only_x, only_features), axis = 0).reshape(len(X),-1)
    #X_extended = np.squeeze(np.stack(big_list))
    y_extended = np.stack(labels)
    return X_extended, y_extended

'''
"""
def get_X_y_extended_with_derivatives(pkl_file, n_frames = 30, stride = 20): #for each data sample makes fixed frames split and saves it as another sample with the same label
    X, Y = get_X_y_list(pkl_file)
    big_list = []
    labels = []
    print(len(X))
    for _, x in enumerate(X):
        print(x.shape)
        print(Y[_])
        print('---------------------------------------------')
        for iter in range(0, x.shape[0], stride): #iter over number of frames
                n_left = x.shape[0] - iter
                #print(n_left)
                if n_left >= n_frames:
                    x_chunk = x[iter:iter + n_frames, :, :]
                    x_chunk_d1 = np.diff(x_chunk, n=1, axis=0)
                    x_chunk_d2 = np.diff(x_chunk, n=2, axis=0)
                    x_all = np.concatenate((x_chunk, x_chunk_d1, x_chunk_d2), axis=0)
                    big_list.append(x_all.reshape(-1, 1))
                    labels.append(Y[_])
    X_extended = np.squeeze(np.stack(big_list))
    y_extended = np.stack(labels)
    return X_extended, y_extended
"""

def predict_average_label(X, trained_model, n_frames = 30, stride = 20, with_derivatives = False, selected_features = None, pca = None):
    Y_pred = []
    for _, x in enumerate(X):
        labels_for_x = []
        x_fragments = []
        #print(x.shape)
        #print('---------------------------------------------')
        for iter in range(0, x.shape[0], stride): #iter over number of frames, x.shape = (n_frames, n_keypoints, n_coordinates)
                n_left = x.shape[0] - iter
                if n_left >= n_frames:
                    if with_derivatives:
                        x_chunk = x[iter:iter + n_frames, :, :]
                        x_chunk_d1 = np.diff(x_chunk, n=1, axis=0)
                        x_chunk_d2 = np.diff(x_chunk, n=2, axis=0)
                        x_fixed_size = np.concatenate((x_chunk, x_chunk_d1, x_chunk_d2), axis=0).reshape(-1, 1)
                        if selected_features is not None:
                            x_fixed_size=np.array(x_fixed_size)[selected_features, :]
                        if pca is not None:
                            x_fixed_size = pca.transform(x_fixed_size.T)

                    else:
                        x_fixed_size = x[iter:iter + n_frames, :, :].reshape(-1, 1)
                        if selected_features is not None:
                            #print(x_fixed_size.shape)
                            x_fixed_size=np.array(x_fixed_size)[selected_features, :]
                        if pca is not None:
                            x_fixed_size = pca.transform(x_fixed_size.T)
                    x_fragments.append(x_fixed_size)
                    #y_pred_fixed = trained_model.predict([np.squeeze(x_fixed_size).T])
                    #labels_for_x.append(y_pred_fixed)
        y_pred = trained_model.predict(np.squeeze(np.array(x_fragments)))
        labels_for_x = y_pred
        #print('QQQQQQQ', labels_for_x)
        #the most frequent label:
        #y_pred = most_frequent_label(labels_for_x)
        values, counts = np.unique(np.array(labels_for_x), return_counts=True)
        index = np.argmax(counts) #index of most frequent
        #print(values[index])
        y_pred = values[index]

        Y_pred.append(y_pred)
    return np.array(Y_pred)


def get_pd_from_array(X, y):
    # creating a list of column names
    column_values = np.array(range(len(X[0])))
    #print(column_values)
    X_df = pd.DataFrame(data = X, columns = column_values)
    print(X_df.head())
    col_val = ['target']
    y_df = pd.DataFrame(data = y, columns = col_val)

    print(y_df.head(5))
    return X_df, y_df

def get_new_features(X_df):
    print(X_df.head(7))
    dif1 = X_df.diff()
    dif2 = X_df.diff(periods = 2)
    col_dic = {}
    for x in X_df.columns:
        col_dic[x]='dif1_'+f'{x}'
    X_dif1 = dif1.rename(columns=col_dic)
    #print(dif1.head(10))
    #print(dif2.head(10))

    col_dic2= {}
    for x in X_df.columns:
        col_dic2[x]='dif2_'+f'{x}'
    X_dif2 = (dif1 - dif2).rename(columns = col_dic2)
    #print(X_dif2.head(7))

    #second derivative
    X_concat_NA= pd.concat([X_df, X_dif1, X_dif2], axis=1)
    X_concat = X_concat_NA.dropna()
    print(X_concat_NA.head(7))
    print(X_concat.head(7))
    return X_concat

if __name__=='__main__':
    #train_data_path = 'data/HDM05-122/HDM05-122-only-annot-subseq-fold-1-of-2.pkl'
    #test_data_path = 'data/HDM05-122/HDM05-122-only-annot-subseq-fold-2-of-2.pkl'

    train_data_path = 'data/HDM05-15/HDM05-15-part1.pkl'
    test_data_path = 'data/HDM05-15/HDM05-15-part2.pkl'

    print('Extracting dataset ...')

    X_train, y_train = get_X_y_extended(train_data_path, with_derivatives = True)

    X_test, y_test = get_X_y_list(test_data_path)
    #X_test, y_test = get_X_y_extended(test_data_path)

    print(f'shape of the train dataset {X_train.shape}')
    print(f'shape of the train labels {y_train.shape}')

    print(f'shape of the test dataset {len(X_test)} * n_frames * 3')
    print(f'shape of the test labels {len(y_test)}')

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    print('Catboost training on original dataset')
    model = CatBoostClassifier(
        iterations=50,
        random_seed=43,
        loss_function='MultiClass'
    )
    model.fit(
        X_train, y_train,
        #cat_features=cat_features,
        eval_set=(X_val, y_val),
        verbose=False,
        plot=True
    )

    #y_pred = model.predict(X_test)
    st = time.time()
    y_pred = predict_average_label(X_test, model, with_derivatives = True)
    fin = time.time()
    total_n_frames  = 266528
    ind_max = 713 #index of element with maxinum namber of frames in test dataset
    print(f'Time for predicting labels on original dataset, total: {fin - st}, per frame: {(fin -st)/total_n_frames }')

    x_len_max = X_test[ind_max]
    y_len_max = y_test[ind_max]
    st = time.time()
    y_pred_max = predict_average_label(np.expand_dims(x_len_max, 0), model, with_derivatives = True)
    fin = time.time()

    print(f'Time for the most long data sequense {fin - st}')
    #y_pred_proba = model.predict_proba(X_train)
    '''
    #print(y_pred_proba)
    #print(y_pred_proba.shape)
    #print(np.squeeze(y_pred))
    #print('----------------------------------------------========================================------------------------------')
    #print(y_test)
    #y_pred = predict_average_label(X_test, model)
    '''

    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy{acc}, original dataset')

    print(metrics.classification_report(y_test, y_pred, digits=3))

    '''
    def plot_confusion_matrix(cm, labels_name, title):
        cm = cm / cm.sum(axis=1)[:, np.newaxis]  # нормализация
        plt.imshow(cm, interpolation='nearest')  # Отобразить изображение в определенном окне
        plt.title(title)  # заголовок изображения
    
        plt.colorbar()
        num_class = np.array(range(len(labels_name)))  # Получить интервал количества меток
        plt.xticks(num_class, labels_name, rotation=90)  # Распечатать этикетку по координате оси x
        plt.yticks(num_class, labels_name)  # Распечатать этикетку по координате оси Y
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    '''

    r = metrics.multilabel_confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    print(r)
    #save confusion seaborn
    cm = r.reshape(r.shape[0], r.shape[1]*r.shape[2])
    cm = np.array(cm)  # Преобразуйте тип списка в тип массива. Если это уже тип массива numpy, пропустите этот шаг.
    labels_name = np.unique(y_test)# здесь набор присвоения меток горизонтальных и вертикальных координат
    #plot_confusion_matrix(cm, labels_name, "confusion_matrix")  # Вызов функции
    cm = metrics.confusion_matrix(y_test, y_pred,)
    fig = sns.heatmap(cm, annot = True, fmt = 'g', cmap = 'Blues', xticklabels = labels_name , yticklabels = labels_name )
    figure = fig.get_figure()
    #figure.savefig('catboost_with_derivatives_X_test_seq.png')
    plt.show()

    #feature importances
    features = range(8091)  # 30*31*3-3*31*3 ##0..8090
    importances = model.feature_importances_
    indices = np.argsort(importances)

    num_features = 30  #top n features
    figure2 = plt.figure()
    plt.title('Feature Importances')
    feature_dic = {}
    for i in range(2790):
        feature_dic[i] = 'coordinate'
    for i in range(2791, 5488):
        feature_dic[i] = 'velocity'
    for i in range(5489, 8091):
        feature_dic[i] = 'acceleration'
    plt.barh(range(num_features), importances[indices[-num_features:]], color='b', align='center')
    plt.yticks(range(num_features), [feature_dic[features[i]] for i in indices[-num_features:]], fontsize=7)
    plt.xlabel('Relative Importance')
    plt.show()
    # figure2.savefig('fi_catboost_withderivatives_X_test_seq.png')

    q = np.quantile(importances, q=0.20)
    #print(q5)
    #print(q, 'BOARDER')
    #print('MAAAAAAAAAAAAAAAAX', importances[indices[-1]])
    #print('Min', importances[indices[0]])
    indices_best = []
    for i, imp in enumerate(importances):
        if (imp > q):
            indices_best.append(i)
    print('Indexes of best features', indices_best)
    print(len(importances), 'INITIAL SIZE')
    print(len(indices_best))

    #count labels of important features
    print('How much features of each class of features have nonzero feature_importance')
    counts = {'coordinate':0, 'velocity':0, 'acceleration':0}
    for i in indices_best:
        if i in range(2790):
            counts['coordinate'] +=1
        if i in range(2791, 5488):
            counts['velocity'] += 1
        if i in range(5489, 8091):
            counts['acceleration']+=1
    print(counts)

    indices_best = np.asarray(indices_best)
    #print(np.array(X_train).shape, 'DDDDDDDDDDDD')
    X_train_best_features = np.array(X_train)[:, indices_best]
    X_val_best_features = np.array(X_val)[:, indices_best]

    X_test_best_features = X_test#[:][indices_best]
    #X_test_best_features, y_test = get_X_y_list(test_data_path, with_derivatives=True, selected_features = indices_best)

    print(f'Train dataset with best features shape {X_train_best_features.shape}')
    model = CatBoostClassifier(iterations=50, random_seed=43, loss_function='MultiClass')
    model.fit(
        X_train_best_features, y_train,
        # cat_features=cat_features,
        eval_set=(X_val_best_features, y_val),
        verbose=False,
        plot=True
    )

    # y_pred = model.predict(X_test)
    y_pred = predict_average_label(X_test_best_features, model, with_derivatives=True, selected_features = indices_best)
    print('### BEST (NON ZERO) FEATURES #####')
    print(metrics.classification_report(y_test, y_pred, digits=3))


    ######################## PCA ################################

    n_components = np.array([250, 150, 50, 20, 10])
    for n in n_components:
        pca = PCA(n_components = n)
        X_train_pca = pca.fit_transform(np.array(X_train))
        X_val_pca = pca.transform(X_val)
        X_test_pca = X_test
        print(f'Shape of dataset with PCA on {n} components: {X_train_pca.shape}')
        model = CatBoostClassifier(iterations=50, random_seed=43, loss_function='MultiClass')
        model.fit(
            X_train_pca, y_train,
            # cat_features=cat_features,
            eval_set=(X_val_pca, y_val),
            verbose=False,
            plot=True
        )
        # y_pred = model.predict(X_test)
        y_pred = predict_average_label(X_test_pca, model, with_derivatives=True, pca = pca)
        print('### PCA WITH'+str(n)+'FEATURES #####')
        print(metrics.classification_report(y_test, y_pred, digits=3))




