import os
import numpy as np

def get_data(data_path, recording_ids):
    files = os.listdir(data_path)
    X_files = [f for f in files if f.endswith('.npy') and f.startswith('X')]
    X_files.sort()
    Y_files = [f for f in files if f.endswith('.npy') and f.startswith('Y')]
    Y_files.sort()

    X_files = np.array(X_files)
    Y_files = np.array(Y_files)

    X_list = [np.load(data_path + f) for f in X_files[recording_ids]]
    Y_list = [np.load(data_path + f) for f in Y_files[recording_ids]]

    return X_list, Y_list

def get_test_train_data(X_list, Y_list, train_split):
    Y_list_mean_subtracted = [Y - np.mean(Y, axis=0, keepdims=True) for Y in Y_list]
    X_list_shuffled = []
    Y_list_shuffled = []
    Y_list_mean_subtracted_shuffled = []

    for i in range(len(X_list)):
        indices = np.arange(X_list[i].shape[0])
        np.random.shuffle(indices)
        X_list_shuffled.append(X_list[i][indices])
        Y_list_mean_subtracted_shuffled.append(Y_list_mean_subtracted[i][indices])
        Y_list_shuffled.append(Y_list[i][indices])

    X_train = [X[:int(train_split * X.shape[0])] for X in X_list_shuffled]
    X_test = [X[int(train_split * X.shape[0]):] for X in X_list_shuffled]
    
    Y_train = [Y[:int(train_split * Y.shape[0])] for Y in Y_list_shuffled]
    Y_test = [Y[int(train_split * Y.shape[0]):] for Y in Y_list_shuffled]

    Y_train_mean_subtracted = [Y[:int(train_split * Y.shape[0])] for Y in Y_list_mean_subtracted_shuffled]
    Y_test_mean_subtracted = [Y[int(train_split * Y.shape[0]):] for Y in Y_list_mean_subtracted_shuffled]

    return X_train, X_test, Y_train, Y_test, Y_train_mean_subtracted, Y_test_mean_subtracted