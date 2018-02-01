import numpy as np
import sys
from random import randrange
from random import seed
from csv import reader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

"""
Housing: This is a regression dataset where the task is to predict the value of houses 
in the suburbs of Boston based on thirteen features that describe different aspects that 
are relevant to determining the value of a house, such as the number of rooms, levels of
pollution in the area, etc.
"""

class Preprocess:
    def __init__(self, filename):
        self.filename = filename

    # Load a CSV file
    def load_csv(self):
        file = open(self.filename, "r")
        lines = reader(file)
        dataset = list(lines)
        return dataset

    # Convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            if len(row) != 0:
                row[column] = float(row[column].strip())

    def pre_process(self):
        # convert string attributes to integers
        dataset = self.load_csv()
        for col in range(len(dataset[0])):
            self.str_column_to_float(dataset, col)

        x = np.array(dataset)[:, 0:-1]
        y = np.array(dataset)[:, -1]

        # normalize features
        x = self.normalize_data(x)

        # concatenate
        dataset = np.concatenate((x, np.array([y]).T), axis=1)
        return dataset

    def normalize_data(self, dataset, type="min_max"):
        if type == "std":
            scaler = StandardScaler().fit(dataset)
            X_scaled = scaler.transform(dataset)
        if type == "l1" or type == "l2":
            scaler = Normalizer(norm=type)
            X_scaled = scaler.fit_transform(dataset)
        if type == "min_max":
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(dataset)
        return X_scaled

    def get_thresholds(self):
        # shuffle is done in the split-n-fold part
        # generate thresholds
        dataset = self.pre_process()
        num_feature = len(dataset[0]) - 1
        num_threshold = len(dataset[:, 0]) - 1  # number of instances - 1
        thresholds = []
        for index in range(num_feature):  # loop all the features (0, 1, 2, 3)
            # sort the dataset by the current index(feature)
            sorted_data = dataset[np.argsort(dataset[:, index])]
            feature = sorted_data[:, index]
            label = sorted_data[:, -1]
            temp = []
            for row in range(num_threshold):
                temp.append((feature[row] + feature[row + 1]) * 1. / 2)
            # add the list of thresholds for the feature
            thresholds.append(set(temp))
        return thresholds

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate SSE
def accuracy_metric(actual, predicted):
    error = 0
    for i in range(len(actual)):
        error += ((actual[i] - predicted[i])**2)
    return error


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['thres']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def evaluation(dataset, thresholds, n_folds=10, mean_ratio=[0.05, 0.10, 0.15, 0.20]):
    folds = cross_validation_split(dataset, n_folds)
    acc = {}
    for ratio in mean_ratio:
        min_size = ratio * len(dataset)
        scores = list()
        for i in range(len(folds)):
            fold = folds[i]
            train_set = list(folds)
            train_set.pop(i)
            train_set = sum(train_set, [])
            node = build_tree(train_set, min_size, thresholds)
            predicted = list()
            for row in fold:
                predicted.append(predict(node, row))
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        acc[ratio] = {}
        acc[ratio]["acc"] = sum(scores) / float(len(scores))
        acc[ratio]["std"] = np.std(np.array(scores))
    return acc


def create_node(dataset, thresholds):
    SSE = sys.maxsize
    node = {}
    for feature in range(len(dataset[0]) - 1):
        curr_node = sum_of_errors(dataset, feature, thresholds)
        if curr_node['SSE'] < SSE:
            SSE = curr_node['SSE']
            node = curr_node
    return node


# calculate the squared sum of one tree node
def squared_error(label):
    # predicted value per tree node
    #print label
    mean = np.mean(label)
    #print mean
    result = 0.0
    for y in label:
        result += ((y - mean) ** 2)
    return result


# pick the best thresholds for one feature
def sum_of_errors(dataset, feature, thresholds):
    min_errors, threshold, group = sys.maxsize, sys.maxsize, None
    for thres in thresholds[feature]:
        left, right = split(dataset, feature, thres)
        left_label = np.array([row[-1] for row in left])
        right_label = np.array([row[-1] for row in right])

        left_err = squared_error(left_label)
        right_err = squared_error(right_label)

        sum_of_err = (float(left_label.shape[0]) / len(dataset)) * left_err + (float(right_label.shape[0]) / len(dataset)) * right_err
        if sum_of_err <= min_errors:
            min_errors, threshold, group = sum_of_err, thres, (left, right)

    return {'SSE': min_errors, 'index': feature, 'group': group, 'thres': threshold}


# for a given threshold number, split the dataset into two group
def split(dataset, feature, threshold):
    left, right = list(), list()
    for row in dataset:
        if row[feature] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)


def create_leaf(group):
    outcomes = np.mean([row[-1] for row in group])
    return outcomes


def build_tree_helper(node, min_size, thresholds):
    left, right = node['group']
    del (node['group'])
    # if left is empty or right is empty, no split
    if len(left) == 0:
        node['left'] = node['right'] = create_leaf(right)
        return

    if len(right) == 0:
        node['right'] = node['left'] = create_leaf(left)
        return

    # left child
    if len(left) <= min_size:
        node['left'] = create_leaf(left)
    else:
        node['left'] = create_node(left, thresholds)
        build_tree_helper(node['left'], min_size, thresholds)

    # right child
    if len(right) <= min_size:
        node['right'] = create_leaf(right)
    else:
        node['right'] = create_node(right, thresholds)
        build_tree_helper(node['right'], min_size, thresholds)


def build_tree(dataset, min_size, thresholds):
    root = create_node(dataset, thresholds)
    build_tree_helper(root, min_size, thresholds)
    return root


def main():
    seed(1)
    preprocess = Preprocess('housing.csv')
    dataset = preprocess.pre_process()
    thres = preprocess.get_thresholds()
    print(evaluation(dataset, thres, n_folds=10, mean_ratio=[0.05, 0.10, 0.15, 0.20]))


if __name__ == '__main__':
    main()

