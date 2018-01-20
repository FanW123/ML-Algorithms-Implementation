import collections, numpy
import numpy as np
from sklearn import preprocessing
import sys
from random import randrange
from random import seed
from csv import reader


class Preprocess:
    def __init__(self, filename):
        self.filename = filename

    def load_csv(self):
        file = open(self.filename, "r")
        lines = reader(file)
        dataset = list(lines)
        return dataset

    def pre_process(self):
        # convert string attributes to integers if needed
        dataset = self.load_csv()
        x = np.array(dataset)[:, 0:-1]
        y = np.array(dataset)[:, -1]
        # normalize features
        # x = preprocessing.normalize(x, axis=0)

        # concatenate
        dataset = np.concatenate((x, np.array([y]).T), axis=1)
        return dataset

    def get_thresholds(self):
        binary_feature_matrix = []  # list
        attri_table = []
        dataset = self.pre_process()
        for feature in range(len(dataset[0]) - 1):
            attri_list = list(set(dataset[:, feature]))
            attri_table.append(attri_list)
            num_attri = len(attri_list)  # the feature has num_attri's attributes
            matrix = np.zeros((dataset.shape[0], num_attri))  # 2-d ndarray
            for row in range(len(dataset)):
                for col in range(len(attri_list)):
                    if dataset[row][feature] == attri_list[col]:
                        matrix[row][col] = 1
                        break
            binary_feature_matrix.append(matrix)
        return binary_feature_matrix, attri_table


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


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] == node['thres']:
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
    # get the number of classes
    # label_list = list(set(row[-1]) for row in dataset)

    info_gain = 0.0
    new_node = {}
    for feature in range(len(dataset[0]) - 1):
        IG, node = max_info_gain(dataset, feature, thresholds)
        if IG >= info_gain:
            info_gain = IG
            new_node = node
    return new_node


def max_info_gain(dataset, feature, thresholds):
    """
        param: feature vector
        return: max information gain for the given feature
    """
    base_ent = calc_ent(np.array(dataset)[:, -1])
    node = calc_condition_ent(dataset, thresholds, feature)
    IG = base_ent - node['condition_ent']
    return IG, node


def calc_ent(label):
    """
        calculate entropy for the feature
    """
    # get the number of classes
    label_list = set([label[i] for i in range(label.shape[0])])
    ent = 0.0
    for l in label_list:
        p = float(label[label == l].shape[0]) / label.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def calc_condition_ent(dataset, thresholds, feature):
    """
        calculate ent H(y|x)
    """
    # calc ent(y|x)
    #     feature_thresholds = thresholds[:, feature]
    # thresholds = [2]
    min_ent, group, key = sys.maxsize, None, ''
    attri_table = thresholds[1][feature]

    for i in range(len(attri_table)):
        left, right = split(dataset, feature, thresholds, i)
        left_label = np.array([row[-1] for row in left])
        right_label = np.array([row[-1] for row in right])

        left_ent = calc_ent(left_label)
        right_ent = calc_ent(right_label)
        ent = (float(left_label.shape[0]) / len(dataset)) * left_ent + (
                float(right_label.shape[0]) / len(dataset)) * right_ent
        if ent < min_ent:
            min_ent, group, key = ent, (left, right), attri_table[i]
    return {'condition_ent': min_ent, 'index': feature, 'group': group, 'thres': key}


def split(dataset, feature, thresholds, col):
    left, right = list(), list()
    # get the index of the feature == 1
    # add the dataset[index] to left list
    print col
    index_list = np.where(thresholds[0][feature][:, col] == 1)[0]
    for i in range(len(dataset)):
        if i in index_list:
            left.append(dataset[i])
        else:
            right.append(dataset[i])
    return np.array(left), np.array(right)


def create_leaf(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def build_tree_helper(node, min_size, thresholds):
    if not thresholds:
        return

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
    if len(left) <= min_size or len(set(left[:, -1])) == 1:
        node['left'] = create_leaf(left)
    else:
        node['left'] = create_node(left, thresholds)
        build_tree_helper(node['left'], min_size, thresholds)

    # right child
    if len(right) <= min_size or len(set(right[:, -1])) == 1:
        node['right'] = create_leaf(right)
    else:
        node['right'] = create_node(right, thresholds)
        build_tree_helper(node['right'], min_size, thresholds)


def build_tree(dataset, min_size, thresholds):
    root = create_node(dataset, thresholds)
    build_tree_helper(root, min_size, thresholds)
    return root

preprocess = Preprocess('mushroom.csv')
dataset = preprocess.pre_process()
thresholds = preprocess.get_thresholds()
print(evaluation(dataset, thresholds, n_folds=10, mean_ratio=[0.05, 0.10, 0.15]))