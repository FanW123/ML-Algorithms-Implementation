import sklearn
from sklearn import datasets
import numpy as np
import math
from sklearn import preprocessing
import sys
from random import randrange
from random import seed
from sklearn.model_selection import KFold

# predict one of the three sub-types of the Irisower given four different physical features.
# These features include the length and width of the sepals and the petals.
# There are a total of 150 instances with each class having 50 instances.
def pre_process():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    # normalize features
    x = preprocessing.normalize(x, axis=0)
    # concatenate
    dataset = np.concatenate((x, np.array([y]).T), axis=1)
    return dataset


def get_thresholds(dataset): # type(dataset) = ndarray
    # shuffle is done in the split-n-fold part
    # generate thresholds
    num_feature = len(dataset[0]) - 1
    num_threshold = len(dataset[:, 0]) - 1
    thresholds = []
    for index in range(num_feature):  # loop all the features (0, 1, 2, 3)
        # sort the dataset by the current index(feature)
        sorted_data = dataset[np.argsort(dataset[:,index])]
        feature = sorted_data[:, index]
        label = sorted_data[:, -1]
        temp = []
        for row in range(num_threshold):
            if row == 0 or label[row] != label[row + 1]:
                temp.append((feature[row] + feature[row + 1]) * 1. / 2)
        #add the list of thresholds for the feature
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


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


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
        acc[ratio]["acc"] = sum(scores) / (float(len(scores)) * 100)
        acc[ratio]["std"] = np.std(np.array(scores))
    return acc

# sklearn cross validation, need to do shuffling in preprocessing
# def evaluation(dataset, thresholds, n_folds = 10, mean_ratio = [0.05, 0.10, 0.15, 0.20]):
#     acc = {}
#     for ratio in mean_ratio:
#         min_size = ratio * len(dataset)
#         scores = list()
#         kf = KFold(n_splits=10)
#         for train, test in kf.split(dataset):
#             node = build_tree(dataset[train], min_size, thresholds)
#             predicted = list()
#             for row in dataset[test]:
#                 predicted.append(predict(node, row))
#             actual = dataset[test][:, -1]
#             accuracy = accuracy_metric(actual, predicted)
#             scores.append(accuracy)
#         acc[ratio] = {}
#         acc[ratio]["acc"] = sum(scores)/float(len(scores))
#         acc[ratio]["std"] = np.std(np.array(scores))
#     return acc


def create_node(dataset, thresholds):
    # get the number of classes
    # label_list = list(set(row[-1]) for row in dataset)

    info_gain = 0.0
    new_node = {}
    for feature in range(len(dataset[0]) - 1):
        if len(thresholds[feature]) == 0:
            continue
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
    min_ent, min_threshold, group = sys.maxsize, sys.maxsize, None
    feature_thresholds = thresholds[feature]

    for threshold in feature_thresholds:
        left, right = split(dataset, feature, threshold)
        left_label = np.array([row[-1] for row in left])
        right_label = np.array([row[-1] for row in right])

        left_ent = calc_ent(left_label)
        right_ent = calc_ent(right_label)
        ent = (float(left_label.shape[0]) / len(dataset)) * left_ent + (
                float(right_label.shape[0]) / len(dataset)) * right_ent
        if ent < min_ent:
            min_ent, min_threshold, group = ent, threshold, (left, right)
    return {'condition_ent': min_ent, 'index': feature, 'group': group, 'thres': min_threshold}


def split(dataset, feature, threshold):
    left, right = list(), list()
    for row in dataset:
        if row[feature] < threshold:
            left.append(row)
        else:
            right.append(row)
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
        if not node['left']:
            thresholds[node['left']['index']].remove(node['left']['thres'])
        build_tree_helper(node['left'], min_size, thresholds)

    # right child
    if len(right) <= min_size or len(set(right[:, -1])) == 1:
        node['right'] = create_leaf(right)
    else:
        node['right'] = create_node(right, thresholds)
        if not node['right']:
            thresholds[node['right']['index']].remove(node['right']['thres'])
        build_tree_helper(node['right'], min_size, thresholds)


def build_tree(dataset, min_size, thresholds):
    root = create_node(dataset, thresholds)
    #print root['index']
    #print root['thres']
    thresholds[root['index']].remove(root['thres'])
    build_tree_helper(root, min_size, thresholds)
    return root

def confusion_matrix_helper(dataset, thresholds, n_folds=10, ratio = 0.5):
    folds = cross_validation_split(dataset, n_folds)
    best_acc = 0.0
    actual = list()
    predicted = list()
    min_size = ratio * len(dataset)
    scores = list()
    for i in range(len(folds)):
        fold = folds[i]
        train_set = list(folds)
        train_set.pop(i)
        train_set = sum(train_set, [])
        node = build_tree(train_set, min_size, thresholds)
        temp_predicted = list()
        for row in fold:
            temp_predicted.append(predict(node, row))
        temp_actual = [row[-1] for row in fold]
        acc = accuracy_metric(temp_actual, temp_predicted)
        if acc > best_acc:
            best_acc = acc
            actual = temp_actual
            predicted = temp_predicted
    return actual, predicted

def confusion_matrix(actual, predicted):
    confusion_matrix = np.zeros((len(actual), len(predicted)))
    for i in range(len(actual)):
        confusion_matrix[int(actual[i]), int(predicted[i])] += 1
    return confusion_matrix

#seed(1)
dataset = pre_process()
thresholds = get_thresholds(dataset)
eval = evaluation(dataset, thresholds)
print(eval)