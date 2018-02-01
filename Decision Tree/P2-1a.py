import collections
from csv import reader
import numpy as np
from random import randrange

"""
  Mushroom: is binary classification dataset and the task is to accurately predict
  whether a mushroom is poisonous or edible given 21 different categorical (ordinal)
  features for each mushroom. These features describe various physical properties of
  the mushrooms such as length, diameter, etc. There are a total of 8124 instances.

   Task: Grow a multiway decision tree using min = {0.05,0.10,0.15}, and calculate the 
   accuracy using ten fold cross-validation for each value of min
"""


class Preprocess:
    def __init__(self, filename):
        self.filename = filename

    def load_csv(self):
        file = open(self.filename, "r")
        lines = reader(file)
        dataset = list(lines)
        return np.array(dataset)

    def get_thresholds(self):
        thresholds = []
        dataset = self.load_csv()
        for feature in range(len(dataset[0])):
            thresholds.append(collections.Counter(dataset[:, feature]))
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
    #     if row[node['index']] not in node['index'].keys():
    #         return

    for i in range(len(node['attri'])):
        if node['attri'][i] == row[node['index']]:
            if isinstance(node['child'][i], dict):
                return predict(node['child'][i], row)
            else:
                return node['child'][i]


def evaluation(dataset, thresholds, n_folds=10, mean_ratio=[0.05, 0.10, 0.15]):
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
    ent, group, attribute_list = 0.0, list(), list()
    attributes = thresholds[feature].keys()
    for f in attributes:
        sub_group = split(dataset, feature, f)  # group nodes with the same attributes in this
        group.append(sub_group)
        attribute_list.append(f)
        label = np.array([row[-1] for row in sub_group])
        sub_ent = calc_ent(label)
        ent += (float(label.shape[0]) / len(dataset)) * sub_ent
    return {'condition_ent': ent, 'index': feature, 'attri': attribute_list, 'group': group}


def split(dataset, feature, f):
    sub_group = list()
    for row in dataset:
        if row[feature] == f:
            sub_group.append(row)
    return np.array(sub_group)


def create_leaf_node(group):
    result = 0
    for i in range(len(group)):
        if len(group[i]) == 0:
            continue
        outcomes = [row[-1] for row in group[i]]
        result = max(set(outcomes), key=outcomes.count)
    return result


def create_leaf(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def build_tree_helper(node, min_size, thresholds):
    group = node['group']
    del (node['group'])
    # if left is empty or right is empty, no split
    node['child'] = list()
    for i in range(len(group)):
        sub_group = group[i]
        if len(sub_group) == 0:
            # find the max size in its sibling
            leaf_node = create_leaf_node(group)
            node['child'].append(leaf_node)
            continue

        if len(sub_group) <= min_size or len(set(sub_group[:, -1])) == 1:
            leaf_node = create_leaf(sub_group)
            node['child'].append(leaf_node)
        else:
            new_node = create_node(sub_group, thresholds)
            node['child'].append(new_node)
            build_tree_helper(new_node, min_size, thresholds)


def build_tree(dataset, min_size, thresholds):
    root = create_node(dataset, thresholds)
    build_tree_helper(root, min_size, thresholds)
    return root


def confusion_matrix_helper(dataset, thresholds, n_folds=10, ratio=0.05):
    classes = len(set(dataset[:, -1]))
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
    return actual, predicted, classes


def confusion_matrix(actual, predicted, classes):
    confusion_matrix = np.zeros([classes, classes])
    for i in range(len(actual)):
        row = 1 if predicted[i] == 'p' else 0
        col = 1 if actual[i] == 'p' else 0
        confusion_matrix[row, col] += 1
    return confusion_matrix


def main():
    preprocess = Preprocess('mushroom.csv')
    dataset = preprocess.load_csv()
    thresholds = preprocess.get_thresholds()
    ev = evaluation(dataset, thresholds)
    for i in ev:
        print("ratio: {0:.2f} acc: {1:.2f} std: {2:.2f}".format(i, ev[i]['acc'], ev[i]['std']))

    actual, predicted, classes = confusion_matrix_helper(dataset, thresholds)
    print(confusion_matrix(actual, predicted, classes))


if __name__ == '__main__':
    main()

