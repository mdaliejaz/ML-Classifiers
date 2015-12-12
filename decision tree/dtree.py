from math import log
from random import shuffle
import random

import pandas as pd

continuity_check = [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
columns = 15
total_data = 600
split_train_test_at = 550

attributes = {0: ['b', 'a'], 3: ['u', 'y', 'l', 't'], 4: ['g', 'p', 'gg'],
              5: ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
              6: ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], 8: ['t', 'f'],
              9: ['t', 'f'], 11: ['t', 'f'], 12: ['g', 'p', 's']}


class TreeNode:
    def __init__(self, column_name, edge_val, children, is_leaf, value):
        self.column_name = column_name
        self.edge_val = edge_val
        self.children = children
        self.is_leaf = is_leaf
        self.value = value


def average_none(table, column):
    sum_not_none = 0
    not_none_count = 0
    for i in xrange(len(table)):
        if table[i][column] is not None:
            not_none_count += 1
            sum_not_none += table[i][column]
    average = sum_not_none / not_none_count
    for i in xrange(len(table)):
        if table[i][column] is None:
            table[i][column] = average


def majority_none(table, column):
    not_none_data = []
    for i in xrange(len(table)):
        if table[i][column] is not None:
            not_none_data.append(table[i][column])
    most_common = max(set(not_none_data), key=not_none_data.count)

    for i in xrange(len(table)):
        if table[i][column] is None:
            table[i][column] = most_common


def replace_none_in_table(table):
    for i in xrange(15):
        if continuity_check[i]:
            average_none(table, i)
        else:
            majority_none(table, i)
    return table


def fill_row(continuous, value):
    if value == '?':
        return None
    if continuous:
        try:
            return int(value)
        except ValueError:
            return float(value)
    else:
        return value


def construct_table():
    table = []
    label = []
    crx_data = open('crx.data.txt', 'r')
    for line in crx_data:
        data_line = line.strip().split(',')
        row_data = [fill_row(continuity_check[i], data_line[i]) for i in xrange(columns)]
        table.append(row_data)
        if data_line[len(data_line) - 1] == '+':
            label.append(1)
        else:
            label.append(0)
    crx_data.close()
    return label, table


def get_shuffled_indices():
    total_data_indices = [i for i in xrange(total_data)]
    shuffle(total_data_indices)
    return total_data_indices


def div_table_train_test(label, table):
    shuffled_indices = get_shuffled_indices()
    train_label = [label[shuffled_indices[i]] for i in xrange(split_train_test_at)]
    train_table = [table[shuffled_indices[i]] for i in xrange(split_train_test_at)]
    test_label = [label[shuffled_indices[i]] for i in xrange(split_train_test_at, 600)]
    test_table = [table[shuffled_indices[i]] for i in xrange(split_train_test_at, 600)]
    return train_label, train_table, test_label, test_table


def find_majority(train_label):
    count1 = 0
    count0 = 0
    for i in xrange(split_train_test_at):
        if train_label[i] == 1:
            count1 += 1
        else:
            count0 += 1
    if count1 > count0:
        return 1
    else:
        return 0


def calculate_entropy(panda_df, col):
    entropy_for_col = 0
    # Find entropy for each attribute in a column
    for attrib in attributes[col]:
        panda_col_value = panda_df.loc[panda_df[col] == attrib]
        if len(panda_col_value.index) == 0:
            # no row for such a attrib
            continue

        plus_probability = len(panda_col_value.loc[panda_col_value[15] == 1]) / float(len(panda_col_value.index))
        minus_probability = len(panda_col_value.loc[panda_col_value[15] == 0]) / float(len(panda_col_value.index))
        # if either plus_probability or minus_probability is 0 then set_entropy to 0 for that attribute
        if plus_probability == 0 or minus_probability == 0:
            entropy_for_each_attrib = 0
        else:
            entropy_for_each_attrib = (-1 * plus_probability * log(plus_probability)) + (
                -1 * minus_probability * log(minus_probability))
        entropy_for_col += entropy_for_each_attrib
    return entropy_for_col


def find_node_with_max_entropy(panda_df):
    # Need to find entropy for each column and return the column number with max entropy
    col_with_max_entropy = -1
    max_col_entropy = 1000
    for col in (panda_df.columns):
        if col in (1, 2, 7, 10, 13, 14, 15):
            continue
        col_entropy = calculate_entropy(panda_df, col)
        if col_entropy < max_col_entropy:
            max_col_entropy = col_entropy
            col_with_max_entropy = col
    return col_with_max_entropy


def build_tree(panda_df, column_name, edge_val, is_leaf):
    if is_leaf is True or column_name == 15:
        value = panda_df.loc[panda_df.index[0], 15]
        return TreeNode(column_name, edge_val, None, 1, value)
    children_edges = list(panda_df[column_name].unique())
    children = []
    child_value_row0 = -1
    for child_edge in children_edges:
        child_df = panda_df.loc[panda_df[column_name] == child_edge]
        child_df = child_df.drop(column_name, 1)
        child_column_name = find_node_with_max_entropy(child_df)
        if child_column_name == -1:
            child_column_name = random.choice(child_df.columns)
        is_leaf = 0
        if len(child_df.loc[child_df[15] == 1]) == len(child_df.index):
            is_leaf = 1
        child_node = build_tree(child_df, child_column_name, child_edge, is_leaf)
        children.append(child_node)
        child_value_row0 = child_df.loc[child_df.index[0], 15]
    return TreeNode(column_name, edge_val, children, is_leaf, child_value_row0)


def construct_tree(panda_df):
    column_name = find_node_with_max_entropy(panda_df)
    return build_tree(panda_df, column_name, None, 0)


def predict_value(node, row):
    edge_val = row[node.column_name]
    if node.is_leaf:
        return node.value
    for child in node.children:
        if child.edge_val == edge_val:
            if child.is_leaf:
                return child.value
            else:
                return predict_value(child, row)


def get_accuracy(root, panda_dft):
    count = 0
    for row in map(list, panda_dft.values):
        predicted_value = predict_value(root, row)
        if predicted_value == row[15]:
            count += 1
    return (count / (len(panda_dft.index) * 1.0)) * 100


if __name__ == '__main__':
    label, table = construct_table()

    table = replace_none_in_table(table)
    train_label, train_table, test_label, test_table = div_table_train_test(label, table)

    panda_df1 = pd.DataFrame(train_table, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    panda_df2 = pd.DataFrame(train_label, columns=[15])
    panda_df = panda_df1.join(panda_df2)

    col_with_max_entropy = find_node_with_max_entropy(panda_df)
    root = construct_tree(panda_df)

    panda_dft1 = pd.DataFrame(test_table, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    panda_dft2 = pd.DataFrame(test_label, columns=[15])
    panda_dft = panda_dft1.join(panda_dft2)

    accuracy = get_accuracy(root, panda_dft)

    print "accuracy:", accuracy, "%"
