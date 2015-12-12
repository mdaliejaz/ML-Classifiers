from collections import defaultdict
from math import log
from random import shuffle
import pandas as pd

continuity_check = [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
columns = 15
total_data = 600
split_train_test_at = 400


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


# def non_continuous(column_set, read_value):
#     if not column_set.has_key(read_value):
#         column_set[read_value] = len(column_set)
#     return column_set[read_value]


def fill_row(continuous, value):
    if value == '?':
        return None
    if continuous:
        try:
            return int(value)
        except ValueError:
            return float(value)
    else:
        # return non_continuous(column_set, value)
        return value


def construct_table():
    table = []
    label = []
    # table_line = {i: {} for i in range(columns)}
    crx_data = open('crx.data.txt', 'r')
    for line in crx_data:
        data_line = line.strip().split(',')
        # row_data = [fill_row(continuity_check[i], table_line[i], data_line[i]) for i in xrange(columns)]
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


def build_tree(train_label, train_table):
    majority = find_majority(train_label)
    return


# def run_method_dtree(train_label, train_table, test_label, test_table):
#     tree = treebuilder.DecisionTree()
#     tree.fit(train_label, train_table, continuity_check)
#
#     return run_test(tree, run_data)


def predict(label, table):
    train_label, train_table, test_label, test_table = div_table_train_test(label, table)
    panda_df1 = pd.DataFrame(train_table, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    panda_df2 = pd.DataFrame(train_label, columns=[15])
    panda_df = panda_df1.join(panda_df2)
    print panda_df

    panda_last_col_1 = panda_df.loc[panda_df[16] == 1]
    panda_last_col_0 = panda_df.loc[panda_df[16] == 0]

    panda_col1_a_p = panda_last_col_1.loc[panda_df[0] == 'a']
    panda_col1_a_n = panda_last_col_0.loc[panda_df[0] == 'b']

        # sum = panda_df.astype(bool).sum(axis=0)[16]
        # test1 = [[1, 2, 3, 4, 5],[1, 2, 3, 4, 5],[0,0,0,0,0],[0,0,0,0,0],[1, 2, 0, 4, 0]]
        # test2 = ['a','b','c','d','e']
        # panda_test1 = pd.DataFrame(test1, columns=[1, 2, 3, 4, 5])
        # panda_test2 = pd.DataFrame(test2, columns=[6])
        # panda_test = panda_test1.join(panda_test2)
        #
        # print panda_test
        # sum = panda_test.astype(bool).sum(axis=0)
        # print sum
        # acc_train, acc_test = run_method_dtree(train_label, train_table, test_label, test_table)
        # run_method_sklean(run_data)
        # return acc_train, acc_test


if __name__ == '__main__':
    label, table = construct_table()
    table = replace_none_in_table(table)
    print table
    predict(label, table)
    # print 'training accuracy is %.2f%%; test accuracy is %.2f%%' % (acc_train * 100, acc_test * 100)
