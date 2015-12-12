from math import log
from random import shuffle

import pandas as pd

continuity_check = [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
columns = 15
total_data = 600
split_train_test_at = 400

attributes = {0: ['b', 'a', 'c'], 3: ['u', 'y', 'l', 't'], 4: ['g', 'p', 'gg'],
              5: ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
              6: ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], 8: ['t', 'f'],
              9: ['t', 'f'], 11: ['t', 'f'], 12: ['g', 'p', 's']}


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

def calculate_entropy(panda_df, col):
    entropy_for_col = 0
    # Find entropy for each attribute in a column
    # print col
    # print attributes[col]
    for attrib in attributes[col]:
        panda_col_value = panda_df.loc[panda_df[col] == attrib]
        if len(panda_col_value.index) == 0:
            # no row for such a attrib
            continue
        # print panda_col_value
        # print len(panda_col_value.index)
        # panda_col_value_plus = panda_col_value.loc[panda_col_value[15] == 1]
        # panda_col_value_minus = panda_col_value.loc[panda_col_value[15] == 0]
        # print len(panda_col_value_plus)
        # print len(panda_col_value_minus)


        plus_probability = len (panda_col_value.loc[panda_col_value[15] == 1]) / float(len(panda_col_value.index))
        minus_probability = len (panda_col_value.loc[panda_col_value[15] == 0]) / float(len(panda_col_value.index))
        # print plus_probability
        # print minus_probability

        # if either plus_probabilty or minus_probability is 0 then set_entropy to 0 for that attribute
        if plus_probability == 0 or minus_probability == 0:
            entropy_for_each_attrib = 0
        else:
            entropy_for_each_attrib = (-1 * plus_probability * log(plus_probability)) + (-1 * minus_probability * log(minus_probability))
            # print entropy_for_each_attrib
        entropy_for_col += entropy_for_each_attrib
    # print entropy_for_col
    return entropy_for_col


def find_node_with_max_entropy(panda_df):

    # print panda_df

    # Need to find entropy for each column and return the column number with max entropy
    col_with_max_entropy = 0
    max_col_entropy = 1000
    # print panda_df.columns
    for col in (panda_df.columns):
        if col in (1, 2, 7, 10, 13, 14, 15):
            continue
        # print col
        col_entropy = calculate_entropy(panda_df, col)
        if col_entropy < max_col_entropy:
            max_col_entropy = col_entropy
            col_with_max_entropy = col
    return col_with_max_entropy

    #
    # panda_last_col_1 = panda_df.loc[panda_df[16] == 1]
    # panda_last_col_0 = panda_df.loc[panda_df[16] == 0]
    #
    # panda_col1_a_p = panda_last_col_1.loc[panda_df[0] == 'a']
    # panda_col1_a_n = panda_last_col_0.loc[panda_df[0] == 'b']


if __name__ == '__main__':
    label, table = construct_table()
    table = replace_none_in_table(table)
    # print table
    train_label, train_table, test_label, test_table = div_table_train_test(label, table)
    panda_df1 = pd.DataFrame(train_table, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    panda_df2 = pd.DataFrame(train_label, columns=[15])
    panda_df = panda_df1.join(panda_df2)
    # construct_tree(panda_df)

    col_with_max_entropy = find_node_with_max_entropy(panda_df)
    print col_with_max_entropy
    # find_node

    # predict(label, table)
    # print 'training accuracy is %.2f%%; test accuracy is %.2f%%' % (acc_train * 100, acc_test * 100)
