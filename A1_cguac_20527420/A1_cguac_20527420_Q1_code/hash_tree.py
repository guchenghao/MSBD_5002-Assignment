#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/DataMing-5002/A1_cguac_20527420/A1_cguac_20527420_Q1_code/hash_tree.py
# Project: /Users/guchenghao/Desktop/DataMing-5002/A1_cguac_20527420/A1_cguac_20527420_Q1_code
# Created Date: Tuesday, September 18th 2018, 10:34:41 am
# Author: Harold Gu
# -----
# Last Modified: Thursday, 27th September 2018 1:24:36 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###


class hash_TreeNode(object):
    def __init__(self, parentNode):
        self.node_value = []  # ! 节点所包含的item
        self.parent = parentNode  # ! 父节点
        self.children = {'left': None, 'right': None, 'middle': None}  # ! 子节点

    def print_hash_tree(self, root):
        temp = []

        # ! 深度优先搜索
        def depth_search(node):
            if node.children['left'] is None and node.children['right'] is None and node.children['middle'] is None:
                temp.append(node.node_value)

            if node.children['left']:
                depth_search(node.children['left'])

            if node.children['middle']:
                depth_search(node.children['middle'])

            if node.children['right']:
                depth_search(node.children['right'])

        depth_search(root)

        return temp


def create_hash_tree(dataset, max_leaf_size):
    '''生成hash_tree'''

    init_hash_tree = hash_TreeNode(None)  # ! 创建hash_tree的根节点

    # ! 通过递归生成哈系树
    update_tree(dataset, init_hash_tree, max_leaf_size)

    return init_hash_tree


def update_tree(dataset, init_hash_tree, max_leaf_size, inx=0):
    if inx > max_leaf_size:  # ! inx代表层级
        return  # ! 程序出口
    for item in dataset:
        # ! 根据题目的hash function or rule来构建hash_tree并插入item
        if item[inx] % 3 == 1.0:
            if init_hash_tree.children['left'] is None:
                init_hash_tree.children['left'] = hash_TreeNode(init_hash_tree)
                init_hash_tree.children['left'].node_value.append(item)
            else:
                init_hash_tree.children['left'].node_value.append(item)

        if item[inx] % 3 == 2.0:
            if init_hash_tree.children['middle'] is None:
                init_hash_tree.children['middle'] = hash_TreeNode(
                    init_hash_tree)
                init_hash_tree.children['middle'].node_value.append(item)
            else:
                init_hash_tree.children['middle'].node_value.append(item)

        if item[inx] % 3 == 0.0:
            if init_hash_tree.children['right'] is None:
                init_hash_tree.children['right'] = hash_TreeNode(
                    init_hash_tree)
                init_hash_tree.children['right'].node_value.append(item)
            else:
                init_hash_tree.children['right'].node_value.append(item)
    # ! 判断节点是否需要继续分裂，大于max_leaf_size则进行分裂，即递归
    if len(init_hash_tree.children['left'].node_value) > max_leaf_size:
        update_tree(
            init_hash_tree.children['left'].node_value, init_hash_tree.children['left'], max_leaf_size, inx + 1)

    if len(init_hash_tree.children['middle'].node_value) > max_leaf_size:
        update_tree(
            init_hash_tree.children['middle'].node_value, init_hash_tree.children['middle'], max_leaf_size, inx + 1)

    if len(init_hash_tree.children['right'].node_value) > max_leaf_size:
        update_tree(
            init_hash_tree.children['right'].node_value, init_hash_tree.children['right'], max_leaf_size, inx + 1)


def print_nested_list(res, inx=1):
    left_tree = []
    middle_tree = []
    right_tree = []
    result = []
    for item in res:
        if item[0][inx] % 3 == 1.0:
            if len(item) == 1:
                left_tree.append(item[0])
                continue
            left_tree.append(item)
        elif item[0][inx] % 3 == 2.0:
            if len(item) == 1:
                middle_tree.append(item[0])
                continue
            middle_tree.append(item)
        else:
            if len(item) == 1:
                right_tree.append(item[0])
                continue
            right_tree.append(item)

    if len(left_tree) == 1:
        result.append(left_tree[0])
    else:
        result.append(left_tree)

    if len(middle_tree) == 1:
        result.append(middle_tree[0])
    else:
        result.append(middle_tree)

    if len(right_tree) == 1:
        result.append(right_tree[0])
    else:
        result.append(right_tree)

    return result


def hash_tree(data, max_leaf_size):
    hash_tree = create_hash_tree(data_mat, max_leaf_size)

    return hash_tree


def get_nested_list(hash_root):
    # ! 通过深度优先搜索输出hash_tree的每个叶节点所保存的数据
    final_res = hash_root.print_hash_tree(hash_root)
    left_tree = []
    middle_tree = []
    right_tree = []
    result = []

    # ! 遍历每个子节点的输出结果，将它们按照层次输出，形成nested list
    for item in final_res:
        if item[0][0] % 3 == 1.0:
            left_tree.append(item)
        elif item[0][0] % 3 == 2.0:
            middle_tree.append(item)
        else:
            right_tree.append(item)

    left_tree = print_nested_list(left_tree)
    middle_tree = print_nested_list(middle_tree)
    right_tree = print_nested_list(right_tree)

    result.append(left_tree)
    result.append(middle_tree)
    result.append(right_tree)

    return result


# ! 输入数据
data_mat = [[1, 2, 4], [1, 2, 9], [1, 3, 5], [1, 3, 9], [1, 4, 7], [1, 5, 8], [1, 6, 7], [1, 7, 9], [1, 8, 9], [2, 3, 5], [2, 4, 7], [2, 5, 6], [2, 5, 7], [2, 5, 8], [2, 6, 7], [2, 6, 8], [2, 6, 9], [2, 7, 8], [3, 4, 5], [3, 4, 7], [3, 5, 7], [3, 5, 8], [3, 6, 8], [3, 7, 9], [3, 8, 9],
            [4, 5, 7], [4, 5, 8], [4, 6, 7], [4, 6, 9], [4, 7, 8],
            [5, 6, 7], [5, 7, 9], [5, 8, 9], [6, 7, 8], [6, 7, 9]]

# ! 生成hash_tree
max_leaf_size = 3
hash_root = hash_tree(data_mat, max_leaf_size)

# ! 生成相应的nested list
result = get_nested_list(hash_root)

print(result)
