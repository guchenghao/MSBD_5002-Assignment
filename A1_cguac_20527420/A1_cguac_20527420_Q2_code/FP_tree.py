#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/DataMing-5002/A1_cguac_20527420/A1_cguac_20527420_Q2_code/FP_tree.py
# Project: /Users/guchenghao/Desktop/DataMing-5002/A1_cguac_20527420/A1_cguac_20527420_Q2_code
# Created Date: Wednesday, September 19th 2018, 12:00:48 am
# Author: Harold Gu
# -----
# Last Modified: Friday, 28th September 2018 11:33:30 am
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###


import copy
import pandas as pd


class FP_TreeNode(object):
    '''定义FP-tree的树节点'''

    def __init__(self, name_val, num_count, parentNode):
        self.name = name_val  # ! 树节点名称
        self.count = num_count  # ! 节点计数值
        self.parent = parentNode  # ! 父节点，便于回溯
        self.children = {}  # ! 子节点
        self.linknode = None  # ! 用于建立链表

    def plus_count(self, num_count):
        '''
            添加出现次数
        '''

        self.count += num_count

    def show_tree(self, inx=1):
        '''
            遍历FP_tree, 并打印出FP_Tree每个节点的信息
        '''
        print(
            ' '*inx, '序号: {0}, 节点名称: {1}, 数量: {2}'.format(inx, self.name, self.count))

        for child in self.children.values():
            child.show_tree(inx + 1)

    def print_cond_tree(self, root):

        # ! 按照层次打印结果
        def print_result(node):
            if node.children:

                level_item = [str(node.name) + ' ' + str(node.count)]
                # ! 判断当前节点的子节点个数
                if len(list(node.children.items())) == 1:
                    # ! 递归
                    new_level = print_result(list(node.children.items())[0][1])

                    level_item.append(new_level)

                else:
                    new_level = []
                    # ! 对子节点按照顺序进行递归处理
                    for next_item in node.children.items():
                        result_level = print_result(next_item[1])
                        if len(result_level) == 1:
                            new_level.append(result_level[0])
                        else:
                            new_level.append(result_level)

                    level_item.append(new_level)

                return level_item
            else:
                return [str(node.name) + ' ' + str(node.count)]

        result = print_result(root)

        return result


class FP_tree_LinkList(object):
    '''
        linklist类
    '''

    def __init__(self):
        self.linkedList = {}


class Fp_Tree_builder(object):
    '''构建FP-tree'''

    def __init__(self):
        self.headLinkedList = FP_tree_LinkList()

    def create_tree(self, dataset, minsup):
        '''创建FP_tree的入口
            第一遍先扫描数据库，排除小于minsup的item，生成初始频繁项集和初始头链表，
            第二遍再扫描数据库，按照频率大小以及字母表顺序逐个插入数据库中的每一个item

        Arguments:
            dataset {[dict]} -- [事务数据字典]
            minsup {[int]} -- [最小支持度]

        Returns:
            [Object] -- [FP_tree的根节点]
        '''
        self.headLinkedList = {}

        # ! 遍历数据集，统计每种item的频数
        for trans in dataset:
            for item in trans:
                if item in list(self.headLinkedList.keys()):
                    self.headLinkedList[item] += dataset[trans]
                else:
                    self.headLinkedList[item] = dataset[trans]

        # ! 在headLinkedList中除去小于minsup值的item
        for k in list(self.headLinkedList.keys()):
            if self.headLinkedList[k] < minsup:
                # ! 将元素删除
                self.headLinkedList.pop(k)

        # ! 获得初始频繁项集
        fre_item_set = list(self.headLinkedList.keys())
        # print(fre_item_set)

        # ! 如果频繁项集为空，则输出空树和空头链表
        if len(fre_item_set) == 0:
            return None, None
        # else:
        #     print('初始频繁项集: {0}'.format(fre_item_set))

        # ! 初始化头链表，为每个频繁项添加空指针
        for i in self.headLinkedList:
            self.headLinkedList[i] = [self.headLinkedList[i], None]

        # ! 创建根节点
        init_tree = FP_TreeNode('Null set', 1, None)

        for trans, num in dataset.items():
            temp = {}  # ! 存储头链表中每个频繁项

            for item in trans:
                if item in fre_item_set:
                    temp[item] = self.headLinkedList[item][0]

            if len(temp) > 0:
                # ! 按照频率大小的顺序，将items进行从大到小的排列，再按照字母表顺序排序
                sortedItems = [item[0] for item in sorted(
                    list(temp.items()), key=lambda x: (-x[1], x[0]))]

                new_tree = init_tree  # ! 创建一个根节点的引用

                # ! 将事务集中每项按顺序插入到FP-tree中
                for item in sortedItems:
                    new_tree = self.insert_treeNode(
                        item, new_tree, self.headLinkedList, num)

        return init_tree, self.headLinkedList

    def insert_treeNode(self, curr_item, init_tree, headLinkedList, num):
        '''插入树节点
            如果即将插入的item已经存在于树中，则增加其计数值；
            如果没有，则生成一个新的树节点
            然后判断头链表是否为空，为空将该改新树节点赋给头链表
            否则，插入相应头链表项的链表尾端

        Arguments:
            curr_item {[List]]} -- [事务数据项]
            init_tree {[Object]} -- [FP-Tree的节点]
            headLinkedList {[dict]} -- [头链表(包含所有的频繁项)]
            num {[int]} -- [事务数据项的计数值]
        '''

        if curr_item in init_tree.children:
            # ! 当前item已经存在，则进行计数
            init_tree.children[curr_item].plus_count(num)
        else:
            init_tree.children[curr_item] = FP_TreeNode(
                curr_item, num, init_tree)
            # ! 更新头指针表或前一个相似元素项节点的指针指向新节点
            if headLinkedList[curr_item][1] is None:
                headLinkedList[curr_item][1] = init_tree.children[curr_item]
            else:
                self.insert_linknode(
                    headLinkedList[curr_item][1], init_tree.children[curr_item])

        return init_tree.children[curr_item]

    def insert_linknode(self, headnode, new_node):
        '''
            从该item项的头节点开始遍历，将该树节点插入headLinkedList链表中
        '''
        while headnode.linknode is not None:
            headnode = headnode.linknode

        headnode.linknode = new_node


class Freq_item_generator(object):

    def __init__(self):
        # ! 该字典dict用于存储每颗子树
        self.Fp_conditional_tree = {}
        self.Tree_builder = Fp_Tree_builder()

    def create_paths(self, base_path, headnode):
        '''
            创建条件基路径
            从头链表中的第一个树节点，进行遍历;
            再以当前基节点回溯至root节点，保存相应的前缀路径

        Arguments:
            base_path {[string]} -- [创建路径的起始点]
            headnode {[Object]} -- [头链表中的头节点]

        Returns:
            [dict] -- [条件路径]
        '''

        con_paths = {}

        # ! 通过linknode来遍历链表，生成条件路径
        while headnode is not None:
            # ! 从left_node进行回溯
            temp_path = []
            leaf_node = copy.copy(headnode)  # ! 浅复制
            # ! 获取基节点的前缀路径
            while leaf_node.parent is not None:
                temp_path.append(leaf_node.name)
                leaf_node = leaf_node.parent

            if len(temp_path) > 1:
                # ! temp_path[1:]将base节点从path中剔除，只留存前缀路径
                con_paths[frozenset(temp_path[1:])] = headnode.count
            headnode = headnode.linknode

        return con_paths

    def generate_freq_pattern(self, headLinkedList, minsup, basepath, fre_items_set):
        '''
            递归生成频繁项集，并在self.Fp_conditional_tree中保存每个FP_cond_tree(条件子树)
        '''
        # ! 头指针表中的元素项按照频繁度排序,从大到小
        table_itemlist = [item[0] for item in sorted(
            list(headLinkedList.items()), key=lambda x: (-x[1][0], x[0]))]

        for base in table_itemlist:
            newFreqSet = copy.deepcopy(basepath)  # ! 深复制
            newFreqSet.update([base])
            fre_items_set.append(newFreqSet)  # ! 将得出的频繁项集添加到list中
            # ! 创建条件基路径
            con_path = self.create_paths(base, headLinkedList[base][1])
            # ! 根据求得的条件路径，生成条件子树
            cond_tree, cond_table = self.Tree_builder.create_tree(
                con_path, minsup)

            # ! 跳过头链表为空的子树
            if cond_table is not None:
                # ! 存储那些height > 1的子树
                self.Fp_conditional_tree[base] = (cond_tree, cond_table)
                # ! 用于测试
                # print('conditional tree for: {0}'.format(newFreqSet))
                # cond_tree.show_tree()
                # ! 利用递归，来完成每个事务项的频繁项集生成
                self.generate_freq_pattern(
                    cond_table, minsup, newFreqSet, fre_items_set)


# ! 测试数据(testData)
# simpDat = [['r', 'z', 'h', 'j', 'p'], ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
#            ['z'],
#            ['r', 'x', 'n', 'o', 's'],
#            ['y', 'r', 'x', 'z', 'q', 't', 'p'],
#            ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

# simpDat = [['a', 'b'], ['b', 'c', 'd'], ['a', 'c', 'd', 'e'],
#            ['a', 'd', 'e'],
#            ['a', 'b', 'c']]

# simpDat = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['a', 'f', 'g'], ['b', 'd', 'e', 'f', 'j'],
#            ['a', 'b', 'd', 'i', 'k'],
#            ['a', 'b', 'e', 'g']]

def load_data():
    '''加载数据'''

    # ! 从文件中读取事务数据
    with open('./groceries.csv', 'r', encoding='utf-8') as f:
        items_dataset = []
        for line in f.readlines():
            items_dataset.append(
                [x for x in line.strip().split(',') if x != ''])
    '''
        对初始数据集中，每个事务项进行统计，并赋予1的计数值
        frozenset可以做为字典的索引值，比较方便对相同的transaction进行计数并存储
    '''
    trans_dict = {}
    for trans in items_dataset:
        # ! 合并包含相同项的事务数据
        if frozenset(trans) in list(trans_dict.keys()):
            trans_dict[frozenset(trans)] += 1
        else:
            trans_dict[frozenset(trans)] = 1

    return trans_dict


# ! 算法入口
def Fp_growth():

    trans_data = load_data()  # ! 加载数据

    minsup = 300  # ! 最小支持度
    tree_builder = Fp_Tree_builder()
    freqItems_builder = Freq_item_generator()

    Fp_tree, LinkTable = tree_builder.create_tree(trans_data, minsup)  # ! 构建FP_tree
    Fp_tree.show_tree()  # ! 打印FP-Tree

    # print('{0}的前缀路径: {1}'.format('beef', create_paths(
    #     'beef', LinkTable['beef'][1])))  # ! 输出beef的前缀路径

    freqItems = []  # ! 存储频繁项集

    freqItems_builder.generate_freq_pattern(
        LinkTable, minsup, set([]), freqItems)  # ! 生成频繁项集

    print('频繁项集: {0}'.format(freqItems))  # ! 打印频繁项集

    # ! 写csv文件操作
    # with open('submission.csv', mode='w+', encoding='utf-8') as f:
    #     for item in freqItems:
    #         f.writelines(str(item) + '\n')
    # ! 以csv的格式输出，生成submission.csv文件
    freqItems_csv = pd.Series(freqItems)
    freqItems_csv.to_csv('./submission.csv', index=False)

    '''
        将Fp_conditional_tree字典中所保存的cond_tree
    '''

    for item in freqItems_builder.Fp_conditional_tree.items():

        print_con_tree = item[1][0].print_cond_tree(item[1][0])

        print("The FP-conditional trees of {0}: {1}".format(item[0], print_con_tree))


# ! 启动算法
Fp_growth()
