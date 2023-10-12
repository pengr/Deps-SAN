# -*- coding: utf-8 -*-
# 由语法依存关系得到语法距离的下三角矩阵的展平脚本(多进程版本)
# Copyright: RoyPeng
# Version: 1.0

from collections import defaultdict
import numpy as np
import copy, re, logging, sys, os, argparse, time
from multiprocessing import Process, Queue, Manager


class Tree:
    def __init__(self, name):
        self.name = name
        self.children = {}

    def __iter__(self):
        return iter(self.children)

    def __str__(self):
        return self.name

    def __repr__(self):
        return 'Tree("{}")'.format(self.name)

    def relationiter(self):
        yield from self.children.items()

    def add_child(self, child, relation):
        self.children[child] = relation
        return child

    # 深度优先查找 返回从根节点到目标节点的路径
    def deep_first_search(self, cur, val, path=[]):
        path.append(cur.name)  # 当前节点值添加路径列表
        if cur.name == val:  # 如果找到目标 返回路径列表
            return path

        if cur.children == {}:  # 如果没有孩子列表 就 返回 no 回溯标记
            return 'no'

        for node in cur.children:  # 对孩子列表里的每个孩子 进行递归
            t_path = copy.deepcopy(path)  # 深拷贝当前路径列表
            res = self.deep_first_search(node, val, t_path)
            if res == 'no':  # 如果返回no，说明找到头没找到,利用临时路径继续找下一个孩子节点
                continue
            else:
                return res  # 如果返回的不是no,说明找到了路径

        return 'no'  # 如果所有孩子都没找到,则回溯

    # 获取最短路径传入两个节点值，返回结果
    def get_shortest_path(self, root, start, end):
        # 分别获取从根节点到start 和end 的路径列表，如果没有目标节点 就返回no
        path1 = self.deep_first_search(root, start, [])
        path2 = self.deep_first_search(root, end, [])

        if path1 == 'no' or path2 == 'no':
            return '无穷大', '无节点'
        # 对两个路径 从尾巴开始向头找到最近的公共根节点LCA，合并根节点
        len1, len2 = len(path1), len(path2)
        for i in range(len1 - 1, -1, -1):
            if path1[i] in path2:
                index = path2.index(path1[i])
                # 使用公式计算距离dist(n1,n2) = dist(root,n1)+dist(root,n2)-2*dist(root,lca(n1,n2)),
                # 其中index即为LCA的索引位置,按照路径与根节点的距离即是index+1
                return len1 + len2 - 2 * (index + 1)

    def dfs(self, include_self=True):
        if include_self:
            yield self
        for child in self.children:
            yield child
            yield from child.dfs(False)

    def bfs(self, include_self=True):
        if include_self:
            yield self
        trees = list(self.children.keys())
        while True:
            if not trees:
                break
            tree = trees.pop(0)
            yield tree
            trees += list(tree.children.keys())

    @classmethod
    def from_data(cls, name, data, treelevels):
        tree = cls(name)
        if data:
            treelevels += 1
            for subname, subdata in data.items():
                tree.add_child(
                    cls.from_data(subname, subdata, treelevels),
                    treelevels)

        return tree


class AutoVivification(dict):
    # Implementation of perl's autovivification feature.
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            value = self[key] = type(self)()
            return value


def list_all_dict(dict_a, temp_value, a, w_num):
    if w_num < 0:                              # 处理递归终止问题
        return

    if temp_value in dict_a.keys():            # 判断该值是否为键，找出该键值对
        all_value = dict_a[temp_value]         # 遍历该键值对的所有值,即一个列表

        for value in all_value :
            w_num -= 1
            list_all_dict(dict_a, value, a[temp_value], w_num)  # 创建一个无限嵌套字典

    else:
        a[temp_value] = {}
        w_num -= 1


def get_sdsa_matirx(s_d):
    if s_d is None or s_d == "":
        pass
    else:
        if ')-' in s_d:  # # 专门处理)这类特殊情况,注意(不会影响下面的re.findall的正常操作
            s_d = s_d.replace(')-', '-RRB-')

        # 预处理,除了()这类特殊情况，可取出所有中括号的元素,之后分隔开成2个单独元素
        s_d = re.findall(r'[(](.*?)[)]', s_d)
        dp_list = [dp.split(', ') for dp in s_d]

        # 定义一键对应多值的字典dp_dict; 同时利用树上各节点均会作为子节点,来创建含有树上各节点的列表; 注:只用数字代替单词作为节点
        dp_dict = defaultdict(list)
        for k, v in dp_list:
            dp_dict[k.split('-')[-1]].append(v.split('-')[-1])

        # 定义可无限嵌套的字典
        parse_tree = AutoVivification()

        # 将"ROOT-0"的子节点命名为根节点; 核心递归算法,由一键对应多值的字典-->递归字典
        root_name = dp_dict['0'][0]
        list_all_dict(dp_dict, root_name, parse_tree, len(dp_list))

        # 由递归字典parse_tree[root_name]创建一棵递归树, 根节点即为root_name, 例如'Thank-1'; 注:每个节点带有自己的层数
        root = Tree.from_data(root_name, parse_tree[root_name], 0)

        # 按照该单词在源句中的位置, 排序好单词列表, 由于节点全是数字故可直接用range生成
        parse_label_sorted = range(1, len(s_d) + 1)

        # 分层遍历排序好的单词列表,按排序逐个找两两节点的最小路径,记录下来并得到两两节点的语法距离
        sdsa_matrix = []
        for w_i in parse_label_sorted:
            sdsa = []
            for w_j in parse_label_sorted[:w_i - 1]:
                sdsa.append(root.get_shortest_path(root, str(w_i), str(w_j)))
            sdsa_matrix += sdsa

        # 转成numpy矩阵,得到该句子的语法距离下三角矩阵，然后展平
        sdsa_matrix = np.array(sdsa_matrix).astype('int8')
        sdsa_matrix = sdsa_matrix.reshape(1, sdsa_matrix.size)

        return sdsa_matrix


def build_sdsa_matrix(args):
    # <<遍历输入的所有dp文件,包括训练集,验证集,测试集>>,然后获取output路径,即将文件后缀dp改为res_mt
    for input_file in args.dp_files:
        output_file = input_file.replace('dp', "sdsa")

        logging.info('Building sdsa_matrix from %s to %s'%(input_file, output_file))

        #######################　<<改写>>,多进程加速数据预处理　##################################
        with open(file=input_file, mode='r', encoding='utf-8') as f, open(file=output_file, mode='a+') as f1:
            sents_queue = Queue()  # 创建一个消费者队列
            process_list = []  # 创建一个进程队列
            sdsa_matrix_buffer = Manager().list()

            for idx, s_d in enumerate(f.readlines()):
                sents_queue.put((idx, s_d))

            class MyProcess(Process):  # 继承Process类
                def __init__(self, i, queue, sdsa_matrix_buffer):
                    super(MyProcess, self).__init__()
                    self.queue = queue
                    self.i = i
                    self.sdsa_matrix_buffer = sdsa_matrix_buffer

                def run(self):
                    print('第{}个进程正在处理数据'.format(self.i))

                    while not self.queue.empty():  # 如果while True 线程永远不会终止
                        idx, s_d = self.queue.get()
                        sdsa_matrix = get_sdsa_matirx(s_d)
                        self.sdsa_matrix_buffer.append((idx, sdsa_matrix)) # 由一个缓存器暂时存储所生成的句子号和res_matirx
                        if len(self.sdsa_matrix_buffer) % 1000 == 0:
                            print('the sdsa_matirx of {}th sent has loaded'.format(len(self.sdsa_matrix_buffer)))

            # 创建num_proc个子进程准备执行run()
            for i in range(args.num_proc):
                p = MyProcess(i, sents_queue, sdsa_matrix_buffer)  # 实例化进程对象,每个子进程都传入的是所创建的消费者队列
                process_list.append(p)  # 存储这6个进程对象
                p.start()

            # 先阻塞主进程,等待其他的子线程执行结束之后,再主线程在终止
            for p in process_list:
                p.join()

            # <<改写>>, Manager.list()-> list(),按照idx的升序进行排序
            examples = sorted(sorted(list(sdsa_matrix_buffer), key=lambda e: e[0]))
            for idx, sdsa_matrix in examples:
                np.savetxt(f1, sdsa_matrix, fmt='%i')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sdsa_matrix")
    parser.add_argument("--dp-files", nargs="+", required=True,
                        help="Path(s) list of all dependency files")
    parser.add_argument('--num-proc', type=int, default=8, metavar='INT',
                        help="the numbers of multi-process")
    args = parser.parse_args()

    level = logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # 使用文件输入
    current_time = time.time()
    logging.info('Beign building sdsa_matrix')
    build_sdsa_matrix(args)
    logging.info('Done')

    print("Cost {}s".format(time.time()-current_time))
