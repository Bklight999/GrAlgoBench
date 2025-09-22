from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter, deque
import random
import re
import json
import numpy as np
import walker
import pandas as pd
from tasks.base import *
from tqdm import tqdm
import igraph as ig
import statistics
import pdb
#from .node_edge_config import n_e_cfg

class PathSum_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(PathSum_Task, self).__init__(data_loc, 'PathSum')
    
    # def check_solution(self, problem_id, response):    
    #     match = re.search(r'\\boxed\{(.*?)\}', response)
    #     if not match:
    #         return False  # 格式不符合要求

    #     user_answer = match.group(1) 
    #     correct_answer = str(self.problem_set[problem_id]['exact_answer'])

    #     return user_answer.strip(), user_answer.strip() == correct_answer.strip() 
    

    def generate_bushy_binary_tree(self, G, start_node, k):
        if start_node not in G:
            print(f"Error: Starting node {start_node} is not in the graph.")
            return nx.Graph()

        if G.number_of_nodes() < k:
            print(f"Warning: Number of nodes in graph ({G.number_of_nodes()}) is less than target {k}. Will use all nodes.")
            k = G.number_of_nodes()

        T = nx.Graph()
        queue = deque([start_node])
        T.add_node(start_node, name=G.nodes[start_node]['name'])

        while queue and T.number_of_nodes() < k:
            u = queue.popleft()
            neighbors = list(G.neighbors(u))
            random.shuffle(neighbors)
            
            children_added = 0
            for v in neighbors:
                if T.number_of_nodes() >= k:
                    break
                if children_added >= 2:
                    break
                if v not in T:
                    T.add_node(v, name=G.nodes[v]['name'])
                    T.add_edge(u, v, weight=random.randint(1, 10))
                    
                    queue.append(v)
                    children_added += 1
            
            if T.number_of_nodes() >= k:
                break
                
        return T

    def generate_dataset(self, count=100, difficulty='easy'):
        G = nx.Graph()
        with open('/mnt/sdc/qifan/GLC-Benchmark/data_generation/real_world_graphs/academic/academic_graph.txt', 'r') as f:
            first_line = True
            error_count = 0
            names_to_id = {}
            for line in f:
                if first_line:
                    first_line = False
                    continue
                try:
                    names = re.findall(r'"([^"]+)"', line)
                    u, v = names[0].strip(), names[1].strip()
                    if u not in names_to_id:
                        id1 = len(names_to_id)
                        names_to_id[u] = id1
                        G.add_node(id1, name=u)
                    if v not in names_to_id:
                        id2 = len(names_to_id)
                        names_to_id[v] = id2
                        G.add_node(id2, name=v)
                    G.add_edge(names_to_id[u], names_to_id[v])
                except Exception as e:
                    error_count += 1
                    continue
            print(f'Edges number: {G.number_of_edges()}, Error count: {error_count}')

        print(f'graph statistics: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')

        for difficulty in ['easy', 'medium', 'hard', 'challenge']: #,'easy', 'medium',  'hard'
            self.problem_set = []
            min_nodes, max_nodes = n_e_cfg.node_config[difficulty]['nodes']
            with tqdm(total=count, desc=f'Generating {difficulty} dataset') as pbar:
                while len(self.problem_set) < count:
                    node_size = sample_node_size(min_nodes, max_nodes)
                    start_node = random.randint(0, G.number_of_nodes() - 1)
                    tree = self.generate_bushy_binary_tree(G, start_node, node_size)

                    if tree.number_of_nodes() != node_size:
                        continue

                    
                    value = self.gen_value_for_comparison(tree, start_node)
                    answer= self.exact_solver(tree, start_node, value)
                    problem_text = self.generate_problem(tree, start_node, value)

                    self.problem_set.append({
                        'id':len(self.problem_set),
                        'problem_text': problem_text,
                        'start_node': start_node,
                        'value': value,
                        'graph': tree,
                        'exact_answer': answer
                    })
                    pbar.update(1)
            self.save_dataset(difficulty)

    def generate_problem(self, tree, start_node, target_sum):
        prompt = []

        prompt.append(
            "You are given a binary tree representing a co-authorship network. Each node is an author, and each edge represents a co-authorship relationship, with the edge's weight indicating the number of papers co-authored by the two authors. Find the number of paths from the root author to any leaf author such that the sum of the edge weights (i.e., the total number of co-authored papers along the path) is greater than the given value."
        )

        prompt.append(
            "- Authors in the network: " + ", ".join([tree.nodes[node]['name'] for node in tree.nodes()])
        )

        co_authorship = ", ".join(
            f"{tree.nodes[u]['name']} and {tree.nodes[v]['name']} (co-authored {tree[u][v]['weight']} papers)"
            for u, v in tree.edges()
        )
        prompt.append(f"- Co-authorship relationships (with number of co-authored papers): {co_authorship}.")
        prompt.append(f"- The root of the tree is {tree.nodes[start_node]['name']}.")
        prompt.append(f"- The target value is {target_sum}.")
        prompt.append(f"- Find the number of paths from the root author to any leaf author such that the sum of the edge weights (i.e., the total number of co-authored papers along the path) is greater than the given value.")
        prompt.append("Present your answer in the following format: '\\boxed{n}'. n is the number of paths.")

        return '\n'.join(prompt)
    
    def gen_value_for_comparison(self, tree, root):
        def all_path_sums(tree, root):
            sums = []
            def dfs(node, parent, acc_sum):
                neighbors = [n for n in tree.neighbors(node) if n != parent]
                if not neighbors:
                    sums.append(acc_sum)
                    return
                for neighbor in neighbors:
                    weight = tree.edges[node, neighbor]['weight']
                    dfs(neighbor, node, acc_sum + weight)
            dfs(root, None, 0)
            return sums

        path_sums = all_path_sums(tree, root)
        value = int(statistics.median(path_sums))
        return value

    @staticmethod
    def exact_solver(tree, root, value, mode='greater'):
        count = 0
        def dfs(node, parent, acc_sum):
            nonlocal count
            neighbors = [n for n in tree.neighbors(node) if n != parent]
            if not neighbors:
                if (mode == 'greater' and acc_sum > value) or (mode == 'less' and acc_sum < value):
                    count += 1
                return
            for neighbor in neighbors:
                weight = tree.edges[node, neighbor]['weight']
                dfs(neighbor, node, acc_sum + weight)
        dfs(root, None, 0)
        return count
    
    
