# from rdkit import Chem
# from rdkit.Chem import rdFMCS
# from rdkit.Chem import Draw
from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter
import random
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tasks.base import *
import walker
#from .node_edge_config import n_e_cfg

class MaxDegree_Task(NPTask):

    def __init__(self, data_loc='./dataset'):
        super(MaxDegree_Task, self).__init__(data_loc, 'MaxDegree')

    # def check_solution(self, problem_id, response):
    #     true_count = self.problem_set[problem_id]['exact_answer']
    #     match = re.search(r'\\boxed\{(\d+)\}', response)
    #     if not match:
    #         return False  # Format doesn't match requirements
    
    #     user_count = int(match.group(1))
    #     return user_count, user_count == true_count  

    def generate_dataset(self, count=100):
        G = nx.Graph()
        with open('/mnt/sdc/qifan/GLC-Benchmark/data_generation/real_world_graphs/city_road/Rome__Italy_street_graph.txt', 'r') as f:
            first_line = True
            names_to_id = {}
            for line in f:
                if first_line:
                    first_line = False
                    continue
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
        
        print(f'graph statistics: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
        
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)

        for difficulty in ['easy', 'medium', 'hard', 'challenge']:
            self.problem_set = []
            min_nodes, max_nodes = n_e_cfg.node_config[difficulty]['nodes']

            with tqdm(total=count, desc=f'Generating {difficulty} dataset') as pbar:
                while len(self.problem_set) < count:
                    node_num = random.randint(min_nodes, max_nodes)
                    cnt = Counter(random.choice(all_walks))
                    node_list = [node for node, _ in cnt.most_common(node_num)]
                    if len(node_list) < node_num:
                        continue

                    H = nx.induced_subgraph(G, node_list)
                    if not nx.is_connected(H):
                        continue


                    answer = self.exact_solution(H)
                    problem_text = self.generate_problem_edge_list(H)
                    self.problem_set.append({
                        'id': len(self.problem_set),
                        'problem_text': problem_text,
                        'exact_answer': answer,
                        'graph': H,
                    })
                    pbar.update(1)
            print(f'{difficulty} dataset generated: {len(self.problem_set)} problems')
            self.save_dataset(difficulty)
            
    def generate_problem_edge_list(self, graph):
        prompt = [f'You are required to find the street in the network that has maximum connections to other streets.']
        prompt.append('- Streets in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        intersections = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        prompt.append('- Intersections in the network: ' + intersections)
        prompt.append(f"Please find the street that intersects with the largest number of other streets (i.e., the street with the maximum number of connections." + " Output only the number of connections that street has, in the following format: '\\boxed{n}'. n is the number of connections of the street.")
        return '\n'.join(prompt)
    
    def exact_solution(self, graph):
        max_degree = 0
        max_degree_node = None
        for node, deg in graph.degree():
            if deg > max_degree:
                max_degree = deg
                max_degree_node = node
        return max_degree