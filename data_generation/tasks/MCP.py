from matplotlib import pyplot as plt
import networkx as nx
from collections import Counter
import random
import re
import json
import numpy as np
# import walker
import pandas as pd
from tasks.base import *
from tqdm import tqdm
# import igraph as ig
import pdb


class MCP_Task(NPTask):
    def __init__(self, data_loc='dataset'):
        super(MCP_Task, self).__init__(data_loc, 'MCP')

    
    def generate_dataset(self, count=100, difficulty='easy'):
        G = nx.Graph()
        with open('/mnt/sdc/qifan/GLC-Benchmark/data_generation/real_world_graphs/Wikipedia/wikipedia.txt', 'r') as f:
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
                    #print(f'Error parsing line: {line.strip()} | Error: {e}, {names_to_id[u]}, {names_to_id[v]}')
                    continue
            print(f'Edges number: {G.number_of_edges()}, Error count: {error_count}')



        print(f'graph statistics: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
        #all_walks = self.random_walks(G, walk_len=1000, alpha=0.2)
        all_walks = walker.random_walks(G, n_walks=1, walk_len = 1000, start_nodes=range(G.number_of_nodes()), alpha=0.2)
        #all_walks = next(nx.generate_random_paths(G, sample_size=1, path_length=1000, seed=42))

        for difficulty in ['easy', 'medium', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (8,15) if difficulty == 'easy' else (16,30) if difficulty == 'medium' else (31,50)
            #min_nodes, max_nodes = (8,15) if difficulty == 'easy' else (16,25) if difficulty == 'medium' else (26,30)
            with tqdm(total=count, desc=f'Generating {difficulty} dataset') as pbar:
                while len(self.problem_set) < count:
                    #pdb.set_trace()
                    node_size = sample_node_size(min_nodes, max_nodes)
                    # randomly select a walkw
                    
                    c = Counter(random.choice(all_walks))
                    node_list = [k for k, v in c.most_common(node_size)]
                    #pdb.set_trace()
                    if len(node_list) < node_size:
                        continue
                    H = nx.induced_subgraph(G, node_list)
                    #print(H.number_of_nodes())
                    answer= self.exact_solver(H)
                    #pdb.set_trace()
                    problem_text = self.generate_problem(H)
                    self.problem_set.append({
                        'id':len(self.problem_set),
                        'problem_text': problem_text,
                        'graph': H,
                        'exact_answer': answer
                    })    
                    pbar.update(1)
            self.save_dataset(difficulty)

    def generate_problem(self, graph):
        prompt = ['You are required to solve the Maximum Clique Problem for an undirected wikipedia network. In this network, nodes represent wikipedia articles and edges represent hyperlinks between articles. Your objective is to find the largest subset of nodes such that every pair of vertices in this subset is connected by an edge.']
        prompt.append('- Articles in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        hyperlinks = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        prompt.append(f"- Hyperlinks between these articles: {hyperlinks}.")
        prompt.append("Identify the clique with the maximum number of articles in this network. Present your answer in the following format: '\boxed{k}'. k is the number of articles of this clique")
        return '\n'.join(prompt)

    @staticmethod
    def exact_solver(graph):
        clique = max(nx.find_cliques(graph), key=len)
        print(len(clique))
        return len(clique)
    
    
    def approx_solver(self, graph, method='greedy'):
        if method == 'random':
            nodes = list(graph.nodes)
            random.shuffle(nodes)
            max_clique = []
            for node in nodes:
                if all(node in graph.neighbors(neighbor) for neighbor in max_clique):
                    max_clique.append(node)
                else:
                    break
        elif method == 'greedy':
            nodes = sorted(graph.nodes, key=lambda x: graph.degree(x), reverse=True)
            max_clique = []
            for node in nodes:
                if all(node in graph.neighbors(neighbor) for neighbor in max_clique):
                    max_clique.append(node)
        elif method == 'chris':
            max_clique = list(nx.approximation.max_clique(graph))
        return len(max_clique), max_clique  
    
# if __name__ == '__main__':
#     task = MCP_Task()
#     task.generate_dataset()
