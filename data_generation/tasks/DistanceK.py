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

class DistanceK_Task(NPTask):
    def __init__(self, data_loc='./dataset'):
        super(DistanceK_Task, self).__init__(data_loc, 'DistanceK')


    def bfs_sample_with_max_distance(self, G, node_num, max_attempts=30, sample_ratio=0.5, min_per_layer=1):
        nodes = list(G.nodes())
        for _ in range(max_attempts):
            center = random.choice(nodes)
            queue = deque([(center, 0)])
            dist_dict = {center: 0}
            visited = set([center])
            while len(dist_dict) < node_num and queue:
                cur_layer = []
                cur_dist = queue[0][1]
                while queue and queue[0][1] == cur_dist:
                    cur_layer.append(queue.popleft()[0])

                random.shuffle(cur_layer)
                n_next = max(int(len(cur_layer) * sample_ratio), min_per_layer)
                selected = cur_layer[:n_next]
                not_selected = cur_layer[n_next:]

                for node in not_selected:
                    visited.add(node)

                for cur in selected:
                    for neighbor in G.neighbors(cur):
                        if neighbor not in visited:
                            dist_dict[neighbor] = cur_dist + 1
                            visited.add(neighbor)
                            queue.append((neighbor, cur_dist + 1))
                            if len(dist_dict) >= node_num:
                                break
                    if len(dist_dict) >= node_num:
                        break

            if len(dist_dict) == node_num:
                subG = G.subgraph(dist_dict.keys()).copy()
                max_dist = max(dist_dict.values())
                return subG, center, max_dist
        return None, None, None

    def generate_dataset(self, count=100):
        G = nx.Graph()
        with open('/mnt/sdc/qifan/GLC-Benchmark/data_generation/real_world_graphs/city_road/London__UK_street_graph.txt', 'r') as f:
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
        
        print(f'graph statistics: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, Error count: {error_count}')
        

        for difficulty in ['easy', 'medium', 'hard']:
            self.problem_set = []
            min_nodes, max_nodes = (8,15) if difficulty == 'easy' else (16,30) if difficulty == 'medium' else (31,50)
            dis_threshold = 5 if difficulty == 'easy' else 7 if difficulty == 'medium' else 10

            with tqdm(total=count, desc=f'Generating {difficulty} dataset') as pbar:
                while len(self.problem_set) < count:
                    node_num = random.randint(min_nodes, max_nodes)
                    H, center, max_dis = self.bfs_sample_with_max_distance(G, node_num)

                    if H is None or max_dis < dis_threshold:
                        continue

                    k = random.randint(2, max_dis)
                    answer = len(self.exact_solution(H, k, center))
                    if answer == 0:
                        continue
                    print(answer)
                    problem_text = self.generate_problem_edge_list(H, k, center)

                    self.problem_set.append({
                        'id': len(self.problem_set),
                        'problem_text': problem_text,
                        'exact_answer': answer,
                        'graph': H,
                        'k': k,
                        'start_node': center
                    })
                    pbar.update(1)
            print(f'{difficulty} dataset generated: {len(self.problem_set)} problems')
            self.save_dataset(difficulty)
            
    def generate_problem_edge_list(self, graph, k, center):
        prompt = [f'You are given an undirected graph representing the London street network, where nodes represent streets and edges represent intersections. The distance between two directly connected nodes is 1. Given a street {graph.nodes[center]["name"]}, find all streets that are exactly distance {k} away from the street.']
        prompt.append('- Streets in the network: ' + ", ".join([graph.nodes[node]['name'] for node in graph.nodes()]))
        intersections = ", ".join(f"{graph.nodes[u]['name']} and {graph.nodes[v]['name']}" for u, v in graph.edges())
        prompt.append('- Intersections in the network: ' + intersections) 
        prompt.append(f'Please find the streets in distance {k} from the street {graph.nodes[center]["name"]} and output the number of these streets.')
        prompt.append(" Present your answer in the following format: '\boxed{n}'. n is the number of these streets")
        return '\n'.join(prompt)
    
    def exact_solution(self, graph, k, center):
        H = graph.copy()
        dist_dict = {center: 0}
        queue = deque([center])
        while queue:
            cur = queue.popleft()
            for neighbor in H.neighbors(cur):
                if neighbor not in dist_dict:
                    dist_dict[neighbor] = dist_dict[cur] + 1
                    queue.append(neighbor)
        
        return [node for node, dist in dist_dict.items() if dist == k]
