import networkx as nx
import random
import pickle
import os
import re
from tqdm import tqdm
from collections import deque
from tasks.base import NPTask
import pandas as pd
import numpy as np
from tasks.base import *


class DistanceThreshold_Task(NPTask):
    def __init__(self, data_loc='dataset', flight_data_dir='/mnt/sdc/qifan/GLC-Benchmark/data_generation/real_world_graphs/flight'):
        super(DistanceThreshold_Task, self).__init__(data_loc, 'DistanceThreshold')
        self.flight_data_dir = flight_data_dir
    

    def load_weighted_flight_graph(self, filename, min_weight=1, max_weight=20):
        G = nx.Graph()
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                u = str(row['source airport'])
                v = str(row['destination airport'])
                weight = float(row['distance_km'])
                G.add_edge(u, v, weight=weight)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        weight = random.randint(min_weight, max_weight)
                        G.add_edge(parts[0], parts[1], weight=weight)
        return G

    def bfs_sample_connected_subgraph(self, G, node_num, max_attempts=30):
        nodes = list(G.nodes())
        for _ in range(max_attempts):
            center = random.choice(nodes)
            queue = deque([center])
            visited = {center}
            sub_nodes = [center]
            while queue and len(sub_nodes) < node_num:
                cur = queue.popleft()
                for neighbor in G.neighbors(cur):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        sub_nodes.append(neighbor)
                        queue.append(neighbor)
                    if len(sub_nodes) >= node_num:
                        break
            if len(sub_nodes) == node_num:
                return G.subgraph(sub_nodes).copy()
        return None

    def get_threshold(self, nodes_list, edges):
        n = len(nodes_list)
        idx_map = {name: i for i, name in enumerate(nodes_list)}
        INF = float('inf')
        dist = [[INF]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v, w in edges:
            i, j = idx_map[u], idx_map[v]
            dist[i][j] = min(dist[i][j], w)
            dist[j][i] = min(dist[j][i], w)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        all_shortest_paths = []
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] < INF:
                    all_shortest_paths.append(dist[i][j])
        if not all_shortest_paths:
            return 1  # 防止空
        return int(np.percentile(all_shortest_paths, 30))

    def generate_dataset(self, count=100):
        files = [f for f in os.listdir(self.flight_data_dir) if f.endswith('.txt') or f.endswith('.csv')]
        for file in files:
            G = self.load_weighted_flight_graph(os.path.join(self.flight_data_dir, file))
            print(f'Loaded {file}: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}')
            for difficulty in ['easy', 'medium', 'hard']:
                self.problem_set = []
                min_nodes, max_nodes = (8, 15) if difficulty == 'easy' else (16, 30) if difficulty == 'medium' else (31, 50)
                with tqdm(total=count, desc=f'Generating {difficulty} dataset for {file}') as pbar:
                    while len(self.problem_set) < count:
                        node_num = random.randint(min_nodes, max_nodes)
                        H = self.bfs_sample_connected_subgraph(G, node_num)
                        if H is None:
                            continue
                        nodes_list = list(H.nodes())
                        edges = []
                        for u, v, data in H.edges(data=True):
                            edges.append([u, v, data['weight'] * 10000])
                        if len(edges) == 0:
                            continue
                        distanceThreshold = self.get_threshold(nodes_list, edges)
                        answer = self.exact_solution(nodes_list, edges, distanceThreshold)
                        problem_text = self.generate_problem_text(nodes_list, edges, distanceThreshold)
                        self.problem_set.append({
                            'id': len(self.problem_set),
                            'problem_text': problem_text,
                            'exact_answer': answer,
                            'graph': H,
                            'distanceThreshold': distanceThreshold
                        })
                        pbar.update(1)
                print(f'{difficulty} dataset generated: {len(self.problem_set)} problems')
                self.save_dataset(difficulty)

    def generate_problem_text(self, nodes_list, edges, distanceThreshold):
        edge_str = ', '.join(f'[{u}, {v}, {w}]' for u, v, w in edges)
        prompt = [
            f'You are given an undirected weighted graph representing the airport network, where nodes represent airports (codes: {", ".join(nodes_list)}) and edges represent direct flights between airports. The weight of each edge is the distance between two airports.',
            f'- The list of airports: {", ".join(nodes_list)}',
            f'- The direct flights (edges) are: {edge_str}',
            f'- The distance threshold is {distanceThreshold}.',
            'For each airport, the distance to another airport is defined as the sum of the weights (distances) along the shortest path connecting them.',
            'Your task is: Return the airport code with the smallest number of other airports that can be reached with a shortest path distance no more than the threshold. If there are multiple such airports, return the one with the lexicographically largest code.',
            "Present your answer in the following format: '\boxed{airport_code}' . The airport_code is the airport code."
        ]
        return '\n'.join(prompt)

    def exact_solution(self, nodes_list, edges, distanceThreshold):
        n = len(nodes_list)
        idx_map = {name: i for i, name in enumerate(nodes_list)}
        INF = float('inf')
        dist = [[INF]*n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v, w in edges:
            i, j = idx_map[u], idx_map[v]
            dist[i][j] = min(dist[i][j], w)
            dist[j][i] = min(dist[j][i], w)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        min_cnt = n+1
        res = None
        for i in range(n):
            cnt = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
            if cnt < min_cnt or (cnt == min_cnt and (res is None or nodes_list[i] > res)):
                min_cnt = cnt
                res = nodes_list[i]
        return res 