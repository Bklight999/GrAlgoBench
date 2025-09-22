import networkx as nx
import random
import pickle
import os
import re
import pandas as pd
from tqdm import tqdm
from tasks.base import *

class MST_Task(NPTask):
    def __init__(self, data_loc='dataset', flight_data_dir='/mnt/sdc/qifan/GLC-Benchmark/data_generation/real_world_graphs/flight'):
        super(MST_Task, self).__init__(data_loc, 'MST')
        self.flight_data_dir = flight_data_dir

    # def check_solution(self, problem_id, response):
    #     if r"\\boxed{" in response:
    #         user_answer = response[response.rfind(r"\\boxed{"):].strip()
    #         # parenthis match
    #         left_count = 0
    #         for i in range(len(user_answer)):
    #             if user_answer[i] == "{":
    #                 left_count += 1
    #             elif user_answer[i] == "}":
    #                 left_count -= 1
    #                 if left_count == 0:
    #                     user_answer = user_answer[:i+1]
    #                     break
    #     else:
    #         return response, False

    #     correct_answer = str(self.problem_set[problem_id]['exact_answer'])

    #     return user_answer.strip(), user_answer.strip() == correct_answer.strip() 

    def load_weighted_flight_graph(self, filename):
        G = nx.Graph()
        df = pd.read_csv(filename)
        for _, row in df.iterrows():
            u = str(row['source airport'])
            v = str(row['destination airport'])
            if pd.isna(row['distance_km']):
                continue  # 跳过无效数据
            weight = int(float(row['distance_km']) * 10000)  # 乘以10000并转为整数
            G.add_edge(u, v, weight=weight)
        return G

    def bfs_sample_connected_subgraph(self, G, node_num, max_attempts=30):
        nodes = list(G.nodes())
        for _ in range(max_attempts):
            center = random.choice(nodes)
            queue = [center]
            visited = {center}
            sub_nodes = [center]
            while queue and len(sub_nodes) < node_num:
                cur = queue.pop(0)
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

    def generate_dataset(self, count=100):
        files = [f for f in os.listdir(self.flight_data_dir) if f.endswith('.csv')]
        for file in files:
            G_full = self.load_weighted_flight_graph(os.path.join(self.flight_data_dir, file))
            print(f'Loaded {file}: Nodes={G_full.number_of_nodes()}, Edges={G_full.number_of_edges()}')
            for difficulty in ['easy', 'medium', 'hard']:
                self.problem_set = []
                min_nodes, max_nodes = (8, 12) if difficulty == 'easy' else (13, 20) if difficulty == 'medium' else (21, 30)
                min_edges, max_edges = (15, 25) if difficulty == 'easy' else (26, 40) if difficulty == 'medium' else (41, 60)
                with tqdm(total=count, desc=f'Generating {difficulty} MST dataset for {file}') as pbar:
                    attempts = 0
                    while len(self.problem_set) < count and attempts < count * 10:
                        attempts += 1
                        if G_full.number_of_nodes() < min_nodes:
                            break
                        node_num = random.randint(min_nodes, min(max_nodes, G_full.number_of_nodes()))
                        H = self.bfs_sample_connected_subgraph(G_full, node_num)
                        if H is None or H.number_of_edges() < min_edges:
                            continue
                        # 确保子图是连通的
                        if not nx.is_connected(H):
                            continue
                        all_edges = list(H.edges(data=True))
                        if len(all_edges) < min_edges:
                            continue
                        sampled_edges = random.sample(all_edges, random.randint(min_edges, min(max_edges, len(all_edges))))
                        G = nx.Graph()
                        for u, v, data in sampled_edges:
                            G.add_edge(u, v, weight=data['weight'])
                        # 确保采样后的图仍然是连通的
                        if not nx.is_connected(G):
                            continue
                        # 计算最小生成树
                        try:
                            mst = nx.minimum_spanning_tree(G, weight='weight')
                            mst_weight = sum(data['weight'] for _, _, data in mst.edges(data=True))
                            mst_edges = [(u, v) for u, v in mst.edges()]
                        except Exception:
                            continue
                        edge_list = [[u, v, G[u][v]['weight']] for u, v in G.edges()]
                        problem_text = self.generate_problem_text(list(G.nodes()), edge_list)
                        self.problem_set.append({
                            'id': len(self.problem_set),
                            'problem_text': problem_text,
                            'exact_answer': mst_edges,
                            'graph': G,
                            'mst_weight': mst_weight
                        })
                        pbar.update(1)
                print(f'{difficulty} dataset generated: {len(self.problem_set)} problems')
                self.save_dataset(difficulty)

    def generate_problem_text(self, nodes_list, edge_list):
        edge_str = ', '.join(f'[{u}, {v}, {w}]' for u, v, w in edge_list)
        prompt = [
            "You are given an undirected weighted graph representing the airport network, where nodes represent airports (e.g., airport codes) and edges represent direct flights between airports. The weight of each edge is the distance between two airports.",
            f"- The list of airports: {', '.join(nodes_list)}",
            f"- The number of airports: {len(nodes_list)}",
            f"- The number of direct flights: {len(edge_list)}",
            "- The direct flights (edges) are listed below:",
            edge_str,
            "",
            "Your task is to find the minimum spanning tree (MST) of this graph. A minimum spanning tree is a subset of edges that connects all vertices with the minimum total weight.",
            "",
            "Input format:",
            "- The first line contains two integers n and m, representing the number of airports and the number of direct flights.",
            "- The next m lines each contain three items: u, v, w, indicating a direct flight between airport u and airport v with distance w.",
            "",
            "Output format:",
            "- Output the edges of the minimum spanning tree in the following format: '\boxed{n}' ",
            " The n is the number of edges",
            "",
            "Data:",
            f"{len(nodes_list)} {len(edge_list)}",
            '\n'.join([f"{u} {v} {w}" for u, v, w in edge_list])
        ]
        return '\n'.join(prompt) 