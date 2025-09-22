import random
import pickle
import re
import signal
import functools
import pandas as pd
import time
import networkx as nx
import numpy as np
import os
# from rdkit import Chem
from matplotlib import pyplot as plt
from collections import deque
from tqdm import tqdm

def sample_node_size(min_nodes, max_nodes):
    return random.randint(min_nodes, max_nodes)

def find_node_by_name(graph, name):
    for node, data in graph.nodes(data=True):
        if data.get('name') == name:
            return node
    return None


class NPTask(object):  # todo NPTask -> GraphTask
    def __init__(self, data_loc='./dataset', task_name='TSP'):
        self.data_loc = data_loc
        self.task_name = task_name
        self.problem_set = []

            
    def generate_dataset(self):
        raise NotImplementedError

    def check_solution(self, problem_id, response):
        if "\\boxed{" in response:
            user_answer = response[response.rfind("\\boxed{"):].strip()
            # parenthis match
            left_count = 0
            for i in range(len(user_answer)):
                if user_answer[i] == "{":
                    left_count += 1
                elif user_answer[i] == "}":
                    left_count -= 1
                    if left_count == 0:
                        user_answer = user_answer[:i+1]
                        break
        else:
            return -1, False
        
        user_answer1 = user_answer.split('\\boxed{')[1].split('}')[0]
        if self.task_name in ['LCA']:
            user_answer1 = find_node_by_name(self.problem_set[problem_id]['graph'], user_answer1)
        correct_answer = self.problem_set[problem_id]['exact_answer']
        print(correct_answer, user_answer1, str(user_answer1) == str(correct_answer))
        if user_answer1 is None:
            return -1, False

        return user_answer1, str(user_answer1) == str(correct_answer)
    
    def save_dataset(self, difficulty=None):
        if difficulty:
            pickle.dump(self.problem_set, open(f'{self.data_loc}/{self.task_name}_{difficulty}.pkl', 'wb'))
        else:
            pickle.dump(self.problem_set, open(f'{self.data_loc}/{self.task_name}.pkl', 'wb'))

        print(f'{len(self.problem_set)} problems generated.')

    def load_dataset(self, difficulty=None):
        if difficulty:
            self.problem_set = pickle.load(open(f'{self.data_loc}/{self.task_name}_{difficulty}.pkl', 'rb'))
        else:
            self.problem_set = pickle.load(open(f'{self.data_loc}/{self.task_name}.pkl', 'rb'))

            