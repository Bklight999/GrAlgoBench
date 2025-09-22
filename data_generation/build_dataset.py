from tasks import *
from openai import OpenAI
import networkx as nx
import random
import os
import argparse


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='LCA', help='task name')
    parser.add_argument('--problem_num', type=int, default=100, help='number of problems')
    parser.add_argument('--loc', type=str, default='dataset', help='dataset location')
    args = parser.parse_args()
    classname = args.task + '_Task'
    set_seed(100)
    task = globals()[classname](args.loc)
    task.generate_dataset(count=args.problem_num)
