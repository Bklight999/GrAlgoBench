import json
from collections import defaultdict
import os
import sys
import re
from collections import Counter
import tiktoken

sys.path.append('/hpc2hdd/home/mpeng885/qifanzhang/GLC-Benchmark/data_generation')
from tasks import *
from tasks.base import *



for task_difficulty in ['easy', 'medium', 'hard', 'challenge']:
    path = '/hpc2hdd/home/mpeng885/qifanzhang/GLC-Benchmark/Inference/final_results_32B/'
    merged_responses = {}

    # First iterate through folders in path, folder names are model names
    model_list = os.listdir(path)



    print(model_list)
    for model in model_list:
        task_list = os.listdir(path+model)
        print(task_list)
        for task in task_list:
            task_name = task.split('-')[0]
            difficulty = task.split('-')[1].split('.')[0]
            if task_name in ['MaximumFlow','LCA']:
                continue
            #print(task_name, difficulty)
            if difficulty != task_difficulty:
                continue
            response = []
            with open(path+model+'/'+task,'r') as f:
                for line in f:
                    response.append(json.loads(line.strip()))
            print(len(response))
            if task_name not in merged_responses:   
                merged_responses[task_name] = {}
            for i in range(len(response)):
                id = str(response[i]['id'])
                if id not in merged_responses[task_name]:
                    merged_responses[task_name][id] = {}
                if model not in merged_responses[task_name][id]:
                    merged_responses[task_name][id][model] = {}
                
                # Store all predict1 to predict8 in the dictionary
                for times in range(1, 9):
                    predict_key = f'predict{times}'
                    if predict_key in response[i]:
                        merged_responses[task_name][id][model][times] = response[i][predict_key]




    dataset_loc = '/hpc2hdd/home/mpeng885/qifanzhang/GLC-Benchmark/data_generation/dataset'
    difficulty = task_difficulty
    task_list = list(merged_responses.keys())
    pass_1_accuracy = {}
    cons_8_accuracy = {}
    output_token_length = {}  # 新增

    problem_num = len(merged_responses[task_list[0]])

    # 初始化 tiktoken 编码器
    # encoding = tiktoken.get_encoding("cl100k_base")
    # 如果你是gpt-3.5-turbo等模型，也可用 encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    for task_name in task_list:
        task = globals()[task_name+'_Task'](dataset_loc)
        task.load_dataset(difficulty)

        pass_1_accuracy[task_name] = {}
        cons_8_accuracy[task_name] = {}
        #output_token_length[task_name] = {}

        for llm in model_list:
            print(f"task: {task_name}, llm: {llm}")
            total_pass_1_score = 0
            total_cons_8_score = 0
            total_token_num = 0
            total_prediction_num = 0
            pass_1_accuracy[task_name][llm] = {}
            cons_8_accuracy[task_name][llm] = {}
            #output_token_length[task_name][llm] = 0

            for i in range(problem_num):
                problem_scores = []
                answer_list = []
                if llm in merged_responses[task_name][str(i)]:
                    for times in range(1, 9):
                        if times in merged_responses[task_name][str(i)][llm]:
                            prediction = merged_responses[task_name][str(i)][llm][times]
                            # 使用tiktoken统计token数
                            # token_num = len(encoding.encode(prediction, disallowed_special=()))
                            # total_token_num += token_num
                            total_prediction_num += 1

                            answer, subscore = task.check_solution(i, prediction)
                            answer_list.append(answer)
                            problem_scores.append(subscore)
                    counter = Counter(answer_list)
                    print(counter)
                    most_common = counter.most_common(1)
                    most_common_answer, count = most_common[0]
                    if str(most_common_answer) == str(task.problem_set[i]['exact_answer']):
                        total_cons_8_score += 1
                    if problem_scores:
                        total_pass_1_score += 1 if sum(problem_scores) > 0 else 0 #sum(problem_scores) / len(problem_scores)
            pass_1_accuracy[task_name][llm] = total_pass_1_score / problem_num
            cons_8_accuracy[task_name][llm] = total_cons_8_score / problem_num

    path = '/hpc2hdd/home/mpeng885/qifanzhang/GLC-Benchmark/accuracy/acc_ckpt/'
    file_name_pass = path + f'pass_1_accuracy_32B_{task_difficulty}.json'
    file_name_cons = path + f'cons_8_accuracy_32B_{task_difficulty}.json'

    with open(file_name_pass, 'w') as f:
        json.dump(pass_1_accuracy, f)

    with open(file_name_cons, 'w') as f:
        json.dump(cons_8_accuracy, f)

    print(f'pass_1_accuracy: {pass_1_accuracy}')
    print(f'cons_8_accuracy: {cons_8_accuracy}')
