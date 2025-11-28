from sympy import symbols, Not, Or, And, to_dnf
import re

from EG_agent.planning.btpg.algos.base.planning_action import PlanningAction


# 读入环境文件
def read_env_file(file_path):
    env_dict = {}
    current_key = None
    current_values = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                if '#' in line:
                    parts = line.split('#', 1)
                    current_key = parts[0].strip()
                    current_values = []
                else:
                    current_values.extend(line.split(', '))
                    env_dict[int(current_key)] = set(current_values)
    return env_dict


# 导入不同的环境
import re
def extract_objects(actions):
    pattern = re.compile(r'\w+\(([^)]+)\)')
    objects = []
    for action in actions:
        match = pattern.search(action)
        if match:
            objects.append(match.group(1))
    return objects

def collect_action_nodes(behavior_lib):
    action_list = []

    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            # print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            if cls.num_args == 0:
                action_list.append(PlanningAction(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(PlanningAction(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(PlanningAction(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    print(f"共收集到{len(action_list)}个实例化动作:")
    # for a in self.action_list:
    #     if "Turn" in a.name:
    #         print(a.name)
    print(action_list)
    print("--------------------\n")

    return action_list


def save_data_txt(output_path,data1):
    # Open the file for writing
    with open(output_path, "w", encoding="utf-8") as f:
        # Loop through each entry in data1 and write the required information
        for idx, entry in enumerate(data1, start=1):
            f.write(f"{idx}\n")
            f.write(f"Environment:{entry['Environment']}\n")
            f.write(f"Instruction: {entry['Instruction']}\n")
            # Use ' & ' to join goals, assuming this is the correct separator
            f.write(f"Goals: {' & '.join(entry['Goals'])}\n")
            # Join actions with a comma
            f.write(f"Actions: {', '.join(entry['Actions'])}\n")
            # Join key predicates with a comma
            f.write(f"Vital Action Predicates: {', '.join(entry['Vital Action Predicates'])}\n")
            # Ensure Key_Object is a list and join it with commas
            key_objects = entry['Vital Objects']
            if isinstance(key_objects, list):
                f.write(f"Vital Objects: {', '.join(key_objects)}\n\n")
            else:
                f.write(f"Vital Objects: {key_objects}\n\n")

    print(f"Data saved to {output_path}")

import os
def write_to_file(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(data + '\n')


def act_str_process(act_str, already_split=False):
    """
    Processes action strings in both underscore-separated and parentheses formats.

    Parameters:
    - act_str (str or list): The string or list of strings to be processed.
    - already_split (bool): Whether the input is already a list of action strings.

    Returns:
    - priority_act_ls (list): A list of formatted action strings in "Verb(Object)" format.
    """
    # Determine if the input is already a list or needs to be split
    if already_split:
        act_str_ls = act_str
    else:
        act_str_ls = act_str.replace(" ", "").split(",")

    priority_act_ls = []

    # Process each action string
    for literal in act_str_ls:
        # Remove only unwanted characters while keeping parentheses intact
        literal = re.sub(r"[\[\]\n]", "", literal)

        # Check for underscore-separated format (e.g., "Walk_pear")
        if '_' in literal and '(' not in literal:
            first_part, rest = literal.split('_', 1)
            literal = f"{first_part}({rest})"
            literal = literal.replace('_', ',')

        # Add parentheses if not already in that format
        elif '(' not in literal:
            literal = literal.replace('_', ',')
            literal = f"{literal}()"

        # Append processed literal to the list
        priority_act_ls.append(literal)

    return priority_act_ls

# Example usage:
# action_str = 'Walk_pear, RightGrab_pear, Walk_kitchentable'
# processed = act_str_process(action_str)
# print(processed)
#
# action_str_parentheses = 'Walk(pear), RightGrab(pear), Walk(kitchentable)'
# processed_parentheses = act_str_process(action_str_parentheses)
# print(processed_parentheses)







def goal_transfer_str(goal):
    goal_dnf = str(to_dnf(goal, simplify=True,force=True))
    # print(goal_dnf)
    goal_set = []
    if ('|' in goal or '&' in goal or 'Not' in goal) or not '(' in goal:
        goal_ls = goal_dnf.split("|")
        for g in goal_ls:
            g_set = set()
            g = g.replace(" ", "").replace("(", "").replace(")", "")
            g = g.split("&")
            for literal in g:
                if '_' in literal:
                    first_part, rest = literal.split('_', 1)
                    literal = first_part + '(' + rest
                    # 添加 ')' 到末尾
                    literal += ')'
                    # 替换剩余的 '_' 为 ','
                    literal = literal.replace('_', ',')
                g_set.add(literal)
            goal_set.append(g_set)

    else:
        g_set = set()
        w = goal.split(")")
        g_set.add(w[0] + ")")
        if len(w) > 1:
            for x in w[1:]:
                if x != "":
                    g_set.add(x[1:] + ")")
        goal_set.append(g_set)
    return goal_set



def act_format_records(act_record_list):
    # 初始化一个空列表来存储格式化后的结果
    formatted_records = []
    predicate = []
    objects_ls= []
    # 遍历列表中的每个记录
    for record in act_record_list:

        if "," not in record:
            # 找到括号的位置
            start = record.find('(')
            end = record.find(')')
            # 提取动作和对象
            action = record[:start]
            obj = record[start+1:end]
            # 格式化为新的字符串格式
            formatted_record = f"{action}_{obj}"
            # 将格式化后的字符串添加到结果列表中
            formatted_records.append(formatted_record)
            predicate.append(action)
            objects_ls.append(obj)
        else:
            # 有逗号，即涉及两个物体
            start = record.find('(')
            end = record.find(')')
            action = record[:start]
            objects = record[start + 1:end].split(',')
            obj1 = objects[0].strip()  # 去除可能的空白字符
            obj2 = objects[1].strip()
            formatted_record = f"{action}_{obj1}_{obj2}"
            # 将格式化后的字符串添加到结果列表中
            formatted_records.append(formatted_record)
            predicate.append(action)
            objects_ls.append(obj1)
            objects_ls.append(obj2)

    from collections import OrderedDict
    return list(formatted_records),list(OrderedDict.fromkeys(predicate)),list(OrderedDict.fromkeys(objects_ls))



def remove_duplicates_using_set(lst):
    return list(set(lst))


def update_objects_from_expressions(expressions, pattern, objects):
    for expr in expressions:
        match = pattern.search(expr)
        if match:
            # 将括号内的内容按逗号分割并加入到集合中
            objects.update(match.group(1).split(','))


