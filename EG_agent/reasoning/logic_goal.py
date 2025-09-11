import time
import numpy as np
import re

from EG_agent.reasoning.llms.internvl3 import VLMInference
from EG_agent.reasoning.tools.data_process_check import format_check, goal_transfer_ls_set
from EG_agent.system.path import AGENT_PROMPT_PATH

class LogicGoalGenerator:
    def __init__(self, 
                 prompt_folder='zh',
                 prompt_file1="scene.txt", 
                 prompt_file2="logic_expression.txt", 
                 test_data_set_file="data100.txt"):
        # 加载数据集和 prompt
        with open(f'{AGENT_PROMPT_PATH}/{prompt_folder}/{test_data_set_file}', 'r', encoding="utf-8") as f:
            self.data_set = f.read()
        with open(f'{AGENT_PROMPT_PATH}/{prompt_folder}/{prompt_file1}', 'r', encoding="utf-8") as f:
            self.prompt1 = f.read()
        with open(f'{AGENT_PROMPT_PATH}/{prompt_folder}/{prompt_file2}', 'r', encoding="utf-8") as f:
            self.prompt2 = f.read()
        self.prompt = self.prompt1 + self.prompt2
        self.sections = re.split(r'\n\s*\n', self.data_set)
        self.llm = VLMInference()
        self.llm.add_system_prompt(self.prompt)
    
    def get_feedback_prompt(self, id, prompt1, prompt2, question, result, error_list, error_black_set):
        error_message = ""
        er_word0 = ""
        er_word1 = ""
        er_word2 = ""
        if error_list[0] is not None:
            error_message = ""
        else:
            if error_list[1] != set():
                error_black_set[0] |= set(error_list[1])
            if error_list[2] != set():
                error_black_set[1] |= set(error_list[2])
            if error_list[3] != set():
                error_black_set[2] |= set(error_list[3])
            er_word0 = ", ".join(list(error_black_set[0]))
            er_word1 = ", ".join(list(error_black_set[1]))
            er_word2 = ", ".join(list(error_black_set[2]))
            error_message += f"\n[Blacklist]\n<Illegal Condition>=[{er_word1}]\n<Illegal Object>=[{er_word2}]\n<Other Illegal Words or Characters>=[{er_word0}]\n"
            error_message += "\n[Blacklist] Contains restricted elements.\n"+\
                "If a word from <Illegal Condition> is encountered, choose the nearest parameter with a similar meaning from the [Condition] table to formulate the answer.\n"+\
                "If a word from <Illegal Object> is encountered, choose the nearest parameter with a similar meaning from the [Object] table to formulate the answer."
        prompt = prompt1 + prompt2 + error_message
        return prompt

    def generate_from_dataset(self, max_retry=6):
        # 同步版本：对每个问题直接调用 infer 并立即处理返回结果
        question_list = []
        correct_answer_list = []
        correct_answer_ls_set = []
        outputs_list = [[] for _ in range(len(self.sections))]

        for i, s in enumerate(self.sections):
            x, y = s.strip().splitlines()
            x = x.strip()
            y = y.strip().replace("Goal: ", "")
            question_list.append(x)
            correct_answer_list.append(y)
            print(f"correct answer {i}: {y}")
            correct_answer_ls_set.append(goal_transfer_ls_set(y))

        total_num = len(question_list)
        error_black_ls = [[set(), set(), set()] for _ in range(total_num)]

        # Metrics
        finish_num = 0
        SR = 0
        GCR = 0
        GR_ls = np.zeros(6)
        results = []

        for idx, question in enumerate(question_list):
            # 初始推断
            outputs = []
            attempt = 0
            correct = False

            while attempt < max_retry:
                attempt += 1
                answer = self.llm.infer(question)
                print(f"answer {idx}: {answer}")
                outputs.append(answer)
                format_correct, error_list = format_check(answer)

                if not format_correct:
                    # 构造反馈提示并重试
                    if attempt < max_retry:
                        new_prompt = self.get_feedback_prompt(idx, self.prompt1, self.prompt2, question, answer, error_list, error_black_ls[idx])
                        # 直接用 infer 发送带反馈的 prompt（同步）
                        question = question  # question text stays same; prompt is passed via system/user in LLM wrapper if supported
                        # 使用 prompt 作为 system prompt temporarily by calling add_system_prompt with caution
                        # 为保持简单性，直接将 new_prompt 作为 system prompt for this infer call if VLMInference supports it:
                        # fallback: call infer with new_prompt as text (many LLM wrappers expect full prompt)
                        answer = self.llm.infer(new_prompt)
                        # replace last recorded answer with feedback-influenced one
                        outputs[-1] = answer
                        format_correct, error_list = format_check(answer)
                        if not format_correct:
                            continue
                        # else fall through to success handling
                    else:
                        # exhausted retries
                        GR_flag = False
                        results.append({
                            "id": idx,
                            "question": question,
                            "answer": outputs,
                            "correct_answer": correct_answer_list[idx],
                            "correct": False
                        })
                        break

                # 格式正确，评估是否语义匹配
                GR_ls[len(outputs)-1] += 1
                answer_ls_set = goal_transfer_ls_set(answer)
                if answer_ls_set == correct_answer_ls_set[idx]:
                    SR += 1
                    GCR += 1
                    correct = True
                else:
                    # partial match score
                    if len(correct_answer_ls_set[idx]) > 0:
                        GCR += len([a_set for a_set in answer_ls_set if a_set in correct_answer_ls_set[idx]]) * 1.0 / len(correct_answer_ls_set[idx])
                results.append({
                    "id": idx,
                    "question": question,
                    "answer": outputs,
                    "correct_answer": correct_answer_list[idx],
                    "correct": correct
                })
                break  # move to next question

        # self.llm.close()  # keep existing behavior
        return results

    def generate_single(self, question, correct_answer=None, max_retry=6):
        """
        对单个 question 进行同步推断和反馈，返回最终结果。
        correct_answer: 可选，若提供则用于正确性判定，否则仅做格式判定。
        """
        outputs = []
        error_black_set = [set(), set(), set()]

        attempt = 0
        correct = False

        while attempt < max_retry:
            attempt += 1
            answer = self.llm.infer(question)
            outputs.append(answer)
            format_correct, error_list = format_check(answer)

            if not format_correct:
                if attempt < max_retry:
                    new_prompt = self.get_feedback_prompt(0, self.prompt1, self.prompt2, question, answer, error_list, error_black_set)
                    # 同步用新 prompt 再次调用 infer
                    answer = self.llm.infer(new_prompt)
                    print(f"answer {idx}: {answer}")
                    outputs[-1] = answer
                    format_correct, error_list = format_check(answer)
                    if not format_correct:
                        continue
                else:
                    return {
                        "question": question,
                        "answer": outputs,
                        "correct_answer": correct_answer,
                        "correct": False
                    }

            # 格式正确或经反馈后正确
            if correct_answer is not None:
                answer_ls_set = goal_transfer_ls_set(answer)
                correct_answer_ls_set = goal_transfer_ls_set(correct_answer)
                if answer_ls_set == correct_answer_ls_set:
                    correct = True
                else:
                    correct = False
            else:
                correct = format_correct

            return {
                "question": question,
                "answer": outputs,
                "correct_answer": correct_answer,
                "correct": correct
            }

        # 超出重试仍未成功
        return {
            "question": question,
            "answer": outputs,
            "correct_answer": correct_answer,
            "correct": False
        }

if __name__ == "__main__":
    generator = LogicGoalGenerator()
    # 批量推理
    results = generator.generate_from_dataset()
    # for r in results:
    #     print(f"Q: {r['question']}\nA: {r['answer']}\nCA: {r['correct_answer']}\nCorrect: {r['correct']}\n")
    # 单个问题推理示例
    single_result = generator.generate_single("请将桌上的苹果拿到厨房", correct_answer="IsNear(self,kitchen) & IsCaptured(apple)")
    print(single_result)
