import re
import json

from EG_agent.reasoning.llms.internvl3 import VLMInference
from EG_agent.reasoning.tools.data_process_check import format_check, goal_transfer_ls_set
from EG_agent.system.module_path import AGENT_PROMPT_PATH


class LogicGoalGenerator:
    def __init__(
        self,
        prompt_folder='zh',
        prompt_scene="scene.txt",
        prompt_goal="logic_expression.txt",
        test_data_set_file="data100.txt",
    ):
        # 加载数据集与 system prompt（仅注入一次，由 VLMInference 管理）
        with open(f'{AGENT_PROMPT_PATH}/{prompt_folder}/{test_data_set_file}', 'r', encoding="utf-8") as f:
            self.data_set = f.read()
        self.sections = re.split(r'\n\s*\n', self.data_set)

        # read prompt_scene as a template and format with module-level object sets
        with open(f'{AGENT_PROMPT_PATH}/{prompt_folder}/{prompt_scene}', 'r', encoding="utf-8") as f:
            self.prompt_scene_template = f.read()
        with open(f'{AGENT_PROMPT_PATH}/{prompt_folder}/{prompt_goal}', 'r', encoding="utf-8") as f:
            self.prompt_goal = f.read()

        self.llm = VLMInference()

    def prepare_prompt(self, object_set=()):
        from EG_agent.prompts import default_objects
        all_cond_str = default_objects.AllCondition
        object_set = object_set or {"AnyObject"}
        # if object_set is None:
        #     # 预定义 object sets 时用这个
        #     object_list = sorted(list(default_objects.AllObject))
        #     object_str = json.dumps(object_list, ensure_ascii=False)
        #     self.prompt_scene = self.prompt_scene_template.format(
        #         AllObject=object_str,
        #         AllCondition=all_cond_str
        #     )
        # else:
        # 需要动态传入 object sets 时用这个在外部调用
        objects_list = sorted(list(object_set))
        objects = json.dumps(objects_list, ensure_ascii=False)
        self.prompt_scene = self.prompt_scene_template.format(
            AllObject=objects,
            AllCondition=all_cond_str
        )
        self.prompt = self.prompt_scene + self.prompt_goal
        self.llm.set_system_prompt(self.prompt)

    def _parse_section(self, section):
        """
        更健壮的解析：取首个非空行作为 question，
        在其后的行里找含 'Goal:' 的一行为答案；找不到则取最后一行。
        """
        lines = [ln.strip() for ln in section.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return None, None
        question = lines[0]
        # 找到含 Goal: 的行（不区分大小写）
        answer_line = None
        for ln in lines[1:]:
            if ln.lower().startswith('goal'):
                answer_line = ln
                break
        if answer_line is None:
            answer_line = lines[-1]
        correct_answer = re.sub(r'^\s*goal\s*:\s*', '', answer_line, flags=re.IGNORECASE).strip()
        return question, correct_answer

    def get_feedback_message(self, question, result, error_list, error_black_set):
        """
        仅返回反馈文本（不再拼接 prompt1/prompt2）。
        累计黑名单并以三类清单形式给出替换建议。
        约定沿用原索引映射：
          error_list[1] -> Other
          error_list[2] -> Condition
          error_list[3] -> Object
        """
        # 容错处理：error_list 可能长度不足
        def _as_set(x):
            try:
                return set(x) if isinstance(x, (set, list, tuple)) else set()
            except Exception:
                return set()

        if isinstance(error_list, (list, tuple)):
            other_set = _as_set(error_list[1]) if len(error_list) > 1 else set()
            cond_set = _as_set(error_list[2]) if len(error_list) > 2 else set()
            obj_set = _as_set(error_list[3]) if len(error_list) > 3 else set()
        else:
            other_set = cond_set = obj_set = set()

        # 累计进入黑名单
        error_black_set[0] |= other_set
        error_black_set[1] |= cond_set
        error_black_set[2] |= obj_set

        er_word0 = ", ".join(sorted(error_black_set[0]))
        er_word1 = ", ".join(sorted(error_black_set[1]))
        er_word2 = ", ".join(sorted(error_black_set[2]))

        # 若三类均为空，则不给额外反馈
        if not (er_word0 or er_word1 or er_word2):
            return ""

        feedback = []
        feedback.append("\n[Blacklist]")
        feedback.append(f"<Illegal Condition>=[{er_word1}]")
        feedback.append(f"<Illegal Object>=[{er_word2}]")
        feedback.append(f"<Other Illegal Words or Characters>=[{er_word0}]")
        feedback.append("\n[Instruction]")
        feedback.append("The above blacklist contains restricted elements.")
        feedback.append("If a word from <Illegal Condition> is encountered, choose the nearest parameter with a similar meaning from the [Condition] table to formulate the answer.")
        feedback.append("If a word from <Illegal Object> is encountered, choose the nearest parameter with a similar meaning from the [Object] table to formulate the answer.")
        feedback.append("Return only a valid logic expression. Do not include explanations or the 'Goal:' prefix.")
        return "\n".join(feedback)

    def generate_from_dataset(self, max_retry=6):
        """
        同步逐条推理 + 反馈重试。
        返回：包含每条样本结果的列表。
        """
        question_list = []
        correct_answer_list = []
        correct_answer_ls_set = []

        for i, s in enumerate(self.sections):
            q, ca = self._parse_section(s)
            if not q or not ca:
                print(f"[Warn] Skip malformed sample at section #{i}.")
                continue
            question_list.append(q)
            correct_answer_list.append(ca)
            correct_answer_ls_set.append(goal_transfer_ls_set(ca))

        results = []

        for idx, question in enumerate(question_list):
            outputs = []
            error_black_set = [set(), set(), set()]  # [Other, Condition, Object]
            last_error_list = None
            correct = False
            success = False  # 标记是否有一次尝试格式通过（不再重试）

            for attempt in range(1, max_retry + 1):
                # 将上一轮的错误转为简短反馈附在问题后面
                feedback = self.get_feedback_message(
                    question,
                    outputs[-1] if outputs else "",
                    last_error_list,
                    error_black_set
                ) if last_error_list is not None else ""

                prompt_text = f"{question}\n{feedback}" if feedback else question
                print(f"[Info] Instruction: {question}")
                answer = self.llm.infer(prompt_text).replace("Goal:", "").strip()
                print(f"[Info] Sample #{idx} Attempt {attempt} Answer: {answer}")
                outputs.append(answer)

                format_correct, error_list = format_check(answer)
                if not format_correct:
                    last_error_list = error_list
                    continue

                # 格式正确（成功取得一次有效格式），不再重试
                success = True

                # 若格式正确，做严格集合匹配评估
                answer_ls_set = goal_transfer_ls_set(answer)
                if answer_ls_set == correct_answer_ls_set[idx]:
                    correct = True

                break  # 不再尝试更多重试
            
            # 尝试结束后统一记录结果
            # if success:
            #     results.append({
            #         "id": idx,
            #         "question": question,
            #         "answer": outputs,  # 历次尝试
            #         "correct_answer": correct_answer_list[idx],
            #         "correct": correct
            #     })
            # else:
            #     # 用尽重试仍失败（格式未通过）
            #     results.append({
            #         "id": idx,
            #         "question": question,
            #         "answer": outputs,
            #         "correct_answer": correct_answer_list[idx],
            #         "correct": False
            #     })

        # return results

    def generate_single(self, question, max_retry=6) -> str:
        """
        Retry-based single-question inference:
        - perform up to max_retry LLM calls,
        - attach concise feedback from previous format errors when available,
        - return the valid answer string on first format-pass,
        - return None if all retries fail.
        """
        latest_answer = ""
        error_black_set = [set(), set(), set()]
        last_error_list = None

        for attempt in range(1, max_retry + 1):
            feedback = self.get_feedback_message(
                question,
                latest_answer,
                last_error_list,
                error_black_set
            ) if last_error_list is not None else ""

            prompt_text = f"{question}\n{feedback}" if feedback else question
            # 仅在第一次请求时记录记忆，供后续对话理解总体意图，后续重试不记录
            answer = self.llm.infer(prompt_text, record_memory=attempt==1).replace("Goal:", "").strip()
            print(f"[Info] Attempt {attempt} Answer: {answer}")
            latest_answer = answer

            # format_correct, error_list = format_check(answer)
            format_correct, error_list = True, []  # 暂时不做格式检查，直接返回结果
            if format_correct:
                return answer
            print(f"[Info] format_correct is: {format_correct}, error_list is: {error_list}!")
            last_error_list = error_list

        # all retries exhausted, no valid format
        return ""
    
    def ask_question(self, question: str, use_system_prompt: bool = True) -> str:
        """
        直接问答接口对外暴露，不存入记忆。
        """
        return self.llm.infer(question, record_memory=False, use_system_prompt=use_system_prompt)


if __name__ == "__main__":
    generator = LogicGoalGenerator()
    # 批量推理
    # results = generator.generate_from_dataset()
    # 单个问题推理示例
    single_result = generator.generate_single("在窗户拍摄并上报火情。")
