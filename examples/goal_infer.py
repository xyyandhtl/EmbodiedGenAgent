from EG_agent.reasoning.logic_goal import LogicGoalGenerator

if __name__ == "__main__":
    generator = LogicGoalGenerator()
    # 批量推理
    results = generator.generate_from_dataset()
    # 单个问题推理示例
    single_result = generator.generate_single("在窗户拍摄并上报火情。")
    print(single_result)