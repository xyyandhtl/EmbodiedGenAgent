### test goal_infer
`python -m examples.goal_infer`
```shell
[Info] Instruction: Instruction: "请前往控制室。"
2025-09-12 09:54:22 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 09:54:22 - INFO - vlm_inference - LLM latency 0.31s
[Info] Sample #0 Attempt 1 Answer: RobotNear_ControlRoom
[Info] Instruction: Instruction: "请在仓库标记异常。"
2025-09-12 09:54:22 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 09:54:22 - INFO - vlm_inference - LLM latency 0.20s
[Info] Sample #1 Attempt 1 Answer: IsMarked_Warehouse
[Info] Instruction: Instruction: "请在仓库标记异常。"
2025-09-12 09:54:22 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 09:54:22 - INFO - vlm_inference - LLM latency 0.18s
[Info] Sample #1 Attempt 2 Answer: IsMarked_Warehouse
[Info] Instruction: Instruction: "请在仓库标记异常。"
2025-09-12 09:54:23 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 09:54:23 - INFO - vlm_inference - LLM latency 0.18s
[Info] Sample #1 Attempt 3 Answer: IsMarked_Warehouse
[Info] Instruction: Instruction: "请在仓库标记异常。"
2025-09-12 09:54:23 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 09:54:23 - INFO - vlm_inference - LLM latency 0.17s
[Info] Sample #1 Attempt 4 Answer: IsMarked_Warehouse
[Info] Instruction: Instruction: "请在仓库标记异常。"
...
```


### test bt_gen
`python -m examples.bt_gen`
!['IsReported_doorway & IsCaptured_victim'](assets/behavior_tree.png) 

