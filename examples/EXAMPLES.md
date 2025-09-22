### test goal_infer
`python -m examples.goal_infer`
```shell
[Info] Instruction: Instruction: "请前往控制室。"
2025-09-12 11:27:14 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:14 - INFO - vlm_inference - LLM latency 0.34s
[Info] Sample #0 Attempt 1 Answer: RobotNear_ControlRoom
[Info] Instruction: Instruction: "请在仓库标记异常设备。"
2025-09-12 11:27:14 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:14 - INFO - vlm_inference - LLM latency 0.21s
[Info] Sample #1 Attempt 1 Answer: IsMarked_Equipment
[Info] Instruction: Instruction: "拍摄并报告楼梯口的烟雾源。"
2025-09-12 11:27:15 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:15 - INFO - vlm_inference - LLM latency 0.38s
[Info] Sample #2 Attempt 1 Answer: IsCaptured_SmokeSource & IsReported_SmokeSource
[Info] Instruction: Instruction: "请在出口处救援受困人员。"
2025-09-12 11:27:15 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:15 - INFO - vlm_inference - LLM latency 0.31s
[Info] Sample #3 Attempt 1 Answer: RobotNear_Doorway & IsReported_Victim
[Info] Instruction: Instruction: "请报告房间A的水泄漏情况。"
2025-09-12 11:27:15 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:15 - INFO - vlm_inference - LLM latency 0.49s
[Info] Sample #4 Attempt 1 Answer: RobotNear_RoomA & IsCaptured_WaterLeak & IsReported_WaterLeak
[Info] Instruction: Instruction: "请报告房间A的水泄漏情况。"
2025-09-12 11:27:16 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:16 - INFO - vlm_inference - LLM latency 0.36s
[Info] Sample #4 Attempt 2 Answer: IsCaptured_WaterLeak & IsReported_WaterLeak
[Info] Instruction: Instruction: "请前往大厅或充电站。"
2025-09-12 11:27:16 - INFO - httpx - HTTP Request: POST http://127.0.0.1:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-09-12 11:27:16 - INFO - vlm_inference - LLM latency 0.29s
[Info] Sample #5 Attempt 1 Answer: RobotNear_Lobby | RobotNear_ChargingStation
...
```


### test bt_gen
`python -m examples.bt_gen`
!['IsReported_doorway & IsCaptured_victim'](assets/behavior_tree.png) 


### test vlmap ros2 runner
`python -m EG_agent.vlmap.vlmap_nav_ros2`
> Hint: this module only requires ros2 rgb, depth and odom topics, you can first use habitat simulator to test, refer to [habitat-data-collector](https://github.com/Eku127/habitat-data-collector.git)

![vlmap_rerun_viewer](assets/vlmap_rerun_viewer.png)
