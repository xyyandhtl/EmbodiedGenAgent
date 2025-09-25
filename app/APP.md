# Application
- the user interface to interact with the agent
- runtime visualization of map, instance, behavior_tree, ... 



![alt text](docs/gui_design.png)

### Build prompts (Use ChatGPT5)

Attempt 1:
> 我要写一个 生成式行为树具身智能体 的用户交互界面，请参考我给的界面设计图 gui_design.png，使用PyQt5写界面代码 window.ui，可以通过 ui2py.sh 转化为 window.py，并确保界面和设计图一致。

> 功能上，我的智能体是由这个 system.py 中的 EGAgentSystem 类来控制的，界面的启动和停止按钮控制智能体的运行。然后给我增加实现界面所需各种交互功能，在和 ui2py.sh 同一个目录下，新增功能实现的文件，太多的话可以分不同文件不同模块写。然后有需要的话完善我的 EGAgentSystem 类以满足要求。界面所需的各种图像、文字、列表等信息，可以都给我先在 EGAgentSystem 预留接口，等我后续自行补上。最后按你的理解，差异化各个小窗信息的更新频率，并美化界面。

Attempt 2:
我现在system.py还没开发完，能给我写个用于简单测试的system_test.py在main.py同目录下吗，能使main.py运行起界面来就行，随便 显示些什么信息在各框里

Attempt 3:
把界面给我改成默认全屏显示

Attempt 4:
当我点击 启动智能体 按钮时，报错
QObject: Cannot create children for a parent that is in a different thread.
(Parent is QTextDocument(0x6075f916ce20), parent's thread is QThread(0x6075f8d8caa0), current thread is QThread(0x7b5a80000d70)
Segmentation fault (core dumped)

Attempt 5: