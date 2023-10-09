"""
请为我编写一个名为run_self.py的Python脚本，该脚本的功能如下：
监测键盘事件。
根据operation.json文件，将数字映射到键盘事件。例如，文件中的"8_": "B"意味着action 8对应于键盘的B键。
使用environment.py中的NoxInteractionEnv类的screenshot方法来获取环境状态（state）。
检查是否存在名为traindata_self的文件夹，如果没有，则创建它。
将上述获取的state保存为图片，文件名按照1, 2, 3…的顺序命名，
并保存到traindata_self文件夹下新建的形如2023-10-04_iteration_1的子文件夹中。
将检测到的键盘事件对应的action保存到traindata_self文件夹下的action.txt文件中，每行一个action。
"""


import os
import json
import keyboard
import threading
from datetime import datetime
from time import sleep
from PIL import Image
import numpy as np
from environment import NoxInteractionEnv

# 配置参数
SCREENSHOT_INTERVAL = 0.7  # 截屏间隔时间（秒）
BASE_PATH = "traindata_self"  # 基础路径
OPERATION_MAPPING_FILE = "operation.json"  # 操作映射文件路径
NO_ACTION_LABEL = "-1"  # 无动作标签

# 加载operation.json中的映射
with open(OPERATION_MAPPING_FILE, "r", encoding='utf-8') as file:
    operation_mapping = json.load(file)

# 初始化环境
env = NoxInteractionEnv(OPERATION_MAPPING_FILE)

# 检查traindata_self文件夹是否存在，如果不存在则创建
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# 创建子文件夹
current_date = datetime.now().strftime("%Y-%m-%d")
base_folder_name = current_date
folder_suffix = 1

while os.path.exists(os.path.join(BASE_PATH, f"{base_folder_name}_iteration_{folder_suffix}")):
    folder_suffix += 1

subfolder_name = f"{base_folder_name}_iteration_{folder_suffix}"
subfolder_path = os.path.join(BASE_PATH, subfolder_name)
os.makedirs(subfolder_path)

# 初始化文件名计数器和action文件
file_counter = 1
action_file = open(os.path.join(subfolder_path, "action.txt"), "a")
last_action = [NO_ACTION_LABEL]

def save_state_and_action(key_event):
    global last_action
    print(f"Key pressed: {key_event.name}")
    if key_event.name == 'esc':  # 检测到esc键按下
        action_file.close()  # 关闭文件
        os._exit(0)  # 终止程序
    if key_event.event_type == 'down':
        for k, v in operation_mapping.items():
            if v == key_event.name.upper():
                if last_action[0] == NO_ACTION_LABEL:  # 如果当前只有无动作标签，先清空列表
                    last_action = []
                last_action.append(k[:-1] if k != "-1" else k)  # 如果动作不是"-1"，则删除最后一个字符
                print(f"Executed action: {v}")
                break

def screenshot_task():
    global file_counter, last_action
    while True:
        state = np.squeeze(env.screenshot())
        image = Image.fromarray(state)
        image.save(os.path.join(subfolder_path, f"{file_counter}.png"))
        action_file.write(','.join(last_action) + "\n")  # 使用逗号连接所有动作并写入文件
        last_action = [NO_ACTION_LABEL]  # 重置动作为无动作标签列表
        file_counter += 1
        print("screenshot_saved")
        sleep(SCREENSHOT_INTERVAL)

# 使用hook来捕获所有键盘事件
keyboard.hook(save_state_and_action)

# 开始截屏任务
screenshot_thread = threading.Thread(target=screenshot_task)
screenshot_thread.start()

# 保持脚本运行
keyboard.wait()
