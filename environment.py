"""
请帮我编写一个名为 `NoxInteractionEnv` 的环境类，用于与夜神模拟器进行交互。这个类应该包含以下几个函数：
1. `reset`：用于执行截屏函数，获取夜神模拟器的屏幕快照。
2. `screenshot`：用于截取夜神模拟器屏幕，并将其转换为灰度图像，尺寸为1x*y*。
3. `execute_act`：根据输入的动作序号，执行指定路径的JSON文件（ `config.jsonpath`）中对应的操作，
例如执行命令 `adb shell input touchscreen swipe 1757 924 1757 924 100`。最后返回截取的屏幕快照（ `state`）。

"""


import json
import subprocess
from PIL import Image
import numpy as np

class NoxInteractionEnv:
    def __init__(self, jsonpath, screenshot_path="screenshot.png"):
        self.jsonpath = jsonpath
        self.screenshot_path = screenshot_path
        self.actions = self.load_actions()

    def load_actions(self):
        """加载JSON文件中的动作列表"""
        with open(self.jsonpath, 'r', encoding='utf-8') as file:
            return json.load(file)

    def screenshot(self):
        """获取屏幕截图并返回其灰度值的numpy数组"""
        # 使用 adb 命令截屏并保存到文件
        with open(self.screenshot_path, "wb") as f:
            subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=f)

        # 使用 PIL 打开并转换为灰度图像
        image = Image.open(self.screenshot_path).convert('L')

        # 转换为 numpy 数组并调整维度
        state = np.array(image)
        state = np.expand_dims(state, axis=0)

        return state

    def execute_act(self, action_index):
        """根据给定的动作索引执行adb命令，并返回新的屏幕快照"""
        action_index = action_index.numpy() + 1
        action = self.actions.get(str(action_index))
        
        if action:
            subprocess.run(action)
            print(self.actions.get(f"_{action_index}"))
        else:
            print(f"Action index {action_index} not found in actions list.")

        return self.screenshot()

    def reset(self):
        """重置环境并返回屏幕快照"""
        return self.screenshot()


