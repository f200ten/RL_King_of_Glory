"""
以下是一个代码，这个代码的任务是实例化名为`Actor`的类，该类位于`model.py`中。
同时，该代码需要监听键盘事件，包括按下Esc键来结束程序，按下'I'键来暂停或继续程序。
如果按下向右箭头键，则reward增加一，如果按下向左箭头键，则reward减少一。
注意打印出按下的按钮对应执行的事件。

在程序启动时，首先通过调用environment.py中`NoxInteractionEnv`类中的`reset`函数来获取一个状态张量`state`。
然后，程序将不断使用`actor`处理`state`以获取`action_vector`，接着从`action_vector`中选择最大值的索引，
将其作为输入传递给`NoxInteractionEnv`的`execute_act`函数。接着，程序会接收来自`execute_act`的输出，并重复这一过程。

额外功能：请检查是否存在一个以当前日期和迭代次数命名的文件夹，如果不存在则创建它。
在这个文件夹内，保存每轮的state和`reward`值。state(图片）的文件名应按顺序编号（例如：1, 2, 3, 4, 5...），
而`reward`值则应记录在一个txt文件中。
"""

"""
改进一下这个代码，键盘事件并不能很好地捕捉到，请另外开辟一个进程，
专门捕捉键盘事件，并修改相应值，循环中检测到对应值修改，就执行相应动作。
"""

"""
修改代码，对于保存图像和reward，如果存在命名为当前日期_1的文件夹，
就创建一个当前日期_2的文件夹。另外，所有文件夹都应按照当前日期_number的格式。
"""


import os
import datetime
import cv2
import torch
from model import Actor  # 请确保model.py在你的PYTHONPATH中
from environment import NoxInteractionEnv  # 请导入正确的环境模块
from multiprocessing import Process, Value, Event, Manager
import keyboard


input_channels = 1
output_dim = 21
jsonpath = "operation.json"

class YourApp:
    def __init__(self, input_channels, output_dim, jsonpath):
        self.actor = Actor(input_channels, output_dim, train=False)  # Assuming Actor is defined elsewhere
        self.env = NoxInteractionEnv(jsonpath)  # Assuming NoxInteractionEnv is defined elsewhere
        self.reward = Value('i', 0)
        self.is_paused = Value('b', False)
        self.screenshot_count = 1
        self.folder_name = self.create_folder()

        # 加载模型
        model_copy_path = "model_copy"  # 指定模型的路径
        if os.path.exists(os.path.join(model_copy_path, 'actor.pth')):
            self.actor.load_state_dict(torch.load(os.path.join(model_copy_path, 'actor.pth')))
            print("Actor model loaded from model_copy.")

    def create_folder(self):
        base_folder_name = datetime.datetime.now().strftime('%Y-%m-%d')
        folder_suffix = 1
        base_path = './traindata'

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        while os.path.exists(os.path.join(base_path, f"{base_folder_name}_iteration_{folder_suffix}")):
            folder_suffix += 1
        
        folder_name = os.path.join(base_path, f"{base_folder_name}_iteration_{folder_suffix}")
        os.makedirs(folder_name)
        
        return folder_name

    def on_key_event(self, e, shared_dict, exit_event):
        if e.event_type == 'down':
            if e.name == 'esc':
                print("Exiting...")
                exit_event.set()
            elif e.name == 'i':
                shared_dict['is_paused'] = not shared_dict['is_paused']
                print("Paused" if shared_dict['is_paused'] else "Resumed")
            elif e.name == 'right':
                shared_dict['reward'] += 1
                print("Right arrow pressed. Reward increased by 1.")
            elif e.name == 'left':
                shared_dict['reward'] -= 1
                print("Left arrow pressed. Reward decreased by 1.")

    def listen_keyboard(self, shared_dict, exit_event):
        keyboard.hook(lambda e: self.on_key_event(e, shared_dict, exit_event))
        keyboard.wait('esc')

    def main_loop(self):
        state = self.env.reset()  # Assuming reset is a method of NoxInteractionEnv
        exit_event = Event()
        
        with Manager() as manager:
            shared_dict = manager.dict({
                'reward': 0,
                'is_paused': False
            })
            
            keyboard_listener = Process(target=self.listen_keyboard, args=(shared_dict, exit_event))
            keyboard_listener.start()

            while True:
                if exit_event.is_set():
                    keyboard_listener.terminate()
                    break
                if not shared_dict['is_paused']:
                    action_vector = self.actor(state)  # Assuming actor returns an action vector
                    max_index = action_vector.argmax()
                    state = self.env.execute_act(max_index)  # Assuming execute_act is a method of NoxInteractionEnv

                    screenshot_filename = os.path.join(self.folder_name, f"{self.screenshot_count}.png")
                    cv2.imwrite(screenshot_filename, state.squeeze())

                    with open(os.path.join(self.folder_name, "rewards.txt"), "a") as f:
                        f.write(f"{shared_dict['reward']}\n")
                        
                    shared_dict['reward'] = 0

                    self.screenshot_count += 1


if __name__ == "__main__":
    app = YourApp(input_channels, output_dim, jsonpath)
    app.main_loop()

