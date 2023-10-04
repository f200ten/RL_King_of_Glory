"""
以下是train.py的代码，用于训练一个神经网络，其中包括了Actor和Critic模型，
这些模型定义在model.py文件中。训练的输入数据存储在./traindata目录下，该目录包含多个子文件夹，
每个子文件夹代表一组训练数据。每个子文件夹中包含了以数字12345依次命名的number.png图像文件，
每个number.png文件对应的奖励值存储在rewards.txt文件中，与数字相对应。

在程序运行时，会监听键盘上的esc键，如果按下esc键，将在当前一轮训练完成后保存模型参数到./models目录中。
程序会首先检查./models目录中是否已存在模型参数文件，如果存在则加载它们并继续训练，如果不存在则从头开始训练。
程序会从./traindata目录中的第一个子文件夹开始，遍历每个子文件夹，并利用其中的图像和对应的奖励值来训练Actor和Critic模型。
一旦训练结束，模型参数将被保存。

"""

BATCH_SIZE = 4
EPOCHS = 5


import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Actor, Critic  # 从 model.py 导入 Actor 和 Critic 类
import keyboard
from custom_dataset import CustomDataset  # 自定义数据集类，需要您根据实际情况编写

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device.")

# 定义一个事件标志来退出训练循环
exit_event = False

# 定义一个回调函数，用于处理键盘事件
def on_key_event(e):
    global exit_event
    if e.name == 'esc':
        print("ESC pressed, saving models.")
        exit_event = True

# 监听键盘事件
keyboard.on_release(on_key_event)

# 检查并加载模型参数
model_path = '.\\model'
model_copy_path = '.\\model_copy'

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_copy_path):
    os.makedirs(model_copy_path)

actor = Actor(1, 21, train=True).to(device)
critic = Critic(1, train=True).to(device)
actor_optimizer = optim.Adam(actor.parameters())
critic_optimizer = optim.Adam(critic.parameters())

if os.path.exists(os.path.join(model_copy_path, 'actor.pth')) and os.path.exists(os.path.join(model_copy_path, 'critic.pth')):
    actor.load_state_dict(torch.load(os.path.join(model_copy_path, 'actor.pth')))
    critic.load_state_dict(torch.load(os.path.join(model_copy_path, 'critic.pth')))
    print("Models loaded from model_copy.")

# 数据加载和预处理
transform = transforms.Compose([transforms.ToTensor()])
dataset = CustomDataset(root='traindata', transform=transform)  # 自定义数据集类
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 训练模型
for epoch in range(EPOCHS):  # 假设我们训练10个epoch
    if exit_event:
        print("ESC pressed, saving models.")
        torch.save(actor.state_dict(), os.path.join(model_path, 'actor.pth'))
        torch.save(critic.state_dict(), os.path.join(model_path, 'critic.pth'))

        # 将 model_copy 中的文件替换为 model 中的文件
        for filename in os.listdir(model_path):
            src = os.path.join(model_path, filename)
            dst = os.path.join(model_copy_path, filename)
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)

        break

    for images, rewards in dataloader:
        images, rewards = images.to(device), rewards.to(device)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # Actor network produces action probabilities
        action_probs = actor(images)
        
        # Critic network produces Q values
        Q_values = critic(images)

        # Calculate the Actor loss
        actor_loss = -torch.mean(Q_values * torch.log(action_probs + 1e-8))
        critic_loss = torch.mean((Q_values - rewards) ** 2)

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()

    print(f"Epoch {epoch+1} completed.")

# 保存模型参数
torch.save(actor.state_dict(), os.path.join(model_path, 'actor.pth'))
torch.save(critic.state_dict(), os.path.join(model_path, 'critic.pth'))
print("Training completed and models saved.")
