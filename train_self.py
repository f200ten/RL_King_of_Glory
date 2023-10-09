"""
你需要一个名为train_self.py的PyTorch脚本来训练一个模型。模型文件名是actor.pth，位于model文件夹中。
训练数据位于traindata_self文件夹中，每个子文件夹包含一系列按顺序命名的图片文件（1.png, 2.png, 等）和一个action.txt文件，
后者的每一行对应一个动作标签（数字）。在训练过程中，你将实例化model.py中的Actor类，使用图片作为输入，
并得到一个名为action_pred的输出向量。你将使用action.txt中的数字（减1）来创建一个（1，21）维的张量，
其中指定的索引位置为1，其余为0（例如，数字2将转换为一个张量，其中索引1的位置为1，其余为0）。
注意可能存在多个动作，这时多个索引位置变为1。
如果动作标签为-1，则张量全为0。模型的损失将基于这个张量和action_pred的差异。
训练完成后，模型将被保存回model/actor.pth。
"""


import torch
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from model import Actor  # 确保 model.py 中有一个 Actor 类

class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.image_files = [f for f in os.listdir(data_folder) if f.endswith('.png')]
        with open(os.path.join(data_folder, 'action.txt'), 'r') as file:
            self.actions = [list(map(int, line.strip().split(','))) for line in file.readlines()]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.image_files[idx])
        image = read_image(img_path).float()
        action = self.actions[idx]
        action_tensor = torch.zeros(21)
        for act in action:
            if act != -1:
                action_tensor[act-1] = 1
        return image, action_tensor

def train_model(data_path, model_path, epochs=5, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Actor(1, 21, train=True).to(device)  # 确保 Actor 类可以直接被实例化
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for folder in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder)
            if os.path.isdir(folder_path):
                dataset = CustomDataset(folder_path)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                for image, action_target in dataloader:
                    image, action_target = image.to(device), action_target.to(device)
                    optimizer.zero_grad()
                    action_pred = model(image)
                    loss = criterion(action_pred, action_target)
                    loss.backward()
                    optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} finished")

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    data_path = 'traindata_self'
    model_path = 'model/actor.pth'
    train_model(data_path, model_path)

