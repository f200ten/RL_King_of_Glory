import os
from torch.utils.data import Dataset
from PIL import Image

# 定义 CustomDataset
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.data = []

        for folder in self.folders:
            img_files = [f for f in os.listdir(os.path.join(root, folder)) if f.endswith('.png')]
            with open(os.path.join(root, folder, 'rewards.txt'), 'r') as f:
                rewards = list(map(float, f.readlines()))

            for img_file in img_files:
                number = int(img_file.split('.')[0])  # 获取图像的数字名称
                reward = rewards[number - 1]  # 从rewards.txt获取对应的奖励值
                self.data.append((os.path.join(root, folder, img_file), reward))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, reward = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, reward

