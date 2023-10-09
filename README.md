# RL_King_of_Glory

强化学习（actor critic模型）玩王者荣耀的项目。

## 运行说明

请按照以下步骤运行项目：

1. 确保打开了夜神模拟器。
2. 运行`run_without_train.py`以获取训练数据，注意按->键奖励值加一，<-键奖励值减一，i键暂停或继续程序，esc键结束程序。
3. 运行`train.py`以训练模型。
4. 可以运行`run_or_train_ui.py`来通过图形化界面交互。
5. 增加了`run_self.py`和`train_self.py`，用于模仿学习。

## 备注

1. 本项目参考自[这个项目](https://github.com/FengQuanLi/WZCQ)。
2. 鄙人不才，希望大佬们多多提建议和改进要求，共同完善这个项目。
3. 没装cudnn可能会报错说张量维度不对。
4. 如果角色不动，需要手动配置operation.json文件，参考[这个链接](https://www.bilibili.com/read/cv18924582/#:~:text=%E6%89%93%E5%BC%80%E6%A8%A1%E6%8B%9F%E5%99%A8%E7%9A%84%E6%8C%89%E9%94%AE,%E7%8E%B0%E5%9C%A8%E7%9A%84XY%E5%9D%90%E6%A0%87%E3%80%82)。
5. 如果出现train时出现输入张量维度报错，尝试删除之前所有训练数据并重新获取。
