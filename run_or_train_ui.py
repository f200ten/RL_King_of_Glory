import tkinter as tk
import subprocess

def run_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

# 创建主窗口
root = tk.Tk()
root.title("GUI Application")
root.geometry("200x300")  # 设置窗口大小
root.attributes("-topmost", True)  # 让窗口置顶

# 创建Run按钮，点击时执行run_without_train.py
run_button = tk.Button(root, text="Run", command=lambda: run_script("run_without_train.py"))
run_button.pack(pady=20)  # 增加垂直间距

# 创建Train按钮，点击时执行train.py
train_button = tk.Button(root, text="Train", command=lambda: run_script("train.py"))
train_button.pack(pady=20)

# 创建Run_s按钮，点击时执行run_self.py
run_s_button = tk.Button(root, text="Run_s", command=lambda: run_script("run_self.py"))
run_s_button.pack(pady=20)

# 创建Train_s按钮，点击时执行train_self.py
train_s_button = tk.Button(root, text="Train_s", command=lambda: run_script("train_self.py"))
train_s_button.pack(pady=20)

# 进入Tkinter的事件循环
root.mainloop()
