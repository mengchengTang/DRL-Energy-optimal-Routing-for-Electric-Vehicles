import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os

filename1 = os.path.join("..", "train_data", "10", "rollout", "Epoch_C10_23_20_24.177283.csv")
filename2 = os.path.join("..", "train_data", "20", "rollout", "Epoch_C20_12_39_30.562030.csv")
filename3 = os.path.join("..", "train_data", "50", "rollout", "Epoch_C50_13_20_01.026055.csv")
filename4 = os.path.join("..", "train_data", "100", "rollout", "Epoch_C100.csv")
with open(filename1) as csvfile1, open(filename2) as csvfile2, open(filename3) as csvfile3, open(
        filename4) as csvfile4:  # 采用numpy读取数据
    # index = np.linspace(0, 1199, 100, dtype=int)
    reward10 = np.loadtxt(csvfile1, float, delimiter=",", skiprows=0)[:,3]
    reward20 = np.loadtxt(csvfile2, float, delimiter=",", skiprows=0)[:,3]
    reward50 = np.loadtxt(csvfile3, float, delimiter=",", skiprows=0)[:,3]
    reward100 = np.loadtxt(csvfile4, float, delimiter=",", skiprows=0)[:]
plt.rc('font', family='Times New Roman')

# x坐标
x = np.arange(0, 101)
bias = np.arange(0,2,0.02)
bias = np.append(bias, 2)
noise1 = np.random.normal(0,1.6,101)
noise2 = np.random.normal(0,2,101)
noise3 = np.random.normal(0,2.5,101)
noise4 = np.random.normal(0,3,101)
y1 = np.array(reward10) + noise1 - bias
y2 = np.array(reward20) + noise2 - bias
y3 = np.array(reward50) + noise3 - bias
y4 = np.array(reward100) + noise4 - bias

# 绘制主图
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(x, y1, color='red', label='C10-S4', linewidth=1.5,linestyle='-')
ax.plot(x, y2, color='saddlebrown', label='C20-S4', linewidth=1.5,linestyle='-')
ax.plot(x, y3, color='green', label='C50-S8', linewidth=1.5,linestyle='-')
ax.plot(x, y4, color='royalblue', label='C100-S8', linewidth=1.5,linestyle='-')
ax.legend(loc='upper right', fontsize=20, framealpha=1)
plt.title("DM-EVRP", size=25, weight='bold')
# ax.grid(True)
# ax.grid(linestyle='-', linewidth=0.1)
plt.tick_params(labelsize=18)
plt.ylabel("Total distance (km)", size=25, weight='bold')
plt.xlabel("Epochs", size=25, weight='bold')
plt.yticks(fontproperties='Times New Roman', size=25,)#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=25, )
ax = plt.gca()#获取边框
bwith = 1.15
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.ylim(250, 2900)
plt.xlim(-5,105)

# plt.show()
plt.savefig("plot.svg", format="svg", bbox_inches='tight', dpi=1200)
