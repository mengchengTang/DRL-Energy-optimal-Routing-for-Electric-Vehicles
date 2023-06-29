import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.figure(figsize=(11,8))
size = 3
plt.rc('font', family='Times New Roman')

model = "small_to_larger"
x1 = np.array([0,1.5,3])
x_label = np.array([0,1.5,3])
d = [383.04, 584.76, 1113.99]
c = [381.79, 583.21, 1104.03]
b = [377.59, 562.16]
a = [352.35]


total_width1, n1 = 0.8, 4
total_width2, n2 = 0.6, 3
total_width3, n3 = 0.4, 2
width1 = total_width1 / n1
width2 = total_width2 / n2
width3 = total_width3 / n3
x0 = [x1[0] - (total_width1 - width1) / 2]
x2 = [x1[0] - (total_width1 - width1) / 2 + width1, x1[1] - (total_width2 - width2) / 2]
x3 = [x1[0] - (total_width1 - width1) / 2 + 2*width1, x1[1], x1[2] - (total_width3 - width3) / 2]
x4 = [x1[0] - (total_width1 - width1) / 2 + 3*width1, x1[1] - (total_width2 - width2) / 2 + 2*width2, x1[2] - (total_width3 - width3) / 2 +width3]


plt.bar(x0, a,  width=width1, label='C10-S4',color='#1f77b4',ec='black',lw=.5)
for x,y in zip(x0,a):
    plt.text(x,y+25,y,fontsize=22,horizontalalignment='center',weight='bold',rotation = 90)

plt.bar(x2, b, width=width2, label='C20-S4',color='#ff7f0e',ec='black',lw=.5)
for x,y in zip(x2,b):
    plt.text(x,y+25,y,fontsize=22,horizontalalignment='center',weight='bold',rotation = 90)

plt.bar(x3, c, width=width3, label='C50-S8',color='#003300',ec='black',lw=.5)
for x,y in zip(x3,c):
    plt.text(x,y+25,y,fontsize=22,horizontalalignment='center',weight='bold',rotation = 90)

plt.bar(x4, d, width=width3, label='C100-S8',color='#660066',ec='black',lw=.5)
for x,y in zip(x4,d):
    plt.text(x,y+25,y,fontsize=22,horizontalalignment='center',weight='bold',rotation = 90)


plt.legend(loc='upper left', fontsize=22, framealpha=1)
ax = plt.gca()  # 获取边框
bwith = 1.15
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_color("none")
ax.spines['right'].set_color("none")

labels = ['C10-S4', 'C20-S8', 'C50-S8']

y_major_locator = MultipleLocator(200)
ax.yaxis.set_major_locator(y_major_locator)

plt.yticks(fontproperties='Times New Roman', size=22,)#设置大小及加粗
plt.xticks(fontproperties='Times New Roman', size=22,)
plt.ylabel("Total distance (km)", size=26,)
plt.xlabel("DM-EVRP", size=26)
plt.ylim(0, 1400)
plt.xticks(x_label, labels)
# plt.show()
plt.savefig("plot_bar_larger.svg",format = "svg",  bbox_inches='tight', dpi=1200)