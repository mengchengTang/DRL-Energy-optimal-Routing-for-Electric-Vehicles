import matplotlib.pyplot as plt
import torch
import numpy as np


def render(static, tour_indices, save_path, dynamic, num_nodes, charging_num):
    """画出图形的解决方案."""

    plt.close('all')
    plt.figure(figsize=(8,8))

    idx = tour_indices[0]                                                   #取出相应的batch的路由路线
    if len(idx.size()) == 1:
        idx = idx.unsqueeze(0)                                              #(1,sequence_len)

    idx = idx.expand(static.size(1), -1)                                    #(2,sequence_len)
    data = torch.gather(static[0].data, 1, idx).cpu().numpy()               #(2,equence_len)其中2为坐标点
    start = static[0, :, 0].cpu().data.numpy()                              #取出原点坐标
    point=static[0,:,:].cpu().numpy()
    demand=dynamic[0,1,:].float().cpu().numpy()*4
    x = np.hstack((start[0], data[0], start[0]))                            #np.hstack()将元素按水平方向进行叠加 x坐标：(1,sequence_len)
    y = np.hstack((start[1], data[1], start[1]))                            #y坐标：(1,squence_len)
    idx = np.hstack((0, tour_indices[0].cpu().numpy().flatten(), 0))        #将起点和终点以及路线图进行拼接：（1，squence_len+2)
    where = np.where(idx == 0)[0]                                           #找出哪些点的坐标为仓库

    plt.rc('font', family='Times New Roman')

    for j in range(len(where) - 1):                                         #画几个往返
        low = where[j]                                                      #往返的起始索引
        high = where[j + 1]                                                 #往返的结束索引

        if low + 1 == high:                                                 #如果两个索引连在一起，则直接进行下一回合
            continue
        plt.plot(x[low: high + 1], y[low: high + 1], zorder=1, linewidth=2,label=f"Vehicle{j}")      #画图

    plt.scatter(point[0, charging_num + 1:], point[1, charging_num + 1:], s=40, c='black', zorder=2, label="Customer")  # 轴.scatter()画散点
    plt.scatter(point[0, 0], point[1, 0], s=200, c='r', marker='*', zorder=3, label="Depot")  # 轴.scatter()画仓库点
    plt.scatter(point[0, 1:charging_num + 1], point[1, 1:charging_num + 1], s=40, c='green',marker='s', zorder=3, label="Station")  # 3个充电站画图
    plt.legend(loc=2, fontsize=12, framealpha=0.2,bbox_to_anchor=(1.05, 1), borderaxespad=0)  # 轴.legend()增加图例
    plt.legend(fontsize=12, framealpha=1)  # 轴.legend()增加图例
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    plt.title(f"C{num_nodes}-S{charging_num}",size=25, weight='bold')
    plt.yticks(fontproperties='Times New Roman', size=20)#设置大小及加粗
    plt.xticks(fontproperties='Times New Roman', size=20)
    # for i in range(charging_num + 1, num_nodes + charging_num + 1):
    #     plt.text((point[0, i]), point[1, i], f"{demand[i]}", size=12, color = "k",
    #             ha="center", va="center",weight = "bold")
    # plt.axis('off')
    # plt.show()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)