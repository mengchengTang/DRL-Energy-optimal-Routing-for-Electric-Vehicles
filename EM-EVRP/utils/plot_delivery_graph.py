import matplotlib.pyplot as plt
import torch
import numpy as np


def render(static, tour_indices, save_path, dynamic, num_nodes, charging_num):
    """画出图形的解决方案."""

    plt.close('all')
    plt.figure(figsize=(8,8))

    idx = tour_indices[0]
    if len(idx.size()) == 1:
        idx = idx.unsqueeze(0)

    idx = idx.expand(static.size(1), -1)
    data = torch.gather(static[0].data, 1, idx).cpu().numpy()
    start = static[0, :, 0].cpu().data.numpy()
    point=static[0,:,:].cpu().numpy()
    demand=dynamic[0,1,:].float().cpu().numpy()*4
    x = np.hstack((start[0], data[0], start[0]))
    y = np.hstack((start[1], data[1], start[1]))
    idx = np.hstack((0, tour_indices[0].cpu().numpy().flatten(), 0))
    where = np.where(idx == 0)[0]

    plt.rc('font', family='Times New Roman')

    for j in range(len(where) - 1):
        low = where[j]
        high = where[j + 1]

        if low + 1 == high:
            continue
        plt.plot(x[low: high + 1], y[low: high + 1], zorder=1, linewidth=2,label=f"Vehicle{j}")

    plt.scatter(point[0, charging_num + 1:], point[1, charging_num + 1:], s=40, c='black', zorder=2, label="Customer")
    plt.scatter(point[0, 0], point[1, 0], s=200, c='r', marker='*', zorder=3, label="Depot")
    plt.scatter(point[0, 1:charging_num + 1], point[1, 1:charging_num + 1], s=40, c='green',marker='s', zorder=3, label="Station")
    plt.legend(loc=2, fontsize=12, framealpha=0.2,bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.legend(fontsize=12, framealpha=1)
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