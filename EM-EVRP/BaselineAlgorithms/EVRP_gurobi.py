# 采用gurobi求解精确解, 测试集统一采用存储在test_data中的数据
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import csv
import datetime
import argparse


def make_instance(args):
    static, dynamic, distance, slope = args
    grid_size = 1
    return {
        'static': np.array(static, ) / grid_size,
        'dynamic': np.array(dynamic, ),  # scale demand
        'distance': np.array(distance, ),
        'slope': np.array(slope, )
    }

def HCVRPDataset(filename=None,  num_samples=256, offset=0):
    """
    :param filename: 需要读取的文件名
    :param num_samples: 读取实例的数量
    :param offset: 从哪个位置开始读
    :return: 读取的数据，列表形式，每一个列表中是一个字典
    """
    assert os.path.splitext(filename)[1] == '.pkl'

    with open(filename, 'rb') as f:
        data = pickle.load(f)  # (N, size, 2)
    data = [make_instance(args) for args in data[offset:offset + num_samples]]
    return data


class VehicleRouting():
    def __init__(self, i, instance, t_limit, Start_SOC, velocity, max_load, custom_num, charging_num, plot_num):
        # 图的参数
        self.static = instance["static"]  # 地理坐标，单位 km
        self.dynamic = instance["dynamic"]
        self.distances = instance["distance"]
        self.slope = instance["slope"]
        self.i = i
        self.max_load = max_load
        self.Start_SOC = Start_SOC
        self.t_limit = t_limit
        self.velocity = velocity
        self.custom_num = custom_num
        self.charging_num = charging_num
        self.plot_num = plot_num  # 画图的张数
        # 车辆能耗参数
        self.mc = 4100
        self.g = 9.81
        self.w = 1000
        self.Cd = 0.7
        self.A = 6.66
        self.Ad = 1.2041
        self.Cr = 0.01
        self.motor_d = 1.18
        self.motor_r = 0.85
        self.battery_d = 1.11
        self.battery_r = 0.93
        self.demands = self.dynamic[1] * max_load


    def build_model(self,):
        # 创建模型
        self.model = Model("EVRP")
        # 设置集合
        custom_num = self.custom_num
        charging_num = self.charging_num
        depot = [0]
        depotChargingStation = [0]
        nc = [i for i in range(charging_num + 1, custom_num +charging_num + 1 )]
        nf = [i for i in range(1, charging_num + 1)]
        v = depot + nf + nc   # 所有顶点集合
        chargingStationSet = depotChargingStation + nf  # 可充电顶点的集合

        # 边的信息
        A = [(i, j) for i in v for j in v]  # 边的集合
        c = {(i, j): self.distances[i, j] for i in v for j in v}  # 距离矩阵
        g = {(i, j): self.slope[i, j] for i in v for j in v}  # 坡度矩阵
        t = {(i, j): self.distances[i, j] / self.velocity for i in v for j in v}  # 时间矩阵
        d = {i: self.demands[i] for i in v}  # 点的需求
        vehicleSpecificConstant = 0.5 * self.Cd * self.A * self.Ad  # 空气阻力系数

        # 添加变量
        x = self.model.addVars(A, vtype=GRB.BINARY, name='x')  # Equal 1 if vehicle goes from node i to node j
        w = self.model.addVars(A, vtype=GRB.CONTINUOUS, name='w')  # 车辆在ij的转载量
        q = self.model.addVars(v, vtype=GRB.CONTINUOUS, name='q')  # 虚拟充电站集合
        y = self.model.addVars(v, vtype=GRB.CONTINUOUS, name='y')  # 车辆到达一点剩余电量
        time_c = self.model.addVars(v, vtype=GRB.CONTINUOUS, name="t_c")  # 车辆到达某一点的时间

        # 设定目标函数
        self.model.setObjective(quicksum(self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
                                         for i, j in A if i != j), GRB.MINIMIZE)
        # self.model.setObjective(quicksum( c[i,j] * x[i, j] for i, j in A if i != j), GRB.MINIMIZE)
        # 添加约束
        # 约束一：每个客户点只能访问一次
        self.model.addConstrs(quicksum(x[i, j] for j in v if j != i) == 1 for i in nc)
        # 约束二：充电站访问可以少于一次
        self.model.addConstrs(quicksum(x[i, j] for j in v if j != i) <= 1 for i in nf)
        # 约束三：流量守恒
        self.model.addConstrs(
            quicksum(x[j, i] for i in v if i != j) - quicksum(x[i, j] for i in v if i != j) == 0 for j in v)
        # 约束四：货物量约束
        self.model.addConstrs(
            quicksum(w[j, i] for j in v if i != j) - quicksum(w[i, j] for j in v if i != j) == d[i] for i in nc + nf)
        # 约束五：小于最大装载量约束在0~max_load之间
        self.model.addConstrs(w[i, j] <= self.max_load * x[i, j] for i, j in A if i != j)
        self.model.addConstrs(w[i, j] >= 0 for i, j in A if i != j)
        # 约束六：车辆返回仓库是载重拉满
        self.model.addConstrs(w[0, j] == self.max_load * x[0, j] for j in v)
        # 约束六：去往客户点的能量消耗约束
        self.model.addConstrs(
            self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
                                             - self.Start_SOC * (1 - x[i, j]) <= y[i] - y[j] for i in v for j in nc)
        self.model.addConstrs(
            y[i] - y[j] <= self.motor_d * self.battery_d * (
                        ((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                        + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[
                i, j] + self.Start_SOC * (1 - x[i, j]) for i in v for j in nc)

        # 约束八：去往充电站的能量消耗约束
        self.model.addConstrs(
            self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
                                             - self.Start_SOC * (1 - x[i, j]) <= y[i] - q[j] for i in v for j in nf)
        self.model.addConstrs(
            y[i] - q[j] <= self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
            + self.Start_SOC * (1 - x[i, j])  for i in v for j in nf)

        self.model.addConstrs(y[i] == self.Start_SOC for i in chargingStationSet)
        # 约束九：总能耗约束
        self.model.addConstrs(y[i] >= self.motor_d * self.battery_d * (((self.g * g[i, 0]) + (self.g * self.Cr)) * (self.mc + w[i, 0] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, 0] / 3600 * x[i, 0]   for i in nc)

        # 约束十：追踪时间约束
        self.model.addConstrs((t[i,j] + 0.33) * x[i,j] - self.t_limit * (1-x[i,j]) <= time_c[j] - time_c[i] for i in v for j in nc)
        self.model.addConstrs(time_c[j] - time_c[i] <= (t[i, j] + 0.33) * x[i, j] + self.t_limit * (1 - x[i, j])  for i in v for j in nc)
        # 到达充电站的约束
        self.model.addConstrs((t[i, j] + 1) * x[i, j] - self.t_limit * (1 - x[i, j]) <= time_c[j] - time_c[i] for i in v for j in nf)
        self.model.addConstrs(time_c[j] - time_c[i] <= (t[i, j] + 1) * x[i, j] + self.t_limit * (1 - x[i, j])  for i in v for j in nf)
        # 从仓库出发的时间约束
        self.model.addConstrs(time_c[i] == 0 for i in depot)
        self.model.addConstrs(time_c[i] <= self.t_limit - t[i,0] * x[i, 0] for i in nc+nf)

        # self.model.Params.TimeLimit = 50
        self.model.Params.MIPGap = 0
        # self.model.Params.Threads = 1
        self.model.optimize()


        if self.i < self.plot_num:
            K = 0
            for i in v:
                # print(y[i].x)
                # print("time:",time_c[i].x)
                if i != 0 and x[0, i].x > 0.9:
                    K += 1
            routes = []

            for i in v:
                if i != 0 and x[0, i].x > 0.9:
                    aux = [0, i]
                    while i != 0:
                        j = i
                        for h in v:
                            if j != h and x[j, h].x > 0.9:
                                aux.append(h)
                                i = h
                    routes.append(aux)

            # for i in routes:
            #     for j in range(len(i)-1):
            #         print(f"点{i[j]}到{i[j+1]}的距离为：{self.distances[i[j],i[j+1]]}")
            #         print(f"点{i[j]}到{i[j + 1]}的坡度为：{self.slope[i[j], i[j + 1]]}")
            #         e = self.motor_d*self.battery_d*(0.5 * self.Cd * self.A * self.Ad * (self.velocity/3.6) ** 2 + (w[i[j],i[j+1]].x *self.w+self.mc) * self.g * self.slope[i[j], i[j + 1]]  + (w[i[j],i[j+1]].x*self.w+self.mc) * self.g * self.Cr) * self.distances[i[j],i[j+1]]/3600
            #         energy.append(e)
            #         print(f"{(i[j],i[j+1])}段能耗为{e}")
            # average_e = np.sum(energy)
            # print("*"*50)
            # print(average_e)

            power = []
            for i in v:
                for j in v:
                    if w[i,j].x > 0.2:
                        power.append(w[i,j].x)
                        power.append(i)
                        power.append(j)
            print(power)

            # 画图部分
            fig = plt.figure(figsize=(10,10))
            xc = self.static[0, :]
            yc = self.static[1, :]
            n = self.custom_num
            f = self.charging_num
            plt.scatter(xc[f+1:n+f+1], yc[f+1:n+f+1], c='b')
            plt.scatter(xc[1:f+1], yc[1:f+1], c='g')
            plt.scatter(xc[0], yc[0], c='r')
            for i in nc:
                # plt.text(xc[i], yc[i] + 3, "C" + format(i) + "D" + "%.2f" % d[i] +"SOC"+ "%.3f" % y[i].x + "time" + "%.3f" % time_c[i].x)
                plt.text(xc[i], yc[i] + 3,"C" + format(i))
            for i in nf:
                plt.text(xc[i], yc[i] + 3, "F" + format(i))
            for k in range(0, len(routes)):
                for i in range(1, len(routes[k])):
                    plt.annotate(text="", xy=(xc[routes[k][i]], yc[routes[k][i]]),
                                 xytext = (xc[routes[k][i - 1]], yc[routes[k][i - 1]]), arrowprops=dict(arrowstyle='->'))
            plt.xlim(-5,105)
            plt.ylim(-5,105)

            if not args.CVRP_lib_test:
                save_path = os.path.join("graph", f"{self.custom_num}", "gurobi")
            else:
                save_path = os.path.join("graph", "CVRPlib")
            name = f'batch%d_%2.4f.png' % (self.i, self.model.ObjVal)
            save_path = os.path.join(save_path, name)
            plt.savefig(save_path, bbox_inches='tight', dpi=100)

        optim_cost = self.model.ObjVal
        solution_time = self.model.Runtime
        # 将结果记录到文件
        if not args.CVRP_lib_test:
            out_path = os.path.join("data_record", f"{self.custom_num}", "gurobi", f"online_C{self.custom_num}_{now}.csv")
            with open(out_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([optim_cost, solution_time])

        return optim_cost, solution_time


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='ACO solving electric vehicle routing problem')
    # 图上参数
    parser.add_argument('--nodes',  default=20, type=int)
    parser.add_argument('--CVRP_lib_test', default=False)
    parser.add_argument('--Start_SOC', default=80, type=float, help='SOC, unit: kwh')
    parser.add_argument('--velocity', default=50, type=float, help='unit: km/h')
    parser.add_argument('--max_load', default=4, type=float, help='the max load of vehicle')
    parser.add_argument('--charging_num', default=5, type=int, help='number of charging_station')
    parser.add_argument('--t_limit', default=10, type=float, help='tour duration time limitation, 12 hours')
    parser.add_argument('--seed', default=12345, type=float, help='test seed')
    parser.add_argument('--plot_num', default=1,  help='whether plot')
    args = parser.parse_args()

    filename = os.path.join("..", "test_data",f"{args.nodes}", "256_seed12345.pkl")  # 上级目录用..
    # filename = os.path.join("..", "test_data", "CVRPlib", "P-n40-k5.txt.pkl")
    date = HCVRPDataset(filename, num_samples=256, offset=0)
    # date = date[53:54]
    costs = []
    times = []
    energy_costs = []
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    output_path = os.path.join("data_record", "20", "gurobi", f"{now}.csv")
    for instance in range(len(date)):
        EVRP = VehicleRouting(instance,date[instance], args.t_limit, args.Start_SOC, args.velocity, args.max_load, args.nodes, args.charging_num, args.plot_num)
        optim_cost, solution_time = EVRP.build_model()
        costs.append(optim_cost)
        times.append(solution_time)
    mean_cost = np.mean(costs)
    mean_time = np.mean(times)

    # 将测试数据写入文件
    if not args.CVRP_lib_test:
        with open(output_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(costs)):
                writer.writerow([costs[i], times[i]])
            writer.writerow("####### mean value ###########")
            writer.writerow([mean_cost, mean_time])


