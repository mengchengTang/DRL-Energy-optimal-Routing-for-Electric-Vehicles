# 采用蚁群算法求解作为baseline, 测试集统一采用存储在test_data中的数据
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import datetime
import csv
import time


# 处理实例数据
def make_instance(args):
    static, dynamic, distance, slope = args
    grid_size = 1
    return {
        'static': np.array(static,) / grid_size,
        'dynamic': np.array(dynamic,),  # scale demand
        'distance': np.array(distance,),
        'slope': np.array(slope, )
    }


def HCVRPDataset(filename=None,  num_samples=128, offset=0):
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


class Sol():
    def __init__(self):
        self.node_seq=None
        self.cost=None
        self.routes=None
        self.cost_list = None


class EVRP():
    def __init__(self, i, instance, t_limit, Start_SOC, velocity, max_load, alpha, beta, rho, epochs, ant_number, plot_num):
        # 图的参数
        self.i = i  # 表示第几个实例
        self.static = instance["static"]  # 地理坐标，单位 km
        self.dynamic = instance["dynamic"]
        self.distances = instance["distance"]  # [num, num]
        self.slope = instance["slope"]

        self.demands = self.dynamic[1] * max_load  # 每个点的需求
        num_dict = {15:10, 25:20, 59:50, 109:100, 26:20} # 每一个问题规模对应的客户人数
        self.num = self.static.shape[1]
        self.custom_num = num_dict[self.num]
        self.charge_num = self.num - self.custom_num - 1
        self.plot_num = plot_num

        self.max_load = max_load
        self.Start_SOC = Start_SOC
        self.t_limit = t_limit
        self.velocity = velocity

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
        # 蚁群算法相关参数
        # :param alpha:信息启发因子
        # :param beta:期望启发因子
        # :param rho:信息挥发参数
        # :param epochs:迭代轮次
        # :param ant_number:蚂蚁的数量
        self.custom_time = 0.33 # 客户服务时间 20分钟
        self.charging_time = 1 # 充电站服务时间 1个小时

        self.alpha = alpha
        self.beta = beta
        self.ant_number = ant_number
        self.rho = rho
        self.epochs = epochs
        self.Q = 5  # 总信息素浓度
        self.pheromone = np.ones(self.distances.shape) * 100   # 初始信息素浓度
        self.best_solution = None
        self.solution_list = []


    def Soc_Consume(self, i , j, load):
        """
        计算两点之间的能耗
        :param i: 起始点
        :param j: 结束点
        :param load: 载重量
        :return: 能耗
        """
        power = (0.5 * self.Cd * self.A * self.Ad * (self.velocity / 3.6) ** 2 + (load * self.w  + self.mc) * self.g * (self.slope[i, j] + self.Cr))

        if power >= 0:
            return self.motor_d * self.battery_d * power * self.distances[i, j] / 3600
        else:
            return self.motor_r * self.battery_r * power * self.distances[i, j] / 3600


    def getsolution(self,):
        # 选择下一访问的点
        solution_list = []
        local_best_sol = Sol()
        local_best_sol.cost = float('inf')  # 每一个轮回目前的最好解决方案
        for k in range(self.ant_number):
            # 随机蚂蚁的位置，去除三个充电站的位置
            node_seq = []
            open_node = np.ones(self.num)  # 目前可以访问的点
            open_node[0:self.charge_num + 1] = 0  # 将仓库到充电站屏蔽
            node_seq = [int(random.randint(self.charge_num + 1, self.num - 1))]
            now_node = node_seq[-1]
            open_node[now_node] = 0
            # 根据信息素确定下一需要访问的点
            while any(open_node):
                next_node = self.searchNextNode(now_node, open_node)
                node_seq.append(next_node)
                open_node[next_node] = 0
                now_node = next_node
            sol = Sol()
            sol.node_seq = node_seq
            sol.cost, sol.routes, sol.cost_list = self.split_routes(node_seq)
            solution_list.append(sol)  # 添加解决方案，这之中是包括路线和路线成本
            if sol.cost < local_best_sol.cost:
                local_best_sol = copy.deepcopy(sol)
        self.solution_list = copy.deepcopy(solution_list)
        if local_best_sol.cost < self.best_solution.cost:  # 这是每一次迭代伦次的最好方案
            self.best_solution = copy.deepcopy(local_best_sol)


    def searchNextNode(self, now_node, open_node):
        """
        根据目前的点选择下一需要访问的点
        :param now_node: 当前蚂蚁所在的点
        :param open_node: 目前可以访问的点
        :return: 下一访问的点
        """
        total_prob = 0.0
        next_node = None  # 定义好这个变量
        prob = np.zeros(len(open_node))
        for i in range(len(open_node)):
            if open_node[i]:
                # 这里考虑到用两种信息启发因子：1.两点间的节约的距离  2.两点之间能耗的倒数 3.两点之间节约的能耗
                eta = abs(self.distances[now_node, 0] + self.distances[0, i] - self.distances[now_node, i])
                # print(self.distances[now_node, 0])
                # print(self.distances[0,i])
                # print(self.distances[now_node, i])
                # print(eta)
                # eta1 = abs(1 / self.Soc_Consume(now_node, i, self.max_load))  # 当不同的两个点位置相同时，eta1可能为无穷
                # eta2 = self.Soc_Consume(0, now_node, self.max_load) + self.Soc_Consume(now_node, 0, self.max_load - self.demands[now_node]) \
                #        + self.Soc_Consume(0, i, self.max_load) + self.Soc_Consume(i, 0, self.max_load - self.demands[i]) \
                #         - self.Soc_Consume(0, now_node, self.max_load) - self.Soc_Consume(now_node, i, self.max_load - self.demands[now_node]) \
                #        - self.Soc_Consume(i, 0, self.max_load - self.demands[now_node] - self.demands[i])
                pheromone = self.pheromone[now_node, i]
                prob[i] = ((eta ** self.alpha) * (pheromone ** self.beta))  # 计算概率：与信息素浓度成正比，与距离成反比

                total_prob += prob[i]
        # 采用轮盘赌决定下一需要访问的节点
        # print(total_prob)
        if total_prob == 0:  # 因为有的实例在会在只有一个open_node的时候出现eta = 0 的情况
            for i in range(len(open_node)):
                if open_node[i] == 1:
                    next_node = i
        # print(open_node)
        # assert total_prob > 0.0,  " The total prob should >0 "
        else:
            temp_prob = random.uniform(0.0, total_prob)  # 产生一个随机概率,0.0-total_prob
            # print(temp_prob)
            for i in range(len(open_node)):
                if open_node[i]:
                    # 轮次相减
                    temp_prob -= prob[i]
                    if temp_prob < 0.0:
                        next_node = i
                        break

        return next_node


    def split_routes(self, node_seq):
        '''
        分离路线：首先根据车辆的容量进行分离，如何根据车辆的电量进行插入充电站
        :param nodes_seq: 选择的节点序列
        :param model: 模型
        :return: 使用车辆的数量及车辆的路由路线
        '''
        # print(node_seq)
        node_seq.insert(0, 0)  # 首先在列表的最前面添加0
        # node_seq.append(0)
        num_vehicle = 0
        vehicle_routes = []
        vehicle_soc_list = []
        route = []
        load = self.max_load
        time = 0
        soc = self.Start_SOC
        time_list = []
        soc_list = []  # 添加一个元组，列表第一个元素是能量消耗，列表第二个是充电电量
        # 首先通过车辆容量以及车辆的行驶时间进行分离路线
        for i in range(1, len(node_seq)):
            # 检查载重是否满足
            check_load = bool(load >= self.demands[node_seq[i]])
            if check_load:
                # 充电去往下一个点，在回到仓库
                check_nextnode_soc = (soc >= (self.Soc_Consume(node_seq[i - 1], node_seq[i], load) + self.Soc_Consume(node_seq[i], 0,load - self.demands[node_seq[i]])))
                if check_nextnode_soc:
                    # 满足能耗的时候，再确定时间约束能不能满足
                    check_nextnode_time = ((time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time + (self.distances[node_seq[i], 0] / self.velocity) ) <= self.t_limit)
                    if check_nextnode_time:
                        # 满足了货物量、时间、能耗这几个要素才能被添加
                        route.append(node_seq[i])
                        # 更新动态信息
                        soc_list.append([self.Soc_Consume(node_seq[i-1], node_seq[i], load),0])
                        soc = soc - self.Soc_Consume(node_seq[i-1], node_seq[i], load)
                        load = load - self.demands[node_seq[i]]
                        time_list.append((self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time)
                        time = time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time
                    else:
                        # 满足载重，满足能耗，不满足时间
                        vehicle_routes.append(route)
                        soc_list.append([self.Soc_Consume(node_seq[i-1], 0, load),0])
                        vehicle_soc_list.extend(soc_list)
                        route = [node_seq[i]]
                        num_vehicle = num_vehicle + 1
                        soc_list=[[self.Soc_Consume(0, node_seq[i], self.max_load),0]]
                        soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                        load = self.max_load - self.demands[node_seq[i]]
                        time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                        time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                else:
                    # 不满足能耗了，检查插入位置
                    SOC_station = [self.Soc_Consume(node_seq[i-1], j, load) for j in range(1, self.charge_num + 1)]
                    min_soc = min(SOC_station)
                    min_index = SOC_station.index(min_soc) + 1  # 取出去充电站能量消耗最小的点
                    if soc >= min_soc:  # 如果可以插入充电站 去完充电站当然可以回去
                        check_nextstation_time = ((time + (self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time +
                                                   (self.distances[min_index, 0] / self.velocity) ) <= self.t_limit)
                        if check_nextstation_time:
                            # 满足了货物量、时间、能耗这几个要素才能被添加
                            route.append(min_index)
                            # 去更新去往充电站的状态更新
                            soc_list.append([self.Soc_Consume(node_seq[i - 1], min_index, load), soc - min_soc - self.Start_SOC]) # 这个要看去了充电站实际到底是多电了还是少电了,添加充电量的负数
                            soc = self.Start_SOC
                            time = time + (self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time
                            time_list.append((self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time)
                            # 因为刚去了充电站，所以这里检查去客户点的时间
                            check_nextnode_time = ((time + (self.distances[min_index, node_seq[i]] / self.velocity) + self.custom_time + (self.distances[node_seq[i], 0] / self.velocity) ) <= self.t_limit)
                            if check_nextnode_time:
                                route.append(node_seq[i])
                                # 更新去点的动态信息
                                soc = soc - self.Soc_Consume(min_index, node_seq[i], load)
                                soc_list.append([self.Soc_Consume(min_index, node_seq[i], load), 0])
                                time = time + (self.distances[min_index, node_seq[i]] / self.velocity) + self.custom_time
                                time_list.append((self.distances[min_index, node_seq[i]] / self.velocity) + self.custom_time)
                                load = load - self.demands[node_seq[i]]
                            else:
                                vehicle_routes.append(route)
                                soc_list.append([self.Soc_Consume(min_index, 0, load),0])
                                vehicle_soc_list.extend(soc_list)
                                route = [node_seq[i]]
                                num_vehicle = num_vehicle + 1
                                soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                                soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load),0]]
                                time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                                time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                                load = self.max_load - self.demands[node_seq[i]]
                        else: # todo: 充电站因为时间的原因不能插入的话怎么办？直接返回！！
                            vehicle_routes.append(route)
                            soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                            vehicle_soc_list.extend(soc_list)
                            route = [node_seq[i]]
                            num_vehicle = num_vehicle + 1
                            soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                            soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load),0]]
                            time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                            time_list.append((self.distances[0, node_seq[i]] / self.velocity) + self.custom_time)
                            load = self.max_load - self.demands[node_seq[i]]
                    else: # todo:因为能耗原因插入不了充电站怎么办,往前检查，如果时间可以则插入充电站，时间不可以还是直接开一条新的。
                        for j in range(1,len(route)):
                            # 首先要计算倒退到前一点时的状态
                            forward_soc = soc + np.array(soc_list[-j:]).sum()
                            forward_time = time - np.array(time_list[-j:]).sum()
                            forward_load = load + sum([self.demands[k] for k in route[-j:]])
                            SOC_station1 = [self.Soc_Consume(node_seq[i-1-j], n, forward_load) for n in range(1, self.charge_num + 1)]
                            min_soc1 = min(SOC_station1)
                            min_index1 = SOC_station1.index(min_soc1) + 1  # 取出去充电站能量消耗最小的点
                            if forward_soc >= min_soc1:
                                # 当前时间减去i-1-j到i-1-j+1，然后在这个之中插入一个充电站
                                check_nextstation_time = ((time - (self.distances[node_seq[i-1-j], node_seq[i-1-j+1]] / self.velocity) + (self.distances[node_seq[i-1-j], min_index1]+
                                                        self.distances[min_index1, node_seq[i-1-j+1]])  / self.velocity + self.charging_time +
                                                           (self.distances[node_seq[i-1], 0] / self.velocity)) <= self.t_limit)
                                if check_nextstation_time: # todo:还是要用insert，直接索引的话会替换
                                    route.insert(-j, min_index1)
                                    # route[-j-1] = min_index1
                                    # 去更新去往充电站的状态更新
                                    soc_list.insert(-j, [self.Soc_Consume(node_seq[i-1-j], min_index1, forward_load), forward_soc - min_soc1 - self.Start_SOC])
                                    soc_list[-j] = [self.Soc_Consume(min_index1, node_seq[i-1-j+1], forward_load), 0]
                                    time_list.insert(-j, self.distances[node_seq[i-1-j], min_index1]/self.velocity + self.charging_time)
                                    time_list[-j] = self.distances[min_index1, node_seq[i-1-j+1]]/self.velocity + self.custom_time
                                    # 对当前node_seq[i-1]时刻的soc和时间进行更新
                                    soc = self.Start_SOC - np.array(soc_list).sum()
                                    time = np.array(time_list).sum()
                                    # 因为刚去了充电站，所以这里检查去客户点的时间
                                    check_nextnode_time = ((time + (self.distances[node_seq[i-1], node_seq[i] ] / self.velocity) + (self.distances[node_seq[i], 0] / self.velocity)+ self.custom_time ) <= self.t_limit)
                                    if check_nextnode_time:
                                        route.append(node_seq[i])
                                        # 更新去点的动态信息
                                        soc = soc - self.Soc_Consume(node_seq[i-1], node_seq[i], load)
                                        soc_list.append([self.Soc_Consume(node_seq[i-1], node_seq[i], load), 0])
                                        time = time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time
                                        time_list.append((self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time)
                                        load = load - self.demands[node_seq[i]]
                                        break
                                    else:
                                        vehicle_routes.append(route)
                                        soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                                        vehicle_soc_list.extend(soc_list)
                                        route = [node_seq[i]]
                                        num_vehicle = num_vehicle + 1
                                        soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                                        soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                                        time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                                        time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                                        load = self.max_load - self.demands[node_seq[i]]
                                        break
                                else:
                                    vehicle_routes.append(route)
                                    soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                                    vehicle_soc_list.extend(soc_list)
                                    route = [node_seq[i]]
                                    num_vehicle = num_vehicle + 1
                                    soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                                    soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                                    time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                                    time_list.append((self.distances[0, node_seq[i]] / self.velocity) + self.custom_time)
                                    load = self.max_load - self.demands[node_seq[i]]
                                    break
                            else:
                                continue
            else:
                vehicle_routes.append(route)
                soc_list.append([self.Soc_Consume(node_seq[i - 1], 0, load), 0])
                vehicle_soc_list.extend(soc_list)
                route = [node_seq[i]]
                num_vehicle = num_vehicle + 1
                soc = self.Start_SOC - self.Soc_Consume(0, node_seq[i], self.max_load)
                soc_list = [[self.Soc_Consume(0, node_seq[i], self.max_load), 0]]
                time = (self.distances[0, node_seq[i]] / self.velocity) + self.custom_time
                time_list = [(self.distances[0, node_seq[i]] / self.velocity) + self.custom_time]
                load = self.max_load - self.demands[node_seq[i]]
        vehicle_routes.append(route)
        soc_list.append([self.Soc_Consume(node_seq[i], 0, load), 0])
        vehicle_soc_list.extend(soc_list)    # todo:还要加上从仓库出发再从最后一个点到达仓库的
        vehicle_soc = np.array(vehicle_soc_list)[:,0].sum()
        return  vehicle_soc, vehicle_routes, vehicle_soc_list


    def upate_pheromone(self,):
        '''
        对模型中的信息素进行修改
        :param model: 模型
        '''
        rho = self.rho
        # print(self.pheromone)
        self.pheromone = (1 - rho) * self.pheromone
        # 更新信息素
        for sol in self.solution_list:
            routes=sol.routes
            for route in routes:
                for i in range(len(route)-1):
                    from_node_no=route[i]
                    to_node_no=route[i+1]
                    self.pheromone[from_node_no,to_node_no] +=  self.Q / sol.cost


    def plot_graph(self, solution):
        # 画图
        routes = solution.routes
        fig = plt.figure(figsize=(10, 10))
        xc = self.static[0, :]
        yc = self.static[1, :]
        n = self.custom_num
        f = self.charge_num
        plt.scatter(xc[f + 1:n + f + 1], yc[f + 1:n + f + 1], c='b')
        plt.scatter(xc[1:f + 1], yc[1:f + 1], c='g')
        plt.scatter(xc[0], yc[0], c='r')
        for i in range(self.charge_num + 1 ,self.custom_num + self.charge_num + 1):
            plt.text(xc[i], yc[i] + 3, "C" + format(i) + "-" + "D" + "%.2f" % self.demands[i])
        for i in range(1, self.charge_num + 1):
            plt.text(xc[i], yc[i] + 3, "F" + format(i))
        for k in range(0, len(routes)):
            routes[k].insert(0,0)
            routes[k].append(0)
            for i in range(1, len(routes[k])):
                plt.annotate(text="", xy=(xc[routes[k][i]], yc[routes[k][i]]),
                             xytext=(xc[routes[k][i - 1]], yc[routes[k][i - 1]]), arrowprops=dict(arrowstyle='->'))
        # plt.xlim(-5, 105)
        # plt.ylim(-5, 105)
        if not args.CVRP_lib_test:
            save_path = os.path.join("graph", f"{self.custom_num}", "ACO")
        else:
            save_path = os.path.join("graph",  "CVRPlib")
        name = f'batch%d_%2.4f.png' % (self.i, solution.cost)
        save_path = os.path.join(save_path, name)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)


    def run(self):
        # main
        start = time.time()
        sol = Sol()
        sol.cost= float('inf')
        self.best_solution = sol
        history_best_obj = []
        for ep in range(self.epochs):
            self.getsolution()
            self.upate_pheromone()
            history_best_obj.append(self.best_solution.cost)
            print("%s/%s， best obj: %s" % (ep, self.epochs, self.best_solution.cost))
        if self.i < self.plot_num:
            self.plot_graph(self.best_solution)
        soluiton_time = time.time() - start
        return self.best_solution.cost , soluiton_time


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='ACO solving electric vehicle routing problem')
    # grapg args
    parser.add_argument('--nodes',  default=20, type=int)
    parser.add_argument('--CVRP_lib_test', default=False)
    parser.add_argument('--Start_SOC', default=80, type=float, help='SOC, unit: kwh')
    parser.add_argument('--velocity', default=50, type=float, help='unit: km/h')
    parser.add_argument('--max_load', default=4, type=float, help='the max load of vehicle')
    parser.add_argument('--charging_num', default=5, type=int, help='number of charging_station')
    parser.add_argument('--t_limit', default=10, type=float, help='tour duration time limitation, 12 hours')
    parser.add_argument('--epoch', default=600, type=int, help='epoch of iteration 10:400 20:600 50:800 100:1000')
    parser.add_argument('--plot_num', default=0, help='画图的个数')
    # 蚁群算法参数
    parser.add_argument('--alpha',  default=3, type=float, help="启发信息权重")
    parser.add_argument('--beta', default=1, type=float, help='')
    parser.add_argument('--rho', default=0.1, type=float, help='信息素挥发因子')
    parser.add_argument('--ant_number', default=80, type=int, help='蚂蚁数量')

    args = parser.parse_args()
    # filename = os.path.join("..","test_data","CVRPlib","P-n101-k4.txt.pkl")
    filename = os.path.join("..", "test_data", "20", "256_seed12345.pkl")
    date = HCVRPDataset(filename, num_samples=256, offset=0)
    costs = []
    times = []
    for i in range(len(date)):
        print(i)
        instance = EVRP(i,date[i], args.t_limit, args.Start_SOC, args.velocity, args.max_load, alpha=args.alpha, beta=args.beta, rho=args.rho, epochs=args.epoch, ant_number = args.ant_number, plot_num = args.plot_num)
        optim_cost, solution_time = instance.run()
        costs.append(optim_cost)
        times.append(solution_time)
    mean_cost = np.mean(costs)
    mean_time = np.mean(times)

    # 将测试数据写入文件
    if not args.CVRP_lib_test:
        now = '%s' % datetime.datetime.now().time()
        now = now.replace(':', '_')
        output_path = os.path.join("data_record",f"{args.nodes}", "ACO", f"{now}.csv")
        with open(output_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(costs)):
                writer.writerow([costs[i], times[i]])
            writer.writerow("####### mean value ###########")
            writer.writerow([mean_cost, mean_time])