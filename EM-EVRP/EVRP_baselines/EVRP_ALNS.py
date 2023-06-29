# 采用自适应大领域搜索算法求解作为baseline, 测试集统一采用存储在test_data中的数据
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


#  处理实例数据
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
    def __init__(self, i, instance, args):
        # 图的参数
        self.i = i  # 表示第几个实例
        self.static = instance["static"]  # 地理坐标，单位 km
        self.dynamic = instance["dynamic"]
        self.distances = instance["distance"]  # [num, num]
        self.slope = instance["slope"]

        self.demands = self.dynamic[1] * args.max_load  # 每个点的需求
        num_dict = {15:10, 25:20, 59:50, 109:50, 26:20} # 每一个问题规模对应的客户人数
        self.num = self.static.shape[1]
        self.custom_num = num_dict[self.num]
        self.charge_num = self.num - self.custom_num - 1
        self.plot_num = args.plot_num

        self.max_load = args.max_load
        self.Start_SOC = args.Start_SOC
        self.t_limit = args.t_limit
        self.velocity = args.velocity

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
        self.custom_time = 0.33  # 客户服务时间 20分钟
        self.charging_time = 1  # 充电站服务时间 1个小时


        # ALNS算法相关参数
        self.epochs = args.epochs
        self.pu = args.pu
        self.rand_d_max = args.rand_d_max  # 随机移除的最大比例
        self.rand_d_min = args.rand_d_min  # 随机移除的最小比例
        self.worst_d_max = args.worst_d_max
        self.worst_d_min = args.worst_d_min
        self.regret_n = args.regret_n
        self.r1 = args.r1
        self.r2 = args.r2
        self.r3 = args.r3
        self.rho = args.rho
        self.phi = args.phi
        self.d_weight = np.ones(2) * 10       # 两种破坏算子，3种修复算子
        self.d_select = np.zeros(2)
        self.d_score = np.zeros(2)
        self.d_history_select = np.zeros(2)
        self.d_history_score = np.zeros(2)
        self.r_weight = np.ones(3) * 10
        self.r_select = np.zeros(3)
        self.r_score = np.zeros(3)
        self.r_history_select = np.zeros(3)
        self.r_history_score = np.zeros(3)

        self.best_solution = None
        self.solution_list = []


    def Soc_Consume(self, i, j, load):
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


    def genInitialSol(self, node_seq):
        """
        得到初始解
        :param node_seq:刚开始为所有客户的排序
        :return:
        """
        node_seq = copy.deepcopy(node_seq)
        random.seed(0)
        random.shuffle(node_seq)
        return node_seq

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


    def createRandomDestory(self,):
        """
        按照一定比例随机移除一些点
        :return: 移除的点
        """
        d = random.uniform(self.rand_d_min, self.rand_d_max)   # 在最大移除比例和最小移除比例中均匀采用一个数
        reomve_list = random.sample(range(self.charge_num + 1, self.custom_num + self.charge_num + 1), int(d * self.custom_num))
        return reomve_list


    def createWorseDestory(self, sol):
        """
        将移除成本最（）的随机个点移除
        :param sol: 当前解决方案
        :return:
        """
        deta_f = []
        for node_no in sol.node_seq:
            nodes_seq_ = copy.deepcopy(sol.node_seq)
            nodes_seq_.remove(node_no)
            nodes_seq_copy = copy.deepcopy(nodes_seq_)
            cost, routes, cost_list = self.split_routes(nodes_seq_copy)
            deta_f.append(sol.cost - cost)
        sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
        d = random.randint(int(self.worst_d_min * self.custom_num), int(self.worst_d_max * self.custom_num))    # 删除的比例
        remove_list = sorted_id[:d]
        return remove_list


    def createRandomRepair(self, remove_list, sol):
        """
        随机修复
        :param remove_list:破坏算子移除的点
        :param sol: 当前解决方案
        :return: 新的解决方案
        """
        unassigned_nodes_seq = []
        assigned_nodes_seq = []
        # remove node from current solution
        for i in range(self.custom_num):
            if sol.node_seq[i] in remove_list:
                unassigned_nodes_seq.append(sol.node_seq[i])    # 移除点的索引
            else:
                assigned_nodes_seq.append(sol.node_seq[i])      # 没有移除的点
        # insert
        for node_no in unassigned_nodes_seq:
            index = random.randint(0, len(assigned_nodes_seq) - 1)
            assigned_nodes_seq.insert(index, node_no)
        new_sol = Sol()
        new_sol.node_seq = copy.deepcopy(assigned_nodes_seq)
        new_sol.cost, new_sol.routes, new_sol.cost_list = self.split_routes(assigned_nodes_seq)
        return new_sol


    def createGreedyRepair(self, remove_list, sol):
        unassigned_nodes_seq = []
        assigned_nodes_seq = []
        # remove node from current solution
        for i in range(self.custom_num):
            if sol.node_seq[i] in remove_list:
                unassigned_nodes_seq.append(sol.node_seq[i])
            else:
                assigned_nodes_seq.append(sol.node_seq[i])
        # insert
        while len(unassigned_nodes_seq) > 0:
            insert_node_no, insert_index = self.findGreedyInsert(unassigned_nodes_seq, assigned_nodes_seq)  # 返回插入成本最小的点以及其索引
            assigned_nodes_seq.insert(insert_index, insert_node_no)
            unassigned_nodes_seq.remove(insert_node_no)
        new_sol = Sol()
        new_sol.node_seq = copy.deepcopy(assigned_nodes_seq)
        new_sol.cost, new_sol.routes, new_sol.cost_list = self.split_routes(assigned_nodes_seq)
        return new_sol


    def findGreedyInsert(self, unassigned_nodes_seq, assigned_nodes_seq):
        best_insert_node_no = None
        best_insert_index = None
        best_insert_cost = float('inf')
        assigned_nodes_seq_copy = copy.deepcopy(assigned_nodes_seq)
        assigned_nodes_seq_obj, _, _ = self.split_routes(assigned_nodes_seq_copy)  # 这伦没插入点时的成本
        for node_no in unassigned_nodes_seq:
            for i in range(len(assigned_nodes_seq)):
                assigned_nodes_seq_ = copy.deepcopy(assigned_nodes_seq)
                assigned_nodes_seq_.insert(i, node_no)
                assigned_nodes_seq__copy = copy.deepcopy(assigned_nodes_seq_)
                obj_, _, _ = self.split_routes(assigned_nodes_seq__copy)
                deta_f = obj_ - assigned_nodes_seq_obj
                if deta_f < best_insert_cost:
                    best_insert_index = i
                    best_insert_node_no = node_no
                    best_insert_cost = deta_f
        return best_insert_node_no, best_insert_index


    def createRegretRepair(self, remove_list, sol):
        unassigned_nodes_seq = []
        assigned_nodes_seq = []
        # remove node from current solution
        for i in range(self.custom_num):
            if sol.node_seq[i] in remove_list:
                unassigned_nodes_seq.append(sol.node_seq[i])
            else:
                assigned_nodes_seq.append(sol.node_seq[i])
        # insert
        while len(unassigned_nodes_seq) > 0:
            insert_node_no, insert_index = self.findRegretInsert(unassigned_nodes_seq, assigned_nodes_seq)
            assigned_nodes_seq.insert(insert_index, insert_node_no)
            unassigned_nodes_seq.remove(insert_node_no)
        new_sol = Sol()
        new_sol.node_seq = copy.deepcopy(assigned_nodes_seq)
        new_sol.cost, new_sol.routes, new_sol.cost_list = self.split_routes(assigned_nodes_seq)
        return new_sol


    def findRegretInsert(self, unassigned_nodes_seq, assigned_nodes_seq):
        opt_insert_node_no = None
        opt_insert_index = None
        opt_insert_cost = -float('inf')
        for node_no in unassigned_nodes_seq:
            n_insert_cost = np.zeros((len(assigned_nodes_seq), 3))
            for i in range(len(assigned_nodes_seq)):
                assigned_nodes_seq_ = copy.deepcopy(assigned_nodes_seq)
                assigned_nodes_seq_.insert(i, node_no)
                assigned_nodes_seq_copy = copy.deepcopy(assigned_nodes_seq_)
                obj_, _, _ = self.split_routes(assigned_nodes_seq_copy)
                n_insert_cost[i, 0] = node_no
                n_insert_cost[i, 1] = i
                n_insert_cost[i, 2] = obj_
            n_insert_cost = n_insert_cost[n_insert_cost[:, 2].argsort()]
            deta_f = 0
            for j in range(1, self.regret_n):
                deta_f = deta_f + n_insert_cost[j, 2] - n_insert_cost[0, 2]
            if deta_f > opt_insert_cost:
                opt_insert_node_no = int(n_insert_cost[0, 0])
                opt_insert_index = int(n_insert_cost[0, 1])
                opt_insert_cost = deta_f
        return opt_insert_node_no, opt_insert_index


    def bestrepair(self):
        pass

    def selectDestoryRepair(self,):
        """
        选择算子，目前两种破坏算子三种修复算子
        :return:选择算子的索引
        """
        d_weight = self.d_weight
        d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
        d_cumsumprob -= np.random.rand()
        destory_id = list(d_cumsumprob > 0).index(True)

        r_weight = self.r_weight
        r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
        r_cumsumprob -= np.random.rand()
        repair_id = list(r_cumsumprob > 0).index(True)  # 这里的index为True的第一个，所以没有问题
        return destory_id, repair_id                    # 选出一个算子的编号


    def doDestory(self, destory_id, sol):
        if destory_id == 0:
            reomve_list = self.createRandomDestory()
        else:
            reomve_list = self.createWorseDestory(sol)
        return reomve_list


    def doRepair(self, repair_id, reomve_list, sol):
        if repair_id == 0:
            new_sol = self.createRandomRepair(reomve_list, sol)
        elif repair_id == 1:
            new_sol = self.createGreedyRepair(reomve_list, sol)
        else:
            new_sol = self.createRegretRepair(reomve_list, sol)
        return new_sol


    def resetScore(self,):
        """
        每个回合开始之前重置一下选择次数和分数
        :return:
        """
        self.d_select = np.zeros(2)  # 选择次数
        self.d_score = np.zeros(2)  # 分数
        self.r_select = np.zeros(3)  # 选择次数
        self.r_score = np.zeros(3)  # 分数


    def updateWeight(self, ):
        """
        更新每一个算子的权重
        :return:
        """
        for i in range(self.d_weight.shape[0]):
            if self.d_select[i] > 0:
                self.d_weight[i] = self.d_weight[i] * (1 - self.rho) + self.rho * self.d_score[i] / self.d_select[i]
            else:
                self.d_weight[i] = self.d_weight[i] * (1 - self.rho)

        for i in range(self.r_weight.shape[0]):
            if self.r_select[i] > 0:
                self.r_weight[i] = self.r_weight[i] * (1 - self.rho) + self.rho * self.r_score[i] / self.r_select[i]
            else:
                self.r_weight[i] = self.r_weight[i] * (1 - self.rho)

        # 记录历史分数以及选择次数
        self.d_history_select = self.d_history_select + self.d_select
        self.d_history_score = self.d_history_score + self.d_score
        self.r_history_select = self.r_history_select + self.r_select
        self.r_history_score = self.r_history_score + self.r_score


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
        plt.xlim(-5, 105)
        plt.ylim(-5, 105)
        save_path = os.path.join("graph", f"{self.custom_num}", "ACO")
        name = f'batch%d_%2.4f.png' % (self.i, solution.cost)
        save_path = os.path.join(save_path, name)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)


    def run(self):
        # 主程序
        start = time.time()
        phi = self.phi
        sol = Sol()
        sol.node_seq = self.genInitialSol(list(range(self.charge_num + 1,self.custom_num + self.charge_num + 1)))  # 这里注意客户的索引在4~23
        node_seqs = copy.deepcopy(sol.node_seq)
        sol.cost, sol.routes, sol.cost_list = self.split_routes(node_seqs)
        self.best_solution = copy.deepcopy(sol)
        history_best_obj = []
        history_best_obj.append(sol.cost)
        for epoch in range(self.epochs):
            T = sol.cost * 0.2
            self.resetScore()
            for k in range(self.pu):
                destory_id, repair_id = self.selectDestoryRepair()
                self.d_select[destory_id] += 1  # 算子使用数量加一
                self.r_select[repair_id] += 1
                reomve_list = self.doDestory(destory_id, sol)            # 破坏算子，得到移除的点
                new_sol = self.doRepair(repair_id, reomve_list, sol)     # 修复算子，得到新的点
                if new_sol.cost < sol.cost:
                    sol = copy.deepcopy(new_sol)
                    if new_sol.cost < self.best_solution.cost:
                        self.best_solution = copy.deepcopy(new_sol)
                        self.d_score[destory_id] += self.r1
                        self.r_score[repair_id] += self.r1
                    else:
                        self.d_score[destory_id] += self.r2
                        self.r_score[repair_id] += self.r2
                elif new_sol.cost - sol.cost < T:
                    sol = copy.deepcopy(new_sol)
                    self.d_score[destory_id] += self.r3
                    self.r_score[repair_id] += self.r3
                T = T * phi
                print("%s/%s:%s/%s， best obj: %s" % (epoch, self.epochs, k, self.pu, self.best_solution.cost))
                history_best_obj.append(self.best_solution.cost)
            self.updateWeight()
        # 如果画图的话
        if self.i < self.plot_num:
            self.plot_graph(self.best_solution)

        soluiton_time = time.time() - start
        return self.best_solution.cost , soluiton_time


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser(description='ALNS solving electric vehicle routing problem')
    # 图上参数
    parser.add_argument('--nodes',  default=20, type=int)
    parser.add_argument('--Start_SOC', default=80, type=float, help='SOC, unit: kwh')
    parser.add_argument('--velocity', default=50, type=float, help='unit: km/h')
    parser.add_argument('--max_load', default=4, type=float, help='the max load of vehicle')
    parser.add_argument('--charging_num', default=5, type=int, help='number of charging_station')
    parser.add_argument('--t_limit', default=10, type=float, help='tour duration time limitation, 12 hours')
    parser.add_argument('--epochs', default=60, type=int, help='epoch of iteration')
    parser.add_argument('--pu', default=5, type=int, help='the frequency of weight adjustment')
    parser.add_argument('--plot_num', default=0, help='画图的个数')
    # ALNS算法参数
    parser.add_argument('--r1',  default=30, type=float, help="大于")
    parser.add_argument('--r2', default=20, type=float, help='')
    parser.add_argument('--r3', default=10, type=float, help='')
    parser.add_argument('--rho', default=0.3, type=int, help='反馈因子')
    parser.add_argument('--rand_d_min', default=0.2, type=int, help='随机破坏最小比例')
    parser.add_argument('--rand_d_max', default=0.5, type=int, help='随机破坏最大比例')
    parser.add_argument('--worst_d_min', default=0.2, type=int, help='最差破坏最小个数')
    parser.add_argument('--worst_d_max', default=0.5, type=int, help='最差破坏最大个数')
    parser.add_argument('--regret_n', default=5, type=int, help='遗憾修复-n')
    parser.add_argument('--phi', default=0.8, type=int, help='')

    args = parser.parse_args()
    filename = os.path.join("..","test_data",f"{args.nodes}","256_seed12345.pkl")
    date = HCVRPDataset(filename, num_samples=256, offset=0)
    # date = date[27:28] #取数据在哪个区间
    costs = []
    times = []
    for i in range(len(date)):
        # 传入的参数名称分别为：
        print(i)
        instance = EVRP(i, date[i], args)
        optim_cost, solution_time = instance.run()
        costs.append(optim_cost)
        times.append(solution_time)
    mean_cost = np.mean(costs)
    mean_time = np.mean(times)

    # 将测试数据写入文件
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    output_path = os.path.join("data_record",f"{args.nodes}", "ALNS", f"{now}.csv")
    with open(output_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(costs)):
            writer.writerow([costs[i], times[i]])
        writer.writerow("####### mean value ###########")
        writer.writerow([mean_cost, mean_time])