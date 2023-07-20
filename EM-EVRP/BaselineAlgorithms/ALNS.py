# -*- coding: utf-8 -*-
# Author: tangmengcheng
# Email: 745274877@qq.com
# Date: 2023-07-20
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

        self.i = i
        self.static = instance["static"]
        self.dynamic = instance["dynamic"]
        self.distances = instance["distance"]
        self.slope = instance["slope"]

        self.demands = self.dynamic[1] * args.max_load
        num_dict = args.nodes
        self.num = self.static.shape[1]
        self.custom_num = num_dict[self.num]
        self.charge_num = args.charging_num
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
        self.custom_time = 0.33
        self.charging_time = 1


        self.epochs = args.epochs
        self.pu = args.pu
        self.rand_d_max = args.rand_d_max
        self.rand_d_min = args.rand_d_min
        self.worst_d_max = args.worst_d_max
        self.worst_d_min = args.worst_d_min
        self.regret_n = args.regret_n
        self.r1 = args.r1
        self.r2 = args.r2
        self.r3 = args.r3
        self.rho = args.rho
        self.phi = args.phi
        self.d_weight = np.ones(2) * 10
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

        power = (0.5 * self.Cd * self.A * self.Ad * (self.velocity / 3.6) ** 2 + (load * self.w  + self.mc) * self.g * (self.slope[i, j] + self.Cr))

        if power >= 0:
            return self.motor_d * self.battery_d * power * self.distances[i, j] / 3600
        else:
            return self.motor_r * self.battery_r * power * self.distances[i, j] / 3600


    def genInitialSol(self, node_seq):

        node_seq = copy.deepcopy(node_seq)
        random.seed(0)
        random.shuffle(node_seq)
        return node_seq

    def split_routes(self, node_seq):

        node_seq.insert(0, 0)

        num_vehicle = 0
        vehicle_routes = []
        vehicle_soc_list = []
        route = []
        load = self.max_load
        time = 0
        soc = self.Start_SOC
        time_list = []
        soc_list = []

        for i in range(1, len(node_seq)):

            check_load = bool(load >= self.demands[node_seq[i]])
            if check_load:

                check_nextnode_soc = (soc >= (self.Soc_Consume(node_seq[i - 1], node_seq[i], load) + self.Soc_Consume(node_seq[i], 0,load - self.demands[node_seq[i]])))
                if check_nextnode_soc:

                    check_nextnode_time = ((time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time + (self.distances[node_seq[i], 0] / self.velocity) ) <= self.t_limit)
                    if check_nextnode_time:

                        route.append(node_seq[i])

                        soc_list.append([self.Soc_Consume(node_seq[i-1], node_seq[i], load),0])
                        soc = soc - self.Soc_Consume(node_seq[i-1], node_seq[i], load)
                        load = load - self.demands[node_seq[i]]
                        time_list.append((self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time)
                        time = time + (self.distances[node_seq[i-1], node_seq[i]] / self.velocity) + self.custom_time
                    else:

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

                    SOC_station = [self.Soc_Consume(node_seq[i-1], j, load) for j in range(1, self.charge_num + 1)]
                    min_soc = min(SOC_station)
                    min_index = SOC_station.index(min_soc) + 1
                    if soc >= min_soc:
                        check_nextstation_time = ((time + (self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time +
                                                   (self.distances[min_index, 0] / self.velocity) ) <= self.t_limit)
                        if check_nextstation_time:

                            route.append(min_index)

                            soc_list.append([self.Soc_Consume(node_seq[i - 1], min_index, load), soc - min_soc - self.Start_SOC])
                            soc = self.Start_SOC
                            time = time + (self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time
                            time_list.append((self.distances[node_seq[i-1], min_index] / self.velocity) + self.charging_time)

                            check_nextnode_time = ((time + (self.distances[min_index, node_seq[i]] / self.velocity) + self.custom_time + (self.distances[node_seq[i], 0] / self.velocity) ) <= self.t_limit)
                            if check_nextnode_time:
                                route.append(node_seq[i])

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
                        else:
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
                    else:
                        for j in range(1,len(route)):

                            forward_soc = soc + np.array(soc_list[-j:]).sum()
                            forward_time = time - np.array(time_list[-j:]).sum()
                            forward_load = load + sum([self.demands[k] for k in route[-j:]])
                            SOC_station1 = [self.Soc_Consume(node_seq[i-1-j], n, forward_load) for n in range(1, self.charge_num + 1)]
                            min_soc1 = min(SOC_station1)
                            min_index1 = SOC_station1.index(min_soc1) + 1
                            if forward_soc >= min_soc1:

                                check_nextstation_time = ((time - (self.distances[node_seq[i-1-j], node_seq[i-1-j+1]] / self.velocity) + (self.distances[node_seq[i-1-j], min_index1]+
                                                        self.distances[min_index1, node_seq[i-1-j+1]])  / self.velocity + self.charging_time +
                                                           (self.distances[node_seq[i-1], 0] / self.velocity)) <= self.t_limit)
                                if check_nextstation_time:
                                    route.insert(-j, min_index1)

                                    soc_list.insert(-j, [self.Soc_Consume(node_seq[i-1-j], min_index1, forward_load), forward_soc - min_soc1 - self.Start_SOC])
                                    soc_list[-j] = [self.Soc_Consume(min_index1, node_seq[i-1-j+1], forward_load), 0]
                                    time_list.insert(-j, self.distances[node_seq[i-1-j], min_index1]/self.velocity + self.charging_time)
                                    time_list[-j] = self.distances[min_index1, node_seq[i-1-j+1]]/self.velocity + self.custom_time

                                    soc = self.Start_SOC - np.array(soc_list).sum()
                                    time = np.array(time_list).sum()

                                    check_nextnode_time = ((time + (self.distances[node_seq[i-1], node_seq[i] ] / self.velocity) + (self.distances[node_seq[i], 0] / self.velocity)+ self.custom_time ) <= self.t_limit)
                                    if check_nextnode_time:
                                        route.append(node_seq[i])

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
        vehicle_soc_list.extend(soc_list)
        vehicle_soc = np.array(vehicle_soc_list)[:,0].sum()
        return  vehicle_soc, vehicle_routes, vehicle_soc_list


    def createRandomDestory(self,):
        d = random.uniform(self.rand_d_min, self.rand_d_max)
        reomve_list = random.sample(range(self.charge_num + 1, self.custom_num + self.charge_num + 1), int(d * self.custom_num))
        return reomve_list


    def createWorseDestory(self, sol):
        deta_f = []
        for node_no in sol.node_seq:
            nodes_seq_ = copy.deepcopy(sol.node_seq)
            nodes_seq_.remove(node_no)
            nodes_seq_copy = copy.deepcopy(nodes_seq_)
            cost, routes, cost_list = self.split_routes(nodes_seq_copy)
            deta_f.append(sol.cost - cost)
        sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
        d = random.randint(int(self.worst_d_min * self.custom_num), int(self.worst_d_max * self.custom_num))
        remove_list = sorted_id[:d]
        return remove_list


    def createRandomRepair(self, remove_list, sol):
        unassigned_nodes_seq = []
        assigned_nodes_seq = []
        # remove node from current solution
        for i in range(self.custom_num):
            if sol.node_seq[i] in remove_list:
                unassigned_nodes_seq.append(sol.node_seq[i])
            else:
                assigned_nodes_seq.append(sol.node_seq[i])
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
            insert_node_no, insert_index = self.findGreedyInsert(unassigned_nodes_seq, assigned_nodes_seq)
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
        assigned_nodes_seq_obj, _, _ = self.split_routes(assigned_nodes_seq_copy)
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
        d_weight = self.d_weight
        d_cumsumprob = (d_weight / sum(d_weight)).cumsum()
        d_cumsumprob -= np.random.rand()
        destory_id = list(d_cumsumprob > 0).index(True)

        r_weight = self.r_weight
        r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
        r_cumsumprob -= np.random.rand()
        repair_id = list(r_cumsumprob > 0).index(True)
        return destory_id, repair_id


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
        self.d_select = np.zeros(2)
        self.d_score = np.zeros(2)
        self.r_select = np.zeros(3)
        self.r_score = np.zeros(3)


    def updateWeight(self, ):
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

        self.d_history_select = self.d_history_select + self.d_select
        self.d_history_score = self.d_history_score + self.d_score
        self.r_history_select = self.r_history_select + self.r_select
        self.r_history_score = self.r_history_score + self.r_score


    def plot_graph(self, solution):
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
            save_path = os.path.join("graph", f"{self.custom_num}", "ALNS")
        else:
            save_path = os.path.join("graph",  "CVRPlib")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        name = f'batch%d_%2.4f.png' % (self.i, solution.cost)
        save_path = os.path.join(save_path, name)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)


    def run(self):
        start = time.time()
        phi = self.phi
        sol = Sol()
        sol.node_seq = self.genInitialSol(list(range(self.charge_num + 1,self.custom_num + self.charge_num + 1)))
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
                self.d_select[destory_id] += 1
                self.r_select[repair_id] += 1
                reomve_list = self.doDestory(destory_id, sol)
                new_sol = self.doRepair(repair_id, reomve_list, sol)
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

        if self.i < self.plot_num:
            self.plot_graph(self.best_solution)

        soluiton_time = time.time() - start
        return self.best_solution.cost , soluiton_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ALNS solving electric vehicle routing problem')

    parser.add_argument('--nodes',  default=20, type=int)
    parser.add_argument('--Start_SOC', default=80, type=float, help='SOC, unit: kwh')
    parser.add_argument('--velocity', default=50, type=float, help='unit: km/h')
    parser.add_argument('--max_load', default=4, type=float, help='the max load of vehicle')
    parser.add_argument('--charging_num', default=5, type=int, help='number of charging_station')
    parser.add_argument('--t_limit', default=10, type=float, help='tour duration time limitation, 12 hours')
    parser.add_argument('--epochs', default=60, type=int, help='epoch of iteration')
    parser.add_argument('--pu', default=5, type=int, help='the frequency of weight adjustment')
    parser.add_argument('--plot_num', default=0,)

    parser.add_argument('--r1',  default=30, type=float,)
    parser.add_argument('--r2', default=20, type=float,)
    parser.add_argument('--r3', default=10, type=float,)
    parser.add_argument('--rho', default=0.3, type=int,)
    parser.add_argument('--rand_d_min', default=0.2, type=int,)
    parser.add_argument('--rand_d_max', default=0.5, type=int,)
    parser.add_argument('--worst_d_min', default=0.2, type=int,)
    parser.add_argument('--worst_d_max', default=0.5, type=int, )
    parser.add_argument('--regret_n', default=5, type=int, )
    parser.add_argument('--phi', default=0.8, type=int, )

    args = parser.parse_args()
    filename = os.path.join()
    date = HCVRPDataset(filename, num_samples=256, offset=0)

    costs = []
    times = []
    for i in range(len(date)):

        print(i)
        instance = EVRP(i, date[i], args)
        optim_cost, solution_time = instance.run()
        costs.append(optim_cost)
        times.append(solution_time)
    mean_cost = np.mean(costs)
    mean_time = np.mean(times)

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    output_path = os.path.join("data_record",f"{args.nodes}", "ALNS", f"{now}.csv")
    with open(output_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(costs)):
            writer.writerow([costs[i], times[i]])
        writer.writerow("####### mean value ###########")
        writer.writerow([mean_cost, mean_time])