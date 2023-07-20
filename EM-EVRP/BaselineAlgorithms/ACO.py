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
        'dynamic': np.array(dynamic,),
        'distance': np.array(distance,),
        'slope': np.array(slope, )
    }


def HCVRPDataset(filename=None,  num_samples=128, offset=0):

    assert os.path.splitext(filename)[1] == '.pkl'

    with open(filename, 'rb') as f:
        data = pickle.load(f)
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

        self.i = i
        self.static = instance["static"]
        self.dynamic = instance["dynamic"]
        self.distances = instance["distance"]
        self.slope = instance["slope"]

        self.demands = self.dynamic[1] * max_load
        num_dict = {15:10, 25:20, 59:50, 109:100, 26:20}
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

        self.custom_time = 0.33
        self.charging_time = 1

        self.alpha = alpha
        self.beta = beta
        self.ant_number = ant_number
        self.rho = rho
        self.epochs = epochs
        self.Q = 5
        self.pheromone = np.ones(self.distances.shape) * 100
        self.best_solution = None
        self.solution_list = []


    def Soc_Consume(self, i , j, load):
        power = (0.5 * self.Cd * self.A * self.Ad * (self.velocity / 3.6) ** 2 + (load * self.w  + self.mc) * self.g * (self.slope[i, j] + self.Cr))

        if power >= 0:
            return self.motor_d * self.battery_d * power * self.distances[i, j] / 3600
        else:
            return self.motor_r * self.battery_r * power * self.distances[i, j] / 3600


    def getsolution(self,):
        solution_list = []
        local_best_sol = Sol()
        local_best_sol.cost = float('inf')
        for k in range(self.ant_number):

            node_seq = []
            open_node = np.ones(self.num)
            open_node[0:self.charge_num + 1] = 0
            node_seq = [int(random.randint(self.charge_num + 1, self.num - 1))]
            now_node = node_seq[-1]
            open_node[now_node] = 0

            while any(open_node):
                next_node = self.searchNextNode(now_node, open_node)
                node_seq.append(next_node)
                open_node[next_node] = 0
                now_node = next_node
            sol = Sol()
            sol.node_seq = node_seq
            sol.cost, sol.routes, sol.cost_list = self.split_routes(node_seq)
            solution_list.append(sol)
            if sol.cost < local_best_sol.cost:
                local_best_sol = copy.deepcopy(sol)
        self.solution_list = copy.deepcopy(solution_list)
        if local_best_sol.cost < self.best_solution.cost:
            self.best_solution = copy.deepcopy(local_best_sol)


    def searchNextNode(self, now_node, open_node):

        total_prob = 0.0
        next_node = None
        prob = np.zeros(len(open_node))
        for i in range(len(open_node)):
            if open_node[i]:

                eta = abs(self.distances[now_node, 0] + self.distances[0, i] - self.distances[now_node, i])
                # print(self.distances[now_node, 0])
                # print(self.distances[0,i])
                # print(self.distances[now_node, i])
                # print(eta)
                # eta1 = abs(1 / self.Soc_Consume(now_node, i, self.max_load))
                # eta2 = self.Soc_Consume(0, now_node, self.max_load) + self.Soc_Consume(now_node, 0, self.max_load - self.demands[now_node]) \
                #        + self.Soc_Consume(0, i, self.max_load) + self.Soc_Consume(i, 0, self.max_load - self.demands[i]) \
                #         - self.Soc_Consume(0, now_node, self.max_load) - self.Soc_Consume(now_node, i, self.max_load - self.demands[now_node]) \
                #        - self.Soc_Consume(i, 0, self.max_load - self.demands[now_node] - self.demands[i])
                pheromone = self.pheromone[now_node, i]
                prob[i] = ((eta ** self.alpha) * (pheromone ** self.beta))

                total_prob += prob[i]

        if total_prob == 0:
            for i in range(len(open_node)):
                if open_node[i] == 1:
                    next_node = i

        # assert total_prob > 0.0,  " The total prob should >0 "
        else:
            temp_prob = random.uniform(0.0, total_prob)

            for i in range(len(open_node)):
                if open_node[i]:

                    temp_prob -= prob[i]
                    if temp_prob < 0.0:
                        next_node = i
                        break

        return next_node


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


    def upate_pheromone(self,):

        rho = self.rho

        self.pheromone = (1 - rho) * self.pheromone

        for sol in self.solution_list:
            routes=sol.routes
            for route in routes:
                for i in range(len(route)-1):
                    from_node_no=route[i]
                    to_node_no=route[i+1]
                    self.pheromone[from_node_no,to_node_no] +=  self.Q / sol.cost


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