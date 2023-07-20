# -*- coding: utf-8 -*-
# Author: tangmengcheng
# Email: 745274877@qq.com
# Date: 2023-07-20
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
        'dynamic': np.array(dynamic, ),
        'distance': np.array(distance, ),
        'slope': np.array(slope, )
    }

def HCVRPDataset(filename=None,  num_samples=256, offset=0):

    assert os.path.splitext(filename)[1] == '.pkl'

    with open(filename, 'rb') as f:
        data = pickle.load(f)  # (N, size, 2)
    data = [make_instance(args) for args in data[offset:offset + num_samples]]
    return data


class VehicleRouting():
    def __init__(self, i, instance, t_limit, Start_SOC, velocity, max_load, custom_num, charging_num, plot_num):

        self.static = instance["static"]
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
        self.plot_num = plot_num

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

        self.model = Model("EVRP")

        custom_num = self.custom_num
        charging_num = self.charging_num
        depot = [0]
        depotChargingStation = [0]
        nc = [i for i in range(charging_num + 1, custom_num +charging_num + 1 )]
        nf = [i for i in range(1, charging_num + 1)]
        v = depot + nf + nc
        chargingStationSet = depotChargingStation + nf


        A = [(i, j) for i in v for j in v]
        c = {(i, j): self.distances[i, j] for i in v for j in v}
        g = {(i, j): self.slope[i, j] for i in v for j in v}
        t = {(i, j): self.distances[i, j] / self.velocity for i in v for j in v}
        d = {i: self.demands[i] for i in v}
        vehicleSpecificConstant = 0.5 * self.Cd * self.A * self.Ad


        x = self.model.addVars(A, vtype=GRB.BINARY, name='x')
        w = self.model.addVars(A, vtype=GRB.CONTINUOUS, name='w')
        q = self.model.addVars(v, vtype=GRB.CONTINUOUS, name='q')
        y = self.model.addVars(v, vtype=GRB.CONTINUOUS, name='y')
        time_c = self.model.addVars(v, vtype=GRB.CONTINUOUS, name="t_c")

        self.model.setObjective(quicksum(c[i, j] * x[i, j] for i, j in A if i != j), GRB.MINIMIZE)

        # 1
        self.model.addConstrs(quicksum(x[i, j] for j in v if j != i) == 1 for i in nc)
        # 2
        self.model.addConstrs(quicksum(x[i, j] for j in v if j != i) <= 1 for i in nf)
        # 3
        self.model.addConstrs(
            quicksum(x[j, i] for i in v if i != j) - quicksum(x[i, j] for i in v if i != j) == 0 for j in v)
        # 4
        self.model.addConstrs(
            quicksum(w[j, i] for j in v if i != j) - quicksum(w[i, j] for j in v if i != j) == d[i] for i in nc + nf)
        # 5
        self.model.addConstrs(w[i, j] <= self.max_load * x[i, j] for i, j in A if i != j)
        self.model.addConstrs(w[i, j] >= 0 for i, j in A if i != j)
        # 6
        self.model.addConstrs(w[0, j] == self.max_load * x[0, j] for j in v)
        # 7
        self.model.addConstrs(
            self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
                                             - self.Start_SOC * (1 - x[i, j]) <= y[i] - y[j] for i in v for j in nc)
        self.model.addConstrs(
            y[i] - y[j] <= self.motor_d * self.battery_d * (
                        ((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                        + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[
                i, j] + self.Start_SOC * (1 - x[i, j]) for i in v for j in nc)

        # 8
        self.model.addConstrs(
            self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
                                             - self.Start_SOC * (1 - x[i, j]) <= y[i] - q[j] for i in v for j in nf)
        self.model.addConstrs(
            y[i] - q[j] <= self.motor_d * self.battery_d * (((self.g * g[i, j]) + (self.g * self.Cr)) * (self.mc + w[i, j] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, j] / 3600 * x[i, j]
            + self.Start_SOC * (1 - x[i, j])  for i in v for j in nf)

        self.model.addConstrs(y[i] == self.Start_SOC for i in chargingStationSet)
        # 9
        self.model.addConstrs(y[i] >= self.motor_d * self.battery_d * (((self.g * g[i, 0]) + (self.g * self.Cr)) * (self.mc + w[i, 0] * self.w)
                                             + (vehicleSpecificConstant * ((self.velocity / 3.6) ** 2))) * c[i, 0] / 3600 * x[i, 0]   for i in nc)

        # 10
        self.model.addConstrs((t[i,j] + 0.33) * x[i,j] - self.t_limit * (1-x[i,j]) <= time_c[j] - time_c[i] for i in v for j in nc)
        self.model.addConstrs(time_c[j] - time_c[i] <= (t[i, j] + 0.33) * x[i, j] + self.t_limit * (1 - x[i, j])  for i in v for j in nc)
        # 11
        self.model.addConstrs((t[i, j] + 1) * x[i, j] - self.t_limit * (1 - x[i, j]) <= time_c[j] - time_c[i] for i in v for j in nf)
        self.model.addConstrs(time_c[j] - time_c[i] <= (t[i, j] + 1) * x[i, j] + self.t_limit * (1 - x[i, j])  for i in v for j in nf)
        # 12
        self.model.addConstrs(time_c[i] == 0 for i in depot)
        self.model.addConstrs(time_c[i] <= self.t_limit - t[i,0] * x[i, 0] for i in nc+nf)


        self.model.Params.MIPGap = 0
        # self.model.Params.Threads = 1
        self.model.optimize()


        if self.i < self.plot_num:
            K = 0
            for i in v:

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
            power = []
            for i in v:
                for j in v:
                    if w[i,j].x > 0.2:
                        power.append(w[i,j].x)
                        power.append(i)
                        power.append(j)

            fig = plt.figure(figsize=(10,10))
            xc = self.static[0, :]
            yc = self.static[1, :]
            n = self.custom_num
            f = self.charging_num
            plt.scatter(xc[f+1:n+f+1], yc[f+1:n+f+1], c='b')
            plt.scatter(xc[1:f+1], yc[1:f+1], c='g')
            plt.scatter(xc[0], yc[0], c='r')
            for i in nc:
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
                save_path = os.path.join("graph", f"{self.custom_num}", "Gurobi")
            else:
                save_path = os.path.join("graph", "CVRPlib")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            name = f'batch%d_%2.4f.png' % (self.i, self.model.ObjVal)
            save_path = os.path.join(save_path, name)
            plt.savefig(save_path, bbox_inches='tight', dpi=100)

        optim_cost = self.model.ObjVal
        solution_time = self.model.Runtime

        if not args.CVRP_lib_test:
            out_path = os.path.join("data_record", f"{self.custom_num}", "Gurobi", f"online_C{self.custom_num}_{now}.csv")
            with open(out_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([optim_cost, solution_time])
        return optim_cost, solution_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ACO solving electric vehicle routing problem')

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

    filename = os.path.join()
    date = HCVRPDataset(filename, num_samples=256, offset=0)
    costs = []
    times = []
    energy_costs = []
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    for instance in range(len(date)):
        EVRP = VehicleRouting(instance,date[instance], args.t_limit, args.Start_SOC, args.velocity, args.max_load, args.nodes, args.charging_num, args.plot_num)
        optim_cost, solution_time = EVRP.build_model()
        costs.append(optim_cost)
        times.append(solution_time)
    mean_cost = np.mean(costs)
    mean_time = np.mean(times)

    output_path = os.path.join("data_record", f"{args.nodes}", "Gurobi", f"{now}.csv")
    if not args.CVRP_lib_test:
        with open(output_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(costs)):
                writer.writerow([costs[i], times[i]])
            writer.writerow("####### mean value ###########")
            writer.writerow([mean_cost, mean_time])


