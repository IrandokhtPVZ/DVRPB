# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 08:46:20 2022

@author: Irandokht
"""

import numpy as np
np.random.seed(1000)
import pandas as pd
from openpyxl.workbook import Workbook
import random
random.seed(1000)
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import copy
from docplex.mp.model import Model
###############################################################################
''' initialization'''
###############################################################################
method = 'CFRS in our second paper'   # CPLEX or CFRS
# for datasets

Path ='C:\\Users\\parvizio\\Dropbox\\Computational Experiments-DVRPB-second version\\Datasets\\'
#Path ='F:\\Research\\DVRPB-Fall 2020 & Spring & Summer & Fall 2021\\Computational Experiments\\Computational Experiments-DVRPB-second version\\Datasets\\'
#Charlotte_40_Q10
Instances = ['120_Q30', '120_Q50', '240_Q50', '360_Q50', '800_Q30', '800_Q50', '1600_Q50', '2400_Q50']
DoD = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # other sheets names in each instance: 'DistanceMatrix', 'ShortestPath'

# for clustering phase
global max_iter, thresholdn
max_iter = 100
threshold = 10

# for routing phase
global alpha
time_limit = 3600
alpha = 1 #alpha = 1 if the VRPB with backhaul solution

###############################################################################
''' reading data '''
###############################################################################
def Data(instance, sheet):
    global S, S_0, L, L_0, off_B, off_B_0
    #global A, A_L, A_B, A_C1, A_C2, A_C3
    global Q, k
    global d, p, c
    global ins, Graph
    
    
    xls = pd.ExcelFile(str(Path)+str(instance)+'.xlsx')
    ins = pd.read_excel(xls, str(sheet))
    distance = pd.read_excel(xls, sheet_name='DistanceMatrix', index_col=0) 
    
        
    Graph = ins['Graph'].iloc[0]
    
    # 0 represents depot
    #V = list(ins[ins['type']!=0]['node_id'])
    #V_0 = [0] + V
    
    # linehaul customers ID
    L = list(ins[ins['type']==1]['node_id'])
    L_0 = [0] + L
    
    # backhaul customers ID
    off_B = list(ins[ins['type']==2]['node_id'])
    off_B_0 = off_B + [0]
    
    # all offline customers
    S = L + off_B
    S_0 = [0] + S
    
    # truck capacity and No. of avaiable trucks
    Q = ins['Q'].iloc[0]    
    k = int(ins['k'].iloc[0])
    
    # all links, completed graph
    A_L = [(i,j) for i in L_0 for j in L if i!=j]
    A_B = [(i,j) for i in off_B for j in off_B_0 if i!=j]
    A_C1 = [(i,j) for i in L for j in off_B_0]
    A_C2 = [(i,j) for i in off_B for j in L]
    A_C3 = [(0,j) for j in off_B]
    
    A = A_L + A_B + A_C1 + A_C2 + A_C3
       
    # distance is computed as the shorthest path for every two nodes based on Haversine distance metric in miles and rounded to 2 decimals     
    c = {(i,j):round(distance[ins.loc[ins['node_id']==i, 'node_id_prev'].iloc[0]][ins.loc[ins['node_id']==j, 'node_id_prev'].iloc[0]],2) for i in S_0 for j in S_0}
        
    # delivery and pickup of all nodes, delivery/pickup is zero for the depot
    d = {i: ins[ins['node_id']==i]['delivery'].iloc[0] for i in S_0}
    p = {i: ins[ins['node_id']==i]['pickup'].iloc[0] for i in S_0}
###############################################################################
'''clustering linehaul and backhaul nodes using k-mean clustering algorithm '''
###############################################################################
def KMeans_Clustring():
    # //step 0: initialization
    ins['cluster'] = 0
    
    # clustering cost
    e_temp = float('inf') 
    # counter for the number of iterations without imporvement in solution
    thresh = 0 
       
    # initalize the center of clusters
    random_nodeid_idx = np.random.choice(S, k, replace = False)
    cluster_centers = {} # the set of cluster centers
    for i in range (k):
        cluster_centers[i+1] = copy.deepcopy(random_nodeid_idx[i])
  
    # //start optimization
    for iter in range (1, max_iter+1):
        #print('************ iteration ', iter)
        # //step1: assign linehaul and backhaul customers to their closest 
        # cluster center with enough delivery/pickup free capacity
        clusters = {}
        free_capacity_delivery = {}
        free_capacity_pickup = {}
        for i in range (k):
            clusters[i+1] = list()
            free_capacity_delivery[i+1] = Q
            free_capacity_pickup[i+1] = Q
            
        for node in S:
            E_temp = float('inf')
            k_temp = 1           
            for id in cluster_centers:
                 # check for the assignment criteria
                if free_capacity_delivery[id] >= d[node] and free_capacity_pickup[id] >= p[node] and c[node,cluster_centers[id]] < E_temp:
                    E_temp = c[node,cluster_centers[id]]
                    k_temp = id
            
            # append customer to its assigned cluster
            clusters[k_temp].append(node)
            # update vehicle capacity
            free_capacity_delivery[k_temp] = free_capacity_delivery[k_temp] - d[node]
            free_capacity_pickup[k_temp] = free_capacity_pickup[k_temp] - p[node]
        # compute the sum of traveling cost from each customer to his assigned cluster center
        e = 0
        for id in clusters:
            for node in clusters[id]:
                if node != cluster_centers[id]:
                    e += c[node, cluster_centers[id]]    
        e = round(e, 2)
        # //step 2: update the center of clusters        
        #cluster_centers_prev = copy.deepcopy(cluster_centers)
        for id in clusters:
            E_temp = float('inf')
            for node1 in clusters[id]:
                e_node1_k = 0
                for node2 in clusters[id]:
                    if node1 != node2:
                        e_node1_k += c[node1,node2]
                if e_node1_k <= E_temp:
                    E_temp = e_node1_k
                    # update the center of clusters
                    cluster_centers[id] = node1
            
        # //step 3: check the convergence of the algorithm
        if e < e_temp:
            #cluster_centers_keep = copy.deepcopy(cluster_centers_prev)
            clusters_keep = copy.deepcopy(clusters)
            e_temp = e
            thresh = 0
        elif e >= e_temp:# or list(cluster_centers.values()) == list(cluster_centers_keep.values()):
          thresh += 1
        
        if thresh >= threshold:
            #print('converged', iter)
            break
        else:
            iter += 1
       
    for id in clusters_keep:
        ins.loc[ins['node_id'].isin(clusters_keep[id]), 'cluster'] = id
###############################################################################
''' solve traveling salesman problem with backhauls for each cluster '''
###############################################################################
'''get the sequence of vertex/node ID for each optimized route strating and ending from depot'''
def findTuple(elem, arcs):
  for t in arcs:
    if t[0] == elem:
      return t
  return None
def node_seq(arcs, start_depot, end_depot):
    startRoutes = list(filter(lambda elem: elem[0]==start_depot, arcs))
    for i in range(len(startRoutes)):
      tempList = list()
      currentTuple = startRoutes[i]
      tempList.append(currentTuple[0])
      tempList.append(currentTuple[1])
      while True:
        if currentTuple[1] == end_depot:
          break
        else:
          nextTuple = findTuple(currentTuple[1], arcs)
          currentTuple = nextTuple
          tempList.append(currentTuple[1])        
    return  tempList
###############################################################################
def VRPB(SS):
    
    SS_0 = [0] + SS
    
    # linehaul customers ID
    LL = [i for i in SS if i in L]
    LL_0 = [0] + LL
    
    # backhaul customers ID
    off_BB = [i for i in SS if i in off_B]
    off_BB_0 = off_BB + [0]
    
    # all links, completed graph
    A_LL = [(i,j) for i in LL_0 for j in LL if i!=j]
    A_BB = [(i,j) for i in off_BB for j in off_BB_0 if i!=j]
    A_CC1 = [(i,j) for i in LL for j in off_BB_0]
    A_CC2 = [(i,j) for i in off_BB for j in LL]
    A_CC3 = [(0,j) for j in off_BB]
    
    AA = A_LL + A_BB + A_CC1 + A_CC2 + A_CC3
    
    mdl = Model('VRPB')
    
    # decision variables
    x = mdl.binary_var_dict(A_LL, name = 'x')
    y = mdl.binary_var_dict(A_BB, name = 'y')
    z1 = mdl.binary_var_dict(A_CC1, name = 'z1')
    z2 = mdl.binary_var_dict(A_CC2, name = 'z2')
    z3 = mdl.binary_var_dict(A_CC3, name = 'z3')
          
    u = mdl.continuous_var_dict(AA, ub = Q, name = 'u')
    w = mdl.continuous_var_dict(AA, ub = Q, name = 'w')
    
    # degree constraints
    mdl.add_constraint(mdl.sum(x[0,j]for j in LL) + (1-alpha)*mdl.sum(z3[0,j] for j in off_BB)==1)
    mdl.add_constraint(mdl.sum(z1[i,0]for i in LL) + mdl.sum(y[i,0]for i in off_BB)==1)
    
    mdl.add_constraints(mdl.sum(x[i,j]for i in LL_0 if i!=j) + (1-alpha)*mdl.sum(z2[i,j]for i in off_BB)==1 for j in LL)
    mdl.add_constraints(mdl.sum(x[i,j]for j in LL if i!=j) + mdl.sum(z1[i,j]for j in off_BB_0)==1 for i in LL)
    
    mdl.add_constraints(mdl.sum(z1[i,j]for i in LL) + mdl.sum(y[i,j]for i in off_BB if i!=j) + (1-alpha)*z3[0,j]==1 for j in off_BB)
    mdl.add_constraints((1-alpha)*mdl.sum(z2[i,j]for j in LL)+ mdl.sum(y[i,j]for j in off_BB_0 if i!=j)==1 for i in off_BB)
    
    # capacity and connectivity constraints
    mdl.add_constraints(mdl.sum(u[j,i] for j in SS_0 if i!=j) - mdl.sum(u[i,j] for j in SS_0 if i!=j) == d[i] for i in SS)
    mdl.add_constraints(mdl.sum(w[i,j] for j in SS_0 if i!=j) - mdl.sum(w[j,i] for j in SS_0 if i!=j) == p[i] for i in SS)
    
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*x[i,j] for i,j in A_LL)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*y[i,j] for i,j in A_BB)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*z1[i,j] for i,j in A_CC1)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*(1-alpha)*z2[i,j] for i,j in A_CC2)
    mdl.add_constraints(u[i,j]+w[i,j]<=Q*(1-alpha)*z3[i,j] for i,j in A_CC3)
    
    
    # objective function
    obj_1 = mdl.sum(c[i,j]*x[i,j]for i,j in A_LL)
    obj_2 = mdl.sum(c[i,j]*y[i,j]for i,j in A_BB)
    obj_3 = mdl.sum(c[i,j]*z1[i,j]for i,j in A_CC1)
    obj_4 = (1-alpha)*mdl.sum(c[i,j]*z2[i,j]for i,j in A_CC2)
    obj_5 = (1-alpha)*mdl.sum(c[i,j]*z3[i,j]for i,j in A_CC3)
    obj = obj_1 + obj_2 + obj_3 + obj_4 + obj_5
    mdl.add_kpi(obj, 'VRPB Cost')
    
    # solution
    mdl.minimize(obj)
    mdl.parameters.timelimit = time_limit 
    solution = mdl.solve(log_output = False) #true if you need to see the steps of slover
    #mdl.report_kpis()
    mdl.export_as_lp()
    #print(mdl.export_as_lp())
    #print(solution)
    if not solution:
        print('fail in solving VRPB, there is no feasible solution')
    x_opt = [a for a in A_LL if x[a].solution_value> 0.9]
    y_opt = [a for a in A_BB if y[a].solution_value> 0.9]
    z1_opt = [a for a in A_CC1 if z1[a].solution_value> 0.9]
    z2_opt = [a for a in A_CC2 if z2[a].solution_value> 0.9]
    z3_opt = [a for a in A_CC3 if z3[a].solution_value> 0.9]
    obj = round(solution.objective_value,2)
    
    arcs = x_opt + y_opt + z1_opt + z2_opt + z3_opt
    route = node_seq(arcs, 0, 0)
    #route.put(node_seq(arcs, 0, 0))
    #cost.put(obj)
    return obj, route
###############################################################################
''' main algorithm'''
###############################################################################
headers = ['instance_name', 'sheet','solution method',
           'Graph', 'V','L','B', 'offline_B', 'online_B','DoD', 'Q', 'k',
           'Static itinerary', 'Static itinerary cost', 'Static cost', 'Static time (sec)','date']

workbook_name = 'StaticVRPB_'+str(datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.xlsx'
wb = Workbook()
page = wb.active
page.title = 'StaticVRPB'
page.append(headers)

for instance_name in Instances:
    for dod in DoD:
        
        Data(str(instance_name), str(dod))
        print('************', instance_name, dod )
        
        start = time.time()
        if k > 1:
            KMeans_Clustring()
        else:
            ins['cluster'] = 0
            ins.loc[ins['type']!=0, 'cluster'] = 1
            
        
        # solving the static VRPB for each vehicle one by one
        
        itinerary = {}
        for i in range (k):
            print('route {}'.format(i+1))
            SS = [nodes for nodes in S if nodes in list(ins.loc[ins['cluster']==i+1] ['node_id'])] 
            cost, route = VRPB(SS)            
            itinerary[i+1] = route
            
        # compute transportation cost
        total_cost = 0
        itinerary_cost = {}
        for keys in itinerary:
            route_k = itinerary[keys]
            cost_k = 0
            for i in range(len(route_k)-1):
                total_cost += c[route_k[i], route_k[i+1]]
                cost_k += c[route_k[i], route_k[i+1]]
            itinerary_cost[keys] = round(cost_k, 2)    
        # compute CPU running time in second
        stop = time.time()
        cpu_time = round(stop - start, 2)
        
        
        V = list(ins[ins['type']!=0]['node_id'])
        B = [i for i in V if i not in L]
        info = [str(instance_name), dod, str(method),
                str(Graph), len(V), len(L), len(B), len(off_B), len(B)-len(off_B), dod, Q, k,
                str(itinerary), str(itinerary_cost), round(total_cost,2), cpu_time, 
                str(datetime.now().strftime('%Y-%m-%d'))]
        
        page.append(info)
        wb.save(filename = workbook_name)










