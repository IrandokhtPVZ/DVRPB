# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:38:08 2022

@author: Irandokht
"""

import numpy as np
np.random.seed(1000)
import pandas as pd
import random
random.seed(1000)
from ast import literal_eval
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
import copy
from openpyxl.workbook import Workbook
##############################################################################
''' initialization '''
print('points that you should discuss before implimentation:')
print("(1) working shift is set to be 480, to resolve the problem with routes that are finished so soon, we merged routes by solving a bin packging problem considering the working shift 480 min and 60 min for loading/unloding at the depot")
print('(2) we solved static VRPB based on distance, to convert it to time we define speed 25 mile per hours')
print('(3) insertion is based on the least detoure/deviation')
##############################################################################
#Path ='F:\\Research\\DVRPB-Fall 2020 & Spring & Summer & Fall 2021\\Computational Experiments\\Computational Experiment-DVRPB-third version\\Datasets\\'
Path ='C:\\Users\\parvizio\\Dropbox\\Computational Experiment-DVRPB-third version\\Datasets\\'
#Path = 'C:\\Users\\Irandokht\\Dropbox\\Computational Experiments-DVRPB-third version\\Datasets\\'
Instances = ['120_Q50', '800_Q50']#, '120_Q30', '800_Q30', '240_Q50', '1600_Q50', '360_Q50', '2400_Q50']
DoD = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # other sheets names in each instance: 'DistanceMatrix', 'ShortestPath'

ave_speed = 25
loading_time = 60
working_shift = 480
lenght_decision_epoch = 60 #1, 60, 240
arrival_time_dist = 'U(0,480)' #U(0,480), U(0,240)

##############################################################################
''' reading data '''
###############################################################################
# reading solutions obtained for the static VRPB
static_VRPB = pd.read_excel('StaticVRPB-V3.xlsx')

# reading randomly generated data for the time that online backhaul customers place their requests
global arrival_time
arrival_time_data = pd.read_excel(str(Path)+'arrival_time.xlsx')
arrival_time = {}
for row in range(arrival_time_data.shape[0]):
    node = arrival_time_data['node_id'].iloc[row]
    a_t = arrival_time_data[str(arrival_time_dist)].iloc[row]
    arrival_time[node] = a_t
    
# reading instance data
def Instance_Data(instance_name, sheet):
    
    global V, L, B, off_B, on_B
    global t
    global Q, K
    global p, status
    
    xls = pd.ExcelFile(str(Path)+str(instance_name)+'.xlsx')
    ins = pd.read_excel(xls, str(sheet))
    distance = pd.read_excel(xls, sheet_name='DistanceMatrix', index_col=0) 
    
    # 0 represents depot
    V = list(ins[ins['type']!=0]['node_id'])
    V_0 = [0] + V
    
    # linehaul customers ID
    L = list(ins[ins['type']==1]['node_id'])
    #L_0 = [0] + L
    
    B = [i for i in V if i not in L]
    # backhaul customers ID
    off_B = list(ins[ins['type']==2]['node_id'])
    #off_B_0 = off_B + [0]   
    # backhaul customers ID
    on_B = list(ins[ins['type']==3]['node_id'])
    #on_B_0 = on_B + [0]
       
    #A = [(i,j) for i in V_0 for j in V_0 if i!=j]
    
    c = {(i,j):round(distance[ins.loc[ins['node_id']==i, 'node_id_prev'].iloc[0]][ins.loc[ins['node_id']==j, 'node_id_prev'].iloc[0]],2) for i in V_0 for j in V_0}
    t = {(i,j):round((60*c[i,j])/ave_speed, 2) for i in V_0 for j in V_0}
    
    # truck capacity and No. of avaiable trucks
    Q = ins['Q'].iloc[0]    
    K = int(ins['k'].iloc[0])
    
    status = {i:0 for i in on_B}
    p = {i: ins[ins['node_id']==i]['pickup'].iloc[0] for i in V_0}
###############################################################################
''' DVRPB '''
###############################################################################
def Temporal_Itinerary(merged_route, spatial_itinerary, temporal_itinerary):            
    starting_time = 0
    for route in merged_route:
        #print(route)
        t_i = starting_time
        temporal_itinerary[route] = [t_i]
        #print(spatial_itinerary[route])
        for i in range (len(spatial_itinerary[route])-1):
            
            #print(i, spatial_itinerary[route][i], spatial_itinerary[route][i+1])
            t_i += t[spatial_itinerary[route][i], spatial_itinerary[route][i+1]]
            temporal_itinerary[route].append(round(t_i,2))
        starting_time = temporal_itinerary[route][-1] + loading_time
    return temporal_itinerary
###############################################################################
def DVRPB(d_t_s, s_i_k, t_i_k, f_c_k, L_tail_k, on):    
    delta = float('inf')   
    s_i_k_temp = copy.deepcopy(s_i_k)
       
    if f_c_k < p[on] or t_i_k[-2] < d_t_s:
        
        status_on = 2
        
    else:
        unvisited_customers = [s_i_k[i] for i in range(len(s_i_k))
                               if t_i_k[i] >= d_t_s and
                               (s_i_k[i] in B or s_i_k[i]==L_tail_k)]
        if len(unvisited_customers) != 0:
            unvisited_customers = unvisited_customers + [0]
            for i in range(len(unvisited_customers)-1):
                delta_temp = t[unvisited_customers[i], on] + t[on, unvisited_customers[i+1]]
                
                if delta_temp < delta:
                    delta = delta_temp
                    s_i_k = copy.deepcopy(s_i_k_temp)                                
                    s_i_k.insert(s_i_k.index(unvisited_customers[i])+ 1, on)
                           
            status_on = 0
            s_i_k_temp = copy.deepcopy(s_i_k)
            
        else:
            
            status_on = 2
              
    s_i_k = copy.deepcopy(s_i_k_temp)
                       
    return s_i_k, delta, status_on

###############################################################################
''' main algorithm '''
###############################################################################
headers = ['instance_name', 'sheet',
           'Graph', 'V','L','B', 'offline_B', 'online_B','DoD', 'Q', 'k',
           'lenght decistion epoch', 'arrival time online requests',
           'Static itinerary', 'Static itinerary cost', 'Static cost', 'Static time (sec)','static merged routes',
           'Dynamic itinerary', 'Dynamic itinerary cost', 'Dynamic cost',
           'Number of online requests after the last decision epoch', 'Number of accpeted online requests', 'node_id accepted online requests',
           'Dynamic time (sec)','date']

workbook_name = 'DynamicVRPB_'+str(datetime.now().strftime('%Y-%m-%d-%H-%M'))+'.xlsx'
wb = Workbook()
page = wb.active
page.title = 'DynamicVRPB'
page.append(headers)

for instance_name in Instances:
    writer = pd.ExcelWriter(str(instance_name)+'_'+str(arrival_time_dist)+'_'+str(lenght_decision_epoch)+'.xlsx')
    for dod in DoD:
        
        print('************\n', instance_name, dod, '\n************')
        
        solution = {}
        solution['instance_name'] = str(instance_name)
        solution['dod'] = dod
        
        Instance_Data(instance_name, dod)
        
        Static_VEPB_temp = static_VRPB[static_VRPB['instance_name']==instance_name]
        Static_VEPB_temp = Static_VEPB_temp[Static_VEPB_temp['sheet']==dod]
        
        spatial_itinerary = literal_eval(Static_VEPB_temp['Static itinerary'].iloc[0])
        merged_routes = literal_eval(Static_VEPB_temp['merged_routes'].iloc[0])
        
        temporal_itinerary = {}
        free_capacity = {}
        traveling_time = {}
        L_tail = {}
        for keys in merged_routes:
            merged_route = merged_routes[keys]            
            
            temporal_itinerary = Temporal_Itinerary(merged_route,
                                                    copy.deepcopy(spatial_itinerary),
                                                    copy.deepcopy(temporal_itinerary))
                
        for route in range(1, K+1):
            traveling_time[route] = temporal_itinerary[route][-1]
            B_temp = [i for i in spatial_itinerary[route] if i in B]
            free_capacity[route] = Q - len(B_temp)
            L_tail[route] = [spatial_itinerary[route][l] for l in range(len(spatial_itinerary[route])-1) if spatial_itinerary[route][l] in L and spatial_itinerary[route][l+1] not in L][0]
            
        
        time_stamp = lenght_decision_epoch
        decision_time_stamps = []
        while time_stamp < working_shift:
            decision_time_stamps.append(time_stamp)
            time_stamp += lenght_decision_epoch
                   
            
        solution['spatial_itinerary'] = [str(spatial_itinerary)]
        solution['temporal_itinerary'] = [str(temporal_itinerary)]
        solution['free_capacity'] = [str(free_capacity)]
        solution['traveling_time'] = [str(traveling_time)]
        solution['L_tail'] = str(L_tail)
        solution['working_shift'] = working_shift
        solution['lenght_decision_epoch'] = lenght_decision_epoch        
        solution['decision_time_stamp'] = [0]
        solution['on_B_l'] = ['none']
        solution['arrival_time'] = ['none']
        solution['status'] = ['none']
        solution['inserted_route'] = ['none']
        solution['detour_deviation'] = ['none']
        
        start = time.time()
        for decision_time_stamp in decision_time_stamps:
            print('decision_time_stamp: ', decision_time_stamp)
            on_B_l_temp = [on for on in on_B if arrival_time[on] >= decision_time_stamp-lenght_decision_epoch and arrival_time[on]<decision_time_stamp]
            arrival_time_temp = {i:arrival_time[i] for i in on_B_l_temp}
            sorted_arrival_time_temp = {keys:values for keys, values in sorted(arrival_time_temp.items(), key=lambda item:item[1])}
            on_B_l = list(sorted_arrival_time_temp.keys())
            
            if len(on_B_l) == 0:
                solution['spatial_itinerary'].append('none')
                solution['temporal_itinerary'].append('none')
                solution['free_capacity'].append('none')
                solution['traveling_time'].append('none')
                solution['decision_time_stamp'].append(decision_time_stamp)
                solution['on_B_l'].append('none')
                solution['arrival_time'].append('none')
                solution['status'].append('none')
                solution['inserted_route'].append('none')
                solution['detour_deviation'].append('none')
                continue
            else:
                for on in on_B_l:
                    k_temp = 0
                    delta_temp = float('inf')                    
                    spatial_itinerary_temp = copy.deepcopy(spatial_itinerary)
                    temporal_itinerary_temp = copy.deepcopy(temporal_itinerary)
                    free_capacity_temp = copy.deepcopy(free_capacity)
                    traveling_time_temp = copy.deepcopy(traveling_time)
                    
                    for keys in merged_routes:
                        merged_route = merged_routes[keys]                       
                        for route in merged_route:
                            spatial_itinerary[route], delta, status_on = DVRPB(decision_time_stamp,
                                                            copy.deepcopy(spatial_itinerary[route]),
                                                            copy.deepcopy(temporal_itinerary[route]),
                                                            free_capacity[route],
                                                            L_tail[route], on)
                            
                            if status_on != 2:
                               temporal_itinerary = Temporal_Itinerary(merged_route,
                                                                       copy.deepcopy(spatial_itinerary),
                                                                       copy.deepcopy(temporal_itinerary))
                            
                            else:
                                continue
                            
                            if temporal_itinerary[merged_route[-1]][-1] <= working_shift: 
                                status_on = 1
                            else:
                                spatial_itinerary[route] = copy.deepcopy(spatial_itinerary_temp[route])
                                status_on = 2
                          
                            
                            if status_on == 1 and delta < delta_temp:
                                k_temp = route
                                delta_temp = delta
                    
                    if k_temp != 0: 
                        spatial_itinerary_temp[k_temp] = copy.deepcopy(spatial_itinerary[k_temp])                        
                        status[on] = 1
                        for keys in merged_routes:
                            merged_route = merged_routes[keys]
                            temporal_itinerary_temp = Temporal_Itinerary(merged_route,
                                                                         copy.deepcopy(spatial_itinerary_temp),
                                                                         copy.deepcopy(temporal_itinerary_temp))
                        free_capacity_temp[k_temp] = free_capacity[k_temp] - p[on]
                        traveling_time_temp = {k:temporal_itinerary_temp[k][-1] for k in range(1, K+1)}
                    
                    else:
                        status[on] = 2
                    
                    spatial_itinerary = copy.deepcopy(spatial_itinerary_temp)
                    temporal_itinerary = copy.deepcopy(temporal_itinerary_temp)
                    free_capacity = copy.deepcopy(free_capacity_temp)
                    traveling_time = copy.deepcopy(traveling_time_temp)
                    
                    solution['spatial_itinerary'].append(str(spatial_itinerary))
                    solution['temporal_itinerary'].append(str(temporal_itinerary))
                    solution['free_capacity'].append(str(free_capacity))
                    solution['traveling_time'].append(str(traveling_time))
                    solution['decision_time_stamp'].append(decision_time_stamp)
                    solution['on_B_l'].append(on)
                    solution['arrival_time'].append(arrival_time[on])
                    solution['status'].append(status[on])
                    solution['inserted_route'].append(k_temp)
                    solution['detour_deviation'].append(delta_temp)
    
        # compute total traveling time    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        TTT = 0
        for keys in merged_routes:
            merged_route = merged_routes[keys]
            TTT += traveling_time[merged_route[-1]]
        # compute number of online consutmers whose requests have been accepted for the same day service
        n_D = 0
        node_id_accepted = []
        for on in on_B:
            if status[on] == 1:
                n_D += 1
                node_id_accepted.append(on)
        
        # compute CPU running time in second
        stop = time.time()
        cpu_time = round(stop - start, 2)
        
        a = [i for i in on_B if arrival_time[i]>=decision_time_stamp]
        info = [str(instance_name), dod,
                str(Static_VEPB_temp['Graph'].iloc[0]), len(V), len(L), len(B), len(off_B), len(on_B), dod, Q, K,
                lenght_decision_epoch, arrival_time_dist, 
                str(Static_VEPB_temp['Static itinerary'].iloc[0]), str(Static_VEPB_temp['Static itinerary cost'].iloc[0]),
                Static_VEPB_temp['Static cost'].iloc[0], Static_VEPB_temp['Static time (sec)'].iloc[0], Static_VEPB_temp['merged_routes'].iloc[0],
                str(spatial_itinerary), str(traveling_time), round(TTT, 2), len(a), n_D, str(node_id_accepted),
                cpu_time, str(datetime.now().strftime('%Y-%m-%d'))]
        
        page.append(info)
        wb.save(filename = workbook_name)
        
        Solution = pd.DataFrame(solution)
        Solution.to_excel(writer, sheet_name=str(dod), index = False)
    writer.save()
   



