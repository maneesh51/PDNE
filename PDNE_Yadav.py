# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:20:56 2023

@author: Dr. Manish Yadav

Cyber-Physical Systems in Mechanical Engineering
Technische Universit ̈at Berlin
Straße des 17. Juni 135
10623 Berlin, Germany
Email: manish.yadav@tu-berlin.de

This code simulates the results for the manuscript "Evolution Beats Random
Chance: Performance-dependent Network Evolution for Enhanced Computational Capacity"
Authors: Manish Yadav, Sudeshna Sinha and Merten Stender

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
import copy
import networkx as nx
import time
from timeit import default_timer as timer
import os
import pickle
from scipy.integrate import solve_ivp


### Reservoir functions#########################################################################################

def Reservoir(GNet, Init, Inps, Winps, N_I, N, alpha):
    Nodes_res = GNet.shape[0]; 
    Npts_U = Inps.shape[1]

    R = np.zeros([N, Npts_U])
    R[:,0] = Init
    
    #time loop
    for t in range(0, Npts_U-1):
        Inp_term=0
        for i in range(N_I):
            Inp_term +=  Winps[i]*Inps[i,t]
        R[:,t+1] = (1 - alpha)*np.asarray(R[:,t]) + alpha*np.tanh(np.dot(GNet, R[:,t].T) + Inp_term)    
    return R

def GNet_SpectralRadius(Gn, Spectral_radius):
    ### Rescaling to a desired spectral radius 
    Spectral_radius_Gn = max(abs(np.linalg.eigvals(Gn)))
    ResMat = Gn*Spectral_radius/Spectral_radius_Gn
    return ResMat

def RC(G, Spectral_radius, alpha, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Inps_test, Outs_test, Trans, RC_reps, Return=0):
    Outs_predict=np.zeros((RC_reps, N_O, Outs_test[0,Trans:].shape[0]))
    Outs_test_pred=np.zeros((RC_reps, N_O, Outs_test[0,Trans:].shape[0]))
    MSEs_train=np.zeros((RC_reps, N_O)); MSEs_test_pred=np.zeros((RC_reps, N_O))
    for r in range(RC_reps):
        ### G to Matrix
        GNet = nx.to_numpy_matrix(G)
        N = G.number_of_nodes()

        GNet = GNet_SpectralRadius(GNet, Spectral_radius)

        #### Input Nodes
        Winps = np.zeros((N_I, N))
        for i in range(N_I):
            Winps[i, InpsNodes[i]] = 0.005 ####0.01 
        # print('A2. In RC, Inp nodes:', InpsNodes, 'Winps:', Winps, 'N:', N)
        ### Reservoir run
        Init = np.random.random(N)*0.1 #0.25
        
        Res = Reservoir(GNet, Init, Inps, Winps, N_I, N, alpha)

        ### Training
        beta = 5e-10 #5e-810
        W_outs = Ridge_Regression(Res[:,Trans:], beta, Outs[:,Trans:], N_O, OutsNodes)
        # print('W_outs:', np.asarray(W_outs).shape, 'OutsNodes:',OutsNodes)
        ### Testing
        Outs_predict[r], MSEs_train[r] = Test_or_Predict(GNet, Init, Inps, Winps, N, N_I, N_O, OutsNodes, alpha, W_outs, Outs, Trans)
        # Plot_t(Out1_no_transients, Out1_predict, MSE_train)

        ### Prediction
        Outs_test_pred[r], MSEs_test_pred[r] = Test_or_Predict(GNet, Init, Inps_test, Winps, N, N_I, N_O, OutsNodes, alpha, W_outs, Outs_test, Trans)
        # Plot_t(Out1_test_no_transients, Out1_test_pred, MSE_test_pred)
    
    ##### Mean over RC reps
    MSEs_MnSD_train = np.array([np.mean(MSEs_train, axis=0), np.std(MSEs_train, axis=0)])
    MSEs_MnSD_test_pred = np.array([np.mean(MSEs_test_pred, axis=0), np.std(MSEs_test_pred, axis=0)])
    
    if Return==0:
        return MSEs_MnSD_train, MSEs_MnSD_test_pred
    
    if Return==1:
        return Outs_predict, Outs_test_pred, MSEs_MnSD_train, MSEs_MnSD_test_pred

#####AutoRC#########################################################################################
def Auto_Reservoir(GNet, Init, Inps, Winps, Wouts, OutsNodes, N_I, N, alpha, TaskType):
    Nodes_res = GNet.shape[0]; 
    Npts_U = Inps.shape[1]

    R = np.zeros([N, Npts_U])
    R[:,0] = Init
    
    Inps_pred = np.zeros([N_I, Npts_U])
        
    #time loop
    for t in range(0, Npts_U-1):
        Inp_term=0
        
        # Inp_term = Winps[0]*Inps[0,t]
        
        #### Warmup phase###############################################
        if t<800:  ##1800:
            for i in range(0,N_I):
                Inp_term +=  Winps[i]*Inps[i,t]
            
        
        if t>=800:  ##1800:
            if TaskType=='Chaos':
                for i in range(0,N_I):
                    Inp_term +=  Winps[i]*Inps_pred[i,t]
            if TaskType=='VDP':
                Inp_term =  Winps[0]*Inps_pred[0,t]+(Winps[1]*Inps[1,t])
            
        
        #############################################################   
            
        R[:,t+1] = (1 - alpha)*np.asarray(R[:,t]) + alpha*np.tanh(np.dot(GNet, R[:,t].T)+ Inp_term )    
        
        ### Next step auto prediction
        for j in range(N_I):
            Inps_pred[j, t+1] = np.dot(Wouts[j], R[OutsNodes[j],t+1])
                
            # Outs_pred[i] = np.dot(W_outs[i], Res_t[OutsNodes[i],Trans:])
                
    return R, Inps_pred


def Auto_RC(G, Spectral_radius, alpha, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Trans, RC_reps, TaskType):
    for r in range(RC_reps):
        ### G to Matrix
        GNet = nx.to_numpy_matrix(G)
        N = G.number_of_nodes()

        GNet = GNet_SpectralRadius(GNet, Spectral_radius)

        #### Input Nodes
        Winps = np.zeros((N_I, N))
        for i in range(N_I):
            Winps[i, InpsNodes[i]] = 0.0075 ###0.01 
        # print('A2. In RC, Inp nodes:', InpsNodes, 'Winps:', Winps, 'N:', N)
        ### Reservoir run
        Init = np.random.random(N)*0.1#0.25
        
        Res = Reservoir(GNet, Init, Inps, Winps, N_I, N, alpha)

        ### Training
        beta = 5e-10 #5e-10 
        W_outs = Ridge_Regression(Res[:,Trans:], beta, Outs[:,Trans:], N_O, OutsNodes)
             
        Res, Inps_pred = Auto_Reservoir(GNet, Init, Inps, Winps, W_outs, OutsNodes, N_I, N, alpha, TaskType)
        
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 4; fig_size[1] = 4
        plt.rcParams["figure.figsize"] = fig_size
        plt.plot(Inps[0, 50:], Inps[1, 50:], lw=0.5)
        plt.plot(Inps_pred[0, 100:], Inps_pred[1, 100:], lw=0.5, c='r')
        plt.show()
        
        for i in range(N_I):
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 8; fig_size[1] = 1.5
            plt.rcParams["figure.figsize"] = fig_size
            plt.plot(Inps[i], lw=0.5)
            plt.plot(Inps_pred[i], lw=0.5, c='r')
            plt.ylim(-5,5)
            plt.show()



###Training#########################################################################################   
def Ridge_Regression(R, beta, V_train, N_O, OutsNodes):
    W_outs = [[]]*N_O #np.zeros((N_O, R.shape[0]))
    for i in range(N_O):
        R_out = R[OutsNodes[i],:]
        W_outs[i] = np.dot(np.dot(V_train[i], R_out.T), np.linalg.inv((np.dot(R_out, R_out.T) + beta*np.identity(R_out.shape[0]))))
    #     print('W_outs[i]:', np.asarray(W_outs[i]).shape,'V_train:',V_train.shape,'R_out:',R_out.shape)
    # print('W_outs:', np.asarray(W_outs).shape)
    return W_outs

# @jit(target_backend='cuda')
def MSE(A, B):
    return np.mean(((A - B)**2))

# @jit(target_backend='cuda')
def Errors(y_predicted, y_actual):
    MSE = np.mean(np.square(np.subtract(y_predicted,y_actual)))
    Variance = (np.mean(np.square(np.subtract(y_actual, np.mean(y_actual) ) ) ) )
    NMSE = MSE/Variance
    NRMSE = np.sqrt(NMSE)
    return NMSE, NRMSE 

### Testing#########################################################################################
# @jit(target_backend='cuda')   
def Test_or_Predict(GNet, Init, Inps_t, Winps, N, N_I, N_O, OutsNodes, alpha, W_outs, Outs, Trans):
    Res_t = Reservoir(GNet, Init, Inps_t, Winps, N_I, N, alpha)
    
    Outs_pred=[[]]*N_O
    NMSEs_t=np.zeros(N_O)
    for i in range(N_O):
        # print('Test or Pred fun. N_O:', N_O, 'i:',i, len(W_outs), Res_t.shape, 'W_outs[i]', W_outs[i].shape, 'OutsNodes:',OutsNodes)
        Outs_pred[i] = np.dot(W_outs[i], Res_t[OutsNodes[i],Trans:])
        NMSEs_t[i] = Errors(Outs_pred[i], Outs[i,Trans:])[0] #MSE(Out1_t_no_transients, Out1_t)
        # print('Res_t[OutsNodes[i],Trans:]:',Res_t[OutsNodes[i],Trans:].shape,'W_outs[i]:',W_outs[i].shape,'Outs_pred[i]:',np.asarray(Outs_pred[i]).shape)
    return Outs_pred, NMSEs_t

def Plot_t(N_O, Outs, Outs_t, Trans, Tillt, MSE_t):
    for i in range(N_O):
        fig_size = plt.rcParams["figure.figsize"]  
        fig_size[0] = 8; fig_size[1] = 2.5
        plt.rcParams["figure.figsize"] = fig_size  

        plt.plot(Outs[i,Trans:Tillt+Trans], lw=0.75)
        plt.title('Error={:.10f}'.format(MSE_t[i]))
        plt.plot(Outs_t[i,:Tillt], lw=0.75)
        plt.show()
        

###### Inputs#####################################################################################################
def NARMA_MemoryX(inp, Memory):
    y = np.zeros(inp.shape[0])
    y[0] = inp[0]
    for i in range(inp.shape[0]-1):
        sumy = 0
        for j in range(1, Memory+1):
            if i>j: 
                sumy += y[i+1-j]
        
        if Memory>10:
            y[i+1] = np.tanh(0.3*y[i] + 0.05*y[i]*sumy + 1.5*inp[i]*inp[i+1-Memory] + 0.1)
        else:
            y[i+1] = 0.3*y[i] + 0.05*y[i]*sumy + 1.5*inp[i]*inp[i+1-Memory] + 0.1
    return y

def NARMA_Call(N_I, Npts_U, NARMA_Props):
    InpDistrib = NARMA_Props[0]; MemoryX = NARMA_Props[1]
    
    Inps=np.zeros((N_I, Npts_U)); Outs=np.zeros((N_I, Npts_U))
    Inps_test=np.zeros((N_I, Npts_U)); Outs_test=np.zeros((N_I, Npts_U))
    
    for i in range(N_I):  
        Inps[i] = np.random.uniform(InpDistrib[i,0], InpDistrib[i,1], Npts_U) 
        Outs[i] = NARMA_MemoryX(Inps[i], MemoryX[i])

        Inps_test[i] = np.random.uniform(InpDistrib[i,0], InpDistrib[i,1], Npts_U) 
        Outs_test[i] = NARMA_MemoryX(Inps_test[i], MemoryX[i])
    return Inps, Outs, Inps_test, Outs_test


#########Tasks#########################################################################################
def SinCos_Inp(Npts_U, h, omega, phi, a_sc, b_sc, P_sc):
    t = np.arange(0, int(Npts_U*h), h)
    #####Input##################
    U = np.sin(omega*t + phi)
    #####Output##################
    Y = a_sc*np.sin(omega*t + phi)**P_sc + b_sc*np.cos(omega*t + phi)**P_sc
    return U, Y
    
def SinCos_Call(N_I, Npts_U, SinCos_Props):
    h=SinCos_Props[0]; Omegas=SinCos_Props[1]; Phis=SinCos_Props[2]
    A_scs=SinCos_Props[3]; B_scs=SinCos_Props[4]; P_scs=SinCos_Props[5] 

    Inps=np.zeros((N_I, Npts_U)); Outs=np.zeros((N_I, Npts_U))
    Inps_test=np.zeros((N_I, Npts_U)); Outs_test=np.zeros((N_I, Npts_U))
    
    for i in range(N_I):  
        Inps[i], Outs[i] = SinCos_Inp(Npts_U, h, Omegas[i], Phis[i], A_scs[i], B_scs[i], P_scs[i])
        Inps_test[i], Outs_test[i] = SinCos_Inp(Npts_U, h, Omegas[i], 2, A_scs[i], B_scs[i], P_scs[i])
    return Inps, Outs, Inps_test, Outs_test

def Lorenz(t, X, s=10, r=28, b=3.0):     
    x, y, z = X
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def SolveLorenz(Npts_U):
    ### numerical time integration
    h=0.02
    T = int(Npts_U*h)
    t_eval = np.arange(start=0, stop=T, step=h)
    
    X0_a=[-10,0,0.5]
    Sol_a = solve_ivp(Lorenz, t_span=[t_eval[0], t_eval[-1]], y0=X0_a, t_eval=t_eval, args=())
    Inps_a = np.array([Sol_a.y[0,:], Sol_a.y[1,:], Sol_a.y[2,:]])
    
    X0_b=[-5,1,-0.5]
    Sol_b = solve_ivp(Lorenz, t_span=[t_eval[0], t_eval[-1]], y0=X0_b, t_eval=t_eval, args=())
    Inps_b = np.array([Sol_b.y[0,:], Sol_b.y[1,:], Sol_b.y[2,:]])
        
    return Inps_a, Inps_a, Inps_b, Inps_b

def VDP(t, X, mu=1):
    x, y = X
    return [y, mu*(1 - x**2)*y - x]

def SolveVDP(Npts_U):
    ### numerical time integration
    h=0.1
    T = int(Npts_U*h)
    t_eval = np.arange(start=0, stop=T, step=h)
    
    X0_a=[1,-1]
    Sol_a = solve_ivp(VDP, t_span=[t_eval[0], t_eval[-1]], y0=X0_a, t_eval=t_eval, args=())
    Inps_a = np.array([Sol_a.y[0,:], Sol_a.y[1,:]])
    
    X0_b=[1,-1]
    Sol_b = solve_ivp(VDP, t_span=[t_eval[0], t_eval[-1]], y0=X0_b, t_eval=t_eval, args=())
    Inps_b = np.array([Sol_b.y[0,:], Sol_b.y[1,:]])
        
    return Inps_a, Inps_a, Inps_b, Inps_b

#########################################################################################

def InpGenerate(TaskType, N_I, Npts_U, InpProps):    
    if TaskType=="NARMA":
        Inps, Outs, Inps_test, Outs_test = NARMA_Call(N_I, Npts_U, InpProps)
    if TaskType=="SinCos":
        Inps, Outs, Inps_test, Outs_test = SinCos_Call(N_I, Npts_U, InpProps)
    if TaskType=="Chaos":
        Inps, Outs, Inps_test, Outs_test = SolveLorenz(Npts_U)
    if TaskType=="VDP":
        Inps, Outs, Inps_test, Outs_test = SolveVDP(Npts_U)
       
    return Inps, Outs, Inps_test, Outs_test

def InpPlot(Inps, Outs, N_I):
    for i in range(N_I):  
        fig_size = plt.rcParams["figure.figsize"]  
        fig_size[0] = 8; fig_size[1] = 1
        plt.rcParams["figure.figsize"] = fig_size 
        plt.plot(Inps[i, :400], c='C0')
        plt.show()
        plt.plot(Outs[i, :400], c='C1')
        plt.show()


def Net_Plot(G,gs):    
    pos=nx.kamada_kawai_layout(G)
    labels = {}
    for idx, node in enumerate(G.nodes()):
        labels[node] = idx

    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = gs; fig_size[1] = gs
    plt.rcParams["figure.figsize"] = fig_size  
    plt.title('')

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='red', alpha=0.5)
    nx.draw_networkx_edges(G, pos, width=0.25)
    nx.draw_networkx_labels(G, pos, labels, font_size=7)
    plt.box(False)
    plt.show()

    
##### Network Measures and performances  #########################################################################################
def Network_Measures(G):
    Msr_names = ['Nodes', 'Avg. CC', 'Avg. In_Deg', 'Avg. Out_Deg', 'Communities', 'Density']
        
    Nodes = G.number_of_nodes()
    CC = nx.average_clustering(G)
    DegIn = np.mean(np.array(list(dict(G.in_degree()).values())))
    DegOut = np.mean(np.array(list(dict(G.out_degree()).values())))
    Communities = len(nx.community.greedy_modularity_communities(G))
    Density = nx.density(G)
    return np.array([Nodes, CC, DegIn, DegOut, Communities, Density]), Msr_names
    
def Plot_NetMsrs(NetMsrs, NetMsrs_Names):
    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 10; fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size  

    fig, ax = plt.subplots(2, 3)
    k=0
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(NetMsrs[:,k], lw=1, label=NetMsrs_Names[k])
            ax[i, j].legend(loc='upper right',fontsize=8)
            k+=1
    plt.show()
    

def Plot_Performance(Scores, Scores_Names, N_O):
    ### Scores.shape[0]-> 2(train, pred), shape[1]->time, shape[2]->2(mean and std), shape[3]->N_O
    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 5; fig_size[1] = 2
    plt.rcParams["figure.figsize"] = fig_size  

    fig, ax = plt.subplots(1, 2)
    k=0
    #### loop for train, test
    for i in range(2):
        #### loop for Num of Outputs performances
        for j in range(N_O):
            ### meanplots
            ax[i].plot(Scores[i][:,0,j], lw=1, label='{:}Out{:d}'.format(Scores_Names[i],j+1))
            ax[i].legend(loc='upper right',fontsize=8)
    ax[0].set_ylim(0.01, 1)
    ax[1].set_ylim(-0.01, )
    ax[0].set_xlabel('time', fontsize=15)
    plt.xlabel('time', fontsize=15)
    plt.show()
    
    
    
#########################################################################################
#########################################################################################
   
def DeleteNode(G_old, alpha, N_I, N_O, InpsNodes_old, OutsNodes_old, Spectral_radius, Inps, Outs, Inps_test, Outs_test, \
               Trans, RC_Reps, NetMsr_init, MSEs_Mn_pred_old, Err_precision, NodesDel_Percent):
    
    Max_DeleteSteps = round((NodesDel_Percent*G_old.number_of_nodes())/100)
    
    NetMeasrs=np.array([]); Scores_train=[]; Scores_pred=[]
    DeleteStep=0; Flag=0; Nodes_Deleted=0
    while(Flag==0):
        
        ### Make copies of G_old, InpsNodes_old, N_I and N_O ##################
        G_temp = G_old.copy()
        InpsNodes_temp = copy.deepcopy(InpsNodes_old)
        OutsNodes_temp = copy.deepcopy(OutsNodes_old)
        
        ### Delete random node#######################################
        G_Nodes = np.array(G_temp.nodes)
        Del_node = np.random.choice(G_Nodes)
        G_temp.remove_node(Del_node)
        
        #### Rename/shift the indexing of nodes and edges###########
        mapping = dict(zip(G_temp, range(0, G_temp.number_of_nodes())))
        G_temp = nx.relabel_nodes(G_temp, mapping)
        
        #### If deleted node was also an input node#################
        for ni in range(N_I):
            # print('Before shifting:', InpsNodes_temp, 'ni:', ni, InpsNodes_temp[ni])
            if Del_node in InpsNodes_temp[ni]:
                InpsNodes_temp[ni].remove(Del_node)
                # print('Deleted node:', Del_node, 'was also inp node. Inp nodes:', InpsNodes_temp)           
            
            ##### shifting inp nodes index number########################
            for i in range(len(InpsNodes_temp[ni])):
                if(InpsNodes_temp[ni][i]>=Del_node):
                    InpsNodes_temp[ni][i] = InpsNodes_temp[ni][i]-1 
                        
        # print('After shifting:', InpsNodes_temp)
        ################################################################
        
        #### If deleted node was also an Output node#################
        for no in range(N_O):
            # print('O-Before shifting:', OutsNodes_temp, 'no:', no, OutsNodes_temp[no])
            if Del_node in OutsNodes_temp[no]:
                OutsNodes_temp[no].remove(Del_node)
                # print('O-Deleted node:', Del_node, 'was also out node. Out nodes:', OutsNodes_temp)           
            
            ##### shifting inp nodes index number########################
            for i in range(len(OutsNodes_temp[no])):
                if(OutsNodes_temp[no][i]>=Del_node):
                    OutsNodes_temp[no][i] = OutsNodes_temp[no][i]-1 
        ################################################################
        
        ### RC run####################################################################
        MSEs_MnSD_train_temp, MSEs_MnSD_pred_temp = RC(G_temp, Spectral_radius, alpha, N_I, N_O, InpsNodes_temp, OutsNodes_temp, Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps)
        
        ######## Rounding the errors for comparing till given nth digit
        ######## If improvement in error in any of the output then update the Network, Inpnodelist and errors 
        ######## but other errors should not deplete
        Temp_Err_Precise=np.round(MSEs_MnSD_pred_temp[0], Err_precision)
        Old_Err_Precise=np.round(MSEs_Mn_pred_old, Err_precision)
        # print('Del fun condn:')
        # print('MSEs_MnSD_pred_temp[0]:',MSEs_MnSD_pred_temp[0],Temp_Err_Precise,'MSEs_Mn_pred_old:',MSEs_Mn_pred_old,Old_Err_Precise)
        if (Temp_Err_Precise <= Old_Err_Precise).all(): 
            G_old = G_temp.copy()    
            Nodes_Deleted = Nodes_Deleted+1
            InpsNodes_old = InpsNodes_temp
            OutsNodes_old = OutsNodes_temp
            MSEs_Mn_pred_old = MSEs_MnSD_pred_temp[0]
            
            ######## Net. properties and scores########
            NetMeasrs, NetMsrs_Names = Network_Measures(G_old)
            Scores_train=MSEs_MnSD_train_temp
            Scores_pred=MSEs_MnSD_pred_temp
            # print('Deletion Accepted.', 'Nodes:', G_old.number_of_nodes(), 'Links:', G_old.number_of_edges(), 'Inp Nodes:', InpsNodes_old\
            #       ,'Out Nodes:', OutsNodes_old, 'OldErr:', MSEs_Mn_pred_old,'TempErr:', MSEs_MnSD_pred_temp[0])
                        
        DeleteStep=DeleteStep+1
        if DeleteStep>=Max_DeleteSteps:
            Flag=1
                    
    return G_old, InpsNodes_old, OutsNodes_old, NetMeasrs, Scores_train, Scores_pred, Nodes_Deleted



# @jit(target_backend='cuda')   
def AddNewNode(t, G_old, alpha, N_I, N_O, MaxNewLinks, Psi, P_inp, P_out, InpsNodes_old, OutsNodes_old, InpNodeType, OutNodeType, Spectral_radius,\
               Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps, MSEs_Mn_pred_old, Err_precision, Max_AddSteps):
    MaxNewLinks_arr = np.arange(1, MaxNewLinks+1)
    Flag=0; AddStep=0
    while(Flag==0):
        
        ##### Making copies of G_old Input nodes list
        G_temp = G_old.copy()
        InpsNodes_temp = copy.deepcopy(InpsNodes_old)
        OutsNodes_temp = copy.deepcopy(OutsNodes_old)
        
        ##### Add new node
        NewNode = G_temp.number_of_nodes()
        G_temp.add_node(NewNode)
        G_Nodes = np.array(G_temp.nodes())
        #### New node can be input node for each input with Prob P_inp 'separately' #################
        N = G_temp.number_of_nodes()
        
        if (InpNodeType==0):
            ##### New node can be input node for other inputs as well (in multiple tasks) with Prob P_inp
            for i in range(N_I):
                if np.random.random()<=P_inp:
                    InpsNodes_temp[i].append(N-1)
        
        if (InpNodeType==1):
            ##### New node can be input node for any 'one' input (in multiple tasks) with Prob P_inp, Strict Exclusiveness
            InpNums=np.arange(N_I)
            Inp_rand=np.random.random()
            if Inp_rand<=P_inp:
                Which_InpNum_Gets_InpNode=np.random.choice(InpNums)
                InpsNodes_temp[Which_InpNum_Gets_InpNode].append(N-1)
            ######################################################################
        
        if (OutNodeType==0):
            #### New node can be output node for each output with Prob P_out 'separately' (in multiple tasks)#################
            for i in range(N_O):
                if np.random.random()<=P_out:
                    OutsNodes_temp[i].append(N-1)
       
        if (OutNodeType==1):
            ##### New node can be output node for any 'one' output (in multiple tasks) with Prob P_out, Strict Exclusiveness
            OutNums=np.arange(N_O)
            Out_rand=np.random.random()
            if Out_rand<=P_out:
                Which_OutNum_Gets_OutNode=np.random.choice(OutNums)
                OutsNodes_temp[Which_OutNum_Gets_OutNode].append(N-1)
        ######################################################################
        
        ###Add new edges   
        NewConnsNum = np.random.choice(MaxNewLinks_arr)
        for n in range(NewConnsNum):        
            if(np.random.rand()<=Psi):
                G_temp.add_edge(NewNode, np.random.choice(G_Nodes))
            else:
                G_temp.add_edge(np.random.choice(G_Nodes), NewNode)

        ### RC run
        MSEs_MnSD_train_temp, MSEs_MnSD_pred_temp = RC(G_temp, Spectral_radius, alpha, N_I, N_O, InpsNodes_temp, OutsNodes_temp, Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps)
        
        ######## Rounding the errors for comparing till given nth digit
        ######## If improvement in error in any of the output then update the Network, Inpnodelist and errors 
        ######## but other errors should not deplete
        Temp_Err_Precise=np.round(MSEs_MnSD_pred_temp[0], Err_precision)
        Old_Err_Precise=np.round(MSEs_Mn_pred_old, Err_precision)
        
        ##### IMPORTANT: '<=' condition will be required for multiple tasks, as all should not improve at the same time. 
        ##### Some improve, some can stay exactly same but should not degrade 
        if (Temp_Err_Precise < Old_Err_Precise).all():  
            t=t+1
            G_old = G_temp.copy()
            InpsNodes_old = InpsNodes_temp
            OutsNodes_old = OutsNodes_temp
            # print(t, 'Nodes Added.', 'Nodes:', G_temp.number_of_nodes(), 'Links:', G_temp.number_of_edges(), 'Inp Nodes:', InpsNodes_old,\
            #       'Out nodes:',OutsNodes_old, 'Old Err:', Old_Err_Precise, 'Temp Err:', Temp_Err_Precise)
            Flag=1
        else:
            if AddStep>=Max_AddSteps:
                t=t+1
                Flag=1
            AddStep=AddStep+1
            ############################################################
            # print(t, 'Addition Rejected.', 'Nodes:', G_old.number_of_nodes(), 'Links:', G_old.number_of_edges(), 'Inp Nodes:', InpsNodes_old,\
            #       'Out nodes:',OutsNodes_old, 'Old Err:',Old_Err_Precise, 'Temp Err:', Temp_Err_Precise)
    
    return G_old, InpsNodes_old, OutsNodes_old, t


# @jit(target_backend='cuda')   
def AddNewNode_Uninformed(t, G_old, N_I, N_O, MaxNewLinks, Psi, P_inp, P_out, InpsNodes_old, OutsNodes_old, InpNodeType, OutNodeType):
    MaxNewLinks_arr = np.arange(1, MaxNewLinks+1)
        
    ##### Making copies of G_old Input nodes list
    G_temp = G_old.copy()
    InpsNodes_temp = copy.deepcopy(InpsNodes_old)
    OutsNodes_temp = copy.deepcopy(OutsNodes_old)

    ##### Add new node
    NewNode = G_temp.number_of_nodes()
    G_temp.add_node(NewNode)
    G_Nodes = np.array(G_temp.nodes())
    #### New node can be input node for each input with Prob P_inp 'separately' #################
    N = G_temp.number_of_nodes()

    if (InpNodeType==0):
        ##### New node can be input node for other inputs as well (in multiple tasks) with Prob P_inp
        for i in range(N_I):
            if np.random.random()<=P_inp:
                InpsNodes_temp[i].append(N-1)

    if (InpNodeType==1):
        ##### New node can be input node for any 'one' input (in multiple tasks) with Prob P_inp, Strict Exclusiveness
        InpNums=np.arange(N_I)
        Inp_rand=np.random.random()
        if Inp_rand<=P_inp:
            Which_InpNum_Gets_InpNode=np.random.choice(InpNums)
            InpsNodes_temp[Which_InpNum_Gets_InpNode].append(N-1)
        ######################################################################

    if (OutNodeType==0):
        #### New node can be output node for each output with Prob P_out 'separately' (in multiple tasks)#################
        for i in range(N_O):
            if np.random.random()<=P_out:
                OutsNodes_temp[i].append(N-1)

    if (OutNodeType==1):
        ##### New node can be output node for any 'one' output (in multiple tasks) with Prob P_out, Strict Exclusiveness
        OutNums=np.arange(N_O)
        Out_rand=np.random.random()
        if Out_rand<=P_out:
            Which_OutNum_Gets_OutNode=np.random.choice(OutNums)
            OutsNodes_temp[Which_OutNum_Gets_OutNode].append(N-1)
    ######################################################################

    ###Add new edges   
    NewConnsNum = np.random.choice(MaxNewLinks_arr)
    for n in range(NewConnsNum):        
        if(np.random.rand()<=Psi):
            G_temp.add_edge(NewNode, np.random.choice(G_Nodes))
        else:
            G_temp.add_edge(np.random.choice(G_Nodes), NewNode)

    t=t+1
    G_old = G_temp.copy()
    InpsNodes_old = InpsNodes_temp
    OutsNodes_old = OutsNodes_temp
    
    return G_old, InpsNodes_old, OutsNodes_old, t



# @jit(target_backend='cuda')   
def Checkpoint_V3(Net_Init, alpha, MaxNewLinks, Psi, P_inp, P_out, N_I, N_O, InpsNodes, OutsNodes, Spectral_radius, T, Delta_Err, PlotEvery, Inps, \
                  Outs, Inps_test, Outs_test, Trans, RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth='Yes',\
                  Delete_Nodes='Yes', InpNodeType=0, OutNodeType=0):
    
    G = nx.DiGraph(Net_Init)
    # Net_Plot(G,3)
        
    ### Init RC run
    MSEs_MnSD_train, MSEs_MnSD_pred = RC(G, Spectral_radius, alpha, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Inps_test, \
                                         Outs_test, Trans, RC_Reps)
    
    ################ Initial Net., properties and performance################################################
    AllGraphs=[]
    AllGraphs.append(G)
    AllInpsNodes=[]; AllOutsNodes=[]
    NetMsr_init, NetMsrs_Names = Network_Measures(G)
    NetMsrs = np.zeros((T+10, NetMsr_init.shape[0]))
    NetMsrs[0] = NetMsr_init
    
    ##### MSEs_MnSD_train.Shape[0]->2(mean and std), MSEs_MnSD_train.Shape[1]->N_O
    ##### Scores_train.shape[0]->time, shape[1]->2(mean and std), shape[2]->N_O
    Scores_train = np.zeros((T+10, MSEs_MnSD_train.shape[0], MSEs_MnSD_train.shape[1]))+np.nan
    Scores_pred = np.zeros((T+10, MSEs_MnSD_pred.shape[0], MSEs_MnSD_pred.shape[1]))+np.nan
    Scores_train[0] = MSEs_MnSD_train
    Scores_pred[0] = MSEs_MnSD_pred
    
    # print(0, 'Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(), ' Err:', NetMsrs[0, 8])
    #########################################################################################################
    ######While loop until desired error is reached or taking too long Time
    t=0
    while( (Scores_pred[t,0,:] > Delta_Err).any() ):
        
        ###### Delete Node#####################################################################################################
        if (Delete_Nodes=='Yes'):
        
            G, InpsNodes, OutsNodes, NetMsrs_AfterDelFun, Scores_train_AfterDelFun, Scores_pred_AfterDelFun, Nodes_deleted =\
            DeleteNode(G, alpha, N_I, N_O, InpsNodes, OutsNodes, Spectral_radius, Inps, Outs, Inps_test, Outs_test,\
            Trans, RC_Reps, NetMsr_init, Scores_pred[t,0], Err_precision, NodesDel_Percent)
            
            if Nodes_deleted>=1:
                t=t+1
                AllGraphs.append(G)
                AllInpsNodes.append(InpsNodes); AllOutsNodes.append(OutsNodes)
                NetMsrs[t] = NetMsrs_AfterDelFun.T
                Scores_train[t] = Scores_train_AfterDelFun
                Scores_pred[t] = Scores_pred_AfterDelFun

            print(t, 'After Delete 1 Fun : Updated Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(),\
                  'Inp Nodes:',len(InpsNodes[0]),'Out nodes:',len(OutsNodes[0]),'Deleted Nodes:',Nodes_deleted,'Err:',Scores_pred[t,0])
        
            ##### Skip if t already reach ==T #######################
            # if (Scores_pred[t,0,:] > Delta_Err).all(): break
            if t>=T:  break
        ##########################################################################################################################
        
        #####Add new node#########################################
        if (Informed_Growth=='Yes'):   
            G, InpsNodes, OutsNodes, t = AddNewNode(t, G, alpha, N_I, N_O, MaxNewLinks, Psi, P_inp, P_out, InpsNodes, OutsNodes,\
            InpNodeType, OutNodeType, Spectral_radius, Inps, Outs, Inps_test, Outs_test, Trans, RC_Reps, Scores_pred[t,0],\
            Max_AddSteps, Err_precision) 
        if (Informed_Growth!='Yes'):
            G, InpsNodes, OutsNodes, t = AddNewNode_Uninformed(t, G, N_I, N_O, MaxNewLinks, Psi, P_inp, P_out, InpsNodes, OutsNodes,\
            InpNodeType, OutNodeType)
        
        ####### Net. properties
        MSEs_MnSD_train, MSEs_MnSD_pred = RC(G, Spectral_radius, alpha, N_I, N_O, InpsNodes, OutsNodes, Inps, Outs, Inps_test, Outs_test,\
                                             Trans, RC_Reps)
        
        AllGraphs.append(G)
        AllInpsNodes.append(InpsNodes); AllOutsNodes.append(OutsNodes)
        Scores_train[t] = MSEs_MnSD_train
        Scores_pred[t] = MSEs_MnSD_pred
        NetMsrs[t], NetMsrs_Names = Network_Measures(G)
        
        print(t, 'After Addition fun: Updated Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(),\
              'Inp Nodes:', len(InpsNodes[0]),'Out nodes:',len(OutsNodes[0]),'Err:', Scores_pred[t,0])
        #########################################################
        
        
        ###### Delete Node Only during last step if condition is met####################################################################
        if ((Scores_pred[t,0,:] < Delta_Err).any() and Delete_Nodes=='Yes'):
        
            G, InpsNodes, OutsNodes, NetMsrs_AfterDelFun, Scores_train_AfterDelFun, Scores_pred_AfterDelFun, Nodes_deleted =\
            DeleteNode(G, alpha, N_I, N_O, InpsNodes, OutsNodes, Spectral_radius, Inps, Outs, Inps_test, Outs_test,\
            Trans, RC_Reps, NetMsr_init, Scores_pred[t,0], Err_precision, NodesDel_Percent)
            
            if Nodes_deleted>=1:
                t=t+1
                AllGraphs.append(G)
                AllInpsNodes.append(InpsNodes); AllOutsNodes.append(OutsNodes)
                NetMsrs[t] = NetMsrs_AfterDelFun.T
                Scores_train[t] = Scores_train_AfterDelFun
                Scores_pred[t] = Scores_pred_AfterDelFun

            print(t, 'After Delete 2 Fun : Updated Net. Nodes:', G.number_of_nodes(), 'Links:', G.number_of_edges(),\
                  'Inp Nodes:',len(InpsNodes[0]),'Out nodes:',len(OutsNodes[0]),'Deleted Nodes:',Nodes_deleted,'Err:',Scores_pred[t,0])
        
            ##### Skip if t already reach ==T #######################
            # if (Scores_pred[t,0,:] > Delta_Err).all(): break
            if t>=T:  break
        ##########################################################################################################################
        
        #### Plot
        if (t%PlotEvery==0):
            Net_Plot(G,3)
        ##### Skip if t already reach ==T #######################
        # if (Scores_pred[t,0,:] > Delta_Err).all(): break
        if t>=T:  break
        ########################################################
    return AllGraphs, NetMsrs, NetMsrs_Names, Scores_train, Scores_pred, AllInpsNodes, AllOutsNodes




def SaveData(SaveDir, TaskType, NetProps, InpProps, P_inp, P_out, ModelRep, NetMsr, NetMsr_Names, Scores, AllGraphs, AllInpsNodes, AllOutsNodes):
    np.save(os.path.join(SaveDir,'NetMeasures_Names.npy'), NetMsr_Names)
    if TaskType=='NARMA':
        np.save(os.path.join(SaveDir,'{:}{:}_NetMeasures_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.npy'.format(TaskType, InpProps[1],\
                                                                            P_inp, P_out, ModelRep)), NetMsr) 
        np.save(os.path.join(SaveDir,'{:}{:}_Scores_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.npy'.format(TaskType, InpProps[1],\
                                                                            P_inp, P_out, ModelRep)), Scores) 

        nx.write_gpickle(AllInpsNodes, os.path.join(SaveDir,'{:}{:}_InpsNodes_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                                        InpProps[1], P_inp, P_out, ModelRep)))
        nx.write_gpickle(AllOutsNodes, os.path.join(SaveDir,'{:}{:}_OutsNodes_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                                    InpProps[1], P_inp, P_out, ModelRep)))
        nx.write_gpickle(AllGraphs, os.path.join(SaveDir,'{:}{:}_Graphs_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                                    InpProps[1], P_inp, P_out, ModelRep)))
    if TaskType=='SinCos':
        np.save(os.path.join(SaveDir,'{:}_NodeDel{:d}_a{:}_b{:}_p{:}_NetMeasures_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.npy'.format(TaskType,\
                                               NetProps[0], InpProps[3],InpProps[4],InpProps[5], P_inp, P_out, ModelRep)), NetMsr) 
        np.save(os.path.join(SaveDir,'{:}_NodeDel{:d}_a{:}_b{:}_p{:}_Scores_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.npy'.format(TaskType,\
                                                NetProps[0], InpProps[3],InpProps[4],InpProps[5], P_inp, P_out, ModelRep)), Scores) 

        nx.write_gpickle(AllInpsNodes, os.path.join(SaveDir,'{:}_NodeDel{:d}_a{:}_b{:}_p{:}_InpsNodes_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                    NetProps[0], InpProps[3],InpProps[4],InpProps[5], P_inp, P_out, ModelRep)))
        nx.write_gpickle(AllOutsNodes, os.path.join(SaveDir,'{:}_NodeDel{:d}_a{:}_b{:}_p{:}_OutsNodes_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                    NetProps[0], InpProps[3],InpProps[4],InpProps[5], P_inp, P_out, ModelRep)))
        nx.write_gpickle(AllGraphs, os.path.join(SaveDir,'{:}_NodeDel{:d}_a{:}_b{:}_p{:}_Graphs_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                 NetProps[0], InpProps[3],InpProps[4],InpProps[5], P_inp, P_out, ModelRep)))
    
    else:
        np.save(os.path.join(SaveDir,'{:}_NetMeasures_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.npy'.format(TaskType,\
                                                                            P_inp, P_out, ModelRep)), NetMsr) 
        np.save(os.path.join(SaveDir,'{:}_Scores_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.npy'.format(TaskType,\
                                                                            P_inp, P_out, ModelRep)), Scores) 

        nx.write_gpickle(AllInpsNodes, os.path.join(SaveDir,'{:}_InpsNodes_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                                     P_inp, P_out, ModelRep)))
        nx.write_gpickle(AllOutsNodes, os.path.join(SaveDir,'{:}_OutsNodes_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                                    P_inp, P_out, ModelRep)))
        nx.write_gpickle(AllGraphs, os.path.join(SaveDir,'{:}_Graphs_Pinp{:.2f}_Pout{:.2f}_Rp{:d}.gpickle'.format(TaskType,\
                                                                    P_inp, P_out, ModelRep)))



##################################################################################################################################################################################
##### Run full model#########################################################################################

# @jit(target_backend='cuda')   
def Run_Full_Model(Net_Init, alpha, Npts_U, TaskType, NetProps, InpProps, MaxNewLinks, Psi, P_inp,\
    P_out, N_I, N_O, InpsNodes_init, OutsNodes_init, Spectral_radius, T, Delta_err, T_plot,\
    Transs, RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth, Delete_Nodes,\
    InpNodeType, OutNodeType, Model_Reps, Scores_Names, SaveDir, SaveDataFlag):
    for Mr in range(Model_Reps):
        start = timer()
        print('Model rep: ', Mr)
        Inps, Outs, Inps_test, Outs_test = InpGenerate(TaskType, N_I, Npts_U, InpProps)
        InpPlot(Inps, Outs, N_I)  

        AllGraphs, NetMsrs, NetMsrs_Names, Scores_train, Scores_pred, AllInpsNodes, AllOutsNodes = Checkpoint_V3(Net_Init, alpha, MaxNewLinks, Psi, P_inp,\
        P_out, N_I, N_O, InpsNodes_init, OutsNodes_init, Spectral_radius, T, Delta_err, T_plot, Inps, Outs, Inps_test,Outs_test,Transs,\
        RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth, Delete_Nodes, InpNodeType, OutNodeType)
        Scores = np.array([Scores_train, Scores_pred])
        print('Model Rep:', Mr, "completed, time elapsed:", (timer()-start)/60, 'mins')
        #### Save Data#######################################
        if(SaveDataFlag=="Yes"):
            SaveData(SaveDir, TaskType, NetProps, InpProps, P_inp, P_out, Mr, NetMsrs, NetMsrs_Names, Scores, AllGraphs, AllInpsNodes, AllOutsNodes)
        #####################################################
        print('Final Inp nodes:', AllInpsNodes[-1])
        print('Final Out nodes:', AllOutsNodes[-1])
        Plot_NetMsrs(NetMsrs, NetMsrs_Names)

        Plot_Performance(Scores, Scores_Names, N_O)
        ##### Output from final Net.####################################
        Outs_predict, Outs_test_pred, MSEs_train, MSEs_pred = RC(AllGraphs[-1], Spectral_radius, alpha, N_I, N_O, AllInpsNodes[-1], AllOutsNodes[-1], Inps, Outs,\
                                                               Inps_test, Outs_test, Transs, 1, 1)
        
        if (TaskType=='Chaos') or (TaskType=='VDP'):
            Auto_RC(AllGraphs[-1], Spectral_radius, alpha, N_I, N_O, AllInpsNodes[-1], AllOutsNodes[-1], Inps, Outs,\
                                                               Transs, 3, TaskType)
            
        Tillt=5000
        Net_Plot(AllGraphs[-1],3.5)
        Plot_t(N_O, Outs, Outs_predict[0], Transs, Tillt, MSEs_train[0])
        Plot_t(N_O, Outs_test, Outs_test_pred[0], Transs, Tillt, MSEs_pred[0])
        ################################################################

        ################################################################


################################################################
### Model type and Parameters


###### Model preparation ##############################
Informed_Growth = 'No'#'Yes'
Delete_Nodes = 'No'#'Yes'  ##### Informed node deletion with 'Yes' or 'No'
InpNodeType = 0  ###### 0-> New node can be input node for other inputs as well (in multiple tasks) with Prob P_inp, 1-> Strictly exclusive to each task
OutNodeType = 0  ###### 0-> New node can be output node for other outputs as well (in multiple tasks) with Prob P_inp, 1-> Strictly exclusive to each task

##### Model Parameters ##################################
NodesDel_Percent = 50 #25 ##### Least no. of attempts to delete nodes 
Max_AddSteps=25 ##### No. of steps the node addition function attempt, if in all steps no new node is added, t+=1

Net_Init = np.array([[0,1],[1,0]])

### Psi = Prob of in or out connections
### Link_prob = Prob of 1 or 2 new connections
### P_inp or P_out is the prob. that a new node is also an input node or not. [0, 1], 0(1)-> No(all) new nodes are inp/out nodes

MaxNewLinks=5; Psi=0.5; Spectral_radius= 1.1


T=500;  #### Max iterations
T_plot=50; RC_Reps=1

NetProps = [NodesDel_Percent, Max_AddSteps, MaxNewLinks, Psi, Spectral_radius]
####### Inputs and tasks###########################################################
P_inp=0.5; P_out= 1 - P_inp; 


TaskType = 'SinCos' #'Chaos' #'VDP' #'Chaos'

InpProps=[]
####NARMA Task###################################################################
if TaskType=='NARMA':
    alpha = 0.9
    N_I=1; N_O=1
    InpsNodes_init = list([[0]])
    OutsNodes_init = list([[1]])
    Npts_U = 2500; Transs=500;
    MemoryX = np.array([10]); InpDistrib=np.array([[0, 0.5], [0, 0.5]])
    InpProps = [InpDistrib, MemoryX]
    Delta_err= 0.105
    ### Desired Error and error precision (rounding off error digits for comparision)
    Err_precision= 6; 
            
####SinCos Task###################################################################
if TaskType=='SinCos':
    alpha = 0.9
    N_I=1; N_O=1
    InpsNodes_init = list([[0]])
    OutsNodes_init = list([[1]])
    Npts_U = 3000; Transs=50;
    h=0.1
    Omegas=np.array([1]); Phis=np.array([0]); 
    
    ####SinCos1
    A_scs=np.array([0]); B_scs=np.array([1]); P_scs=np.array([1])
    Delta_err= 0.000001 
    Err_precision= 10; 
    
    ####SinCos2 
    # A_scs=np.array([1]); B_scs=np.array([0.25]); P_scs=np.array([3])
    # Delta_err= 0.001 
    # Err_precision= 8; 
    
    
    InpProps = [h, Omegas, Phis, A_scs, B_scs, P_scs]
    
    
####Chaos Task###################################################################   
if TaskType=='Chaos':
    alpha = 0.1
    N_I=3; N_O=3
    InpsNodes_init = list([[0],[0],[0]])
    OutsNodes_init = list([[1],[1],[1]])
    Npts_U = 3000; Transs=50;
    Delta_err= 0.001
    ### Desired Error and error precision (rounding off error digits for comparision)
    Err_precision= 6; 
    
####VDP Task###################################################################  
if TaskType=='VDP':
    alpha = 0.23 
    N_I=2; N_O=2
    InpsNodes_init = list([[0],[0]])
    OutsNodes_init = list([[1],[1]])
    Npts_U = 2000; Transs=25;
    Delta_err= 0.001
    ### Desired Error and error precision (rounding off error digits for comparision)
    Err_precision= 6; 
######################################################################################  



#### Test Inp_Out Plot
print('Test Inp_Out Plot:')
Inps, Outs, Inps_test, Outs_test = InpGenerate(TaskType, N_I, Npts_U, InpProps)
InpPlot(Inps, Outs, N_I)

#### Scores_Names#################################################################
Scores_Names=['NMSE train', 'NMSE pred']


### Model Repetitions
Model_Reps= 10 #100

######### Save Directory

### for N1
SaveDir = 'C:/Work/TU-Berlin/Projects/2.RC_DynamicNets/1.NetGrowth_Models/Data/Checkpoint-Model_V3/1Task/N1/SinCos/'
### for N2
# SaveDir = 'C:/Work/TU-Berlin/Projects/2.RC_DynamicNets/1.NetGrowth_Models/Data/Checkpoint-Model_V3/1Task/N2/SinCos/'
# ### for N3
# SaveDir = 'C:/Work/TU-Berlin/Projects/2.RC_DynamicNets/1.NetGrowth_Models/Data/Checkpoint-Model_V3/1Task/N3/SinCos/'


####if you want to save data give 'Yes'
SaveDataFlag="No"


################################################################
########### Run Model
start = time.time()

Run_Full_Model(Net_Init, alpha, Npts_U, TaskType, NetProps, InpProps, MaxNewLinks, Psi, P_inp,\
    P_out, N_I, N_O, InpsNodes_init, OutsNodes_init, Spectral_radius, T, Delta_err, T_plot,\
    Transs, RC_Reps, Err_precision, Max_AddSteps, NodesDel_Percent, Informed_Growth, Delete_Nodes,\
    InpNodeType, OutNodeType, Model_Reps, Scores_Names, SaveDir, SaveDataFlag)

print('Run time', time.time()-start, 'seconds.')



 










