# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:34:23 2020

@author: install
"""
#cd "E:///Users/install/Documents/CogMaster2/Computational Neuroscience Methods/Compu_Neuro"


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.io import loadmat
#%%
plt.style.use('ggplot')

fig_width = 9# width in inches
fig_height = 12  # height in inches
fig_size =  [fig_width,fig_height]
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.size'] = 20#9
plt.rcParams['legend.fontsize'] = 20#7. 
#plt.rcParams['lines.linewidth'] = 1.2
#plt.rcParams['lines.markeredgewidth'] = 0.003
#plt.rcParams['lines.markersize'] = 10
###
##
#plt.rcParams['axes.facecolor'] = '1'
#plt.rcParams['axes.edgecolor'] = '0'
#plt.rcParams['axes.linewidth'] = '0.7'
#
plt.rcParams['axes.labelcolor'] = '0'
plt.rcParams['axes.labelsize'] = 20#9
plt.rcParams['xtick.labelsize'] = 17#7
plt.rcParams['ytick.labelsize'] = 17#7
plt.rcParams['xtick.color'] = '0'
plt.rcParams['ytick.color'] = '0'
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

plt.rcParams['font.sans-serif'] = 'Arial'

#%% 1) param
NGG = 1000
pGG=.1
pz=1
pGz=1
gGz=1
gGG=1.5
tau=10
delta_t=0.1

#%% 2) generating weights matrices
def generate():
#setting the weights randomly
JGg = np.random.normal(0,1/(pGG*NGG) , size=(NGG,NGG))
JGz = np.random.uniform(-1,1,size=NGG)[np.newaxis].T
w = np.random.normal(0,1/(pz*NGG),size=NGG)[np.newaxis].T


#generating x% random indices to replace some weights by zero in JGG and JGz
indices = [[i,j] for i in range(NGG) for j in range(NGG)]

#JGG
indices_zero_JGg = np.random.permutation(indices)
nb_zero_JGg= round((1 - pGG)*NGG*NGG)
indices_zero_JGg = indices_zero_JGg[0:nb_zero_JGg]
JGg[indices_zero_JGg[:,0], indices_zero_JGg[:,1]]= 0

#JGz
indices1d = [i for i in range(NGG)]

indices_zero_JGz = np.random.permutation(indices1d)
nb_zero_JGz= round((1 - pGz)*NGG)
indices_zero_JGz = indices_zero_JGz[0:nb_zero_JGz]
JGz[indices_zero_JGz[:]]= 0

#w
indices_zero_w = np.random.permutation(indices1d)
nb_zero_w= round((1 - pz)*NGG)
indices_zero_w = indices_zero_w[0:nb_zero_w]
w[indices_zero_w[:]]= 0


#sanity check
print ( int(np.mean(sum(JGg==0)*100/NGG)),'% of JGG =0\n',
       int(np.mean(sum(JGz==0)*100/NGG)),'% of JGz =0\n',
       int(np.mean(sum(w==0)*100/NGG)),'% of w =0\n')

re

#%% 3) simulation

#initializing the network activity radomly
x=np.random.normal(0, 1
                   #/(pz*NGG)
                   , NGG)[np.newaxis].T #make it vertical
r=np.tanh(x)
plt.hist(r)

time=np.linspace(0,.1,1001) #10 000dt = 1000ms = 1s

#stock for plot
x_time_array = list()
x_time_array.append(x)

readout= np.dot(w.T,r)
readout_array = list()
readout_array.append(readout)

r_array = list()
r_array.append(r)

#loop
for t in range(len(time)-1):
    #not that the [np.newaxis].T ain the next line only reflects weird behavior of sum fct°
    x= x + (-x + ((gGG*JGg*r).sum(1))[np.newaxis].T + gGz*JGz*readout)*delta_t/tau
    x_time_array.append(x)

    r=np.tanh(x)
    r_array.append(r)
    readout= np.dot(w.T,r)
    readout_array.append(readout)


    if t%100 ==0:
        print(int(t/10), 'ms')



#plot
plt.subplot(3,1,1)
mat_activity = np.squeeze(np.asarray(x_time_array))
plt.plot(time[:], mat_activity[:, 1:10])
plt.xlabel('Time (s)')
plt.ylabel('membrane potential x')

plt.subplot(3,1,2)
plt.plot(time[:],np.squeeze(readout_array[:]))
plt.xlabel('Time (s)')
plt.ylabel('Readout')

plt.subplot(3,1,3)
mat_activity_r = np.squeeze(np.array(r_array))
plt.plot(time[:], mat_activity_r[:, 1:10])
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')



#%%troubleshhoting
plt.hist(x)
plt.hist(r)




#%% 4) same but with weight updating
#param for updating
delta_up=1
alpha=50
P = np.identity(NGG)/alpha
#function
f = lambda x : np.cos(x*1000)


#initializing the network activity radomly
x=np.random.normal(0, 1
                   #/(pz*NGG)
                   , NGG)[np.newaxis].T #make it verticalr=np.tanh(x*100)
plt.hist(r)

time=np.linspace(0,.1,1001) #10 000dt = 1000ms = 1s

#stock for plot
x_time_array = list()
x_time_array.append(x)

readout= np.dot(w.T,r)
readout_array = list()
readout_array.append(readout)

r_array = list()
r_array.append(r)

#loop
for t in range(len(time)-1):
    #not that the [np.newaxis].T ain the next line only reflects weird behavior of sum fct°
    x= x + (-x + ((gGG*JGg*r).sum(1))[np.newaxis].T + gGz*JGz*readout)*delta_t/tau
    x_time_array.append(x)

    r=np.tanh(x)
    r_array.append(r)
    readout= np.dot(w.T,r)
    readout_array.append(readout)



    if t%10 ==0:
        e_m = readout - f(t)
        w = w - e_m * np.dot(P,r)
        P = P - np.dot(np.dot(P,r), np.dot(r.T,P))/(1-float(np.dot(np.dot(r.T,P),r)))        
    
    if t%100 ==0:
        print(int(t/10), 'ms')


#plot
plt.subplot(3,1,1)
mat_activity = np.squeeze(np.asarray(x_time_array))
plt.plot(time[:], mat_activity[:, 1:10])
plt.xlabel('Time (s)')
plt.ylabel('membrane potential x')

plt.subplot(3,1,2)
plt.plot(time[:],np.squeeze(readout_array[:]))
plt.xlabel('Time (s)')
plt.ylabel('Readout')

plt.subplot(3,1,3)
mat_activity_r = np.squeeze(np.array(r_array))
plt.plot(time[:], mat_activity_r[:, 1:10])
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')
