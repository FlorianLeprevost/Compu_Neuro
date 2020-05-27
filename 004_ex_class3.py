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
#plt.style.use('ggplot')

fig_width = 9# width in inches
fig_height = 6  # height in inches
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

#%%

time = range(1000)
spikes = np.asarray([np.random.rand() for i in time])
spike_event = np.where(spikes <.25)

plt.rcParams['figure.figsize'] = [9,1]
plt.eventplot(spike_event, linewidths = 1, lineoffsets =.1, linelength=1)
#plt.legend(loc='best')
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Spikes')
plt.ylim([0,1])
plt.yticks([],[])
plt.yticks([],[])


plt.savefig('report3_fig1.png', dpi=600)
plt.clf
#%% b)

time = range(500)
spikes = np.asarray([np.random.rand() for i in time])
spike_event = np.where(spikes <.05)

plt.rcParams['figure.figsize'] = [9,1]
plt.eventplot(spike_event, linewidths = 1, lineoffsets =.1, linelength=1)
#plt.legend(loc='best')
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Spikes')
plt.ylim([0,1])
plt.yticks([],[])
plt.xticks([100,200,300,400,500],[200,400,600,800,1000])


plt.savefig('report3_fig2.png', dpi=600)
plt.clf


#%% c)
#time = range(1000)
#sum_array = []
#train_array=[]
#for i in range(200):
#    spikes = np.asarray([np.random.rand() for j in time])
#    spike_event = np.where(spikes <.05)
#    train_array = np.append(train_array, spike_event) #axis=0
#    while i<50:
#        plt.eventplot(spike_event, linewidths = 1, lineoffsets =1, linelength=1)
##
#
#plt.hist(train_array, bins=10, rwidth =.9)
#plt.ylim([900,1100])
#


#%% c) ?
plt.rcParams['figure.figsize'] = [9,5]


trials=200
time=range(1000)
high_value = np.max(time)*2
train_array= [None]*trials

for i in range(trials):
    spikes = np.asarray([np.random.rand() for j in time])
    spike_event = np.where(spikes <.05)
    train_array[i] = spike_event[0] #,axis=0)
    
plt.eventplot(train_array[1:50], color='black')
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Trial number')
plt.xlim([0, 1000])
plt.ylim([.5, 50])

plt.savefig('report3_fig3.png', dpi=600)
plt.clf
#%%) d) timepoints

out = np.concatenate(train_array).ravel()

plt.rcParams['figure.figsize'] = [6,6]

plt.hist(out)
plt.ylim([900, 1100])
plt.xlabel(r'Time in  ms')
plt.ylabel(r'Count')

plt.savefig('report3_fig4.png', dpi=600)
plt.clf
#%% d) intertrials intervals
intertrial = []
for i in train_array:
    intertrial.append(np.diff(i))
intertrial = np.concatenate(intertrial).ravel()

plt.hist(intertrial, 50)
plt.xlabel(r'ISI (ms)')
plt.ylabel(r'Count')
plt.xlim([0,120])

plt.savefig('report3_fig5.png', dpi=600)
plt.clf
#%%
#%% II) 

sim_data = loadmat('simdata.mat')

spt=sim_data['spt']

time= sim_data['t']
time = time.tolist()
time = time[0]

freq=sim_data['f1']
freq = freq.tolist()
freq = freq[0]

#%% a) 
train = spt[0,0]
high_value = np.max(time)*2
train_array= [None]*len(train)

for i in range(len(train)):
    spike_event = np.where(train[i] ==1)
    train_array[i] = spike_event[0] #,axis=0)

plt.rcParams['figure.figsize'] = [13,3]
plt.eventplot(train_array,linewidths = 3, lineoffsets =1, linelength=.5, color='black')
plt.ylim([-.5,10])
plt.xlim([0,200])
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Trials')

plt.savefig('report3_fig6.png', dpi=600)
plt.clf

#%% b)
count_trial = 0
train_array= [None]*100
ticks = [None]*8
for j in range(8):
    train = spt[0,j]
    high_value = np.max(time)*2
    ticks[j]=count_trial + len(train)/2
    if j%2 !=0:
        plt.axhspan(count_trial-.5, count_trial-.5 + len(train), facecolor='0.2', alpha=0.2, zorder=-100)
    for i in range(len(train)):
        spike_event = np.where(train[i] ==1)
        train_array[count_trial] = spike_event[0] #,axis=0)
        count_trial +=1

plt.rcParams['figure.figsize'] = [13,8]
plt.rcParams['ytick.major.size'] = 0

plt.eventplot(train_array,linewidths = 3, lineoffsets =1, linelength=.5, color= 'black')
plt.ylim([-.5,100])
plt.xlim([2,200])


index= list(range(200))
plt.yticks(ticks,freq)
plt.xticks(index[0:200:20],time[0:200:20])

plt.xlabel(r'Time (ms)')
plt.ylabel(r'Trials by frequency of stimulation (Hz)')

plt.savefig('report3_fig7.png', dpi=600)
plt.clf

#%% c)
mean_array = []#[None]*8
std_array = []
sem_array = []
for j in range(8):
    train = spt[0,j]
    sc_array= [] #[None]*len(train)
    FR_array=[]
    for i in train:
        sc_array.append(sum(i[40:140]))
    
    mean_array.append(np.mean(sc_array))
    std_array.append(np.std(sc_array))
    
    sem_array.append(np.std(sc_array)/np.sqrt(len(train)))

freq_leg = [str(i) for i in freq]

plt.rcParams['figure.figsize'] = [8,6]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Frequency of stimulation for trials (Hz)')
ax1.set_ylabel('Mean spike count', color=color)
ax1.bar(freq_leg,mean_array,yerr=std_array, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(10,60)


#plt.bar(freq_leg,mean_array,yerr=std_array)
#plt.ylabel(r'Mean spike count')
#plt.xlabel(r'Frequency of stimulation for trials (Hz)')


#firing rate


FR= np.array(mean_array)*2
sem_fr= np.array(sem_array)*2
#plt.errorbar(freq_leg,FR,sem_fr)
#plt.ylabel(r'Average firing rate')
#plt.xlabel(r'Frequency of stimulation for trials (Hz)')


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Average firing rate', color=color)  # we already handled the x-label with ax1
ax2.errorbar(freq_leg,FR,sem_fr, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0,110)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('report3_fig8.png', dpi=600)
plt.clf
#%% III)
#%% a)
El=-70
I=1
gl=.1

time = range(100)
V_array = [El]
Vt=El
for i in time:
    Vt= Vt + gl*(El - Vt) + I
    V_array.append(Vt)
    
plt.plot(V_array, label="$I=1$")
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Membrane potential (mV)')
plt.hlines([-59.8,-39.8], 0, 100, linestyle='dashed')
#b)
V_array = [El]
Vt=El
I=3
for i in time:
    Vt= Vt + gl*(El - Vt) + I
    V_array.append(Vt)
plt.plot(V_array, label="$I=3$")
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Membrane potential (mV)')
plt.legend(loc='best')

plt.xlim([0,100])
plt.savefig('report3_fig9.png', dpi=600)
plt.clf
#c ???

#%% d)?

#%% e)
El=-70
I=1
gl=.1
Vth = -63
time = range(100)
V_array = []
Vt=El
spike_times=[]



for i in time:
    V_array.append(Vt)
    Vt= Vt + gl*(El - Vt) + I
    if Vt >= Vth:
        Vt=El
        spike_times.append(i+.1)
    
plt.plot(V_array, color='steelblue', linewidth=2.5, label="$RP=0 $ms")
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Membrane potential (mV)')
plt.vlines(spike_times, -65,-50, color='steelblue', linewidth=1.5)
#plt.savefig('report3_fig10.png', dpi=600)
#plt.clf
#
##%% f)

El=-70
I=1
gl=.1
Vth = -63
time = range(100)
V_array = []
Vt=El
spike_times=[]

RP=0
RPt=0
for i in time:
    if RP>0:
        RPt=1
        RP += -1
    else:
        RPt=0
    V_array.append(Vt)
    Vt= Vt + gl*(El - Vt) + I - RPt
    if Vt >= Vth:
        Vt=El
        spike_times.append(i+.1)
        RP=3
    
plt.plot(V_array, color='indianred', linewidth=2.5, label="$RP= 3$ ms", linestyle='dashed')
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Membrane potential (mV)')
plt.vlines(spike_times, -65,-50, color='indianred', linewidth=1.5, linestyle='dashed')
plt.legend(loc='best')
plt.xlim([0,100])

plt.savefig('report3_fig11.png', dpi=600)
plt.clf

#%% g)

El=-70
I=1
gl=.1
Vth = -63
time = range(100)
V_array = []
Vt=El
spike_times=[]

sigma=.2

RP=0
RPt=0

for i in time:
    if RP>0:
        RPt=1
        RP += -1
    else:
        RPt=0
    V_array.append(Vt)
    Vt= Vt + gl*(El - Vt) + I - RPt + sigma*np.random.randn()
    if Vt >= Vth:
        Vt=El
        spike_times.append(i+.1)
        RP=3
    
plt.plot(V_array,  linewidth=2.5, color='steelblue', label="$\sigma=.5$")
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Membrane potential (mV)')
plt.vlines(spike_times, -65,-50, linewidth=1.5, color='steelblue')


##%% g)2
V_array = []
Vt=El
spike_times=[]
sigma=1.5
RP=0
RPt=0
for i in time:
    if RP>0:
        RPt=1
        RP += -1
    else:
        RPt=0
    V_array.append(Vt)
    Vt= Vt + gl*(El - Vt) + I - RPt + sigma*np.random.randn()
    if Vt >= Vth:
        Vt=El
        spike_times.append(i+.1)
        RP=3
    
plt.plot(V_array,  linewidth=2.5, color='indianred', label="$\sigma=1.5$", linestyle='dashed')#, label="$V_{th}=-63, \; RP= 3$ms")
plt.xlabel(r'Time (ms)')
plt.ylabel(r'Membrane potential (mV)')
plt.vlines(spike_times, -65,-50, linewidth=1.5, color='indianred', linestyle='dashed')


plt.xlim([0,100])
plt.legend(loc='best')
plt.savefig('report3_fig12.png', dpi=600)
plt.clf

#%% h)
inv_freq = 1000/np.array(freq)
I=[]
for i in range(8):
    for k in range(10):
        x=[.1 for j in range(200)] + ([3 if j %inv_freq[i]<16 else .1 for j in range(500) ] + [.1 for j in range(300)])
        I.append(x)

#param for V model
El=-70
gl=.1
Vth = -63
time = range(1000)
V_array = []
Vt=El

sigma=1.2

RP=0
RPt=0

#param for spike trains
train_array= []


#loop
for j in range(80):
    spike_times=[]
    if j%20==0:
        plt.axhspan(j-.5, j+9.5, facecolor='0.2', alpha=0.2, zorder=-100)
    current=I[j]
    for i in time:
        if RP>0:
            RPt=2
            RP += -1
        else:
            RPt=0
        V_array.append(Vt)
        Vt= Vt + gl*(El - Vt) + current[i] - RPt + sigma*np.random.randn()
        if Vt >= Vth:
            Vt=El
            spike_times.append(i+.1)
            RP=5
    train_array.append(spike_times)
        #####



plt.eventplot(train_array,linewidths = 3, lineoffsets =1, linelength=.5, color= 'black')


plt.rcParams['figure.figsize'] = [13,8]
plt.rcParams['ytick.major.size'] = 0


plt.yticks(list(range(5,95,10)),freq)
plt.ylim([-.5,80])
plt.xlim([2,1000])

plt.xlabel(r'Time in  ms')
plt.ylabel(r'Frequency of stimulation for trials (Hz)')

plt.savefig('report3_fig13.png', dpi=600)
plt.clf
