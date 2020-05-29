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
from scipy import signal
from scipy.integrate import odeint
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  
from scipy import stats

#%%
plt.style.use('ggplot')

fig_width = 18# width in inches
fig_height = 15  # height in inches
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
delta_t=1


#%% 2) generating weights matrices
def generate():
    #setting the weights randomly
    JGg = np.random.normal(0,1/np.sqrt((pGG*NGG)) , size=(NGG,NGG))
    JGz = np.random.uniform(-1,1,size=NGG)[np.newaxis].T
    w = np.random.normal(0,1/np.sqrt((pz*NGG)),size=NGG)[np.newaxis].T
    
    
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
    
    return JGg, JGz, w, indices_zero_w



#%% 4.0) functions that create time series
# q= make time series = better


# x = np.linspace(0, 2000, 2000)

def f_per_pap(time_ms):
    freq=1/200
    ft = lambda x : (np.sin(x*np.pi*freq)*1.3 \
        + np.sin(x*2*np.pi*freq)*1.3/2   \
            + np.sin(x*3*np.pi*freq)*1.3/6  \
                +np.sin(x*4*np.pi*freq)*1.3/3)/1.5
    ts= ft(np.array(range(time_ms)))
    return ts
    

def f_triangle(time_ms):
    ft= lambda x : np.abs(signal.sawtooth(x/30))*2-1
    ts= ft(np.array(range(time_ms)))
    return ts

def f_per(time_ms):
    ft = lambda x : np.cos(x/60) + np.sin(x/15)*.5
    ts= ft(np.array(range(time_ms)))
    return ts

def f_per_comp (time_ms):
    ft =  lambda x : np.cos(x/72) + np.cos(x/7)*.3 + np.sin(x/36)*.5+ np.cos(x/18)*.3
    ts= ft(np.array(range(time_ms)))
    return ts


def f_noise(time_ms):
    ts=list()
    ft = lambda x : np.cos(x/100) + np.sin(x/25)*.5
    for i in range(time_ms):
        ts.append(ft(i*2) + np.random.normal(0,.2))
    return ts

def f_lorenz(time_ms):
    # Lorenz paramters and initial conditions
    sigma, beta, rho = 10, 8/3, 28
    u0, v0, w0 = 0, 1, 1.05
      
    def lorenz(X, t, sigma, beta, rho):
        """The Lorenz equations."""
        u, v, w = X
        up = -sigma*(u - v)
        vp = rho*u - v - u*w
        wp = -beta*w + u*v
        return up, vp, wp
    # Integrate the Lorenz equations on the time grid t
    t = np.linspace(0, time_ms, time_ms)
    f = odeint(lorenz, (u0, v0, w0), t/60, args=(sigma, beta, rho))/10
    x, y, z = f.T
    return x

def f_disco(time_ms):
    ts=list()
    point=1
    for i in range(time_ms):
        ts.append(point)
        if i%100==0:
            point=-point
    return ts

#%% 6) weightsv2
def simul_weight2(time_ms, time_min, time_max, funct):
    #param for updating
    delta_up=6
    alpha=20
    P = np.identity(NGG)/alpha
    ts = funct(time_ms)
    JGg, JGz, w, indices_zero_w = generate()
    #initializing the network activity radomly
    #w = np.zeros(NGG)[np.newaxis].T  
    x=.5*np.random.normal(0, 1, NGG)[np.newaxis].T #make it vertical
    r=np.tanh(x)
    readout= np.dot(w.T,r)   
    time=np.linspace(0,time_ms,time_ms) #10 000dt = 1000ms = 1s
    
    readout_array = list()
    x_time_array = list()
    r_array = list()
    dx_array = list()
    #error
    e_m_array=list()
    e_p_array=list()
    wup_array=list()
    wn_array=list()

    #loop
    for t in range(len(time)):
        dx = (-x + np.dot(gGG*JGg, r) +gGz*JGz*readout)/tau
        x= x + dx*delta_t
        r=np.tanh(x)
        readout= np.dot(w.T,r)
        
        x_time_array.append(x[:10])
        r_array.append(r[:10])
        readout_array.append(readout)  
        dx_array.append(dx[:10])

        if t%100 ==0:
            print(int(t), 'ms')
        
        #update weights between t_min and t_max
        if t%delta_up ==0:
            if t<time_min:
                e_m=0
                wup=0
                e_p=0

            elif t>time_max:
                e_m = readout - ts[t]
                wup=0
                e_p=0
            else:            

                P = P - np.dot(np.dot(P,r), np.dot(r.T,P))/(1+float(np.dot(np.dot(r.T,P),r)))
                e_p = np.dot(w.T,r) - ts[t]
                e_m = readout - ts[t]
                w = w - e_m * np.dot(P,r)
                wup=np.linalg.norm(e_m * np.dot(P,r))
                #w[indices_zero_w[:]]= 0
            e_p_array.append(float(e_p))
            e_m_array.append(float(e_m))
            wup_array.append(float(wup))
            wn_array.append(float(np.sqrt(np.dot(w.T,w))))

        

    #plot
#    offset
    offset=np.array([(np.linspace(0,18,10)) for i in range(time_ms)])
    # plt.subplot(4,1,1)
    # mat_activity = np.squeeze(np.asarray(x_time_array))
    # plt.plot(time[:], mat_activity+offset)
    # plt.ylabel('membrane potential x')
    # plt.xlim([0,time_ms])
    # plt.axvline(time_min,c="black")
    # plt.axvline(time_max,c="black")
    plt.subplot2grid((4,1),(2,0),1,1)
    plt.plot(time[:],np.squeeze(readout_array[:]))
    plt.plot(time[:], ts[:])
    plt.ylabel('Readout')
    plt.xlim([0,time_ms])
    plt.ylim([-2,2])
    plt.xticks([])
    plt.axvline(time_min,c="black", lw=3, alpha=.3)
    plt.axvline(time_max,c="black", lw=3, alpha=.3)

    plt.subplot2grid((4,1),(0,0),2,1)
    mat_activity_r = np.squeeze(np.array(r_array))
    plt.plot(time[:], mat_activity_r+offset)
    plt.ylabel('Firing rate (Hz)')
    plt.xlim([0,time_ms])
    plt.xticks([])
    plt.axvline(time_min,c="black", lw=3, alpha=.3)
    plt.axvline(time_max,c="black", lw=3, alpha=.3)
    
    plt.subplot2grid((4,1),(3,0),1,1)
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), e_m_array, label='error')
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wup_array)*10, label='update ')
    #plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(e_p_array)*10, label='error_p')
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wn_array), label='w norm')
    plt.xlabel('Time (ms)')
    plt.ylabel('error')
    plt.xlim([0,time_ms])
    plt.axvline(time_min,c="black", lw=3, alpha=.3)
    plt.axvline(time_max,c="black", lw=3, alpha=.3)
    plt.legend(loc='best')    


#%% 4) weights author
def simul_weight4(time_ms, time_min, time_max,funct):
    #param for updating
    delta_up=2
    alpha=1
    delta_t=.1
    P = np.identity(NGG)/alpha
    
    n_sample=time_ms*10
    ts = funct(n_sample)


        
        
    #error
    e_m_array=list()
    e_m_array.append(0)    
    e_p_array=list()
    e_p_array.append(0)
    wup_array=list()
    wup_array.append(0)    
    wn_array=list()
    wn_array.append(0)
    
    
    JGg, JGz, w, indices_zero_w = generate()
    #w = np.zeros(NGG)[np.newaxis].T

    #initializing the network activity radomly
    x=.5*np.random.normal(0, 1, NGG)[np.newaxis].T #make it vertical
    r=np.tanh(x)
    plt.hist(r)
    
    time=np.array(range(int(time_ms/delta_t)))/10 #10 000dt = 1000ms = 1s    
    #stock for plot
    x_time_array = list()
    x_time_array.append(x[:10])
    
    readout= np.dot(w.T,r)
    readout_array = list()
    readout_array.append(readout)
    
    r_array = list()
    r_array.append(r[:10])
    
    dx = (-x + np.dot(gGG*JGg,r) + gGz*JGz*readout)
    dx_array = list()
    dx_array.append(dx[:10])
    
    #loop
    ti=0
    for t in time[:-1]:
        ti+=1
        
        dx = (-x + np.dot(gGG*JGg,r) + gGz*JGz*readout)
        x= x + dx*delta_t
    
        r=np.tanh(x)
        readout= np.dot(w.T,r)
        
        r_array.append(r[:10])
        readout_array.append(readout)
        x_time_array.append(x[:10])
        dx_array.append(dx[:10])
    
    
        if t % 100 <0.01:
            print(int(t), 'ms')
            
        
        #update weights between t_min and t_max
        if t % delta_up <0.01:
            if t<time_min:
                e_m=0
                wn=np.sqrt(np.dot(w.T,w))/10
                e_p=0
                wup=0
            elif t>time_max:
                e_m = readout - ts[ti]
                wn=np.sqrt(np.dot(w.T,w))/10
                e_p=0
                wup=0
            else:            
                e_m = readout - ts[ti]

                wn=np.sqrt(np.dot(w.T,w))/10
                #w[indices_zero_w[:]]= 0
                k = np.dot(P,r);
                rPr = np.dot(r.T,k)
                c = 1.0/(1.0 + rPr)
                P = P - k*(k.T*c)                               
                e_p = np.dot(w.T,r) - ts[ti]
                w = w - e_m * k*c
                wup= np.linalg.norm(- e_m * k*c)

            e_m_array.append(float(e_m))
            e_p_array.append(float(e_p))
            wn_array.append(float(wn))
            wup_array.append(float(wup))

        
    
    #plot
    #offset
    offset=np.array([(np.linspace(0,18,10)) for i in range(time_ms*10)])
    # plt.subplot(4,1,1)
    # mat_activity = np.squeeze(np.asarray(x_time_array))
    # plt.plot(time[:], mat_activity+offset)
    # plt.ylabel('membrane potential x')
    # plt.xlim([0,time_ms])
    # plt.axvline(time_min)
    # plt.axvline(time_max)
    
    plt.subplot(3,1,2)
    plt.plot(time[:],np.squeeze(readout_array[:]))
    plt.plot(time, ts)
    plt.ylabel('Readout')
    plt.xlim([0,time_ms])
    plt.ylim([-2,2])
    plt.xticks([])
    plt.axvline(time_min)
    plt.axvline(time_max)

    plt.subplot(3,1,1)
    mat_activity_r = np.squeeze(np.array(r_array))
    plt.plot(time[:], mat_activity_r+offset)
    plt.ylabel('Firing rate (Hz)')
    plt.xlim([0,time_ms])
    plt.axvline(time_min)
    plt.axvline(time_max)
    
    plt.subplot(3,1,3)
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), e_m_array, label='error')
    #plt.plot(np.linspace(0,time_ms,len(e_m_array)), e_p_array, label='error_p')
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wup_array)*10, label='(w update)')
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wn_array)*10, label='(w norm)')
    plt.xlabel('Time (ms)')
    plt.ylabel('error')
    plt.xlim([0,time_ms])
    plt.axvline(time_min)
    plt.axvline(time_max)
    plt.legend(loc='best')    


#%% 6) weightsv5 double readout
def simul_weight_2read(time_ms, time_min, time_max, funct1,funct2):
    #param for updating
    delta_up=8
    alpha=1
    P = np.identity(NGG)/alpha
    P2 = np.identity(NGG)/alpha
    ts = np.array(funct1(time_ms))/2
    ts2 = np.array(funct2(time_ms))/2
    #error
    e_m_array=list()
    e_m_array.append(0)       
    e_m2_array=list()
    e_m2_array.append(0)    
    e_p_array=list()
    e_p_array.append(0)
    wup_array=list()
    wup_array.append(0)    

    
    
    JGg, JGz, w, indices_zero_w = generate()
    #initializing the network activity radomly
    JGg2, JGz2, w2, indices_zero_w2 = generate()
    JGz= JGz/2
    JGz2= JGz2/2
    w=w/2
    w2=w2/2

    # wn_array=list()
    # wn_array.append(np.sqrt(np.dot(w.T,w)))
    x=.5*np.random.normal(0, 1, NGG)[np.newaxis].T #make it vertical
    r=np.tanh(x)
    plt.hist(r)
    
    time=np.linspace(0,time_ms,time_ms) #10 000dt = 1000ms = 1s
    
    #stock for plot
    x_time_array = list()
    x_time_array.append(x[:10])
    
    readout= np.dot(w.T,r)
    readout_array = list()
    readout_array.append(readout)
    readout2= np.dot(w2.T,r)
    readout_array2 = list()
    readout_array2.append(readout)
    
    r_array = list()
    r_array.append(r[:10])
    
    dx = (-x + np.dot(gGG*JGg, r) + gGz*JGz*readout)/tau
    dx_array = list()
    dx_array.append(dx[:10])
    
    #loop
    alternance =1
    for t in range(len(time)-1):
        dx = (-x + np.dot(gGG*JGg, r) +gGz*JGz*readout+gGz*JGz2*readout2)/tau
        x= x + dx*delta_t
        x_time_array.append(x[:10])
    
        r=np.tanh(x)
        r_array.append(r[:10])
        readout_array.append(np.dot(w.T,r))
        readout_array2.append(np.dot(w2.T,r))
        
        dx_array.append(dx[:10])
    
    
        if t%100 ==0:
            print(int(t), 'ms')
            alternance=-alternance        
        if t%452==0:
            alternance=-alternance
        
        #update weights between t_min and t_max
        if t%delta_up ==0:
            if t<time_min:
                e_m=0
                wup=0
                e_p=0

            elif t>time_max:
                e_m = np.dot(w.T,r) - ts[t]
                wup=0
                e_p=0
            else:            
                if alternance ==1:
                    P = P - np.dot(np.dot(P,r), np.dot(r.T,P))/(1+float(np.dot(np.dot(r.T,P),r)))
                    e_m = np.dot(w.T,r) - ts[t]
                    w = w - e_m * np.dot(P,r)
                else:                
                    P2 = P2 - np.dot(np.dot(P2,r), np.dot(r.T,P2))/(1+float(np.dot(np.dot(r.T,P2),r)))
                    e_m2 = np.dot(w2.T,r) - ts2[t]
                    w2 = w2 - e_m2 * np.dot(P2,r)

                
                #wup=np.linalg.norm(e_m * np.dot(P,r))
                #w[indices_zero_w[:]]= 0
            e_p_array.append(float(e_p))
            e_m_array.append(float(e_m))
            #wup_array.append(float(wup))
            # wn_array.append(float(np.sqrt(np.dot(w.T,w))))
                    #update weights between t_min and t_max

    #plot
#    offset
    offset=np.array([(np.linspace(0,18,10)) for i in range(time_ms)])
    # plt.subplot(4,1,1)
    # mat_activity = np.squeeze(np.asarray(x_time_array))
    # plt.plot(time[:], mat_activity+offset)
    # plt.ylabel('membrane potential x')
    # plt.xlim([0,time_ms])
    # plt.axvline(time_min,c="black")
    # plt.axvline(time_max,c="black")
    plt.subplot2grid((5,1),(2,0),1,1)
    plt.plot(time[:],np.squeeze(readout_array[:]))
    plt.plot(time[:], ts[:])
    plt.ylabel('Readout')
    plt.xlim([0,time_ms])
    plt.ylim([-2,2])
    plt.xticks([])
    plt.axvline(time_min,c="black")
    plt.axvline(time_max,c="black")

    plt.subplot2grid((5,1),(0,0),2,1)
    mat_activity_r = np.squeeze(np.array(r_array))
    plt.plot(time[:], mat_activity_r+offset)
    plt.ylabel('Firing rate (Hz)')
    plt.xlim([0,time_ms])
    plt.xticks([])
    plt.axvline(time_min,c="black")
    plt.axvline(time_max,c="black")
    
    plt.subplot2grid((5,1),(3,0),1,1)
    plt.plot(np.linspace(0,time_ms,len(e_m_array)), e_m_array, label='error')
    #plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wup_array)*10, label='update ')
    #plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(e_p_array)*10, label='error_p')
    #plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wn_array), label='w norm')
    plt.xlabel('Time (ms)')
    plt.ylabel('error')
    plt.xlim([0,time_ms])
    plt.axvline(time_min,c="black")
    plt.axvline(time_max,c="black")
    plt.legend(loc='best')    
    
    plt.subplot2grid((5,1),(4,0),1,1)
    plt.plot(time[:],np.squeeze(readout_array2[:]))
    plt.plot(time[:], ts2[:])
    plt.ylabel('Readout')
    plt.xlim([0,time_ms])
    plt.ylim([-2,2])
    plt.xticks([])
    plt.axvline(time_min,c="black")
    plt.axvline(time_max,c="black")

#%% 6) weightsv2
def simul_weight_pca(time_ms, time_min, time_max, funct):
    #param for updating
    delta_up=8
    alpha=1
    P = np.identity(NGG)/alpha
    ts = funct(time_ms)
    #error
    e_m_array=list()
    e_m_array.append(0)    
    e_p_array=list()
    e_p_array.append(0)
    wup_array=list()
    wup_array.append(0)      
    w_array=list()
    w_array.append(0)    

    
    
    JGg, JGz, w, indices_zero_w = generate()
    #initializing the network activity radomly
    wn_array=list()
    wn_array.append(np.sqrt(np.dot(w.T,w)))
    w_array=list()
    w_array.append(w)  
    
    x=.5*np.random.normal(0, 1, NGG)[np.newaxis].T #make it vertical
    r=np.tanh(x)
    
    time=np.linspace(0,time_ms,time_ms) #10 000dt = 1000ms = 1s
    
    #stock for plot
    x_time_array = list()
    x_time_array.append(x[:10])
    
    readout= np.dot(w.T,r)
    readout_array = list()
    readout_array.append(readout)
    
    r_array = list()
    r_array.append(r)
    
    dx = (-x + np.dot(gGG*JGg, r) + gGz*JGz*readout)/tau
    dx_array = list()
    dx_array.append(dx[:10])
    
    #loop
    for t in range(len(time)-1):
        dx = (-x + np.dot(gGG*JGg, r) +gGz*JGz*readout)/tau
        x= x + dx*delta_t
        x_time_array.append(x[:10])
    
        r=np.tanh(x)
        r_array.append(r)
        readout= np.dot(w.T,r)
        readout_array.append(readout)
    
    
        if t%100 ==0:
            print(int(t), 'ms')
        
        #update weights between t_min and t_max
        if t%delta_up ==0:
            if t<time_min:
                e_m=0
                wup=0

            elif t>time_max:
                e_m = readout - ts[t]
                wup=0
            else:            
                P = P - np.dot(np.dot(P,r), np.dot(r.T,P))/(1+float(np.dot(np.dot(r.T,P),r)))
                e_m = readout - ts[t]
                w = w - e_m * np.dot(P,r)
                wup=np.linalg.norm(e_m * np.dot(P,r))
                w_array.append(w)  

                #w[indices_zero_w[:]]= 0
            e_m_array.append(float(e_m))
            wup_array.append(float(wup))
            wn_array.append(float(np.sqrt(np.dot(w.T,w))))

        

#     #plot
# #    offset
#     offset=np.array([(np.linspace(0,18,10)) for i in range(time_ms)])
#     # plt.subplot(4,1,1)
#     # mat_activity = np.squeeze(np.asarray(x_time_array))
#     # plt.plot(time[:], mat_activity+offset)
#     # plt.ylabel('membrane potential x')
#     # plt.xlim([0,time_ms])
#     # plt.axvline(time_min,c="black")
#     # plt.axvline(time_max,c="black")
#     plt.subplot2grid((4,1),(2,0),1,1)
#     plt.plot(time[:],np.squeeze(readout_array[:]))
#     plt.plot(time[:], ts[:])
#     plt.ylabel('Readout')
#     plt.xlim([0,time_ms])
#     plt.ylim([-2,2])
#     plt.xticks([])
#     plt.axvline(time_min,c="black")
#     plt.axvline(time_max,c="black")

#     # plt.subplot2grid((4,1),(0,0),2,1)
#     # mat_activity_r = np.squeeze(np.array(r_array[10,:]))
#     # plt.plot(time[:], mat_activity_r+offset)
#     # plt.ylabel('Firing rate (Hz)')
#     # plt.xlim([0,time_ms])
#     # plt.xticks([])
#     # plt.axvline(time_min,c="black")
#     # plt.axvline(time_max,c="black")
    
#     plt.subplot2grid((4,1),(3,0),1,1)
#     plt.plot(np.linspace(0,time_ms,len(e_m_array)), e_m_array, label='error')
#     plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wup_array)*10, label='update ')
#     #plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(e_p_array)*10, label='error_p')
#     plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(wn_array), label='w norm')
#     plt.xlabel('Time (ms)')
#     plt.ylabel('error')
#     plt.xlim([0,time_ms])
#     plt.axvline(time_min,c="black")
#     plt.axvline(time_max,c="black")
#     plt.legend(loc='best')    

    return w_array, r_array

#%% PCA w
w_array, r_array =simul_weight_pca(5000,100,4000, f_per_pap)

n_comp=100
test=np.squeeze(w_array)
pca=PCA(n_components=n_comp)
pca_full =pca.fit(test)

pca_compo = pca_full.components_
pca_w =pca.fit(test).transform(test)

cumul=list()
for i in range(n_comp):
    cumul.append(sum(pca_full.explained_variance_ratio_[:i])*100)
plt.plot(cumul)

#%% PCA w several
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(2):
    w_array, r_array =simul_weight_pca(5000,100,4000, f_per_pap)
    w_ok =np.squeeze(np.array(w_array))
    pca_w =pca.fit(w_ok).transform(w_ok)
    ax.plot(pca_w[:,0],pca_w[:,1], pca_w[:,99], lw=4)#,c=np.linspace(0,100,len(pca_w[:,1])))



ax.set_ylabel('test')
ax.set_title('test2')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
# ax.set_xlim([-.2,.2])
# ax.set_ylim([-.2,.2])
# ax.set_zlim([-.1,.1])
plt.grid(True)
# ax.xaxis.set_ticklabels([])
# ax.xaxis.set_ticks_position('none')
# ax.yaxis.set_ticklabels([])
# ax.yaxis.set_ticks_position('none')
# ax.zaxis.set_ticklabels([])
# ax.zaxis.set_ticks_position('none')
plt.show()
    

#%% PCA activity post learning
#%% PCA on r
test=np.squeeze(r_array)
pca=PCA(n_components=3)
pca_test =pca.fit(test).transform(test)
#%% PCA on r better
w_5, r_5 =simul_weight_pca(6000,100,4000, f_per_pap)

last_w=w_5[-1]
n_comp=10
r_array=np.squeeze(np.array(r_5))
pca=PCA(n_components=n_comp)

pca_full =pca.fit(r_array[4000:6000,:])
pca_compo = pca_full.components_

projections = list()
proj2=list()
for i in range(10):
    projections.append(pca_compo[i]*r_array[4000:6000,:])
    proj2.append(np.dot(projections[i],last_w))
    plt.plot(proj2[i], alpha=.3)


plt.plot(np.sum(proj2[i],1), lw=4)
plt.plot(ts[4000:6000]/10, lw=4)


cumul=list()
for i in range(n_comp):
    cumul.append(sum(pca_full.explained_variance_ratio_[:i])*100)
plt.plot(cumul)


# #%% PCA on r better
# w_5, r_5 =simul_weight_pca(6000,100,4000, f_per_pap)

# last_w=w_5[-1]
# n_comp=10
# r_array=np.squeeze(np.array(r_5))
# pca=PCA(n_components=n_comp)
# # pca_full =pca.fit(r_array[4000:6000,:])
# # pca_compo = pca_full.components_


# pca_projections =pca.fit(r_array[4000:6000,:]).transform(r_array[4000:6000,:])

# plt.plot(pca_projections, alpha=.3)


# plt.plot(np.sum(pca_projections,1), lw=4)
# plt.plot(ts[4000:6000]*10, lw=4)

# plt.plot(np.dot(pca_projections[:,0],last_w), alpha=.3)
# cumul=list()
# for i in range(n_comp):
#     cumul.append(sum(pca_full.explained_variance_ratio_[:i])*100)
# plt.plot(cumul)
#%% 6) weights_for g test
def simul_weight_g(time_ms, time_min, time_max, funct, gGG):
    #param for updating
    delta_up=10
    alpha=1
    P = np.identity(NGG)/alpha
    ts = funct(time_ms)
    #error
    e_m_array=list()
    e_p_array=list()
    wup_array=list()

    
    
    JGg, JGz, w, indices_zero_w = generate()
    w = np.zeros(NGG)[np.newaxis].T
    #initializing the network activity radomly
    wn_array=list()
    x=.5*np.random.normal(0, 1, NGG)[np.newaxis].T #make it vertical
    r=np.tanh(x)
    plt.hist(r)
    
    time=np.linspace(0,time_ms,time_ms) #10 000dt = 1000ms = 1s
    
    #stock for plot
    x_time_array = list()
    
    readout= np.dot(w.T,r)
    readout_array = list()
    
    r_array = list()
    
    dx = (-x + np.dot(gGG*JGg, r) + gGz*JGz*readout)/tau
    dx_array = list()
    
    #creating sliding window mean error
    sl_RMS = 10
    t=0
    #loop
    while sl_RMS >.01 and t<time_max:
        t+=1
        dx = (-x + np.dot(gGG*JGg, r) +gGz*JGz*readout)/tau
        #dx_array.append(dx[:10])
        
        x= x + dx*delta_t
        #x_time_array.append(x[:10])
    
        r=np.tanh(x)
        r_array.append(r[:10])
        readout= np.dot(w.T,r)
        readout_array.append(readout)
        
        
    
    
        if t%100 ==0:
            print(int(t), 'ms', sl_RMS)
        
        #update weights between t_min and t_max
        if t%delta_up ==0:
            if t<time_min:
                e_m=readout - ts[t]
                wup=0
                #e_p=0

            elif t>time_max:
                e_m = readout - ts[t]
                wup=0
                #e_p=0
            else:            

                P = P - np.dot(np.dot(P,r), np.dot(r.T,P))/(1+float(np.dot(np.dot(r.T,P),r)))
                #e_p = np.dot(w.T,r) - ts[t]
                e_m = readout - ts[t]
                w = w - e_m * np.dot(P,r)
                wup=np.linalg.norm(e_m * np.dot(P,r))
                #w[indices_zero_w[:]]= 0
                sl_RMS = np.mean(np.abs(e_m_array[-20:]))
                #print(sl_RMS)
                
            #e_p_array.append(float(e_p))
            e_m_array.append(float(e_m))
            wup_array.append(float(wup))
            wn_array.append(float(np.sqrt(np.dot(w.T,w))))
            
    e_m_RMS_array=list()
    for t2 in (np.array(range(1000))+t):
        dx = (-x + np.dot(gGG*JGg, r) +gGz*JGz*readout)/tau
        #dx_array.append(dx[:10])

        x= x + dx*delta_t
        #x_time_array.append(x[:10])
    
        r=np.tanh(x)
        r_array.append(r[:10])
        
        readout= np.dot(w.T,r)
        readout_array.append(readout)
        e_m=readout - ts[t2]
        e_m_RMS_array.append(float(e_m))

        
        if t2%delta_up ==0: 
            e_m=readout - ts[t2]
            wup=0
            e_m_array.append(float(e_m))
            wup_array.append(float(wup))
            wn_array.append(float(np.sqrt(np.dot(w.T,w))))


    #plot
#    offset
    offset=np.array([(np.linspace(0,18,10)) for i in range(t+1000)])
    # plt.subplot(4,1,1)
    # mat_activity = np.squeeze(np.asarray(x_time_array))
    # plt.plot(time[:], mat_activity+offset)
    # plt.ylabel('membrane potential x')
    # plt.xlim([0,time_ms])
    # plt.axvline(time_min,c="black")
    # plt.axvline(time_max,c="black")
    plt.subplot2grid((4,1),(2,0),1,1)
    plt.plot(np.squeeze(readout_array[:]))
    plt.plot(ts[:])
    plt.ylabel('Readout')
    plt.xlim([0,t2])
    plt.ylim([-2,2])
    plt.xticks([])
    plt.axvline(time_min,c="black")
    plt.axvline(t,c="black")

    plt.subplot2grid((4,1),(0,0),2,1)
    mat_activity_r = np.squeeze(np.array(r_array))
    plt.plot(mat_activity_r+offset)
    plt.ylabel('Firing rate (Hz)')
    plt.xlim([0,t2])
    plt.xticks([])
    plt.axvline(time_min,c="black")
    plt.axvline(t,c="black")
    
    plt.subplot2grid((4,1),(3,0),1,1)
    plt.plot(np.linspace(0,t+1000,len(e_m_array)), e_m_array, label='error')
    plt.plot(np.linspace(0,t+1000,len(e_m_array)), np.array(wup_array)*10, label='update ')
    #plt.plot(np.linspace(0,time_ms,len(e_m_array)), np.array(e_p_array)*10, label='error_p')
    plt.plot(np.linspace(0,t+1000,len(e_m_array)), np.array(wn_array), label='w norm')
    plt.xlabel('Time (ms)')
    plt.ylabel('error')
    plt.xlim([0,t2])
    plt.axvline(time_min,c="black")
    plt.axvline(t,c="black")
    plt.legend(loc='best')    
    
    RMS = np.sqrt(sum(np.array(e_m_RMS_array)*np.array(e_m_RMS_array)))
    wn=np.sqrt(np.dot(w.T,w))
    last_error=sl_RMS
    return wn, RMS, t, last_error

#%%run many simulations with different gs
#condition to consider that learned = mean absolute error measured 
# only at the time of updating = when probably highest
# so this mean of the last 20 updates <0.01
wn_list=list()
RMS_list=list()
t_list = list()
last_error_array=list()

for gGG in [.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.56,1.6]:
    for k in range(10):
        try:
            wn, RMS,t, l_e =simul_weight_g(41400,400,40400, f_per_pap, gGG)
        except:
            wn, RMS,t = 'Na','Na','Na'
        wn_list.append(wn)
        RMS_list.append(RMS)
        t_list.append(t)
        last_error_array.append(l_e)

#%%
import pandas as pd
dtf=pd.read_csv('gGG_sim.csv')
gGG_l=np.repeat([.8,.9,1,1.1,1.2,1.3,1.4,1.5,1.56,1.6], 10)
l_time = (np.array(t_list)-400)/400
wn_list=np.squeeze(np.array(wn_list))

dtf = {'w':wn_list, 'RMS error':RMS_list, '# periods trained':l_time, 'last error':last_error_array, 'gGG':gGG_l }
dtf = pd.DataFrame(dtf)
dtf.to_csv('gGG_sim.csv',index=False)    
dtf_less=dtf.loc[10:]
dtf_less.boxplot(column='# periods trained',by='gGG')
dtf_less.boxplot(column='RMS error',by='gGG')
dtf_less.boxplot(column='w',by='gGG')

plt.figure()
dtf.plot.box()
plt.title('test')
bp = dtf_less.boxplot(by='gGG')
bp.title('test')
