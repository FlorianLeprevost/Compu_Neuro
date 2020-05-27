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

#%% 1) a)

w=.04
I= -2
f= lambda s : 50*(1+np.tanh(s))
x =np.linspace(-4, 4, 1000)

plt.plot(x, f(x))
plt.vlines(0, 0,100, linestyles='dashed')
plt.ylabel(r'$f(s)$')
plt.xlabel(r'$s$')

plt.savefig('exercice5fig1.png', dpi=600)
plt.clf
#%% b)
x =np.linspace(0,150, 1500)
dx = -x + f(w*x +I)


plt.plot(x, dx)
plt.hlines(0, 0,150)

plt.xlim([0,150])
plt.ylabel(r'$\dot{x}(t)$')
plt.xlabel(r'$x(t)$')

plt.scatter([2.1,97.8], [0,0], s= 150, zorder=10)
plt.scatter(50, 0, s=150, facecolors='white', edgecolors='b', zorder=10, alpha=1)
plt.arrow(43,0,-15,0, color='red', head_width= 2)
plt.arrow(58,0,15,0, color='red', head_width= 2)
plt.arrow(140,0,-15,0, color='red', head_width= 2)

plt.savefig('exercice5fig2.png', dpi=600)
plt.clf
#%% c) EUler x = x + x'*dt BUT as x'= -x THEN x=x'dt
plt.rcParams['figure.figsize'] = [12,8]
fig, axes2d = plt.subplots(nrows=3, ncols=3,
                           sharex=True, sharey=True)
#fig.text(0.5, 0.04, 'Time (s)', ha='center')
#fig.text(0.04, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')

x0=[49,50,51]
dt=.1
plt.subplot(2,2,1)
colors=['orange', 'green', 'royalblue']
for j in range(3):
    xt=x0[j]
    x_array = [xt]
    for i in range(100):
        xt += (-xt + f(w*xt+I)) *dt
        x_array.append(xt)
    plt.plot(np.linspace(0,10, 101),x_array, label='$x_0=$'+ str(x0[j]), color=colors[j])
plt.legend(loc='best')
#plt.ylabel('Firing rate (Hz)')
#plt.xlabel('Time (s)')
    
    #reason =because 50 = unstable\repeller --> go to the next stable point or/infinity if there is not
#plt.savefig('exercice5fig3.png', dpi=600)
plt.hlines(2.1, 0,100, linestyles='dashed')
plt.hlines(97.8, 0,100, linestyles='dashed')
plt.xlim([0,10])




#% d) 
x0=[49,50,51]
dt=.1
sigma=.5
colors=['orange', 'green', 'royalblue']
plt.subplot(2,2,2)
for j in range(3):
    for k in range(10):
        xt=x0[j]
        x_array = [xt]
        for i in range(100):
            xt += (-xt + f(w*xt+I))*dt + sigma*np.random.randn()*np.sqrt(dt)
            x_array.append(xt)
        if k==0:
            plt.plot(np.linspace(0,10, 101),x_array, color=colors[j], label='$x_0=$'+ str(x0[j]))
        else:
            plt.plot(np.linspace(0,10, 101),x_array, color=colors[j], alpha=i)
#plt.legend(loc='center right', fontsize=15)
#plt.xlabel('Time (s)')
#plt.ylabel('Firing rate (Hz)')
plt.hlines(2.1, 0,100, linestyles='dashed')
plt.hlines(97.8, 0,100, linestyles='dashed')
plt.xlim([0,10])
           
x0=[49,50,51]
dt=.1
sigma=5
colors=['orange', 'green', 'royalblue']
plt.subplot(2,2,3)
for j in range(3):
    for k in range(10):
        xt=x0[j]
        x_array = [xt]
        for i in range(100):
            xt += (-xt + f(w*xt+I))*dt + sigma*np.random.randn()*np.sqrt(dt)
            x_array.append(xt)
        if k==0:
            plt.plot(np.linspace(0,10, 101),x_array, color=colors[j], label='$x_0=$'+ str(x0[j]))
        else:
            plt.plot(np.linspace(0,10, 101),x_array, color=colors[j], alpha=i)
#plt.legend(loc='center right')
#plt.xlabel('Time (s)')
#plt.ylabel('Firing rate (Hz)')
plt.hlines(2.1, 0,100, linestyles='dashed')
plt.hlines(97.8, 0,100, linestyles='dashed')
plt.xlim([0,10])

x0=[49,50,51]
dt=.1
sigma=80
colors=['orange', 'green', 'royalblue']
plt.subplot(2,2,4)
for j in range(3):
    for k in range(10):
        xt=x0[j]
        x_array = [xt]
        for i in range(100):
            xt += (-xt + f(w*xt+I))*dt + sigma*np.random.randn()*np.sqrt(dt)
            x_array.append(xt)
        if k==0:
            plt.plot(np.linspace(0,10, 101),x_array, color=colors[j], label='$x_0=$'+ str(x0[j]))
        else:
            plt.plot(np.linspace(0,10, 101),x_array, color=colors[j], alpha=i)
#plt.legend(loc='center right')
#plt.xlabel('Time (s)')
#plt.ylabel('Firing rate (Hz)')
plt.hlines(2.1, 0,100, linestyles='dashed')
plt.hlines(97.8, 0,100, linestyles='dashed')
plt.xlim([0,10])





fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time (s)')
plt.ylabel('Firing rate (Hz)')

plt.savefig('exercice5fig3.png', dpi=600)
plt.clf 
#%% 2) a)
plt.rcParams['figure.figsize'] = fig_size



w=-.1
I= 5
x1 =np.linspace(-10,120, 100)
x2 = np.linspace(-10,120, 100)
x1n=f(w*x2 + I)
x2n=f(w*x1 + I)

plt.plot(x1, x1n, label='$\dot{x_2}=0$')
plt.plot(x2n, x2, label='$\dot{x_1}=0$')
plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.legend(loc='best')
plt.scatter([0,50,100], [100,50, 0], s= 100, zorder=10, color='black')


plt.savefig('exercice5fig4.png', dpi=600)
plt.clf 
#%% b)
plt.rcParams['figure.figsize'] = [12,6]



x0_1=[0,10,30]
x0_2=[0,0,40]
dt=.1
colors=['orange', 'green', 'royalblue']
legend2=['$x_{1,0}=x_{2,0}=0$','$x_{1,0}= 10 ; x_{2,0}=0$', '$x_{1,0}= 30 ; x_{2,0}=40$']

for j in range(len(x0_1)):
    xt1=x0_1[j]
    xt2=x0_2[j]
    x_array1 = [xt1]
    x_array2 = [xt2]
    plt.subplot(1,2,1)
    for i in range(50):
        dxt1 =  f(w*xt2+I)
        dxt2 =  f(w*xt1+I)
        xt1 += (-xt1 + dxt1) *dt
        xt2 += (-xt2 + dxt2) *dt
        x_array1.append(xt1)
        x_array2.append(xt2)
        
    plt.plot(np.linspace(0,5, 51),x_array1, label=legend2[j], color=colors[j])
    plt.plot(np.linspace(0,5, 51),x_array2,  color=colors[j], alpha=.5)
    plt.hlines(100, 0,5, linestyles='dashed')
    plt.hlines(0, 0,5, linestyles='dashed')
    plt.legend(loc='best', prop={'size': 18})    
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (Hz)')
    plt.xlim([0,5])
    
    # plotting on 2d space
    plt.subplot(1,2,2)
    plt.plot(x_array1, x_array2, color=colors[j])#, label=legend2[j])
    plt.scatter(x_array1[0], x_array2[0], s=20, color=colors[j])
    plt.subplot(1,2,1)


plt.subplot(1,2,2)
w=-.1
I= 5
x1 =np.linspace(-10,120, 100)
x2 = np.linspace(-10,120, 100)
x1n=f(w*x2 + I)
x2n=f(w*x1 + I)

plt.plot(x1, x1n, label='$\dot{x_2}=0$', color='black', lw=2)
plt.plot(x2n, x2, label='$\dot{x_1}=0$', color='grey', lw=2)

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.legend(loc='best', prop={'size': 18})
plt.scatter([50,100,0], [50,0, 100], s= 100, zorder=10, color=colors)




plt.savefig('exercice5fig5.png', dpi=600)
plt.clf 
#%% b.2 showing difference


np.random.seed(2020)

x0_1=[0,10,0,20,70] + [np.random.randint(0,100) for i in range(30)]
x0_2=[0,0, 10,20,70] + [np.random.randint(0,100) for i in range(30)]
dt=.1
legendscheat = ['$x_{1,0}=x_{2,0}$','$x_{1,0}>x_{2,0}$', '$x_{1,0}<x_{2,0}$']
colors=['orange', 'green', 'royalblue']

plt.subplot(1,2,1)
for j in range(len(x0_1)):
    xt1=x0_1[j]
    xt2=x0_2[j]
    x_array1 = [xt1]
    x_array2 = [xt2]
    x_array3 = [xt1-xt2]
    for i in range(50):
        dxt1 =  f(w*xt2+I)
        dxt2 =  f(w*xt1+I)
        xt1 += (-xt1 + dxt1) *dt
        xt2 += (-xt2 + dxt2) *dt
        x_array1.append(xt1)
        x_array2.append(xt2)
        x_array3.append(xt1-xt2)
        
    if j <3:
        plt.plot(np.linspace(0,5, 51),x_array3, label=legendscheat[j], color=colors[j])
        
        plt.subplot(1,2,2)
        plt.plot(x_array1, x_array2, color=colors[j])#, label=legendscheat[j])
        plt.scatter(x_array1[0], x_array2[0], s=20, color=colors[j])
        plt.subplot(1,2,1)
    if j>=3 and x0_1[j] < x0_2[j]:
        plt.plot(np.linspace(0,5, 51),x_array3, color=colors[2])
        
        plt.subplot(1,2,2)
        plt.plot(x_array1, x_array2, color=colors[2])
        plt.scatter(x_array1[0], x_array2[0], s=20, color=colors[2])
        plt.subplot(1,2,1)
        
    if j>=3 and x0_1[j] > x0_2[j]:
        plt.plot(np.linspace(0,5, 51),x_array3, color=colors[1])
        
        plt.subplot(1,2,2)
        plt.plot(x_array1, x_array2, color=colors[1])
        plt.scatter(x_array1[0], x_array2[0], s=20, color=colors[1])
        plt.subplot(1,2,1)
    if j>=3 and x0_1[j] == x0_2[j]:
        plt.plot(np.linspace(0,5, 51),x_array3, color=colors[0])
        
        plt.subplot(1,2,2)
        plt.plot(x_array1, x_array2, color=colors[0])
        plt.scatter(x_array1[0], x_array2[0], s=20, color=colors[0])
        plt.subplot(1,2,1)
        




plt.hlines(100, 0,5, linestyles='dashed')
plt.hlines(-100, 0,5, linestyles='dashed')

plt.legend(loc='best', prop={'size': 18})
plt.xlabel('Time (s)')
plt.ylabel('Firing rate difference ($x_1 -x_2$)')
plt.xlim([0,5])


##
plt.subplot(1,2,2)
w=-.1
I= 5
x1 =np.linspace(-10,120, 100)
x2 = np.linspace(-10,120, 100)
x1n=f(w*x2 + I)
x2n=f(w*x1 + I)

plt.plot(x1, x1n, label='$\dot{x_2}=0$', color='black', lw=2)
plt.plot(x2n, x2, label='$\dot{x_1}=0$', color='grey', lw=2)
plt.ylabel('$x_2$')
plt.xlabel('$x_1$')

plt.legend(loc='best')
plt.scatter([50,100,0], [50,0, 100], s= 100, zorder=10, color=colors)
#plt.plot(x_array1, x_array2)
plt.grid()

plt.savefig('exercice5fig6.png', dpi=600)
plt.clf 
#%% c)
plt.rcParams['figure.figsize'] = fig_size


x0=np.array([[0,0], [10,0], [30,40]])

dt=.1
W=np.mat('0 -0.1; -0.1 0')

colors=['orange', 'green', 'royalblue']
legend2=['$x_{1,0}=x_{2,0}=0$','$x_{1,0}= 10 ; x_{2,0}=0$', '$x_{1,0}= 30 ; x_{2,0}=40$']

for j in range(len(x0)):
    xt=x0[j]

    x_array = np.array(np.array(xt))
    plt.figure(1)
    for i in range(50):
        dxt= np.squeeze(f(np.array(np.dot(W,xt))+I))
        xt = xt + np.dot((-xt + dxt),dt)
        x_array = np.vstack([x_array, xt])
                            
    plt.plot(np.linspace(0,5, 51),x_array[:,0], label=legend2[j], color=colors[j])
    plt.plot(np.linspace(0,5, 51),x_array[:,1], color=colors[j], alpha=.5)
    plt.hlines(100, 0,5, linestyles='dashed')
    plt.hlines(0, 0,5, linestyles='dashed')
    plt.legend(loc='center right')
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (Hz)')
    plt.xlim([0,5])
    
    # plotting on 2d space
    plt.figure(2)
    plt.plot(x_array[:,0], x_array[:,1], color=colors[j], label=legend2[j])
    plt.scatter(x_array[:,0][0],x_array[:,1][0], s=20, color=colors[j])
    plt.figure(1)


plt.figure(2)
w=-.1
I= 5
x1 =np.linspace(-10,120, 100)
x2 = np.linspace(-10,120, 100)
x1n=f(w*x2 + I)
x2n=f(w*x1 + I)

plt.plot(x1, x1n, label='$\dot{x_2}=0$', color='black', lw=2)
plt.plot(x2n, x2, label='$\dot{x_1}=0$', color='grey', lw=2)

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.legend(loc='best', prop={'size': 18})
plt.scatter([50,100,0], [50,0, 100], s= 100, zorder=10, color=colors)
plt.savefig('exercice5fig7png', dpi=600)
plt.clf 

#%% d)
X, Y=np.meshgrid(np.linspace(-20,120,13), np.linspace(-20,120,13))
U,V=np.meshgrid(np.linspace(-20,120,13), np.linspace(-20,120,13))
for i in range(13):
    for j in range(13):
        xt=np.array([X[i][j],Y[i][j]])
        for k in range(100):
            dxt= np.squeeze(f(np.array(np.dot(W,xt))+I))
            xt = xt + np.dot((-xt + dxt),dt)
        U[i][j]= xt[0] - X[i][j]
        V[i][j]= xt[1] - Y[i][j]

plt.streamplot(X, Y, U,V, color='black')


##
w=-.1
I= 5
x1 =np.linspace(-20,120, 100)
x2 = np.linspace(-20,120, 100)
x1n=f(w*x2 + I)
x2n=f(w*x1 + I)
plt.plot(x1, x1n, label='$\dot{x_2}=0$',  lw=2)
plt.plot(x2n, x2, label='$\dot{x_1}=0$',  lw=2)
plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.legend(loc='best', prop={'size': 18})
plt.scatter([50,100,0], [50,0, 100], s= 100, zorder=10, color='black')

plt.savefig('exercice5fig8.png', dpi=600)
plt.clf 

#%% 3)
n_cell=64
p=np.array([-1,-1,-1,-1,1,1,-1,-1,
           -1,-1,-1,-1,1,1,-1,-1,
           -1,-1,-1,-1,1,1,-1,-1,
            1,1,1,1,1,1,-1,-1,
            1,1,1,1,1,1,1,-1,
            1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1])
    




viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
beige = np.array([249/256, 228/256, 183/256, 1])
blue = np.array([135/256, 206/256, 235/256, 1])
newcolors[:128,:] = blue
newcolors[128:256,:] = beige
newcmp = ListedColormap(newcolors)


q=np.flip(p)

plt.subplot(1,2,1)
plt.imshow(p.reshape((8,8)), cmap=newcmp)
plt.xticks(np.arange(0.5, 8.5, 1), np.arange(1,9,1))
plt.yticks(np.arange(0.5, 8.5, 1), np.arange(1,9,1))
plt.grid()
plt.subplot(1,2,2)
plt.imshow(q.reshape((8,8)), cmap=newcmp)
plt.xticks(np.arange(0.5, 8.5, 1), np.arange(1,9,1))
plt.yticks(np.arange(0.5, 8.5, 1), np.arange(1,9,1))
plt.grid()

plt.savefig('exercice5fig9.png', dpi=600)
plt.clf 

#%% b)
res = []

W= np.array([np.array([p[i]*p[j]/n_cell for i in range(n_cell)]) for j in range(n_cell)])

for i in range(8):
    state_t1 = np.array([None for _ in range(n_cell)])
    state = [ np.random.randint(0,2) for i in range(n_cell)]
    state = np.array([x if x!=0 else -1 for x in state])
    plt.subplot(2,8,i+1)

    plt.imshow(state.reshape((8,8)), cmap=newcmp)
    plt.xticks([])
    plt.yticks([])
     
    while (state != state_t1).all():
        stoc    = state
        state   = np.sign(np.dot(state, W))
        state_t1=stoc
        
        plt.subplot(2,8,(i+9))
        plt.imshow(state.reshape((8,8)), cmap=newcmp)
        plt.xticks([])
        plt.yticks([])
    res.append(state)
    
plt.savefig('exercice5fig10.png', dpi=600)
plt.clf 
#%%c)


#
W= np.array([np.array([(p[i]*p[j] + q[i]*q[j])/n_cell for i in range(n_cell)]) for j in range(n_cell)])


for i in range(8):
    state_t1 = np.array([None for _ in range(n_cell)])
    state = [ np.random.randint(0,2) for i in range(n_cell)]
    state = np.array([x if x!=0 else -1 for x in state])
    plt.subplot(2,8,i+1)

    plt.imshow(state.reshape((8,8)), cmap=newcmp)
    plt.xticks([])
    plt.yticks([])
     
    while (state != state_t1).all():
        stoc    = state
        state   = np.sign(np.dot(state, W))
        state_t1=stoc
        
        plt.subplot(2,8,(i+9))
        plt.imshow(state.reshape((8,8)), cmap=newcmp)
        plt.xticks([])
        plt.yticks([])
    res.append(state)

plt.savefig('exercice5fig11.png', dpi=600)
plt.clf 
#%% c bis)

for i in range(8):
    q_m = np.array(q, copy=True)
    random_list = [np.random.randint(0,64) for i in range((i+1)*2)]
    q_m[random_list] = q_m[random_list]*-1
    
    state_t1 = np.array([None for _ in range(n_cell)])
    state = q_m
    
    plt.subplot(2,8,i+1)
    plt.imshow(state.reshape((8,8)), cmap=newcmp)
    plt.xticks([])
    plt.yticks([])

    while (state != state_t1).all():
        stoc    = state
        state   = np.sign(np.dot(state, W))
        state_t1=stoc
        plt.subplot(2,8,(i+9))
        plt.imshow(state.reshape((8,8)), cmap=newcmp)
        plt.xticks([])
        plt.yticks([])

plt.savefig('exercice5fig12.png', dpi=600)
plt.clf 

plt.clf 
#%% c ter)

for i in range(8):
    q_m = np.array(q, copy=True)
    random_list = [np.random.randint(0,64) for i in range(18+i*2)]
    q_m[random_list] = q_m[random_list]*-1
    
    state_t1 = np.array([None for _ in range(n_cell)])
    state = q_m
    
    plt.subplot(2,8,i+1)
    plt.imshow(state.reshape((8,8)), cmap=newcmp)
    plt.xticks([])
    plt.yticks([])

    while (state != state_t1).all():
        stoc    = state
        state   = np.sign(np.dot(state, W))
        state_t1=stoc
        plt.subplot(2,8,(i+9))
        plt.imshow(state.reshape((8,8)), cmap=newcmp)
        plt.xticks([])
        plt.yticks([])

plt.savefig('exercice5fig13.png', dpi=600)
plt.clf 
#%% d)
        
for i in range(10):
    state_t1 = np.array([None for _ in range(n_cell)])
    state = [ np.random.randint(0,2) for i in range(n_cell)]
    state = np.array([x if x!=0 else -1 for x in state])
    plt.matshow(state.reshape((8,8)), cmap=newcmp)
    plt.title('depart')

    while (state != state_t1).all():
        stoc    = state
        state   = f(np.dot(state, W))
        state_t1=stoc
        plt.matshow(state.reshape((8,8)), cmap=newcmp)
        plt.title('step')


    res.append(state)
    
plt.savefig('exercice5fig14.png', dpi=600)
plt.clf 



