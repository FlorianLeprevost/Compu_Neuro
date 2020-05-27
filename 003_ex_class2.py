# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:37:33 2020

@author: install
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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



#%% a)

US = [1 for i in range(50)]
CS = [1 for i in range(25)] + [0 for i in range(25)]

US= np.asarray(US)
CS= np.asarray(CS)
index= range(50)

plt.plot(index, US, label= 'US')
plt.scatter(index, CS, label= 'CS', color = "orange")
plt.legend()#loc='best')
plt.xlabel(r'trial n°')
#plt.ylabel(r'presence/absence')
#plt.savefig('fig2_report1.png', dpi=600)
#plt.clf


#%  b)

LR=.1
w = 0
W_array=[]
for trial in index:
    w  = w + LR*(CS[trial] - w*US[trial])*US[trial]
    W_array = np.append(W_array, w)
    
plt.scatter(index, W_array, label = r"$w$", color="green")
plt.plot(index, W_array, color="green")

plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')
plt.savefig('fig2_report2.png', dpi=600)
plt.clf

    
#%%  c)
LR=.05
w = 0
W_array=[]
for trial in index:
    w  = w + LR*(CS[trial] - w*US[trial])*US[trial]
    W_array = np.append(W_array, w)
    
plt.scatter(index, W_array, label= r'$\epsilon = .05$')
plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')
##

LR=.1
w = 0
W_array=[]
for trial in index:
    w  = w + LR*(CS[trial] - w*US[trial])*US[trial]
    W_array = np.append(W_array, w)
    
plt.scatter(index, W_array, label= r'$\epsilon = .1$')
plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')

##

LR=.2
w = 0
W_array=[]
for trial in index:
    w  = w + LR*(CS[trial] - w*US[trial])*US[trial]
    W_array = np.append(W_array, w)
    
plt.scatter(index, W_array, label= r'$\epsilon = .2$')
plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')


plt.savefig('fig2_report3.png', dpi=600)
plt.clf

    
#%% d)
#US = np.random.randint(0,1)
#CS = np.random.randint(0,1)
np.random.seed(2020)

US = [1 for i in range(50)]
proba = [np.random.rand() for i in range(50)]
CS= np.asarray(proba)<.4
sum(CS>.5)

plt.plot(index, US, label= 'Unconditionned stimulus')
plt.scatter(index, CS, label= 'Conditionned stimulus', color= "orange")
plt.legend()#loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'presence/absence')
axes = plt.gca()
#axes.xaxis.set_ticks([-10,0,10])
axes.yaxis.set_ticks([0,1])


#plt.savefig('fig2_report4.png', dpi=600)
#plt.clf

#%

LR=.1
w = 0
W_array=[]
for trial in index:
    w  = w + LR*(CS[trial] - w*US[trial])*US[trial]
    W_array = np.append(W_array, w)
    
plt.plot(index, W_array, label="internal estimate w", color="green")
plt.scatter(index, W_array, color="green")
plt.legend(loc='upper right', bbox_to_anchor=(.65,.9))
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')
plt.savefig('fig2_report5.png', dpi=600)
plt.clf


#%% e)
US1 = [1 for i in range(50)]
US2 = [0 for i in range(25)] + [1 for i in range(25)]
CS = [1 for i in range(50)]

US1= np.asarray(US1)
US2= np.asarray(US2)
CS= np.asarray(CS)

LR=.1
w1 = 0
w2 = 0
W1_array=[]
W2_array=[]
PE_array=[]

for trial in index:
    v = w1*US1[trial] + w2*US2[trial]
    PE = CS[trial] - v
    PE_array = np.append(PE_array, PE)
    
    w1  = w1 + LR*PE*US1[trial]
    W1_array = np.append(W1_array, w1)
    w2  = w2 + LR*PE*US2[trial]
    W2_array = np.append(W2_array, w2)

plt.plot(index, W1_array, label=r"$w1$")
plt.plot(index, W2_array, label=r"$w2$")
plt.plot(index, PE_array, label=r'$\delta = r - v$')


plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')
plt.savefig('fig2_report6.png', dpi=600)
plt.clf

#%% f)
US1 = [1 for i in range(50)]
US2 = [1 for i in range(50)]
CS = [1 for i in range(50)]

US1= np.asarray(US1)
US2= np.asarray(US2)
CS= np.asarray(CS)

LR1=.1
LR2=.2
w1 = 0
w2 = 0
W1_array=[]
W2_array=[]
PE_array=[]

for trial in index:
    v = w1*US1[trial] + w2*US2[trial]
    PE = CS[trial] - v
    PE_array = np.append(PE_array, PE)
    
    w1  = w1 + LR1*PE*US1[trial]
    W1_array = np.append(W1_array, w1)
    w2  = w2 + LR2*PE*US2[trial]
    W2_array = np.append(W2_array, w2)

plt.plot(index, W1_array, label=r"$w1$ (with  $\epsilon = .1$)")
plt.plot(index, W2_array, label=r"$w2$ (with  $\epsilon = .2$)")
plt.plot(index, PE_array, label=r'$\delta = r - v$')


plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')
plt.savefig('fig2_report7.png', dpi=600)
plt.clf

#%% g)
US1 = [1 for i in range(50)]
US2 = [0 for i in range(25)] + [1 for i in range(25)]
CS = [1 for i in range(25)] + [0 for i in range(25)]

US1= np.asarray(US1)
US2= np.asarray(US2)
CS= np.asarray(CS)

LR=.1
w1 = 0
w2 = 0
W1_array=[]
W2_array=[]
PE_array=[]

for trial in index:
    v = w1*US1[trial] + w2*US2[trial]
    PE = CS[trial] - v
    PE_array = np.append(PE_array, PE)
    
    w1  = w1 + LR*PE*US1[trial]
    W1_array = np.append(W1_array, w1)
    w2  = w2 + LR*PE*US2[trial]
    W2_array = np.append(W2_array, w2)

plt.plot(index, W1_array, label=r"$w1$")
plt.plot(index, W2_array, label=r"$w2$")
plt.plot(index, PE_array, label=r'$\delta = r - v$')


plt.legend(loc='best')
plt.xlabel(r'trial n°')
plt.ylabel(r'Internal estimate')
plt.savefig('fig2_report8.png', dpi=600)
plt.clf

#%% Exercice 2 

#%% a)

beta0 = 0
beta10 = 10
d= np.linspace (-10,10,200)
z1 = 1/(1 + np.exp(d))
z0 = 1/(1 + np.exp(d*beta0))
z10 = 1/(1 + np.exp(d*beta10))


plt.plot(d,z0, label= r'$\beta = 0$')
plt.plot(d,z1, label= r'$\beta = 1$')
plt.plot(d,z10, label= r'$\beta = 10$')

plt.legend(loc='best')

plt.xlabel(r'$m_y - m_b$')
plt.ylabel(r'$p_b$')
axes = plt.gca()
axes.xaxis.set_ticks([-10,0,10])
axes.yaxis.set_ticks([0,.5,1])

plt.savefig('fig2_report9.png', dpi=600)
plt.clf()


#%%

d=1
dim =10
d2 = -1
beta= np.linspace (-10,10,200)
z = 1/(1 + np.exp(d*beta))
z2 = 1/(1 + np.exp(d2*beta))



plt.plot(beta,z, label= r'$m_y - m_b = 1$')
plt.plot(beta,z2, label= r'$m_y - m_b = -1$')


plt.legend(loc='best')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$p_b$')
axes = plt.gca()
axes.xaxis.set_ticks([-10,0,10])
axes.yaxis.set_ticks([0,.5,1])

plt.savefig('fig2_report10.png', dpi=600)
plt.clf()

#%% b)
#fig_width = 12# width in inches
#fig_height = 5  # height in inches
#fig_size =  [fig_width,fig_height]
#plt.rcParams['figure.figsize'] = fig_size
#plt.rcParams['figure.autolayout'] = True
#



my=5
mb= 0
beta = .8
proba_b = 1/(1 + np.exp(beta*(my-mb)))
x=range(200)

np.random.seed(2020)
proba_b_list = [np.random.rand() for i in range(200)]
choice_b = np.asarray(proba_b_list)< proba_b
sum(choice_b)

#color
viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
yellow = np.array([255/256, 255/256, 0/256, 1])
blue = np.array([0/256, 0/256, 256/256, 1])
newcolors[:128,:] = yellow
newcolors[128:256,:] = blue
newcmp = ListedColormap(newcolors)

plt.subplot(211)
plt.scatter(x,choice_b, c=choice_b, cmap = newcmp, s=10)

plt.ylabel('Choice\n'
           r'$\beta = .8$')
plt.yticks([1,0], ["blue", "yellow"])
plt.xticks([], [])

#########
my=5
mb= 0
beta = 0
proba_b = 1/(1 + np.exp(beta*(my-mb)))
x=range(200)

np.random.seed(2020)
proba_b_list = [np.random.rand() for i in range(200)]
choice_b = np.asarray(proba_b_list)< proba_b
sum(choice_b)



plt.subplot(212)
plt.scatter(x,choice_b, c=choice_b, cmap = newcmp, s=10)
#plt.vlines(100,0,1)



plt.xlabel(r'Time (in flower)')
plt.ylabel('Choice\n'
           r'$\beta = 0$')
plt.yticks([1,0], ["blue", "yellow"])


plt.savefig('fig2_report11.png', dpi=600)
plt.clf()

#%% c)
r_b = [8 for i in range(100)] + [2 for i in range(100)]
r_y = [2 for i in range(100)] + [8 for i in range(100)]
index= range(200)

np.random.seed(2020)
r_b= np.asarray(r_b)
r_y= np.asarray(r_y)
beta = 1
LR=.2


my = 5
mb = 0
my_array=[]
mb_array=[]
choice_array = []

cum_reward = 0

for trial in index:
    proba_b = 1/(1 + np.exp(beta*(my-mb)))
    choice =  np.random.rand() < proba_b            #if True (1) = blue
    choice_array = np.append(choice_array, choice)
    if choice == True:
         mb = mb + LR*(r_b[trial]- mb)
         cum_reward += r_b[trial]
    else:
         my = my + LR*(r_y[trial] - my)
         cum_reward += r_y[trial]

    mb_array = np.append(mb_array, mb)
    my_array = np.append(my_array, my)


###### estimate plot
plt.subplot2grid((4,4), (1,0), colspan=4, rowspan=3)
plt.plot(index, mb_array, label=r"$m_b$", color="blue")
plt.plot(index, my_array, label=r"$m_y$", color="yellow")

plt.annotate(" day 1\n"
             "$r_b=8$\n$r_y=2$", (35,4))
plt.annotate(" day 2\n"
             "$r_b=2$\n$r_y=8$", (140,4))
plt.vlines(100,1,8)

plt.annotate("Cumulative reward ="+str(cum_reward), (35,.3))

plt.legend(loc='best')
plt.xlabel(r'Time in flower')
plt.ylabel(r'Internal estimate')

##### choice plot
plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=1)
plt.scatter(index, choice_array, c=choice_array, cmap = newcmp, s=7)
plt.vlines(100,0,1)

###


plt.ylabel('Choice')
plt.yticks([1,0], ["b", "y"])
plt.xticks([], [])

plt.savefig('fig2_report12.png', dpi=600)
plt.clf

#%%
#%%
#%% Exercice 3
np.random.seed(2020)

m_a = 1
m_b = .95
dt = .1
sigma=.5


accum = 0
accum_array = []


ite = range(1,10000)
for i in ite:
    accum += dt*(m_a-m_b) + sigma*np.sqrt(dt)*np.random.normal(0,1)
    accum_array = np.append(accum_array, accum)


###grid
grid = plt.GridSpec(2, 4)

plt.subplot(grid[0:, 0:3])
#plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=2)
plt.plot(ite, accum_array, label=r"$m_b$", color="blue")
plt.xlabel("Time in ms")
plt.ylabel("Evidence")

plt.xticks([0,5000,10000], [0,500,1000])


####################### b)
m_a = 1
m_b = .95
dt = .1
sigma=.5
threshold = 10

accum_array = []

RT_A=[]
RT_B=[]

ite = range(1,10000)

for i in range(1000):
    accum = 0
    accum_array = []
    for i in ite:
        accum += dt*(m_a-m_b) + sigma*np.sqrt(dt)*np.random.normal(0,1.01)
        accum_array = np.append(accum_array, accum)
        if accum >= threshold:
            RT_A = np.append(RT_A, i/10 + 100 )
            break
        elif accum <= -threshold:
            RT_B = np.append(RT_B, i/10 + 100 )
            break
        

ax1 = plt.subplot(grid[0, 3])

plt.hist(RT_A)


plt.subplot(grid[1, 3], sharex = ax1)
plt.hist(RT_B)
plt.ylabel("Count")
plt.xlabel("Reaction Time (ms)")

plt.savefig('fig2_report13.png', dpi=600)
plt.clf

#%% c)
m_e = np.linspace(-.2,.2,8)

ite = range(5000)

arr_of_arr=np.array(ite)


plt.subplot(grid[0:, 0:3])

for i in m_e:
    accum = 0
    accum_array = []
    for j in ite:
        accum += dt*(i) + sigma*np.sqrt(dt)*np.random.normal(0,1)
        accum_array = np.append(accum_array, accum)
    arr_of_arr = np.vstack([arr_of_arr, accum_array])
    plt.plot(ite, accum_array, label=r"$m_e=$"+str(round(i, 2)))
plt.legend(loc='lower left', fontsize = 12, ncol=2)
plt.xlabel("Time in ms")
plt.ylabel("Evidence")

plt.xticks([0,2500,5000], [0,250,500])


#######
m_e = np.linspace(-.2,.2,100)
beta=2*threshold/(sigma*sigma)
proba_array=[]
proba_choice = 1/(1 + np.exp(beta*(m_e)))
   
plt.subplot(grid[0:, 3])
 
plt.plot(m_e,proba_choice)
plt.yticks([0,1], [0,1])
plt.xlabel(r"$m_e$")


plt.savefig('fig2_report14.png', dpi=600)
plt.clf


    









