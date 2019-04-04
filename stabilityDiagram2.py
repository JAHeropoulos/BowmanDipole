import schrodinger 
import numpy as np
import shutil as sh
import os
import multiprocessing as mp
import tensorflow as tf
import Energy as en
cwd = os.getcwd()
if not os.path.exists(cwd + '/Tests'):
    os.makedirs(cwd + '/Tests')

def SHO_2D_V(x,y):
    return .5*(x**2+y**2) 


def interaction(g):
    return lambda x,y: np.full(x.shape,g)


def start(p):
    p.run()

def makeGs(x): 
    gList = [[x,-x],
             [-x,x]]
    VintMap = {}

    for i in range(0,len(gList)):
        gi = gList[i]
        for j in range(0,len(gi)):
            VintMap[(i,j)] = interaction(gi[j])
    return VintMap

gs = np.linspace(0,100,50)
Gs = list(map(lambda x: makeGs(x),gs))
L = []
v = np.linspace(.1,1,10)

for i in range(0,len(v)):
    for j in range(0,len(gs)):
        test = schrodinger.SchrodingerSolver(V=SHO_2D_V,VintMap=Gs[j],gamma=.5,params=[v[i],1,.2,gs[j]],dim=100,xlim=20,maxitr=800,activation=en.gauss,cutoff = 10**(-7),hiddenlayersize=100,testname = 'stability_diagram2', subtestname = 'V'+str(i)+'G'+str(j),Nstates=2)
        test.run()

