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

os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
gList = [[121,-121],
         [-121,121]]
VintMap = {} 

for i in range(0,len(gList)):
    gi = gList[i]
    for j in range(0,len(gi)):
        VintMap[(i,j)] = interaction(gi[j])

L = [] 

L.append(schrodinger.SchrodingerSolver(V=SHO_2D_V,VintMap=VintMap,gamma=.5,params=[.1,1,.6,gList[0][0]],dim=100,xlim=20,maxitr=1000,activation=en.gauss,method=None,cutoff = 10**(-9),hiddenlayersize=100,testname = 'test', subtestname = 'subtest',Nstates=2))
        

for trial in L:
    trial.run()
