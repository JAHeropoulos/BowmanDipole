import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil as sh
import os
import math as m
from scipy.special import erfc
import Energy as en
import time 
class SchrodingerSolver(object):
    def __init__(self,V,VintMap,gamma=.5,params=[],dim=256,xlim=5,maxitr=400,cutoff = 10**(-6),hiddenlayersize=100,activation = tf.nn.tanh,method=None,layers=((2048,10),(10,1024),(1024,1024),(1024,1)),testname = 'test',subtestname = 'subtest',Nstates=1):
        self.cutoff = cutoff
        self.Elist = []
        self.HLS = hiddenlayersize
        self.testname = testname
        self.subtestname = subtestname
        print('Name: ' + subtestname)
        self.path = os.getcwd()+'/'+'Tests'+'/'+self.testname
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.path = self.path + '/' + self.subtestname
        if os.path.exists(self.path):
            sh.rmtree(self.path)
            os.makedirs(self.path)
        else:
            os.makedirs(self.path)
        self.activation = activation
        tf.set_random_seed(np.random.randint(1000))
        self.maxitr = maxitr
        self.method = method
        self.config = tf.ConfigProto(log_device_placement=False)
        

        self.config.intra_op_parallelism_threads = 16
        self.config.inter_op_parallelism_threads = 16
        self.sess = tf.Session(config=self.config)
        self.plotdim = dim
        self.ydim = int(np.square(dim))
        self.Vfun = V
        self.xlim = xlim
        self.x = np.linspace(-xlim,xlim,self.plotdim)
        self.d = self.x[1]-self.x[0]
        self.dk = 1/self.x.shape[0]/self.d
        self.y = np.linspace(-xlim,xlim,self.plotdim)
        self.kx = np.linspace(-self.x.shape[0]/2,self.x.shape[0]/2-1,self.plotdim)*2*np.pi*self.dk
        self.ky = np.linspace(-self.y.shape[0]/2,self.y.shape[0]/2-1,self.plotdim)*2*np.pi*self.dk
        self.xx,self.yy = np.meshgrid(self.x,self.y)
        self.kxx,self.kyy = np.meshgrid(self.kx,self.ky)
        self.Nstates = Nstates 
        self.V = V(self.xx,self.yy)
        self.VintMap = {}
        for Vkey in VintMap.keys():
            tmp = VintMap[Vkey](self.xx,self.yy)*en.G(np.sqrt((self.kxx**2+self.kyy**2)/2.0)) 
            
            self.VintMap[Vkey]=[tf.constant(np.real(tmp.reshape(self.ydim,1)),dtype=tf.float32),tf.constant(np.imag(tmp.reshape(self.ydim,1)),dtype=tf.float32)]
        
        self.xo = 1*self.xx
        self.yo = 1*self.yy
        self.xx = np.tile(self.xx.reshape(self.ydim,1),(self.HLS,1,1))
        self.yy = np.tile(self.yy.reshape(self.ydim,1),(self.HLS,1,1))
        #self.xx = self.xx.reshape(self.ydim,1)
        #self.yy = self.yy.reshape(self.ydim,1)
        self.V = tf.constant(self.V.reshape(self.ydim,1),dtype=tf.float32)
        self.V = [self.V,tf.zeros([self.ydim,1],tf.float32)]

        self.inputkx = tf.constant(self.kxx.reshape(self.ydim,1),dtype=tf.float32)

        self.inputky = tf.constant(self.kyy.reshape(self.ydim,1),dtype=tf.float32)

        
        self.RFW,self.IFW = en.makeFourierMatrices(dim)
        self.RFW = [tf.constant(np.real(self.RFW),dtype=tf.float32),tf.constant(np.imag(self.RFW),dtype=tf.float32)]
        
        self.IFW = [tf.constant(np.real(self.IFW),dtype=tf.float32),tf.constant(np.imag(self.IFW),dtype=tf.float32)]

        self.inputx = tf.placeholder(tf.float32,shape =[self.HLS,self.ydim,1])
        self.inputy = tf.placeholder(tf.float32,shape =[self.HLS,self.ydim,1])
        self.gamma=gamma
        initializer = tf.random_uniform_initializer
        self.C0 = tf.get_variable(subtestname + 'C0',shape=[self.HLS,1,2],initializer=initializer())
        self.C1 = tf.get_variable(subtestname + 'C1',shape=[self.HLS,1,2],initializer=initializer())
        self.T0 = tf.get_variable(subtestname + 'T0',shape=[self.HLS,1],initializer=initializer())
        self.T1 = tf.get_variable(subtestname + 'T1',shape=[self.HLS,1],initializer=initializer())
        self.A0 = tf.get_variable(subtestname + 'A0',shape=[self.HLS,1,2],initializer=initializer())
        self.A1 = tf.get_variable(subtestname + 'A1',shape=[self.HLS,1,2],initializer=initializer())
        self.W0 = tf.get_variable(subtestname + 'W0',shape=[self.HLS,2,2],initializer=initializer())
        self.W1 = tf.get_variable(subtestname + 'W1',shape=[self.HLS,2,2],initializer=initializer())
        self.activtion = activation
        self.P1 = [self.W0,self.C0,self.T0,self.A0,self.activation,self.Nstates,self.HLS]
        self.P2 = [self.W1,self.C1,self.T1,self.A1,self.activation,self.Nstates,self.HLS]
        self.params = params
        self.E = self.energy()
        
        self.loss = self.E[0]
        #self.boundaryConditions()
        
    def energy(self):
        return en.energy(self.inputx,self.inputy,self.inputkx,self.inputky,self.V,self.d,self.P1,self.P2,self.gamma,self.VintMap,self.RFW,self.IFW,self.params,self.ydim,self.plotdim)
    
    def getThetas(self):
        return [self.T0,self.T1]
    
    def getCenters(self):
        return [self.C0,self.C1]
    
   
    
    def run(self):
        #train_op = tf.train.MomentumOptimizer(.01,.001,use_nesterov=True).minimize(self.loss)
        start = time.time()
        if self.method is not None: 
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss=self.loss,method=self.method,options={'gtol': 1e-08})
            tf.global_variables_initializer().run(session=self.sess)
            optimizer.minimize(self.sess,feed_dict={self.inputx: self.xx,self.inputy: self.yy})
        else:
            lr = .01
            learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder).minimize(self.loss)
            tf.global_variables_initializer().run(session=self.sess)
            E0 = 1
            E1 = 999999
            itr = 0
            switched = False
            
            while  abs(E0-E1)/abs(E0) > self.cutoff and itr < self.maxitr:
                
                print('Iteration: ', itr)
                self.sess.run(train_op,feed_dict={self.inputx: self.xx,self.inputy: self.yy,learning_rate_placeholder:lr})
                #loss = self.sess.run(self.loss,feed_dict={self.inputx: self.xx,self.inputy: self.yy})
                E = self.sess.run(self.E,feed_dict={self.inputx:self.xx,self.inputy:self.yy})
           
            
                self.Elist.append(E[0][0])
                #print('E',E[0][0])
                #print('loss: ', loss[0][0])
                E0 = E1
                E1 = E[0][0]
                
                if E1 > E0 and itr > 300:
                    lr = lr*.99
                itr+=1
                if E1 < 0:
                    break
        print('LR_final: ',lr)
        end = time.time()
        print('time: ', str(end-start))
        psiL = []
        centers = []
        thetas = [] 
        psiL.append(self.sess.run(en.nnOut(self.inputx,self.inputy,self.W0,self.C0,self.T0,self.A0,self.activation,self.Nstates,self.HLS),feed_dict={self.inputx:self.xx,self.inputy:self.yy}))
        psiL.append(self.sess.run(en.nnOut(self.inputx,self.inputy,self.W1,self.C1,self.T1,self.A1,self.activation,self.Nstates,self.HLS),feed_dict={self.inputx:self.xx,self.inputy:self.yy}))
        centers = [self.C0.eval(session=self.sess),self.C1.eval(session=self.sess)]
        thetas = [self.T0.eval(session=self.sess),self.T1.eval(session=self.sess)]
        print('Loss final: ',self.Elist[-1])
        w = [self.W0.eval(session=self.sess),self.W1.eval(session=self.sess)]
        Vxy = self.Vfun(self.xo,self.yo)
        print(self.Vfun(1,1))
        np.savez('Tests/' + self.testname+'/'+self.subtestname + '/' + self.subtestname,dim=self.plotdim,x=self.xo,y=self.yo,w = w,centers=centers,thetas=thetas,psiL = psiL,V = Vxy,Elist=self.Elist,Efinal=self.Elist[-1],dx=self.d,params=self.params)
        self.sess.close()
       
        
