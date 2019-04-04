import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil as sh
import os
import math as m
from scipy.special import erfc

def complexAdd(A,B):
    return [A[0]+B[0],A[1]+B[1]]

def complexConj(A):
    return [tf.transpose(A[0]),-tf.transpose(A[1])]

def transpose(A):
    return [tf.transpose(A[0]),tf.transpose(A[1])]

def complexMultiply(A,B):
    return [A[0]*B[0]-A[1]*B[1],A[1]*B[0]+A[0]*B[1]]


def complexMatmul(A,B):
    return [tf.matmul(A[0],B[0])-tf.matmul(A[1],B[1]),tf.matmul(A[1],B[0])+tf.matmul(A[0],B[1])]

def gauss(X):
    return tf.exp(-tf.square(X))

def G(k):
    res = 2.0-3.0*np.sqrt(np.pi)*k*np.exp(k**2)*erfc(k)
    kbig = k[k>20]
    res[k>20] = -1+3.0/(2.0*kbig**2)-9.0/(4.0*kbig**4);
    return res

def shape2D(A,dim):
    return [tf.reshape(A[0],[dim,dim]),tf.reshape(A[1],[dim,dim])]

def shape1D(A,dim):
    return [tf.reshape(A[0],[dim,1]),tf.reshape(A[1],[dim,1])]

def makeFourierMatrices(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    W = np.dot(M,np.eye(N))
    Winv = np.linalg.inv(W)
    return W,Winv

def fftshift(x):
    dim = int(x[0].shape[0].value/2)
    return [tf.manip.roll(x[0],[dim,dim],axis=[0,1]),tf.manip.roll(x[1],[dim,dim],axis=[0,1])]

def ifftshift(x):
    dim = int(x[0].shape[0].value/2)
    if int(x[0].shape[0].value) % 2 == 0:
        return [tf.manip.roll(x[0],[dim,dim],axis=[1,0]),tf.manip.roll(x[1],[dim,dim],axis=[1,0])]
    return [tf.manip.roll(x[0],[dim+1,dim+1],axis=[1,0]),tf.manip.roll(x[1],[dim+1,dim+1],axis=[1,0])]

def DFT_2D(X,W):
    X = fftshift(X)
    res = complexMatmul(complexMatmul(W,X),transpose(W))
    return ifftshift(res)

def IDFT_2D(X,Winv):
    X = fftshift(X)
    res = complexMatmul(complexMatmul(Winv,X),transpose(Winv))
    return ifftshift(res)

def convolution(V,psisq,W,Winv):
    dim = W[0].shape[0]
    
    V[0] = tf.reshape(V[0],[W[0].shape[0],W[0].shape[1]])
    V[1] =  tf.reshape(V[1],[W[0].shape[0],W[0].shape[1]])
    psisq[0] = tf.reshape(psisq[0],[W[0].shape[0],W[0].shape[1]])
    psisq[1] = tf.reshape(psisq[1],[W[0].shape[0],W[0].shape[1]])
    out = IDFT_2D(complexMultiply(V,DFT_2D(psisq,W)),Winv)
    out[0] = tf.reshape(out[0],[int(W[0].shape[0]*W[0].shape[1]),1])
    out[1] = tf.reshape(out[1],[int(W[0].shape[0]*W[0].shape[1]),1])
    V[0] = tf.reshape(V[0],[int(W[0].shape[0]*W[0].shape[1]),1])
    V[1] =  tf.reshape(V[1],[int(W[0].shape[0]*W[0].shape[1]),1])
    return out

def nnOut(X,Y,W,C,T,A,activation,Nstates,HLS):
    #Xx = tf.tile(tf.reshape(X,[1,X.shape[0],X.shape[1]]),[HLS,1,1])
    #Yy = tf.tile(tf.reshape(Y,[1,Y.shape[0],Y.shape[1]]),[HLS,1,1])
    R = tf.reshape(tf.stack([tf.cos(T),-tf.sin(T),tf.sin(T),tf.cos(T)],axis=1),shape=[HLS,2,2])
    Z = tf.concat([X,Y],axis=2)

    out1 = (Z-C)
    out2=out1@R
    out3=activation(out2@W)
    out4 = tf.reduce_sum(A*out3,0)

    return [tf.reshape(out4[:,0],[X.shape[1],1]),tf.reshape(out4[:,1],[X.shape[1],1])]
    
def get_norm(A):
    return complexMatmul(complexConj(A),A)

def get_norm2(A,B):
    return complexAdd(complexMatmul(complexConj(A),A),complexMatmul(complexConj(B),B))

def square(A):
    return [A[0]**2,A[1]**2]


def complexDivide(A,B):
    return complexMultiply(A,[B[0]/(B[0]**2+B[1]**2),-B[1]/(B[0]**2+B[1]**2)])

def sqAbsolute(A):
    return [A[0]**2+A[1]**2,tf.zeros([A[0].shape[0],1],tf.float32)]

def gradient(psi,X):
    psiR = tf.reshape(psi[0],[psi[0].shape[0],1])
    psiI = tf.reshape(psi[1],[psi[0].shape[0],1])

    dxRpsi = tf.reduce_sum(tf.gradients(psiR,X),1)[0]
    dxIpsi = tf.reduce_sum(tf.gradients(psiI,X),1)[0]
   
   
    return [dxRpsi,dxIpsi]

def energy(X,Y,KX,KY,V,d,P1,P2,gamma,VintMap,RFW,IFW,params,dim1D,dim2D):
    v = params[0]
    delta = params[1]
    beta = params[2]
    E = [tf.zeros([1,1],tf.float32),tf.zeros([1,1],tf.float32)]
    Uint = [tf.zeros([1,1],tf.float32),tf.zeros([1,1],tf.float32)]
    Knet = [tf.zeros([1,1],tf.float32),tf.zeros([1,1],tf.float32)]
    Unet = [tf.zeros([1,1],tf.float32),tf.zeros([1,1],tf.float32)]
    Ue = [tf.zeros([1,1],tf.float32),tf.zeros([1,1],tf.float32)]
    grads = []
    L = []
    
    #L.append(nnOut(self.inputx,self.inputy,self.W0,self.C0,self.T0,self.A0,self.activation,self.Nstates,self.HLS))
    L.append(nnOut(X,Y,P1[0],P1[1],P1[2],P1[3],P1[4],P1[5],P1[6]))
    #L.append(nnOut(self.inputx,self.inputy,self.W1,self.C1,self.T1,self.A1,self.activation,self.Nstates,self.HLS))
    L.append(nnOut(X,Y,P2[0],P2[1],P2[2],P2[3],P2[4],P2[5],P2[6]))
    C = 0 
    #for i in range(0,len(L)):
    #    grads.append([gradient(L[i],X),gradient(L[i],Y)])
    norm = [tf.zeros([1,1],tf.float32),tf.zeros([1,1],tf.float32)]
    for i in range(0,len(L)):
        norm = complexAdd(norm,get_norm(L[i]))
    for i in range(0,len(L)):
        psiR_i,psiI_i = L[i]
        #ddxRpsi_i,ddxIpsi_i = grads[i][0]
        #ddyRpsi_i,ddyIpsi_i = grads[i][1]
        #dxRpsi_i,dxIpsi_i = grads[i][0]
        #dyRpsi_i,dyIpsi_i = grads[i][1]
        
        for j in range(0,len(L)):
            psiR_j,psiI_j = L[j]
            #ddxRpsi_j,ddxIpsi_j = grads[j][0]
            #ddyRpsi_j,ddyIpsi_j = grads[j][1]
            if i == 0 and j == 0:
                EMuu = complexDivide(complexMatmul(complexConj([psiR_i,psiI_i]),complexMultiply([delta/2.0, 0.0],[psiR_i,psiI_i])),norm)
                Ue = complexAdd(Ue,EMuu)
            elif i == 1 and j == 1:
                EMdd = complexDivide(complexMatmul(complexConj([psiR_j,psiI_j]),complexMultiply([2*beta*delta+delta/2.0,0.0],[psiR_j,psiI_j])),norm)
                Ue = complexAdd(Ue,EMdd)
                                
            elif i == 0 and j == 1:
                EMud = complexDivide(complexMatmul(complexConj([psiR_i,psiI_i]),complexMultiply([-delta/2.0, 0.0],[psiR_j,psiI_j])),norm)
                EMdu = complexDivide(complexMatmul(complexConj([psiR_j,psiI_j]),complexMultiply([-delta/2.0, 0.0],[psiR_i,psiI_i])),norm)
                Ue = complexAdd(Ue,EMud)
                Ue = complexAdd(Ue,EMdu)

            conv = convolution(VintMap[(i,j)],sqAbsolute([psiR_j,psiI_j]),RFW,IFW)
            #conv = complexMultiply([-1.0,0],conv)
            Uintij_norm = complexMatmul(norm,norm)
            
            Uintij = complexDivide(complexMatmul(complexConj([psiR_i,psiI_i]),complexMultiply(conv,[psiR_i,psiI_i])),Uintij_norm)

           
            Uint = complexAdd(Uint,complexMultiply([1.0/2.0/d**2,0.0],Uintij))


        EK = complexMultiply([1.0,0],shape1D(IDFT_2D(shape2D(complexMultiply([(KX**2.0+KY**2.0),0.0],shape1D(DFT_2D(shape2D([psiR_i,psiI_i],dim2D),RFW),dim1D)),dim2D),IFW),dim1D))
        K = complexMultiply([1.0/2.0,0],complexDivide(complexMatmul(complexConj([psiR_i,psiI_i]),EK),norm))
        
        #K = complexDivide(complexMultiply([1.0/2.0,0],complexAdd(complexMatmul(complexConj([dxRpsi_i,dxIpsi_i]),[dxRpsi_i,dxIpsi_i]),complexMatmul(complexConj([dyRpsi_i,dyIpsi_i]),[dyRpsi_i,dyIpsi_i]))),norm)
        
        #K = complexMultiply([d**2,0],complexMatmul(complexConj([psiR_i,psiI_i]),EK))
       
        U = complexDivide(complexMatmul(complexConj([psiR_i,psiI_i]),complexMultiply(complexMultiply([v**2, 0.0],V),[psiR_i,psiI_i])),norm)
        #E = complexAdd(E,complexDivide(complexMultiply([C,0],complexAdd(K,U)),get_norm([psiR_i,psiI_i])))
        Knet = complexAdd(Knet,K)
        Unet = complexAdd(Unet,U)
    Kp = tf.Print(Knet, [Knet], "Kinetic energy: ")
    Up = tf.Print(Unet,[Unet], "Potential energy: ")
    Uintp = tf.Print(Uint,[Uint],"Interaction energy: ")
    Uep = tf.Print(Ue,[Ue],"Electric field energy: ")
    E = complexAdd(E,complexAdd(Uintp,complexAdd(Uep,complexAdd(Kp,Up))))
    Ep = tf.Print(E,[E],"Total Energy: ")
    Ep = complexAdd([0,0],Ep)
    return Ep 
