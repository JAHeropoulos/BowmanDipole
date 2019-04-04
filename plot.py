import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import shutil as sh
from matplotlib.patches import Ellipse
#np.savez('Tests/' + self.testname+'/'+self.subtestname + '/' + self.subtestname,dim=self.plotdim**2,layers=self.layers,x=self.xx,y=self.yy,psi = psi,V = Vxy,itrlist = self.itrlist,Elist=self.Elist,Efinal=self.Elist[-1])
def complexConj(A):
    return np.transpose(np.conj(A))

figureType = '.png'
subtestnames = os.listdir('Tests/test/')
print(subtestnames)
testnames = ['test']

colormap = cm.Spectral
betas = np.zeros(len(testnames))
Gs = np.zeros(len(testnames))
Rsqs = np.zeros(len(testnames))
itr = 0
for testname in testnames:
    for subtestname in subtestnames:
        plotDirectory = 'Tests'+ '/'+testname + '/' + subtestname + '/' + 'plots'
        print('plots in: ',plotDirectory)
        if os.path.exists(plotDirectory):
            sh.rmtree(plotDirectory)
        os.makedirs(plotDirectory)
            
        fileToLoad = 'Tests'+ '/'+testname + '/' + subtestname + '/' + subtestname + '.npz'
        data = np.load(fileToLoad)
        plotdim = data['dim']
        x = data['x']
        y = data['y']
        psiL = data['psiL']
        V = data['V']
        Elist = data['Elist']
        Efinal = data['Efinal']
        dx = data['dx']
        norm = 0
        centers = data['centers']
        thetas = data['thetas']
        w = data['w']
        params = data['params']
        for i in range(0,len(psiL)):
            c = centers[i].reshape(centers[i].shape[0],2)
            xc = c[:,0]
            yc = c[:,1]
            plt.scatter(xc,yc)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim([-2,2])
            plt.ylim([-2,2])
            plt.title('Basis function centers'+str(i))
            plt.savefig(plotDirectory + '/'+ 'centers'+str(i))
            plt.close()
            t = thetas[i].reshape(thetas[i].shape[0],1)
            plt.hist(t*180/np.pi,bins = 10)
            plt.xlabel('angles (deg)')
            plt.ylabel('frequency')
            plt.title('theta Distribution'+str(i))
            plt.savefig(plotDirectory+'/'+'theta_distribution'+str(i))
            plt.close()
            wxy = w[i][:,:,0]
            xw = 1/wxy[:,0]
            yw = 1/wxy[:,1]
            fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
            for j in range(0,xc.shape[0]):
                #plt.xlim([int(np.min(xc)-1),int(np.max(xc)+1)])
                #plt.ylim([int(np.min(yc)-1),int(np.max(yc)+1)])
                plt.xlim([-10,10])
                plt.ylim([-10,10])
                e = Ellipse((xc[j],yc[j]),xw[j],yw[j],180*t[j]/np.pi)
                e.set_alpha(.5)
                e.set_facecolor(np.random.rand(3))
                ax.add_artist(e)
            plt.title('Basis Functions'+str(i))
            fig.savefig(plotDirectory + '/' + 'basis_functions' +str(i))
            plt.close()
        for j in range(0,len(psiL)):
            psi = psiL[j][0] + psiL[j][1]*1.0j
            norm += np.transpose(np.conj(psi))@psi
        norm = np.sqrt(norm*dx**2)
        L = [] 
        for j in range(0,len(psiL)):
            psi = psiL[j][0] + psiL[j][1]*1.0j
            psi = psi/norm
            psisq = np.absolute(np.conj(psi)*psi)
            x = x.reshape(psi.shape[0],)
            y = y.reshape(psi.shape[0],)
            psi = psi.reshape(psi.shape[0],)
            if j == 0:
                Rsq = np.absolute(np.matmul(np.transpose(np.conj(psi)),(x**2+y**2)*psi*dx**2))
                Rsqs[itr] = Rsq
                betas[itr] = params[2]
                Gs[itr] = params[3] 
                print('Rsq: ',Rsq)
                 
            print('c'+str(j)+': ',np.sum(psisq)*dx**2)
            x = x.reshape(plotdim,plotdim)
            y = y.reshape(plotdim,plotdim)
            psisq = psisq.reshape(plotdim,plotdim)
            L.append(psisq)
            V = V.reshape(plotdim,plotdim)
            
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.plot_surface(x,y,psisq,rstride=1,cstride=1,linewidths=0,cmap=colormap, antialiased=True)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('|psi|^2')
            ax.set_title('State ' + str(j))
            fig.savefig(plotDirectory + '/' +'psisq' + str(j) + figureType)
            plt.close(fig)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111) 
        ax2.plot(np.arange(len(Elist)),np.array(Elist))
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Energy')
        fig2.savefig(plotDirectory + '/' + 'energyIterations' + figureType)
        plt.close(fig2)
        
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111,projection='3d')
        ax3.plot_surface(x,y,V)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('V(x,y)')
        fig3.savefig(plotDirectory + '/' +'potential' + figureType)
        plt.close(fig3)

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111,projection='3d')
        ax4.plot_surface(x,y,L[0]-L[1],rstride=1,cstride=1,linewidths=0,cmap=colormap, antialiased=True)
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('P(x,y)')
        fig4.savefig(plotDirectory + '/' +'polarization' + figureType)
        plt.close(fig4)
