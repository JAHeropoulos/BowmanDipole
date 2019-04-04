import numpy as np


def DFT(x): 
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    t = k*n
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    t = k*n
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def DFT_2D(x):
    
    W= DFT(np.eye(x.shape[0]))
    return ifftshift(W@fftshift(x)@W.T)

def IDFT_2D(x):
    W = DFT(np.eye(x.shape[0]))
    Wp = np.linalg.inv(W)
    return ifftshift(Wp@fftshift(x)@Wp.T)

def fftshift(x):
    return np.roll(x,int(x.shape[0]/2),axis=(0,1))

def ifftshift(x):
    if int(x.shape[0]) %2 == 0:
        return np.roll(x,int(x.shape[0]/2),axis=(1,0))
    return  np.roll(x,int(x.shape[0]/2+1),axis=(1,0))

def main():
    x = np.arange(16).reshape(4,4)
    y = DFT_2D(x)
    z = np.fft.fft2(np.fft.fftshift(x))
    print(np.allclose(y,z))
    print('inverse')
    print()
    a = ifftshift(x)
    b = np.fft.ifftshift(x)
    print(a)
    print(b)
    x = np.random.rand(100,100)
    out1 = DFT_2D(x)
    out2 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
    print(np.allclose(out1,out2))
    out3 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(out2)))
    out4 = IDFT_2D(out1)
    print(np.allclose(out3,out4))
    #print(out3[0,:5])
    #print(out4[0,:5])
    
if __name__ == '__main__':
    main()
