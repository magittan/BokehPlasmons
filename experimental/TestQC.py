from common.baseclasses import AWA
from common.numerical_recipes import QuickConvolver as QC
import matplotlib.pyplot as plt
import numpy as np

image=AWA(np.zeros((101,101)),axes=[np.linspace(-.5,.5,101)]*2)
image[50,50]=1

kernel_function=lambda x,y: 1/np.sqrt(x**2+y**2+1e-8**2)
qc1=QC(size=(1,1),shape=(101,101),pad_by=.5,kernel_function=kernel_function)
result1=qc1(image)
result1-=result1.min() #overall offset, while correct, should not be meaningful
result1[result1==result1.max()]=0 #point at center is controlled by 1e-8

kernel_function_fourier=lambda kx,ky: 2*np.pi/np.sqrt(kx**2+ky**2+1e-8**2)
qc2=QC(size=(1,1),shape=(101,101),pad_by=.5,kernel_function_fourier=kernel_function_fourier)
result2=qc2(image)
result2-=result2.min() #overall offset is controlled by 1e-8

plt.figure();result1.cslice[0].plot()
result2.cslice[0].plot()
plt.gca().set_xscale('symlog',linthreshx=1e-2)
plt.show()
