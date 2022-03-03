import numpy as np
import torch
from torch import nn



#all start at 2015 01 01, 64 *128 for 500 km 
#404 802 for 50km
#go over gulf for a year





#get the data 
temp_data = np.swapaxes(np.load('ts.npy'),0,1)
rad_data = np.swapaxes(np.load('rsds.npy'),0,1)
chlo_data = np.swapaxes(np.load('chlos.npy'),0,1)
#cmip filtering
print(temp_data)
print(temp_data.shape)
print(rad_data)
print(rad_data.shape)
print(chlo_data)
print(chlo_data.shape)
"""

chlo_data = np.swapaxes(np.load('lev.npy'),0,1)
#chlo_data[chlo_data>=9.999e+19] = 0
#chlo_data[chlo_data<=0.51]= 0



chlo_data = np.true_divide(chlo_data,1e19)
cD = torch.empty(240,1,20,20)
for i in range(0,240):
	p = chlo_data[i].reshape(180,360)
	p = p[105:125,260:280].reshape(20,20)
	t = torch.from_numpy(p)
	cD[i,0] = t
#torch tensor to feed to nn
tD = torch.empty(240,1,20,20)
for i in range(0,240):
	p = temp_data[i].reshape(180,360)
	p = p[105:125,260:280].reshape(20,20)
	t = torch.from_numpy(p)
	tD[i,0] = t
torch.save(tD,"tempData.pt")
torch.save(cD,"chloroData.pt")
print(cD)
"""