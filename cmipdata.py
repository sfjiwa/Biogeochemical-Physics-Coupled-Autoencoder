#file2read = netCDF4.Dataset('tasmax_day_IPSL-CM6A-LR_amip-lfmip-pdLC_r1i1p1f1_gr_19800101-21001231.nc','r')
import os, os.path
import datetime as dt
import numpy as np
from numpy.core.defchararray import join
import pandas as pd
from matplotlib import pyplot as plt
from netCDF4 import Dataset, date2index, num2date, date2num
from torchvision.io import read_image

def WriteData(training=True):

    print("reading data")

    if training:
        data_type = 'training'
    else: data_type = 'test'

    # import input data from /data/

    filenames = os.listdir('data')

    if data_type not in filenames:
        dir_path = os.path.join(os.getcwd(), 'data/' + data_type)
        os.makedirs(dir_path, exist_ok=True)

    if 'training' in filenames: filenames.remove('training')
    if 'test' in filenames: filenames.remove('test')

    data_set = np.empty(len(filenames), dtype=object)
    output = []

    for i in range(0, len(filenames)):
        name = filenames[i]
        # get variable id
        var = 'chlos'
        """for letter in filenames[i]:
            if letter != '_':
                var += letter
            else: break
"""

        print()
        data_set[i] = Dataset('data/' + filenames[i])

        # input dimensions
        print(data_set[i])
        n_t = len(data_set[i].variables['time'])
        n_lat = len(data_set[i].variables['latitude'])
        n_lon = len(data_set[i].variables['longitude'])
        #print(data_set[i].variables['latitude(0,0)'])
        print('processing variable: ' + var)

        data_out = data_set[i].variables[var]

        data_out = np.array(np.ma.filled(data_out[:], 0))
        print(data_out.shape)
    # average over depth if applicabl   
        data_out = data_out.reshape(n_t, -)1
        data_out = np.transpose(data_out)

        print('saving input file')
        print(filenames[i] + ", " + var)

        np.save(name, data_out)

    print("done")
WriteData()