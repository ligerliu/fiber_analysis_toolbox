import sys
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
sys.path.append('/nsls2/xf16id1/Staff/jiliang/2020-1/cellulose_analysis')
from scan_data_format import data_format,data_segment
from cellulose_toolbox import data_ana

def rel_scan_data(d,num):
    """
    f is h5py.File.dataset
    num is num of patterns
    """
    return d[num,:,:]

def fly_scan_data(d,num,id2):
    """
    f is h5py.File.dataset
    num is num of patterns
    """
    idx1 = int(num/id2)
    idx2 = int(num%id2)
    return d[idx1,idx2,:,:]

def bkgd_cal(num,f,em,xs_type='saxs',data_type='rel_scan',id2=None):
    """
    f is h5py.File
    num must int or array or list of int
    """
    if xs_type == 'saxs':
        xs_dir = f[list(f.keys())[0]].name+'/primary/data/pil1M_ext_image'
    elif xs_type == 'waxs1':
        xs_dir = f[list(f.keys())[0]].name+'/primary/data/pilW1_ext_image'
    elif xs_type == 'waxs2':
        xs_dir = f[list(f.keys())[0]].name+'/primary/data/pilW2_ext_image'
    if type(num) == int:
        num = np.array([num])
    i = 0
    for _ in num:
        if data_type == 'rel_scan':
            d = rel_scan_data(f[xs_dir],_)
        elif data_type == 'fly_scan':
            if not id2:
                raise "should give data points of fast axis of fly scan"
            d = fly_scan_data(f[xs_dir],_,id2)
        if i == 0:
            xs = np.copy(d).astype(float)
        else:
            xs += d.astype(float)*(em[num[0]]/em[_])
        i += 1
    return xs/i,em[num[0]]

