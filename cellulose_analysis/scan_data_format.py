import numpy as np
from py4xs.hdf import h5exp,h5xs,lsh5
from scipy.optimize import minimize
from py4xs.data2d import Data2d
from py4xs.exp_para import ExpParaLiX
from py4xs.detector_config import DetectorConfig
import fabio
import h5py
import time
import os
import sys
import matplotlib.pyplot as plt
import glob
#sys.path.append('/nsls2/xf16id1/Staff/jiliang/2019-3/305000/305360/')
from cellulose_toolbox import data_ana

global bk_em,bk_S1,bk_W2,pil1M,pilW2

## need to access to folder include h5 file first

class data_format():
    def __init__(self,h5_file,sample_name=None):
        self.f = h5py.File(h5_file,'r')
        # here only handle one sample for h5 data
        # for multiple sample like Uminn sample need to further working on
        if not sample_name:
            #self.sn = self.f.filename.split('/')[-1][:-3]
            self.sn = self.f[list(self.f.keys())[0]].name.split('/')[-1]
        else:
            self.sn = sample_name
        self.saxs_dir = self.sn+'/primary/data/pil1M_ext_image'
        self.waxs1_dir = self.sn+'/primary/data/pilW1_ext_image'
        self.waxs2_dir = self.sn+'/primary/data/pilW2_ext_image'
        stage_tmstmp_dir = self.sn+'/primary/timestamps'
        self.axis_list = []
        for _ in self.f[stage_tmstmp_dir].keys():
            if _[:3] == 'ss_':
                self.axis_list.append(_)
        self.axis1 = self.sn+'/primary/timestamps/' + self.axis_list[0]
        self.axis2 = self.sn+'/primary/timestamps/' + self.axis_list[1]
        if np.size(self.f[self.axis1][:].flatten()) > np.size(self.f[self.axis2][:].flatten()):
            self.tmstmp_pos = self.f[self.axis1][:].flatten()
        else:
            self.tmstmp_pos = self.f[self.axis2][:].flatten()
    
    def fly_scan(self,
                 em1_scale_factor_area=np.array([0]),
                 em2_scale_factor_area=np.array([0])):
        '''
        scale_factor area indicate the points chosen as dark field pattern to correct intensity
        flucutation due to beam intensity variation.
        '''
        tm_em1  = self.f[self.sn]['em1_sum_all_mean_value_monitor/timestamps/em1_sum_all_mean_value'][:]
        tm_em2  = self.f[self.sn]['em2_sum_all_mean_value_monitor/timestamps/em2_sum_all_mean_value'][:]
        val_em2 = self.f[self.sn]['em2_sum_all_mean_value_monitor/data/em2_sum_all_mean_value'][:]
        val_em1 = self.f[self.sn]['em1_sum_all_mean_value_monitor/data/em1_sum_all_mean_value'][:]
        #this calibrate time on em1 and em2
        tm_em1 *= tm_em2[0]/tm_em1[0]
        tm_em1 = np.linspace(tm_em2[0],tm_em2[-1],len(tm_em1))
        #plt.close('all')
        #plt.figure()
        #plt.subplot(211)
        #plt.semilogy(tm_em1,val_em1,tm_em2,val_em2)
        #plt.subplot(212)
        #plt.semilogy(np.linspace(tm_em2[0],tm_em2[-1],len(tm_em1)),val_em1,
        #             tm_em2,val_em2)
        self.em1 = np.interp(self.tmstmp_pos,tm_em1,val_em1,3)
        self.em2 = np.interp(self.tmstmp_pos,tm_em2,val_em2,3)
        self.em1_scale = np.mean(self.em1[em1_scale_factor_area])
        self.em2_scale = np.mean(self.em2[em2_scale_factor_area])
        return self.tmstmp_pos,self.em1,self.em2,self.em1_scale,self.em2_scale
    
    def rel_scan(self,
                 em1_scale_factor_area=np.array([0]),
                 em2_scale_factor_area=np.array([0])):
        tm_em1  = self.f[self.sn]['primary/timestamps/em1_sum_all_mean_value'][:]
        tm_em2  = self.f[self.sn]['primary/timestamps/em2_sum_all_mean_value'][:]
        val_em2 = self.f[self.sn]['primary/data/em2_sum_all_mean_value'][:]
        val_em1 = self.f[self.sn]['primary/data/em1_sum_all_mean_value'][:] 
        tm_em1 *= tm_em2[0]/tm_em1[0]
        #this calibrate time on em1 and em2
        self.em1 = np.interp(self.tmstmp_pos,tm_em1,val_em1,3)
        self.em2 = np.interp(self.tmstmp_pos,tm_em2,val_em2,3)
        self.em1_scale = np.mean(self.em1[em1_scale_factor_area])
        self.em2_scale = np.mean(self.em2[em2_scale_factor_area])
        return self.tmstmp_pos,self.em1,self.em2,self.em1_scale,self.em2_scale

class data_segment(data_format):
    def __init__(self,h5_file,detector='saxs',sample_name=None,scan_type='fly_scan'):
        super().__init__(h5_file,sample_name=None)
        if detector == 'saxs':
            d_dir = self.saxs_dir
        elif detector == 'waxs1':
            d_dir = self.waxs1_dir
        elif detector == 'waxs2':
            d_dir = self.waxs2_dir        
        self.xs_dataset = self.f[d_dir]
        
        if scan_type == 'fly_scan':
            super().fly_scan()
            self.id1 = self.xs_dataset.shape[0]
            self.id2 = self.xs_dataset.shape[1]
        else:
            super().rel_scan()
        
    def slice_index(self,slice_axis=0,slice_num=3):
        total_num = self.xs_dataset.shape[slice_axis]
        idx1 = np.ceil(total_num/slice_num).astype(int)
        self.index1 = []
        self.index2 = []
        for _ in range(slice_num):
            self.index1.append(_*idx1)
            self.index2.append((_+1)*idx1)
        if self.index2[-1] > total_num:
            self.index2[-1] = total_num
    
    def fly_scan_slice_em(self,slice_axis=0,slice_num=3):
        if slice_axis == 0:
            id2 = self.id2
        else:
            id2 = self.id1
        self.slice_index(slice_axis=slice_axis,slice_num=slice_num)
        index1 = np.array(self.index1)*id2
        index2 = np.array(self.index2)*id2
        # directly list multiply is duplication of original list
        self.em1_split = np.split(self.em1,index2[:-1])
        self.em2_split = np.split(self.em2,index2[:-1])
    
    def rel_scan_slice_em(self,slice_axis=0,slice_num=3):
        self.slice_index(slice_axis=slice_axis,slice_num=slice_num)
        self.em1_split = np.split(self.em1,self.index2[:-1])
        self.em2_split = np.split(self.em2,self.index2[:-1])
            
        