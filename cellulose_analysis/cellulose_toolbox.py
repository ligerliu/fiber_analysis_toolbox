import numpy as np
from scipy.signal import medfilt
from py4xs.hdf import h5exp,h5xs,lsh5
from scipy.optimize import minimize
from py4xs.data2d import Data2d,Axes2dPlot,DataType,MatrixWithCoords#,ExpPfrom scipy.optimize import minimize
from py4xs.exp_para import ExpParaLiX
from py4xs.detector_config import DetectorConfig
from py4xs.slnxs import Data1d, average
import fabio
import h5py
import time
import os 
import sys
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

def gaussian(q,x):
    return (x[0]*np.exp(-(q-x[1])**2/2/x[2]**2))

# merge two numpy arrays that each have missing data, as represented as np.nan
def merge(a, b):
    a1 = a.copy()
    b1 = b.copy()
    if a.shape!=b.shape:
        raise Exception("merge: the two data sets must have the same shape: ", a.shape, b.shape)
    idx = np.isnan(a1) & (~ np.isnan(b1))
    a1[idx]=b1[idx]
    idx = (~np.isnan(a1)) & np.isnan(b1)
    b1[idx]=a1[idx]
    return (a1+b1)/2

def fill_gap_qphi(data, interpolate=True, do_2nd_pass=True, in_place=True):
    """ data is an instance of the class MatrixWithCoords
        this works for a q-phi map only (data.xc is q and data.yc is phi)
        assuming centrosymmetry, i.e. I(q, phi) and I(q+-180) have the same value
        
            apply centrosymmetry first for each q
            fill the gap in q for each phi
            appy centrosymmetry again if do_2nd_pass is True
        
        modify data.d if in_place is True
        otherwise reture a modified MatricWithCoords
    """
    if not isinstance(data, MatrixWithCoords):
        raise Exception("data must be an instance of the class MatrixWithCoords.")

    dd = data.copy()
    for i in range(len(dd.xc)):
        xx = np.hstack([dd.yc-180, dd.yc+180])
        yy = np.hstack([dd.d[:,i], dd.d[:,i]])
        yy1 = np.interp(dd.yc, xx, yy)
        dd.d[:,i] = merge(dd.d[:,i],yy1)
    
    # need to remove the empty data at the end to avoid creating artifacts
    if interpolate:
        for i in range(len(data.yc)):
            xx1 = dd.xc
            yy1 = dd.d[i,:]    
            idx = ~np.isnan(yy1)
            idx1 = idx.copy()
            for k1 in range(len(idx)):
                if idx[k1]: break
            for k2 in range(len(idx)):
                if idx[-(k2+1)]: break
            idx1[k1:-(k2+1)] = True
            dd.d[i,idx1] = np.interp(xx1[idx1], xx1[idx], yy1[idx])

    if do_2nd_pass:
        for i in range(len(data.xc)):
            yy = np.hstack([dd.d[:,i], dd.d[:,i]])
            yy2 = np.interp(dd.yc, xx, yy)
            dd.d[:,i] = merge(dd.d[:,i],yy2)

    if in_place:
        data.d = dd.d
        return
    else:
        return dd
    
class data_ana:
    '''
    process and extract the structural parameters of cellulose from scattering pattern
    '''
    def __init__(self,xs,dexp,q,em=1,em_scale=1,bkg_data=None,name='test',**kargs):
        '''
        xs 2D scattering pattern, numpy array
        dexp is LiX experiment configuration
        q is 1D array define q range
        tmstmp is time stamp could be calculated from above tmstmp function
        bkgd_data include correlated bkgd pattern and time stamp
        after normalization, all patterns intensity had been corrected to bkgd intensity.
        '''
        self.q      = q
        self.exp    = dexp
        self.em     = em_scale/em
        self.name   = name
        self.xs = xs.astype(float)
        if bkg_data == None:
            #self.bkg_data  = np.zeros(np.shape(xs))
            self.xs  *= self.em
        else:
            self.bkg_data  = bkg_data['xs']
            self.xs  = xs*self.em - self.bkg_data/bkgd['em']*em_scale
        
        self.data = Data2d(self.xs,exp=self.exp)
    
    def thickness_cal(self,em1=1,em2=1,em1_scale=1,em2_scale=1,thickness_correction=True):
        '''
        basing on penetration attenuation incident beam I0 and penetrated beam I follow 
        rule: I = I0*exp(-coefficient*length).
        '''
        thickness = -np.log((em2/em2_scale)/(em1/em2_scale))
        if thickness_correction:
            #here assume density is uniform throughout scattering length
            sel.xs /= thickness
   
    def cir_ave(self, q = None, smooth=False,smooth_frac=0.1,smooth_it=0):
        '''
        calculate the circularly averaged intensity
        '''
        if not q:
            q = self.q
        (self.I_ave,dI)  = self.data.conv_Iq(q,mask=self.exp.mask,)
                                             #cor_factor=self.exp.FSA*self.exp.FPol)
        if smooth == True:
            y = lowess(self.I_ave, q, frac= smooth_frac,it=smooth_it)
            self.I_ave = np.interp(q,y[:,0],y[:,1])
        return self.I_ave
    
    def azi_map(self,qmin=None,qmax=None,
                amin=None,amax=None,
                Nq=None,Nphi = 180,
                sym_heal=False):
        '''
        The phi range is from -180 to 180
        qmin and qmax define the specific q range qphi map calculation
        amin and amax define the specific azimuth range
        qphi_map is 2D array of remeshed scattering in qphi coordinator
        yc is qphi_map correlated phi array
        xc is qphi_map correlated q array
        '''
        if not Nq:
            Nq  = len(self.q) 
        self.data.conv_Iqphi(Nq,Nphi,mask=self.exp.mask,cor_factor=self.exp.FSA*self.exp.FPol)
        if sym_heal:
            fill_gap_qphi(self.data.qphi_data)
        phi = self.data.qphi_data.yc
        if qmin or qmax or amin or amax:
            if not qmin:
                qmin = self.q[0]
            if not qmax:
                qmax = self.q[-1]
            if not amin:
                amin = phi[0]
            if not amax:
                amax = phi[-1]
            qphi_roi = self.data.qphi_data.roi(qmin,qmax,amin,amax)
            return qphi_roi
        else:
            qphi_map = self.data.qphi_data
            return qphi_map
    
    def azi_fit(self,
                qphi_data,
                bkgd = None, 
                bkgd_type='mean',
                bkgd_scale=1,
                symetric=False,
                med_filt=False,
                multi_ort=True,
                uni_ahw =True,
                ort_num=2,
                roll=False,
                roll_deg = 0,
                int_thrhd = 1.,
                **kargs):
        '''
        qphi_data is class of class MatrixWithCoords
        int_thrhd correlate to intensity threshold for judging the existence of signal.
        '''
        if symetric == True:
            # here assume the cellulose fiber diffraction is center symmetric
            # this may not be sure for tilted fiber diffraction pattern
            fill_gap_qphi(qphi_data)
        azi_I   = qphi_data.flatten(axis=1)
        azi_deg = qphi_data.yc
        if bkgd:
            bkgd = bkgd
        else:
            if bkgd_type == 'mean':
                bkgd = np.ones(np.shape(azi_I))
                bkgd[:int(len(azi_I)/2)] *= np.nanmean(azi_I[:int(len(azi_I)/2)])*0.9
                bkgd[int(len(azi_I)/2):] *= np.nanmean(azi_I[int(len(azi_I)/2):])*0.9
            elif bkgd_type == 'minimum':
                bkgd = np.ones(np.shape(azi_I))
                bkgd[:int(len(azi_I)/2)] *= np.nanmin(azi_I[:int(len(azi_I)/2)])
                bkgd[int(len(azi_I)/2):] *= np.nanmin(azi_I[int(len(azi_I)/2):])
            elif bkgd_type == 'scale':
                bkgd = np.ones(np.shape(azi_I))
                bkgd[:int(len(azi_I)/2)] *= bkgd_scale
                bkgd[int(len(azi_I)/2):] *= bkgd_scale
            elif bkgd_type == 'zero':
                bkgd = np.zeros(np.shape(azi_I))
        
        if med_filt == True:
            azi_I[np.isnan(azi_I)==0] = medfilt(azi_I[np.isnan(azi_I)==0],kernel_size=9)
            
        azi_I = np.interp(azi_deg,azi_deg[np.isnan(azi_I)==0],azi_I[np.isnan(azi_I)==0])
        azi_I -= bkgd
        fit_range = azi_deg[int(len(azi_I)/2):]
        fit_I = azi_I[int(len(azi_I)/2):]
        #### here have problem is difference of degree is not 1, then roll_deg is not correct here
        ###!!!!!
        if roll == True:
            fit_I = np.roll(fit_I,roll_deg)
        fit_I[fit_I<0.] = 0.
        ort = fit_range[np.argmax(fit_I)]  
        ort_array = np.zeros((ort_num,))*np.nan
        ort_ahw_array = np.zeros((ort_num,))*np.nan
        ort_int_array = np.zeros((ort_num,))*np.nan
        
        if (np.nanmax(fit_I)<2*np.nanstd(fit_I)) or (np.nanmax(fit_I)<int_thrhd):
            return ort_int_array,ort_array,ort_ahw_array
        else:
            x0 = np.array([np.nanmax(fit_I),
                          fit_range[np.nanargmax(fit_I)],
                          np.nansum(fit_I)/np.nanmax(fit_I)])
            bnd1 = (x0*0.1,x0*2)
            def func1(x,deg,y):
                return np.abs(gaussian(deg,x)-y)
            res1 = least_squares(func1,x0,args=(fit_range,fit_I),bounds=bnd1)
            if (multi_ort == True) and (ort_num > 1):
                xm0 = np.tile(np.array([np.nanmax(fit_I),
                                fit_range[np.nanargmax(fit_I)],
                                np.nansum(fit_I)/np.nanmax(fit_I)/ort_num]),ort_num)
                for _ in range(ort_num):
                    xm0[int(_*3+1)] = ((fit_range[int(_*len(fit_I)/ort_num):
                                                  int((_+1)*len(fit_I)/ort_num)])
                                              [np.nanargmax(fit_I[int(_*len(fit_I)/ort_num):
                                              int((_+1)*len(fit_I)/ort_num)])])
                                   
                    if xm0[int(_*3+1)] >180:
                        xm0[int(_*3+1)] = 180.
                    if xm0[int(_*3+1)] < 0:
                        xm0[int(_*3+1)] = 0.
                bndm = tuple(np.vstack((xm0*0.,xm0*5)))
                for _ in range(ort_num):
                    bndm[0][_*3+0] = np.nanmin(fit_I)
                    bndm[0][_*3+1] = xm0[_*3+1]-180/ort_num
                    bndm[0][_*3+2] = np.nanmean(np.diff(fit_range))*3
                    #bndm[1][_*3+0] = np.nanmax(fit_I)
                    bndm[1][_*3+1] = xm0[_*3+1]+180/ort_num
                    bndm[1][_*3+2] = np.nanmax(fit_range)/2
                #print(xm0<bndm[0],xm0>bndm[1])    
                def funm(x,deg,y,num=ort_num,uni_ahw=uni_ahw):
                    I = np.zeros(np.shape(y))
                    for _ in range(num):
                        if uni_ahw == True:
                            if _ > 0:
                                x[int(3*_)+2] = np.copy(x[2])
                        x1 = x[int(3*_):int(3*(_+1))]
                        I += gaussian(deg,x1)      
                    return np.abs(I-y)
                resm = least_squares(funm,xm0,args=(fit_range,fit_I),bounds=bndm)
                
                if ((np.mean(np.abs(res1.fun)) <= 1.1*np.mean(np.abs(resm.fun))) or 
                    (np.abs(resm.x[1]-resm.x[4])<= 15)):
                    ort_array[0]     = res1.x[1]-roll_deg
                    ort_ahw_array[0] = res1.x[2]
                    ort_int_array[0] = res1.x[0]
                else:
                    for _ in range(ort_num):
                        ort_array[_]     = resm.x[int(3*_+1)]-roll_deg
                        ort_ahw_array[_] = resm.x[int(3*_+2)]
                        ort_int_array[_] = resm.x[int(3*_)]
            else:
                ort_array[0]     = res1.x[1]-roll_deg
                ort_ahw_array[0] = res1.x[2]
                ort_int_array[0] = res1.x[0]
            return ort_int_array,ort_array,ort_ahw_array
        
    def lincut_xs(self,qphi_data,q,ort,ort_std,
                     qmin=None,qmax=None,flip=True,smooth=True):
        if not qmin:
            qmin = q[0]
        if not qmax:
            qmax = q[-1]
            
        if flip == True:
            if equator_ort >=0:
                I_xs1 = qphi_data.roi(qmin,qmax,
                                            ort-ort_std,
                                            ort+ort_std).flatten(axis=0)
                #I_xs2 is center symmetrical region of I_xs1, average I_xs1 and I_xs2 is 
                #to cover more q range of averaging calculation.
                I_xs2 = qphi_data.roi(qmin,qmax,
                                            ort-180-ort_std,
                                            ort-180+ort_std).flatten(axis=0)
                I_xs = np.nanmean(np.vstack((I_xs1,I_xs2)),axis=0)
            else:
                I_xs1 = qphi_data.roi(qmin,qmax,
                                            ort-ort_std,
                                            ort+ort_std).flatten(axis=0)
                I_xs2 = qphi_data.roi(qmin,qmax,
                                            ort+180-ort_std,
                                            ort+180+ort_std).flatten(axis=0)
                I_xs = np.nanmean(np.vstack((I_xs1,I_xs2)),axis=0)
        else:
            I_xs = qphi_data.roi(qmin,qmax,
                                   ort-ort_std,
                                   ort+ort_std).flatten(axis=0)
        qw = qphi_data.roi(qmin,qmax,
                           ort-ort_std,
                           ort+ort_std).xc
        qwmin = np.min(qw[np.isnan(I_xs)==0])
        argmin = np.argmin(np.abs(qw-qwmin))
        qwmax = np.max(qw[np.isnan(I_xs)==0])
        argmax = np.argmin(np.abs(qw-qwmax))
        
        I = np.zeros((len(q),))*np.nan
        if np.size((I_xs[argmin:argmax])[np.isnan(I_xs[argmin:argmax])])>0:
            ## if there is not NaN data point available for I_xs
            I_xs[argmin:argmax] = np.interp(qw[argmin:argmax],
                                       (qw[argmin:argmax])[np.isnan(I_xs[argmin:argmax])==0],
                                       (I_xs[argmin:argmax])[np.isnan(I_xs[argmin:argmax])==0])
            if smooth == True:
                y = lowess(I_xs, qw, frac= 0.1,it=0)
                I_xs[argmin:argmax] = np.interp(qw[argmin:argmax],y[:,0],y[:,1])
            qargmin = np.argmin(np.abs(q-qwmin))
            qargmax = np.argmin(np.abs(q-qwmax))
            I[qargmin:qargmax] = np.interp(q[qargmin:qargmax],qw[argmin:argmax],I[argmin:argmax])
            
        return I,q
    
    def CI_cal(self,CI_type='peak_height',plot=True):
        '''
        calculate the crystallinity of cellulose, three methods available:
        peak height ratio
        gaussian deconvolution
        crystal structure decouple
        '''
        if CI_type == 'peak_height':
            self.CI = ((np.max(self.I_waxs[(self.qw>1.5)&(self.qw<1.65)])-
                        np.min(self.I_waxs[(self.qw>1.3)&(self.qw<1.4)]))/
                        np.max(self.I_waxs[(self.qw>1.5)&(self.qw<1.65)]))
            amp_bkg_p = np.polyfit(np.hstack((self.qw[(self.qw>0.7)&(self.qw<0.9)],
                                              self.qw[(self.qw>1.28)&(self.qw<1.38)],
                                              self.qw[(self.qw>1.8)&(self.qw<1.95)])),
                                 np.hstack((self.I_waxs[(self.qw>0.7)&(self.qw<0.9)],
                                            self.I_waxs[(self.qw>1.28)&(self.qw<1.38)],
                                            self.I_waxs[(self.qw>1.8)&(self.qw<1.95)])),2)
            amp_bkg = np.polyval(amp_bkg_p,self.qw)
            y = self.I_waxs-amp_bkg
            y[self.qw<1.28] = 0
            y[self.qw>1.95] = 0
            x0 = np.array([np.nanmax(y),
                          self.qw[np.nanargmax(y)],
                          (1.95-1.35)/2/2.355])
            bnd1 = ([0,1.5,0],[x0[0],1.7,np.max(self.qw)])
            def func(x,q,y):
                return np.abs(gaussian(q,x)-y)
            
            res = least_squares(func,x0,args=(self.qw,y),bounds=bnd1)
            wl = self.ew2.wavelength
            Dd = self.ew2.Dd
            theta = np.arcsin(res.x[1]/2/np.pi/2*wl)
            beta = np.arcsin((2.355*res.x[2])/2/np.pi/2*wl)*2
            #scherr equation tau = K*wl/(delta(2theta)*cos(theta)) delta(2theta) = fwhm of peak in radians
            self.width = 0.88*wl/(beta*np.cos(theta))
            if plot == True:
                plt.subplots()
                plt.plot(self.qw,self.I_waxs,
                         self.qw,(amp_bkg+gaussian(self.qw,res.x)),
                         self.qw,amp_bkg)
            return self.CI,self.width
        elif CI_type == 'gaussian_convolution':
            pass  
    
    def qrqz_map(self,x_ind=None,y_ind=None,rot_phi = 0,rot_tilt = 0):
        '''
        transform scattering pattern to fiber diffraction (cylinderical coordinate) system
        x_ind is column index of 2D array
        y_ind is row index of 2D array
        '''
        if x_ind == None:
            x_ind = int(self.xs.shape[1]/10)
        if y_ind == None:
            y_ind = int(self.xs.shape[0]/10)
        self.exp.det_phi  = rot_phi
        self.exp.det_tilt = roi_tilt
        self.exp.calc_rot_matrix()
        self.exp.init_coordinates()
        self.data.conv_Iqrqz(y_ind,x_ind,mask=exp.mask)
        qrqz_map = self.data.qrqz_data.d
        qr       = self.data.qrqz_data.xc
        qz       = self.data.qrqz_data.yc
        return qrqz_map,qr,qz
