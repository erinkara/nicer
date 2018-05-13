#!/usr/bin/env python
# vim: shiftwidth=4 
import argparse
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
import matplotlib.pyplot as plt
import sys, os
try:
    import pyfftw
    USE_FFTW = True
except ImportError:
    USE_FFTW = False
from astropy.io import fits
from matplotlib.ticker import ScalarFormatter
from fast_histogram import histogram2d
import pickle

def create_binned_lc(time,pi,enbins,dt):

    tbin = int(time[-1]//dt-time[0]//dt)
    
    if len(enbins)==3:
        all_en_lcs = np.histogram2d(time, pi, bins=(tbin, enbins))[0] 

    if len(enbins)==2:
        ebin = 1
        erange = enbins
        all_en_lcs = histogram2d(time, pi, range=[[time[0],time[-1]],erange], bins=[tbin, ebin]) 
    if len(enbins) > 2 :
        pi = np.log10(pi)   
        ebin = len(enbins)-1
        erange = [np.log10(enbins[0]),np.log10(enbins[-1])] 
        all_en_lcs = histogram2d(time, pi, range=[[time[0]//dt*dt,time[-1]//dt*dt],erange], \
            bins=[tbin, ebin]) 
   
    all_en_lcs = all_en_lcs.T/dt 
    tbin_arr = np.linspace(time[0],time[-1]//dt*dt,tbin, endpoint=False)
    t = tbin_arr+(dt/2)

    t = t[np.where(np.logical_not(np.all(all_en_lcs==0,axis=0)))[0]]
    all_en_lcs = all_en_lcs[:,np.where(np.logical_not(np.all(all_en_lcs==0,axis=0)))[0]]

    return t, all_en_lcs

def main():
    p = argparse.ArgumentParser(
        description="Calculate lag-frequency spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--obsids", metavar="obsids", type=str, default='',help="Space separated list of obsids in chronological order")
    p.add_argument("--dt", metavar="dt", type=float, default='0.01',help="dt, bin size (Default: 0.01 s).") 
    p.add_argument("--segL", metavar="segL", type=float, default='10',help="Segment size in seconds (Default: 10 s).") 
    p.add_argument("--min_freq_bins", metavar="min_freq_bins", type=float, default=10,\
                help="Minimum number of frequencies per bin. ") 
    p.add_argument("--emin", metavar="emin", type=float, default=0.5,\
                help="Minimum energy for lag-energy analysis ") 
    p.add_argument("--emax", metavar="emax", type=float, default=10,\
                help="Maximum energy for lag-energy analysis ") 
    p.add_argument("--freq_bin"  , metavar="freq_bins", type=str, default=None, \
                help="Frequency binning. If 1 value, then geometric binning at said value, \
                if 2 values, then freq bounds, e.g. '1e-2 5e-2'")
    p.add_argument("--lc",action='store_true',default=False,help="Flag to plot lightcurve.") 
    p.add_argument("--lag",action='store_true',default=False,help="Flag to plot lag.") 
    p.add_argument("--psd",action='store_true',default=False,help="Flag to plot psd.") 
    p.add_argument("--phase",action='store_true',default=False,help="If set, plot phase instead of lag.") 
    p.add_argument("--pfp",action='store_true',default=False,help="Flag for frequency*Power y-axis.") 
    p.add_argument("--leahy",action='store_true',default=False,help="Flag for Leahy normalization.") 
    p.add_argument("--noise_subtracted",action='store_true',default=False,help="Flag for noise-subtracted PSD") 
    p.add_argument("--enbins"  , metavar="enbins", type=str, default='0.3 10', \
                 help="Soft band energies, e.g. '0.3 10'")
    args = p.parse_args()


    #***************************************************
    obsids = np.array(args.obsids.split())
    evt_files = [obsid + "/ni%s_0mpu7_cl_bcorr.evt"%(obsid) for obsid in obsids]
    obsid_txt = '_'.join(obsids)

    dt = args.dt
    segL_in_seconds = args.segL
    segL=int(segL_in_seconds/dt)  
    min_freq_bins = args.min_freq_bins
    nyquist = 1/(2.*dt)
  
    fbin_txt = '-'.join(args.freq_bin.split())
    enbin_txt = '-'.join(args.enbins.split())
    enbin_true = np.array(args.enbins.split(),float)
    if args.lag and len(enbin_true)==2:
        enbins = str(input("Enter at least 2 energy bins for lag analysis: "))
        enbin_true = np.array(enbins.split(),float)
           
 
    if len(enbin_true)==1:
        enbin_true = np.logspace(np.log10(args.emin),np.log10(args.emax),int(enbin_true))
    enbins = enbin_true*100.

    #***************************************************
        
    lc_gtis, t_gtis = [], []
    for f in np.arange(len(evt_files)):
        print("*******************************")
        print("Loading events file: ", evt_files[f]) 
        EVTpicklefile = "./pickles/evt_%s.p"%(obsids[f])
        if os.path.isfile(EVTpicklefile):
            print("There's a pickle for that")
            time, pi = pickle.load( open( EVTpicklefile, "rb" ) )
        else:
            fs = fits.open(evt_files[f],memmap=True)
            data = fs[1].data
            pi   = [float(x) for x in data.field(12)]
            time = data.field(0)
            #pickle.dump( (time, pi), open( EVTpicklefile, "wb" ) )
            del data
            del fs
        if f==0:
            t0 = time[0]
        t0_arr = np.empty(len(time))
        t0_arr[:] = t0
        time = time - t0_arr

        
        print("Obtaining binned light curve...") 
        LCpicklefile = "./pickles/obsid%s_enbins%s_dt%s.p"%(obsids[f],enbin_txt,dt)
        if os.path.isfile(LCpicklefile):
            print("Pickle lc found...") 
            t, all_en_lcs = pickle.load( open( LCpicklefile, "rb" ) )
        else:     
            print("No Pickle lc. Need to bin light curve...") 
            t, all_en_lcs = create_binned_lc(time,pi,enbins,dt)    
            pickle.dump( (t, all_en_lcs), open( LCpicklefile, "wb" ) )
        del time; del pi

        # get gtis 
        t_edge=np.arange(len(t))[np.insert((t[1:]-t[:-1]>1),0,False)]

        lc_gtis += np.hsplit(all_en_lcs, t_edge)
        t_gtis += np.hsplit(t, t_edge)  

    # segment each GTI into user-defined segment length
    print("Break light curves into segments...")
    lc_seg,t_seg = [],[]
    for i in np.arange(len(t_gtis)):
        nSeg = len(t_gtis[i])//segL
        if len(t_gtis[i]) < segL:
            print("GTI wasn't long enough: ",len(t_gtis[i])*dt," seconds removed") 
        else:
            lc_seg.append(np.hsplit(lc_gtis[i][:,0:nSeg*segL],nSeg))
            t_seg.append(np.hsplit(t_gtis[i][0:nSeg*segL],nSeg))
    lc_seg=np.vstack(np.array(lc_seg)[:]) ; t_seg=np.vstack(np.array(t_seg)[:])

    # get frequency bins and number of bin
    if not args.lc:
        freq_bin = np.array(args.freq_bin.split(),float)
        if len(freq_bin)==1:
            geomBinFactor=freq_bin[0]
            fbounds = []
            f = 1/segL_in_seconds
            while f < nyquist:
                f = f * geomBinFactor
                fbounds.append(f)
            fbounds = np.array(fbounds)

            freq = pyfftw.interfaces.numpy_fft.fftfreq(segL,d=dt)[1:segL//2]
            nbinned=np.histogram(freq, bins=fbounds)[0]

            b=0
            f_edge=[fbounds[0]]
            for i in np.arange(len(nbinned)):
                b+=nbinned[i]*lc_seg.shape[0]
                #b+=nbinned[i]
                if b >= min_freq_bins:
                    f_edge.append(fbounds[i])
                    b=0
            f_edge=np.array(f_edge)
            fbounds = f_edge
        else:
            fbounds = freq_bin

        fbin = (fbounds[1:] + fbounds[:-1])/2.       
        dfbin = (fbounds[1:] - fbounds[:-1])/2.       

    #print("Exposure:", t_seg.shape[0]*t_seg.shape[1]/1000., "sec")

    if args.psd:
        print("Making PSD...")
        mean_lc = np.mean(lc_seg,axis=2)
        fs= pyfftw.interfaces.numpy_fft.fft(lc_seg)[:,:,1:segL//2]
        freq = pyfftw.interfaces.numpy_fft.fftfreq(segL,d=dt)[1:segL//2]
        powspec = np.abs(fs)**2
        if args.leahy:
            powspec = 2*dt/(segL*mean_lc[:,:,np.newaxis])*powspec
            if args.noise_subtracted:
                powspec = powspec - 2.
        else:
            powspec = 2*dt/(segL*mean_lc[:,:,np.newaxis]**2)*powspec
            if args.noise_subtracted:
                powspec = powspec - 2./mean_lc[:,:,np.newaxis]

        # is there a more pythonic way to do this without loop???
        all_psds=[]
        all_dpsds=[]
        for e_index in np.arange(len(enbins)-1):
            all_psds.append(binned_statistic(freq,powspec[:,e_index],statistic='mean',bins=fbounds)[0])
        avg_psd = np.nanmean(np.array(all_psds), axis=1)
        m = np.histogram(freq, bins=fbounds)[0]*lc_seg.shape[0]
        avg_dpsd = avg_psd/np.sqrt(m)

        if args.pfp:
            p = fbin*avg_psd
            dp = fbin*avg_dpsd
            label = "Power * Frequency"
        else:
            p = avg_psd
            dp = fbin*avg_dpsd
            label = "Power"

        print("Saving to ascii file")
        fp = open( './ascii_4vsz/PSD_o%s_E%s_F%s_4vsz.dat'%(obsid_txt,enbin_txt,fbin_txt), 'w' )
        out = 'descriptor freq_%s,+- '%(obsid_txt)+' '.join(['psd_%s'%(obsid_txt)+'_e%s-%s,+-'%(enbin_true[i],enbin_true[i+1]) for i in range(len(enbin_true)-1)])+'\n'

        out += ''.join(['%3.3g %3.3g %s\n'%(fbin[i],dfbin[i],''.join(['%3.3g %3.3g '%(p[j][i],dp[j][i]) for j in range(len(p)) ]) ) for i in range(len(fbin))])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for e in range(len(avg_psd)):
            ax.errorbar(fbin,p[e],xerr=dfbin,yerr=dp[e])
            ax.set_ylabel(label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Frequency (Hz)")
       
        fp.write(out)
        fp.close()

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.show()

    if args.lag:
        # is there a way to do this without looping over enbins???
        print("Making lag spectrum...")
        binnedCrossR, binnedCrossI, binnedPowSoft, binnedPowHard, nbinned = [], [], [], [], []
        for i in np.arange(len(enbins)-1):
            
            print("fft #:",i)        
            refBand = lc_seg.sum(axis=1) - lc_seg[:,i]
            f_int = pyfftw.interfaces.numpy_fft.fft(lc_seg)[:,i,1:segL//2]
            f_ref = pyfftw.interfaces.numpy_fft.fft(refBand)[:,1:segL//2]
            freq = pyfftw.interfaces.numpy_fft.fftfreq(segL,d=dt)[1:segL//2]
            del refBand

            c = (2.*dt/segL) * f_int * f_ref.conjugate()
            c_real = c.real
            c_imag = c.imag
            pows = (2.*dt/segL) * np.abs(f_int)**2
            powh = (2.*dt/segL) * np.abs(f_ref)**2
            del f_int; del f_ref

            binnedCrossR.append(binned_statistic(freq, c_real, statistic='mean', bins=fbounds)[0])
            binnedCrossI.append(binned_statistic(freq, c_imag, statistic='mean', bins=fbounds)[0])
            binnedPowSoft.append(binned_statistic(freq, pows, statistic='mean', bins=fbounds)[0])
            binnedPowHard.append(binned_statistic(freq, powh, statistic='mean', bins=fbounds)[0])
            nbinned.append(np.histogram(freq, bins=fbounds)[0])
    
        avgc_real = np.nanmean(np.array(binnedCrossR), axis=1)
        avgc_imag = np.nanmean(np.array(binnedCrossI), axis=1)
        avgc = np.array(avgc_real) + 1j*np.array(avgc_imag)
        avgpows = np.nanmean(binnedPowSoft, axis=1)
        avgpowh = np.nanmean(binnedPowHard, axis=1)
        totalBins = np.array(nbinned)*lc_seg.shape[0]
        
        phase = np.angle(avgc)
        lag = phase/(2*np.pi*fbin)
        g = abs(avgc)**2/(avgpows * avgpowh)
        dphase = totalBins**(-0.5) * np.sqrt((1.-g)/(2.*g))
        dlag = dphase/(2*np.pi*fbin)
        
        lag,dlag=np.squeeze(lag),np.squeeze(dlag)

        # lag-energy spectrum
        if len(enbin_true)-1>2:

            e = (enbin_true[1:] + enbin_true[:-1])/2.0 
            de = (enbin_true[1:] - enbin_true[:-1])/2.0

            fp = open( './ascii_4vsz/LAGE_o%s_E%s_F%s_4vsz.dat'%(obsid_txt,enbin_txt,fbin_txt), 'w' )
            out = 'descriptor en_%s,+- lagen_%s,+-\n'%(obsid_txt,obsid_txt)
            out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(e[i],de[i],\
            -lag[i],dlag[i]) for i in range(len(e)))
            fp.write(out)
            fp.close()


            fig = plt.figure()
            ax = fig.add_subplot(111)
            if args.phase:
                ax.errorbar(e, -phase,xerr=de,yerr=dphase,
                    capsize=0,linestyle='None',marker='o',color='black')
                ax.set_ylabel("Phase (rad)")
            else:
                ax.errorbar(e, -lag,xerr=de,yerr=dlag,
                    capsize=0,linestyle='None',marker='o',color='black')
                ax.set_ylabel("Lag (s)")
            ax.set_xscale('log')
            ax.set_xlabel("Energy (keV)")
            ax.set_xlim(e[0]-de[0],e[-1]+de[-1])
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            #plt.savefig('%s_lagen.eps'%(name), format='eps')
            plt.show()
        # lag-frequency spectrum
        else:

            fp = open( './ascii_4vsz/LAGF_o%s_E%s_F%s_4vsz.dat'%(obsid_txt,enbin_txt,fbin_txt), 'w' )
            out = 'descriptor freql_%s,+- lag_%s,+-\n'%(obsid_txt,obsid_txt)
            out += ''.join('%3.3g %3.3g %3.3g %3.3g\n'%(fbin[i],dfbin[i],\
            lag[0][i],dlag[0][i]) for i in range(len(fbin)))
            fp.write(out)
            fp.close()


            fig = plt.figure()
            ax = fig.add_subplot(111)
            if args.phase:    
                ax.errorbar(fbin,lag[0],xerr=dfbin,yerr=dlag[0],    
                    capsize=0,linestyle='None',marker='o',color='black')
                ax.set_ylabel("Phase (rad)")
            else:    
                ax.errorbar(fbin,lag[0],xerr=dfbin,yerr=dlag[0],    
                    capsize=0,linestyle='None',marker='o',color='black')
                ax.set_ylabel("Lag (s)")
            ax.set_xscale('log')
            ax.set_xlabel("Frequency (Hz)")
            l = plt.axhline(y=0)
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            #plt.savefig('%s_lagen.eps'%(name), format='eps')
            plt.show()
 
    if args.lc:

        fig = plt.figure()
        ax = fig.add_subplot(111)
        lcs_interp=[]
        all_uniform_t=[]
        for gti in np.arange(lc_seg.shape[0]): 
            uniform_t = np.arange(t_seg[gti][0], t_seg[gti][-1], dt)
            f = interpolate.interp1d(t_seg[gti], lc_seg[gti],fill_value="extrapolate",kind='cubic')
            lc_interp=f(uniform_t)
            lcs_interp.append(lc_interp)
            all_uniform_t.append(uniform_t)
            
            for lc in lc_interp:
                ax.plot(uniform_t,lc,marker='o',linestyle=None,markersize=1)
        plt.show()

    

if __name__ == "__main__":
    import ipdb

    main()

