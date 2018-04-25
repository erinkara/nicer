#!/usr/bin/env python
# vim: shiftwidth=4 
import argparse
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
try:
    import pyfftw
    USE_FFTW = True
except ImportError:
    USE_FFTW = False
#import pyfits
from astropy.io import fits
from matplotlib.ticker import ScalarFormatter
from fast_histogram import histogram2d


def main():
    p = argparse.ArgumentParser(
        description="Calculate lag-frequency spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--evt_files", metavar="evt_files", type=str, default='',help="String with names of input files")
    p.add_argument("--dt", metavar="dt", type=float, default='0.01',help="dt, bin size (Default: 0.01 s).") 
    p.add_argument("--segL", metavar="segL", type=float, default='10',help="Segment size in seconds (Default: 10 s).") 
    p.add_argument("--min_freq_bins", metavar="min_freq_bins", type=float, default=10,\
                help="Minimum number of frequencies binned.") 
    p.add_argument("--freq_bin"  , metavar="freq_bins", type=str, default=None, \
                help="Frequency binning. If 1 value, then geometric binning at said value, \
                if 2 values, then freq bounds, e.g. '1e-2 5e-2'")
    p.add_argument("--lc",action='store_true',default=False,help="Flag to plot lightcurve.") 
    p.add_argument("--lag",action='store_true',default=False,help="Flag to plot lag.") 
    p.add_argument("--psd",action='store_true',default=False,help="Flag to plot psd.") 
    p.add_argument("--pfp",action='store_true',default=False,help="Flag for frequency*Power y-axis.") 
    p.add_argument("--leahy",action='store_true',default=False,help="Flag for Leahy normalization.") 
    p.add_argument("--noise_subtracted",action='store_true',default=False,help="Flag for noise-subtracted PSD") 
    p.add_argument("--enbins"  , metavar="enbins", type=str, default='0.3 10', \
                 help="Soft band energies, e.g. '0.3 10'")
    args = p.parse_args()


    #***************************************************
    evt_files = np.array(args.evt_files.split())
    dt = args.dt
    segL_in_seconds = args.segL
    segL=int(segL_in_seconds/dt)  
    min_freq_bins = args.min_freq_bins
    nyquist = 1/(2.*dt)
    

    enbin_true = np.array(args.enbins.split(),float)
    if args.lag and len(enbin_true)==2:
        enbins = str(input("Enter at least 2 energy bins for lag analysis: "))
        enbin_true = np.array(enbins.split(),float)
    
    if len(enbin_true)==1:
        enbin_true = np.logspace(np.log10(0.5),np.log10(10),int(enbin_true))
    enbins = enbin_true*100.
    enbins = [int(i) for i in enbins]
   
 
    #***************************************************
    
    lc_gtis, t_gtis = [], []
    for f in np.arange(len(evt_files)):
        print("Loading events file: ", evt_files[f]) 
        fs = fits.open(evt_files[f],memmap=True)
        data = fs[1].data
        pi   = data.field(12)
        time = data.field(0)
        del data
        del fs

        if f==0:
            t0 = time[0]
        t0_arr = np.empty(len(time))
        t0_arr[:] = t0
        time = time - t0_arr

        tbin = np.arange(time[0],time[-1]//dt)*dt
     
        # create binned light curves (in energy and time)
        # FASTER THAN HISTOGRAM???

        # NEEDS TO TAKE SCALAR BINS NOT ARRAYS
        #all_en_lcs, tbin, enbins = histogram2d(time, pi, bins=(tbin, enbins), range=[[tbin[0],tbin[-1]],[enbins[0],enbins[-1]]]) 
        print("Creating binned light curve...") 
        all_en_lcs, tbin, enbins = np.histogram2d(time, pi, bins=(tbin, enbins)) 
        all_en_lcs = all_en_lcs.T/dt 
        t = tbin[:-1]+(dt/2)
       
        t = t[all_en_lcs[0] !=0]
        all_en_lcs = all_en_lcs[:,all_en_lcs[0]!=0]

        # get gtis 
        t_edge=np.arange(len(t))[np.insert((t[1:]-t[:-1]>1),0,False)]

        lc_gtis += np.hsplit(all_en_lcs, t_edge)
        t_gtis += np.hsplit(t, t_edge)  

        # THINK ABOUT WHETHER TO LOOP OVER EVT FILES FOR LAG AND PSD ANALYSIS, TOO

    # segment each GTI into user-defined segment length
    # alternatives to loop?
    print("Segmenting light curves...")
    lc_seg,t_seg = [],[]
    for i in np.arange(len(t_gtis)):
        nSeg = len(t_gtis[i])//segL
        if len(t_gtis[i]) < segL:
            print("The chosen segment length excludes ",len(t_gtis[i])*dt," seconds of data") 
        else:
            lc_seg.append(np.hsplit(lc_gtis[i][:,0:nSeg*segL],nSeg))
            t_seg.append(np.hsplit(t_gtis[i][0:nSeg*segL],nSeg))
    lc_seg=np.vstack(np.array(lc_seg)[:]) ; t_seg=np.vstack(np.array(t_seg)[:])

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

    print("Exposure:", t_seg.shape[0]*t_seg.shape[1]/1000., "sec")

    if args.psd:
        print("Making PSD...")
        mean_lc = np.mean(lc_seg,axis=2)
        fs= pyfftw.interfaces.numpy_fft.fft(lc_seg)[:,:,1:segL//2]
        freq = pyfftw.interfaces.numpy_fft.fftfreq(segL,d=dt)[1:segL//2]
        if args.leahy:
            powspec = 2*dt/(segL*mean_lc[:,:,np.newaxis])*np.abs(fs)**2
            if args.noise_subtracted:
                powspec = powspec - 2.
        else:
            powspec = 2*dt/(segL*mean_lc[:,:,np.newaxis]**2)*np.abs(fs)**2
            if args.noise_subtracted:
                powspec = powspec - 2./mean_lc[:,:,np.newaxis]

        # is there a more pythonic way to do this without loop???
        all_psds=[]
        for e_index in np.arange(len(enbins)-1):
            all_psds.append(binned_statistic(freq,powspec[:,e_index],statistic='mean',bins=fbounds)[0])
        avg_psd = np.nanmean(np.array(all_psds), axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for en_psd in avg_psd:
            if args.pfp:
                ax.step(fbin,fbin*en_psd)
                ax.set_ylabel("Power*Frequency")
            else:
                ax.step(fbin,en_psd)
                ax.set_ylabel("Power")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Frequency (Hz)")

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        plt.show()

    if args.lag:
        # is there a way to do this without looping over enbins???
        binnedCrossR, binnedCrossI, binnedPowSoft, binnedPowHard, nbinned = [], [], [], [], []
        for i in np.arange(len(enbins)-1):
            
            refBand = lc_seg.sum(axis=1) - lc_seg[:,i]
            f_int = pyfftw.interfaces.numpy_fft.fft(lc_seg)[:,i,1:segL//2]
            f_ref = pyfftw.interfaces.numpy_fft.fft(refBand)[:,1:segL//2]
            freq = pyfftw.interfaces.numpy_fft.fftfreq(segL,d=dt)[1:segL//2]

            c = (2.*dt/segL) * f_int * f_ref.conjugate()
            c_real = c.real
            c_imag = c.imag
            pows = (2.*dt/segL) * np.abs(f_int)**2
            powh = (2.*dt/segL) * np.abs(f_ref)**2

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

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(e, -lag,xerr=de,yerr=dlag,
                capsize=0,linestyle='None',marker='o',color='black')
            ax.set_xscale('log')
            ax.set_xlabel("Energy (keV)")
            ax.set_ylabel("Lag")
            ax.set_xlim(e[0]-de[0],e[-1]+de[-1])
            for axis in [ax.xaxis, ax.yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            #plt.savefig('%s_lagen.eps'%(name), format='eps')
            plt.show()
        # lag-frequency spectrum
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(fbin,lag[0],xerr=dfbin,yerr=dlag[0],    
                capsize=0,linestyle='None',marker='o',color='black')
            ax.set_xscale('log')
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Lag (s)")
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

    plt.close()        
    

if __name__ == "__main__":
    import ipdb

    main()

    sys.exit()
