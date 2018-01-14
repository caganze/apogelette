import numpy as np
import pandas as pd
#from astropy.table import Table, vstack, join
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import splat
import os
import collections
from operator import itemgetter
from itertools import groupby, chain
#from urllib.request import urlretrieve
from astropy import units as u
from lmfit.models import LorentzianModel

from scipy.integrate import quad
from scipy.optimize import curve_fit
import emcee
import corner
import seaborn
#import apogee
seaborn.set_style("ticks")

FIGURES= '/Users/caganze/research/apogee_all/apogelette/data/figures/'
CATALOGS= '/Users/caganze/research/apogee_all/apogelette/data/catalogs/'
SPECTRA_DIR= '/Users/caganze/research/apogee_all/apogelette/data/spectra/'


"""

MAIN APOGEE DATA ANALYSIS CODE
AUTHORS: CHRISTIAN AGANZE AND JESSICA BIRKY


"""
DR13_HTML = 'https://data.sdss.org/sas/dr13/'
APOGEE_REDUX = 'apogee/spectro/redux/'
DR13_DATABASE= 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allStar-130e.2.fits'
LOCAL_DATABASE='allStar-l30e.2.fits'
HOME= '/Users/caganze/Research/APOGEE/SPECTRA/'
DIR='/Users/caganze/Desktop/Research/APOGEE/'
FLUX_UNITS=u.erg/u.s/u.centimeter**2/u.micron

regions = [[1.514823381871448, 1.5799960455729714],[1.5862732478022437, 1.6402951470486746],[1.6502053850596359, 1.689981692712261]]
gaps = [[1.5,1.514823381871448],[1.5799960455729714,1.5862732478022437],[1.6402951470486746,1.6502053850596359],[1.689981692712261,1.7]]


class Spectrum():
  #apogee spectrum object (made of all combined spectra)
    def __init__(self, **kwargs):
        self.sky = kwargs.get("sky", [])
        self.wave = kwargs.get("wave", [])
        self.flux = kwargs.get("flux", [])
        self.visits = kwargs.get("visits", [])
        self.nvisits = kwargs.get("nvisits", 0)
        self.noise= kwargs.get("noise",[])
    def normalize(self, **kwargs):
        scale=kwargs.get('scale', 1)
        #normalization range
        xrng=kwargs.get('xrange', [1.675, 1.68])
        flx=(np.array(self.flux))[np.where((self.wave.value > xrng[0]) & (self.wave.value < xrng[1]))[0]]
        self.flux= (self.flux*scale/np.mean(flx))*FLUX_UNITS
        return self
    def scale(self, s):
        self.flux= s*self.flux
        return self
    def zoom(self, xzoom):
        region=np.where((self.wave.value>xzoom[0])&(self.wave.value<xzoom[1]))[0]
        self.flux= self.flux.value[region]*FLUX_UNITS
        self.wave=self.wave.value[region]*u.micron
        self.noise=self.noise.value[region]*FLUX_UNITS
        #self.sky=self.sky[region]*FLUX_UNITS
        return self
    def smooth(self):
        """
        smoothing the spectra using the sky lines
        input: splat spectrum object and sky flux array
        retruns: splat spectrum object smoothed and sky
        calculate how much of the spectrum you're masking
    
        #
        """
         #remove the high noises at the filters
        gaps=[[0.0, 1.5144], [1.58, 1.5859], [1.6430, 1.6502], [1.695, 1.7]]
        #regions = [[1.514823381871448, 1.5799960455729714],[1.5862732478022437, 1.6402951470486746],[1.6502053850596359, 1.689981692712261]]
        masks1=[ np.where((self.wave.value>(gaps[i])[0]) & (self.wave.value<(gaps[i])[1]))[0] for i in np.arange(4)]
        masks1= np.array(np.concatenate(masks1, axis=0))
        self.flux.value[masks1] = np.nan
        print (masks1)
        
        for r in regions:
           select= np.where((self.wave.value>r[0]) & (self.wave.value<r[1]))[0]
           r_flx=self.flux.value[select]
           sigma=abs(np.std(r_flx))
           print (sigma)
           masks2=np.where((r_flx <1.0* sigma ) | (r_flx < 1.0*sigma))[0]
           self.flux.value[masks2]=np.nan
        
       
        self.plot()
        return self
    def fitModel(self, **kwargs):
        m=Model(**kwargs)
        m.runMCMC(self, guess=m.parameters, **kwargs)
    def plot(self, **kwargs):
        plt.figure(figsize=(12, 4))
        plt.plot(self.wave.value, self.flux.value, 'k')
        if kwargs.get('showNoise', False):
            plt.plot(self.wave, self.noise, 'c')
        #ylim=kwargs.get('ylim', [50, 100])
        #plt.ylim(ylim)
        plt.title(kwargs.get('title', ''))
        plt.show()
        plt.close()

class Continuum():
    def __init__(self,**kwargs):
        self.spectrum = kwargs.get('spectrum')
        self.sigmaNum = kwargs.get('sigmaNum')
        self.xzoom = kwargs.get('xzoom')

    def defContinuum(self, **kwargs):
        showCont = kwargs.get('showCont',False)

        waves, fluxes = [], []
        continuum = []
        for r in regions:
            print (self.spectrum.wave)
            wave_= np.array(self.spectrum.wave)[np.where((self.spectrum.wave.value> r[0]) & (self.spectrum.wave.value < r[1]))[0]]
            flux_ = np.array(self.spectrum.flux)[np.where((self.spectrum.wave.value > r[0]) & (self.spectrum.wave.value < r[1]))[0]]
            waves.append(wave_)
            fluxes.append(flux_)

            #replace np.nans from sky mask with average value
            flux_avg = np.mean([val for val in flux_ if str(val) != 'nan'])
            flux_ = [flux_avg if str(val) == 'nan' else val for val in flux_ ]

            #fit a 1000 degree polynomial to the regions that are not masked out
            pol = np.poly1d(np.polyfit(wave_, flux_, 1000))
            polynomial = pol(np.linspace(wave_[0], wave_[-1], len(wave_)))
            continuum.append(polynomial)
        continuum = np.concatenate(continuum, axis=0) #sum of the 3 polynomials
        waves = list(chain(*waves)) #flatten lists
        fluxes = list(chain(*fluxes))
        sigma = np.std([val for val in fluxes if str(val) != 'nan'])

        peakCutoff = continuum + self.sigmaNum*sigma
        dipCutoff = continuum - self.sigmaNum*sigma

        wave_in_bands, flux_in_bands = waves, fluxes
        if showCont == True:
            plt.plot(waves,fluxes, alpha=.5)
            plt.plot(waves,continuum,color='k')
            plt.plot(waves,peakCutoff)
            plt.plot(waves,dipCutoff)
            plt.xlim(self.xzoom)
            plt.show()
            plt.close()

        return continuum, peakCutoff, dipCutoff, wave_in_bands, flux_in_bands

    def findFeat(self, continuum, peakCutoff, dipCutoff, wave_in_bands, flux_in_bands):
        w, f = wave_in_bands, flux_in_bands

        #Find positions of dips below continuum and peaks above continuum
        dip_mask = [f.index(f[i]) for i in range(len(f)) if f[i] < continuum[i]]
        peak_mask = [f.index(f[i]) for i in range(len(f)) if f[i] > continuum[i]]

        dips = [] #all points below the continuum
        peaks = []

        #Find groups of points where flux is below continuum
        for k, g in groupby(enumerate(dip_mask), lambda ix : ix[0] - ix[1]):
            dips.append(list(map(itemgetter(1), g)))
        for k, g in groupby(enumerate(peak_mask), lambda ix : ix[0] - ix[1]):
            peaks.append(list(map(itemgetter(1), g)))

        #See if the feature is deeper than dipCutoff
        features = [] #list positions of all dips that pass below dipCutoff
        for dip in dips:
            for point in dip:
                if f[point] < dipCutoff[point]:
                    features.append(dip)
                    break
                else: pass

        #See if the feature is higher than peakCutoff
        emissions = [] #list positions of all peaks that pass above dipCutoff
        for peak in peaks:
            for point in peak:
                if f[point] > peakCutoff[point]:
                    emissions.append(peak)
                    break
                else: pass
        return dips, features, emissions


class smoothSpectrum():
    def __init__(self, filepath, **kwargs):
        self.file = filepath
        self.showSpec = kwargs.get('showSpec', False)
        self.showCont = kwargs.get('showCont', False)
        self.xzoom = kwargs.get('xzoom', [regions[0][0], regions[2][1]])

    def readSpectrum(self, **kwargs):

        return readSpectrum(filename=self.file)

    def removeSky(self, spectrum, sky):
        #mask areas outside of bands
        masks1 = [np.where((spectrum.wave.value > gap[0]) & (spectrum.wave.value < gap[1]))[0] for gap in gaps]
        masks1 = np.array(np.concatenate(masks1, axis=0))
        spectrum.flux.value[masks1] = np.nan #remove flux values not in the regions

        #remove sky
        sky = np.nan_to_num(sky)
        masks0 = np.where((abs(sky.value) >= abs(2*np.std(sky.value))))[0]
        spectrum.flux.value[masks0] = np.nan
        noskysp = spectrum

        return noskysp

    def removeOutliers(self, spectrum, **kwargs):
        sp = Spectrum(wave=spectrum.wave.value, flux=spectrum.flux.value) #remove the units from the input spectrum
        fitCont = Continuum(spectrum=sp, sigmaNum=4, xzoom=self.xzoom)
        continuum, peakCutoff, dipCutoff, wave_in_bands, flux_in_bands = fitCont.defContinuum(showCont=self.showCont)
        feat_info = fitCont.findFeat(continuum, peakCutoff, dipCutoff, wave_in_bands, flux_in_bands)
        outlier_masks = feat_info[1] + feat_info[2]
        outlier_masks = list(chain(*outlier_masks))
        outlier_masks = [mask for mask in outlier_masks if str(mask) != '0']

        f = list(spectrum.flux.value)

        remove_mask = [f.index(flux_in_bands[mask]) for mask in outlier_masks]

        for mask in remove_mask:
            f[mask] = np.nan
        f = [np.nan if val == 0 else val for val in f]
        smoothsp = Spectrum(wave=spectrum.wave.value, flux=f)

        return smoothsp

    def returnSmoothed(self, **kwargs):
#        showSpec = kwargs.get('showSpec', False)
#        showCont = kwargs.get('showCont', False)

        sm = smoothSpectrum(self.file, xzoom=self.xzoom, showCont=self.showCont, showSpec=self.showSpec)
        sp, sky = sm.readSpectrum()
        noskysp = sm.removeSky(sp, sky)
        smoothsp = sm.removeOutliers(noskysp)

        if self.showSpec == True:
            plt.plot(sp.wave.value, sp.flux.value, color='b', alpha=.6, label='Original')
            plt.plot(smoothsp.wave, [f+400 for f in smoothsp.flux], color='g', alpha=.6, label='Smoothed')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
            plt.close()

        return smoothsp

class Identify():
    "This scontains routines to help identify lines"
    def __init__(self, spec, **kwargs):
        self.spectrum = spec
        self.loadNist = kwargs.get('loadNist',True)
        self.loadMol = kwargs.get('loadMol',True)
        self.xzoom = kwargs.get('xzoom')
        self.removeLines = kwargs.get('removeLines', [])
        self.linepath = kwargs.get('linepath')

        print(self.xzoom)

    def readNist(self, **kwargs):
        max_energy = kwargs.get('me',30)

        #Load NIST data from saved csv
        nistdata = pd.read_csv(self.linepath+'\\NIST\\nistlines'+str(max_energy)+'.csv')
        nistlines = nistdata['Wavelength']
        nistlines = [float(item) for item in list(nistlines)]
        linenames = list(nistdata['Line'])

        #Remove lines that we don't want to see
        goodlinepos = [i for i in range(len(linenames)) if linenames[i] not in self.removeLines]
        nistlines = np.array(nistlines)[goodlinepos]
        linenames = np.array(linenames)[goodlinepos]

        return nistlines, linenames


    def readMol(self, **kwargs):
        molpath = self.linepath + '\\Molecular\\'
        molfiles = os.listdir(molpath) #['CO2.csv',...]

        def wl(wavenumber): #convert wavenumber to wavelength
            wavelength = 10000/wavenumber
            return wavelength

        mollines, molnames = [], []
        for file in molfiles:
            lines_in_file = list(wl(pd.read_csv(molpath+file)['Wavenumber']))
            mollines.append(lines_in_file)
            for i in lines_in_file:
                molnames.append(file.split('.')[0])
        mollines = list(chain(*mollines)) #flatten list

        return mollines, molnames

    def fitcurve(self, w, f, xrange, nistlines, **kwargs):
        w = np.array(w)[xrange]
        f = np.array(f)[xrange]
        x = np.array(w)
        y = np.array(-f)+np.max(np.array(f)) #invert the spectrum upside down

        mod = LorentzianModel()
        pars = mod.guess(y, x=x)
        out  = mod.fit(y, pars, x=x)
        report = out.fit_report(min_correl=0.25)

        """extract lorentzian parameters from the report output"""
        center = float(report.split('center:')[1].split('+/-',1)[0]) #x (wavelenth) value of the maximum
        amp = float(report.split('amplitude:')[1].split('+/-',1)[0]) #y (flux) value of the maximum
        fwhm = float(report.split('fwhm:')[1].split('+/-',1)[0]) #full width at half maximum
        #get chi-squared value of the fit
        chi_sq = float(report.split('reduced chi-square')[0].split('chi-square         = ')[1])
        iterations = int(report.split('# data points')[0].split('# function evals   = ')[1])

        #Plot inverted data points, Lorentzian curve, and absorption lines from NIST
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, y, 'bo', label='Inverted')
        plt.plot(x, out.best_fit, 'r-', color='g', label='Best Lorentz fit')
        line_error = []
        for i in range(len(nistlines)):
            line_error.append(center-nistlines[i])
            plt.axvline(x=nistlines[i], ymin=0, ymax=1, linewidth=.5, color = 'r')
        plt.axvline(x=center-fwhm, ymin=0, ymax=1, linewidth=.5, color = 'k')
        plt.axvline(x=center+fwhm, ymin=0, ymax=1, linewidth=.5, color = 'k')

        #Print summary for each fitted curve
        if kwargs.get('showCurv',True) == True:
            plt.show()
            #print'Center of function: ',center
            #print'Fit iterations: ',iterations
            #print'Chi-squared: ',chi_sq
            #print'Wavelength range: ',w[0],'-',w[-1]
        else:
            plt.close() #don't show plot

        return center, amp, fwhm, chi_sq, line_error

    def find_extremum(self, w, f, xrange):
        x = np.array(w)[xrange]
        y = np.array(f)[xrange]
        slopes = np.diff(y)/np.diff(x) #d(flux)/d(wavelenth)
        min_zero = np.where(np.diff(np.sign(slopes)) > 0)[0] # returns element position before neg->pos sign change
        max_zero = np.where(np.diff(np.sign(slopes)) < 0)[0]
        return min_zero

    def findFeat(self):
        #Find continuum of the whole spectrum
        fitCont = Continuum(spectrum=self.spectrum, sigmaNum=1, xzoom=self.xzoom)
        continuum, peakCutoff, dipCutoff, wave_in_bands, flux_in_bands = fitCont.defContinuum(showCont=True)

        #Now look at the piece of the continuum within xzoom
        xrange = [np.where((np.array(wave_in_bands) >= self.xzoom[0]) & (np.array(wave_in_bands) <= self.xzoom[1]))[0]]
        w = np.array(wave_in_bands)[xrange]
        f = np.array(flux_in_bands)[xrange]
        continuum = continuum[xrange]
        dipCutoff = dipCutoff[xrange]

        features = fitCont.findFeat(continuum, peakCutoff, dipCutoff, list(w), list(f))[1]
        features = [feat for feat in features if len(feat) > 3] #take out all features where there is <= 4 data points

        #Find NIST lines within the zoomed region
        zoom_spec = Spectrum(wave=w, flux=f)
        if self.loadNist == True:
            nistlines, linenames = Identify(zoom_spec, loadNist=self.loadNist, loadMol=self.loadMol, xzoom=self.xzoom, removeLines=self.removeLines, linepath=self.linepath).readNist()
        linedict = dict(zip(nistlines, linenames)) #zip nistlines with their names

        mollines=[]
        molnames=[]
        if self.loadMol == True:
            mollines, molnames = Identify(zoom_spec, loadNist=self.loadNist, loadMol=self.loadMol, xzoom=self.xzoom, removeLines=self.removeLines, linepath=self.linepath).readMol()
        linedict.update(dict(zip(mollines, molnames))) #add molecular lines to the dictionary

        linedict = collections.OrderedDict(sorted(linedict.items())) #sort the dictionary with lowest wavelength first
        lines = list(linedict.keys()) #wavelength value of each nist or molecular line
        names = list(linedict.values())

        num_feat = len(features)
        x0 = [features[i][0] for i in range(num_feat)]
        x1 = [features[i][-1] for i in range(num_feat)]

        """For each absorption feature (where flux is less than 1 stdev), fit a Lorentzian curve"""
       #find NIST lines in feature range
        pos_line = [] #list of lines that could correspond to features
        fposition = [] #list positions in nistlines for pos_line
        centers = [] #closest center to each nist line; contains repeated values
        line_error = [] #center - line
        for i in range(num_feat):
            lines_in_range = np.where((np.array(lines) >= w[x0[i]]) & (np.array(lines) <= w[x1[i]]))[0]
            nearlines = [lines[i] for i in lines_in_range]
            pos_line.append(nearlines)
            fposition.append(lines_in_range)
            center, amp, fwhm, chi_sq, err = Identify(self.spectrum).fitcurve(w, f, features[i], nearlines, show=True)
            for j in range(len(nearlines)):
                centers.append(center)
            line_error.append(err)
            print('Number of peaks: ', len(Identify(self.spectrum).find_extremum(w, f, features[i])))
        pos_line = list(chain(*pos_line)) #flatten list of lists into one
        fposition = list(chain(*fposition))
        line_error = list(chain(*line_error))

        """Throw away lines that are not within a maximum error"""
        max_error = 1e-2
        idlines = [] #lines that are within the max error of a feature
        unidfeat = [] #wavelengths of undefined features (no line found in NIST)
        highlines = [names[fposition[i]] for i in range(len(fposition))]

        """Plot Spectrum and NIST lines"""
        fig = plt.figure(figsize=(9, 5))

        #Plot all nist lines in light red
        for i in range(len(lines)):
            plt.axvline(x=lines[i], ymin=0, ymax=1, linewidth=.5, color='r', alpha=.2)
        for i in range(len(pos_line)):
            #Hightlight lines GREEN if they are within a max error
            if abs(line_error[i]) < max_error:
                plt.axvline(x=pos_line[i], ymin=0, ymax=1, linewidth=.5, color='g')
                plt.text(pos_line[i], continuum[i]+20, str(highlines[i]))
            #Highlight lines RED if the line is nearby, but not within max error
            else:
                plt.axvline(x=pos_line[i], ymin=0, ymax=1, linewidth=.5, color='r')
                unidfeat.append(centers[i])

        #Plot spectrum, continuum and cutoff lines
        plt.plot(w, f, color='b', alpha=.8, label='spectrum') #plot smoothed spectrum
        plt.plot(w, [c for c in continuum], color='k')
        plt.plot(w, [d for d in dipCutoff], color='m')
        plt.legend(loc='upper right')
        plt.xlim(self.xzoom)
        plt.tight_layout()

        unidfeat = list(set(unidfeat)) #remove repeated elements
        print('Undefined features around', unidfeat)

        plt.show()

        t = Table()
        t['NIST Line'], t['Wavelength'], t['Error'] = highlines, pos_line, line_error
        print(t)
#    def returnid(self):

class Model():
    #model: sum of lorentzians given heights, centers, widths, scales in one array
    #format 
    #paramaters=[h1, h2, ...hn, c1, c2...cn, w1, w2,..wn, s1, s2, ...sn]
    def __init__(self, **kwargs):
        self.shape = kwargs.get("shape", 'lorentzian')
        self.nlines= kwargs.get("nlines", 2)
        self.parameters= kwargs.get("parameters", np.concatenate([[0, 0], [1.66505, 1.66574], [0.00005, 0.0001], [0.0007, 0.001]]))
        self.continuum = kwargs.get("continuum", np.arange(100))
        self.flux= kwargs.get("mflux", [])
    
    def lorentzian(self, x):
        #inputs: height, width, center, scale
        hs=self.parameters[:self.nlines]
        cs=self.parameters[:(self.nlines*2)][(self.nlines):]
        ws=self.parameters[:(self.nlines*3)][(self.nlines*2):]
        ss=self.parameters[:(self.nlines*4)][(self.nlines*3):]
        

        lorentzs= np.sum(np.array([(h-(s*w/((x-c)**2+w**2))) for h, w, c, s in zip(hs,ws, cs, ss)]), axis=0)
        m=self.continuum+lorentzs
        return m
    
    def lnprior(self):
        #use flat priors for all lines
        hs=self.parameters[:self.nlines]
        cs=self.parameters[:(self.nlines*2)][(self.nlines):]
        ws=self.parameters[:(self.nlines*3)][(self.nlines*2):]
        ss=self.parameters[:(self.nlines*4)][(self.nlines*3):]
        
        lp= -np.inf
        hps=np.sum([0 if not np.isnan(h) else lp for h in hs ])
        cps=np.sum([0  if (c-w)<c<(c+w)  else lp for c, w in zip(cs, ws)])
        wps=np.sum([0  if 0<w<0.01 else lp  for w in ws])
        sps=np.sum([0  if 0<s<1 else lp  for s in ss])
        
        lp= hps*cps*wps*sps
        return lp
        
    def lnprob(self, ps, sp):
        md= Model(parameters=ps, continuum=self.continuum, nlines=2)
        model=md.lorentzian(sp.wave.value)
        lprior= md.lnprior()
        lposterior=-0.5*np.sum((sp.flux.value-model)**2/(sp.noise.value**2))
        if not np.isnan(lprior+lposterior):
            return lprior+lposterior
        else:
            return -pow(10, 90)        
    def runMCMC(self, sp, **kwargs):
        #MCMC the lorenztian fit then measure the EW"""
        #create fake data #
        params=kwargs.get('guess', self.parameters)
        params=np.array(params)
        #noise=kwargs.get('noise', spectrum.noise.value)
        ndim=self.nlines*4
        nwalkers=kwargs.get('nwalkers', 10)
        p0=[[] for n in range(0,nwalkers)]
        p0[0]=params
        for n in range(1, nwalkers):
                p0[n]=[p+ np.random.random(1)[0]*0.0001*p for p in params]
        p0= np.array(p0)
        #print 'initial conditions',p0
        print ("nwalkers", nwalkers)
        nsamples=kwargs.get('nsamples', 10000)
        print( "samples", nsamples)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=[sp])
        ps, lnps, rstate= sampler.run_mcmc(p0,nsamples)
        samples = sampler.chain.reshape((-1, ndim))
        pmean=[np.mean((samples.T)[i]) for i in range(0, ndim)]
        #ps_meanunc=[np.std((samples.T)[i]) for i in range(0, ndim)]
        pf= [np.mean(((samples.T)[i])[-10:]) for i in range(0, ndim)]
        #print "Fit parameters and uncertainty.........", zip(ps, ps_ers)
        initial_model= Model(parameters=p0[0], continuum=self.continuum, nlines=self.nlines)
        avg_model= Model(parameters=pmean, continuum=self.continuum, nlines=self.nlines)
        final_model= Model(parameters=pf, continuum=self.continuum, nlines=self.nlines)
        final_model.flux=final_model.lorentzian(sp.wave.value)
        self.flux=final_model.flux
        #USE the distribution to measure EQWS
        eqws=[[np.nan for i in np.arange(self.nlines)]]  #this is just to faciliate appending 
        for p in ps:
            model=Model(parameters=p, continuum=self.continuum, nlines=self.nlines)
            model.flux=model.lorentzian(sp.wave.value)
            ctr, wdth, es=measure_EQW(model, sp.wave.value)
            eqws=np.append(eqws, [es], axis=0)
        #remove the first elt
        eqws= eqws[1:]
        EWs= [np.mean(x) for x in eqws.T]
        Unc= [np.std(x) for x in eqws.T]
        lndata=pd.DataFrame()
        lndata['Center']=ctr
        lndata['Width']=wdth
        lndata['EW']=EWs
        lndata['EW_unc']=Unc
        
        if kwargs.get('show_corner', False):
            plt.figure()
            labels=[[] for i in np.arange(4)]
            labels[0] =["$h_"+str(i)+'$' for i in np.arange(ndim/4)]
            labels[1] =["$C_"+str(i)+'$' for i in np.arange(ndim/4)]
            labels[2] =["$\gamma"+str(i)+'$' for i in np.arange(ndim/4)]
            labels[3] =["$S_"+str(i)+'$' for i in np.arange(ndim/4)]
            ls=np.concatenate(labels)
            fig = corner.corner(samples, labels=ls)
            fig.show()
            plt.figure()
            fig, ax = plt.subplots(ndim, sharex=True, figsize=(12, 6))
            for i in range(ndim):
                ax[i].plot(sampler.chain[:, :, i].T, '-k', alpha=0.2)
            fig.show()
        if kwargs.get('show_fits', False):
            plt.figure()
            plt.plot(sp.wave, sp.flux, 'k', label='SPECTRUM', alpha=0.5)
            plt.plot(sp.wave.value, initial_model.lorentzian(sp.wave.value), '--r', label='GUESS', alpha=0.5)
            plt.plot(sp.wave.value, avg_model.lorentzian(sp.wave.value),  'g' ,label='MEAN FIT')
            plt.plot(sp.wave.value, final_model.flux,  'b',label='FINAL FIT')
            plt.legend()
            plt.show()
       
        return lndata

def measure_eqw(model, wave ):
    #measuring EQW given a  Model and a wwavelegnth array
    cfs=model.parameters[:(model.nlines*2)][(model.nlines):]
    wfs=model.parameters[:(model.nlines*3)][(model.nlines*2):]
    eqws=[]
    for c, w in zip(cfs, wfs):
        select= np.where((c-w<=wave)&(wave<c+w))[0]
        dlambda= np.linspace(c-w, c+w, len(select))
        eqws.append(abs(np.sum((1-(model.continuum[select]/model.flux[select]))*dlambda)))
    return cfs, wfs, eqws

def estimate_distance(**kwargs):
    #apsMag-color relations from bochanski et al. 2010 and schmidt et.al. 2010
    #color=kwargs.get('color', 'r-i')
    rel= kwargs.get('relation',['bochanski2010'])
    
    r, rer=kwargs.get('r', (np.nan, np.nan))
    j, jer=kwargs.get('j', (np.nan, np.nan))
    k, ker=kwargs.get('k', (np.nan, np.nan))
    i, ier=kwargs.get('i', (np.nan, np.nan))
    z, zer=kwargs.get('z', (np.nan, np.nan))
   
    distance=dict()
    if ('bochanski2010' in rel):
        #print "using color-mag relation from Bochanski et al. 2010"
        rel1={'x':(r-z, rer**2+zer**2),'name': 'r-z','coeffs':[5.190, 2.474,  0.4340, -0.08635 ],'unc':  0.394,'range':[0.5, 4.53]}
        rel2={'x':(r-i, rer**2+ier**2),'name': 'r-i','coeffs':[5.025, -4.548, 0.4175, -0.18315 ],'unc':  0.403,'range':[0.62, 2.82]}
        rel3={'x':(i-z, ier**2+zer**2),'name': 'i-z','coeffs':[4.748,  8.275, 2.2789,  -1.5337 ],'unc':  0.403,'range':[0.32, 1.85]}
        for  rel in [rel1, rel2, rel3]:
            if (rel['range'][0]<rel['x'][0]<rel['range'][1]):
                absmag= np.sum([c*rel['x'][0]**p for p, c in enumerate(rel['coeffs'])])
                absmagunc= np.sqrt(np.sum(rel['x'][1], rel['unc']**2))
                d1= pow(10.0, ((r-absmag)/5.0)+1.0)
                #error propagation
                d1_er=(d1*np.log(10)/5)*np.sqrt(absmagunc**2-rer**2)
                
                distance['D('+str(rel['name'])+')']=d1
                distance['D('+str(rel['name']+str(')unc'))]=d1_er
            else:
                print ("color out of range, will try Schmidt2016")
                distance[str('D('+rel['name'])+')']=np.nan
                distance[str('D('+rel['name']+str(')unc'))]=np.nan
                rel=['bochanski2010', 'Schmidt2016']
                
    if ('Schmidt2016' in rel):
        #print "using color-mag relation from Schmidt et al. 2016"
        rel1={'x':(i-z, ier**2+zer**2),'name': 'i-z','coeffs':[7.13, 4.88],'unc':  np.sqrt(0.23**2+0.15**2),'range':[1.0, 2.9]}
        rel2={'x':(i-j, jer**2+ier**2),'name': 'i-j','coeffs':[5.17, 2.61],'unc':  np.sqrt(0.27**2+0.008**2),'range':[2.5, 5.8]}
        rel3={'x':(i-k, ier**2+ker**2),'name': 'i-k','coeffs':[5.41, 1.95],'unc':  np.sqrt(0.30**2+0.007**2),'range':[5.41, 1.95]}
        rel4={'x':(r-z, rer**2+zer**2),'name': 'r-z','coeffs':[5.190, 2.474,  0.4340, -0.08635 ],'unc':  0.394,'range':[0.5, 4.53]}
        #rel5={'x':(r-i, rer**2+ier**2),'name': 'r-i','coeffs':[5.025, -4.548, 0.4175, -0.18315 ],'unc':  0.403,'range':[0.62, 2.82]}
        rel6={'x':(i-z, ier**2+zer**2),'name': 'i-z','coeffs':[4.748,  8.275, 2.2789,  -1.5337 ],'unc':  0.403,'range':[0.32, 1.85]}
        for rel in [rel1, rel2, rel3, rel4,  rel6]:
            if (rel['range'][0]<rel['x'][0]<rel['range'][1]) :
                absmag= np.sum([c*rel['x'][0]**p for p, c in enumerate(rel['coeffs'])])
                absmagunc= np.sqrt(np.sum(rel['x'][1]**2, rel['unc']**2))
                d2= pow(10.0, ((r-absmag)/5.0)+1.0)
                #error propagation
                d2_er=(d2*np.log(10)/5)*np.sqrt(absmagunc**2.0+rer**2.0)
                print (d2, d2_er, np.sqrt(absmagunc**2.0-rer**2.0))
                distance[str('D('+rel['name'])+')']=d2
                distance[str('D('+rel['name']+str(')unc'))]=d2_er
            else:
                print ("outside range")
                distance[str('D('+rel['name'])+')']=np.nan
                distance[str('D('+rel['name']+str(')unc'))]=np.nan

                #distance[str('D('+rel['name'])+')']=np.nan
                #distance[str('D('+rel['name']+str(')unc'))]=np.nan
    return distance
    
def readSpectrum(**kwargs):
   return apogee.readSpectrum(**kwargs)
if __name__=="__main__":
      # A demo of the usage: simple correlation of EW VS SPT
    #1. the lines that I'm measuring 
    wrng=[1.6745, 1.6775 ]
    line1=[1.0, 1.6755, 0.00005, 0.001]
    line2=[1.0, 1.6770, 0.00005, 0.0005]
    lines =np.array([line1, line2]).T
    ps= np.concatenate(lines)
    
    #2. measure the line for all spectra
    sample=pd.read_csv('short_list.csv')
    spts=[]
    eqws=[]
    for  apid, spt in zip(sample['APOGEE_ID'], sample['SPECTTYPE']):
        try:
            print (apid)
            s, sk= readSpectrum(apogee_id=apid)
            s.zoom([1.674, 1.678])  #this is very important
            #cont= Continuum()
            #cont.spectrum=s
            #cont.sigmaNum=5
            #cont=[np.mean(cont.defContinuum()[0]) for w in s.wave]
            cont=np.array([np.mean(s.flux.value[:10]) for w in s.wave])
            model= Model(parameters=ps, continuum=cont, nlines=2)
            lines=model.runMCMC(s, guess=model.parameters, nwalkers=40, nsamples=1000, show_fits=True)    
            eqws.append(lines.ix[0])
            spts.append(splat.typeToNum(spt))
            #h
        except :
             continue
    linesdata=pd.concat(eqws)
    select= [~np.isnan(spts)]
    spts=np.array(spts)[select]
    f=plt.figure()
    ax= f.add_subplot(111)
    index= np.arange(len(linesdata))[select]
    print( index)
    print (spts.shape)
    print (linesdata.ix[index]['EW'])
    
    
    ax.plot( spts, np.array(linesdata.ix[index]['EW']), fmt='o', c='k')

    ax.errorbar(spts, np.array(linesdata.ix[index]['EW']), yerr=np.array(linesdata.ix[index]['EW_unc']), fmt='o', c='k')
    ax.set_xlabel('Spectral Type')
    ax.set_ylabel('EW')