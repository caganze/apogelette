import numpy as np
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import csv
import random
import math
import glob
import splat
import os
import urllib
import collections
from operator import itemgetter
from itertools import groupby, chain
#from urllib.request import urlretrieve
from astropy import units as u
from lmfit.models import LorentzianModel

DR13_HTML = 'https://data.sdss.org/sas/dr13/'
APOGEE_REDUX = 'apogee/spectro/redux/'
DR13_DATABASE= 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allStar-130e.2.fits'
LOCAL_DATABASE='allStar-l30e.2.fits'
HOME= 'C:\\Users\\caganze\\Desktop\\Research\\APOGEE\\SPECTRA\\'

Flux_units=u.erg/u.s/u.centimeter**2/u.micron
class Spectrum():
  #apogee spectrum object (made of all combined spectra)
    def __init__(self, **kwargs):
        self.sky = kwargs.get("sky", [])
        self.wave = kwargs.get("wave", [])
        self.flux = kwargs.get("flux", [])
        self.visits = kwargs.get("visits", [])
        self.nvisits = kwargs.get("nvisits", 0)

regions = [[1.514823381871448, 1.5799960455729714],[1.5862732478022437, 1.6402951470486746],[1.6502053850596359, 1.689981692712261]]
gaps = [[1.5,1.514823381871448],[1.5799960455729714,1.5862732478022437],[1.6402951470486746,1.6502053850596359],[1.689981692712261,1.7]]

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
            print self.spectrum.wave
            wave_= np.array(self.spectrum.wave)[np.where((self.spectrum.wave> r[0]) & (self.spectrum.wave < r[1]))[0]]
            flux_ = np.array(self.spectrum.flux)[np.where((self.spectrum.wave > r[0]) & (self.spectrum.wave < r[1]))[0]]
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

        return readSpectrum(self.file)

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
            print'Center of function: ',center
            print'Fit iterations: ',iterations
            print'Chi-squared: ',chi_sq
            print'Wavelength range: ',w[0],'-',w[-1]
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


filepath = HOME+'apStar-r6-2M05320969+2754534.fits'
smooth = smoothSpectrum(filepath, showCont=True, showSpec=True, xzoom=[regions[0][0], regions[2][1]])
spec = smooth.returnSmoothed()

#select=np.where(spec.wave<1.57)[0]
#print select
#print spec.wave


#sp= Spectrum(wave=np.array(spec.wave)[select], flux=np.array(spec.flux)[select])

print('\nfinished smoothing\n')

removeLines = ['Th I', 'Th II', 'Th III']

linepath ='C:\\Users\\caganze\\Desktop\\Research\\APOGEE\\Absorption_Lines'
iden = Identify(spec, loadNist=True, loadMol=False, xzoom=[1.515, 1.57], removeLines=removeLines, linepath=linepath)
lines = iden.findFeat()

#10e-4
