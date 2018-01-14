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
from operator import itemgetter
from itertools import groupby, chain
from urllib.request import urlretrieve
from astropy import units as u
from lmfit.models import LorentzianModel

DR13_HTML = 'https://data.sdss.org/sas/dr13/'
APOGEE_REDUX = 'apogee/spectro/redux/'
DR13_DATABASE= 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allStar-130e.2.fits'
LOCAL_DATABASE='/Users/Jessica/Downloads/allStar-l30e.2.fits'
HOME= '/Users/Jessica/Desktop/APOGEE_SPECS/'

class Spectrum():
  #apogee spectrum object (made of all combined spectra)
  def __init__(self, **kwargs):
      self.combined= kwargs.get("combined", [])
      self.sky= kwargs.get("sky", [])
      self.wave= kwargs.get("wave", [])
      self.visits=kwargs.get("visits", [])
      self.nvisits=kwargs.get("nvisits", 0)
      
def readSpectrum(**kwargs):
    """ Information contained in APOGEE files
    HDU0: master header with target information
    HDU1: spectra: combined and individual
    HDU2: error spectra
    HDU3: mask spectra
    HDU4: sky spectra
    HDU5: sky error spectra
    HDU6: telluric spectra
    HDU7: telluric error spectra
    HDU8: table with LSF coefficients
    HDU9: table with RV/binary information
    Information found at https://data.sdss.org/datamodel/files/APOGEE_REDUX/APRED_VERS/APSTAR_VERS/TELESCOPE/LOCATION_ID/apStar.html
    """
    
    filename=kwargs.get('filename', 'apStar-test.fits')
    master = fits.open(HOME+filename)[0]
    spectra = fits.open(HOME+filename)[1]
    noise = fits.open(HOME+filename)[2]
    sky = fits.open(HOME+filename)[4]
    sky_err = fits.open(HOME+filename)[5]
    telluric = fits.open(HOME+filename)[6]
    tell_err = fits.open(HOME+filename)[7]
    rv_info = fits.open(HOME+filename)[9]

    " conversion from pixel to wavelength, info available in the hdu header"
    crval=spectra.header['CRVAL1']
    cdelt= spectra.header['CDELT1']
    wave=np.array(pow(10, crval+cdelt*np.arange(spectra.header['NAXIS1']))/10000)*u.micron #microns
    # convert fluxes from  (10^-17 erg/s/cm^2/Ang) to  ( erg/s/cm^2/Mircon)
    spectras=[1e-13*np.array(f)*u.erg/u.s/u.centimeter**2/u.micron for f in spectra.data]
    noises= [1e-13*np.array(f)*u.erg/u.s/u.centimeter**2/u.micron for f in noise.data]
    skys=[1e-13*np.array(f)*u.erg/u.s/u.centimeter**2/u.micron for f in sky.data]
   #create a splat Spectrum object just for the combine
    combined= splat.Spectrum(wave=wave, flux= spectra.data[0], noise= noise.data[0])
    #create APOGEE spectrum object
    sp= Spectrum(wave=combined.wave, combined=combined, noise=noises, sky=skys, visits= spectras, nvisits = len(spectra.data))
    return sp

def smooth(spectrum, sky):
    """
    smoothing the spectra using the sky lines
    input: splat spectrum object and sky flux array
    retruns: splat spectrum object smoothed and sky
    """
     #remove the high noises at the filters
    gaps=[[0, 1.5144], [1.58, 1.5859], [1.6430, 1.6502], [1.695, 1.7]]
    masks1=[np.where((spectrum.wave.value>x[0]) & (spectrum.wave.value<x[1]))[0] for x in gaps]
    masks1= np.array(np.concatenate(masks1, axis=0))
    spectrum.flux.value[masks1] = np.nan
    #remove sky
    sky=np.nan_to_num(sky)
    masks0=np.where((abs(sky.value) >= abs(2*np.std(sky.value))))[0]
    spectrum.flux.value[masks0] = np.nan
    # now find where sigma flux  is still high
    flux=np.nan_to_num(spectrum.flux.value)
    masky= np.where(((abs(flux) > 5* abs(np.std(flux))) == True))
    spectrum.flux.value[masky] = np.nan
    #fit a polynomial to the flux
    flux=np.nan_to_num(spectrum.flux.value)
    pol=np.poly1d(np.polyfit(spectrum.wave, flux, 1000))
    initial=spectrum.wave.value[0]
    final=spectrum.wave.value[-1]
    polynomial= pol(np.linspace(initial, final, len(spectrum.wave)))
    ratio=  polynomial/spectrum.flux
    ratio_wonans= np.nan_to_num(ratio.value) #convert nans to zeros for comparison
    masksfinal= np.where((ratio_wonans < 0.5) |(ratio_wonans > 1.5))[0]
    spectrum.flux.value[masksfinal]= np.nan
    plt.plot(spectrum.wave, polynomial)
    #
    return spectrum, sky
    
def fitcurve(w, f, xrange, nistlines, **kwargs):
    w = w[xrange]
    f = f[xrange]
    x = np.array(w)
    y = np.array(-f)+np.max(np.array(f))

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
    
    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, y, 'bo', label='Inverted')
    plt.plot(x, out.best_fit, 'r-', color='g', label='Best Lorentz fit')
    line_error = []
    for i in range(len(nistlines)):
        line_error.append(center-nistlines[i])
        plt.axvline(x=nistlines[i], ymin=0, ymax=1, linewidth=.5, color = 'r')
    plt.axvline(x=center-fwhm, ymin=0, ymax=1, linewidth=.5, color = 'k')
    plt.axvline(x=center+fwhm, ymin=0, ymax=1, linewidth=.5, color = 'k')
    
    if kwargs.get('show') == True:
        plt.show()
        print('Center of function: ',center)
        print('Fit iterations: ',iterations)
        print('Chi-squared: ',chi_sq)
        print('Wavelength range: ',w[0].value,'-',w[-1].value)
    else:
        plt.close() #don't show plot
        
    return center, amp, fwhm, chi_sq, line_error
 
"""Takes the numerical derivative of the data and outputs the local extremum"""   
def find_extremum(w, f, xrange):
    x = w[xrange]
    y = f[xrange]
    slopes = (np.diff(y).value/np.diff(x).value) #d(flux)/d(wavelenth)
    min_zero = np.where(np.diff(np.sign(slopes)) > 0)[0] # returns element position before neg->pos sign change
    max_zero = np.where(np.diff(np.sign(slopes)) < 0)[0]
    return min_zero
    
def loadlines(**kwargs):
    max_energy = kwargs.get('me')
    xzoom = kwargs.get('xzoom')
    
    #Load APOGEE spectra
    file = kwargs.get('file') 
    obj_name = list(file.split('.fits'))[0].split('r6-')[1] #2M00022661+6411431
    sp = readSpectrum(filename=file) #spectrum and radial velocity
    sp.combined
    sp_smooth, h= smooth(sp.combined, sp.sky[0])
    
    #Load NIST data from saved csv
    nistdata = pd.read_csv('/Users/Jessica/Desktop/NIST/nistlines'+str(max_energy)+'.csv')
    nistlines = nistdata['Wavelength']
    nistlines = [float(item) for item in list(nistlines)]
    linenames = list(nistdata['Line'])
    
    zoom_mask = np.where((sp_smooth.wave.value >= xzoom[0]) & (sp_smooth.wave.value <= xzoom[1]))
    w = sp_smooth.wave[zoom_mask] #smoothed wavelength
    f = sp_smooth.flux[zoom_mask] #smoothed flux
    
    return w, f, obj_name, nistlines, linenames, xzoom
    
def def_continuum(w,f):
    """Define continuum as a line equal to the mean of the flux"""
    flux_avg = np.mean([val.value for val in f if str(val.value) != 'nan']) #continuum line of unmasked flux values
    continuum = flux_avg
    sigma = np.std([val.value for val in f if str(val.value) != 'nan'])
    dip_mask = np.where(f.value <= continuum) #list positions of f, where flux is less than the mean
    dips = [] #all points below the mean
    #Find groups of points where flux is below average
    for k, g in groupby(enumerate(dip_mask[0]), lambda ix : ix[0] - ix[1]):
        dips.append(list(map(itemgetter(1), g)))
    #See if the feature is deeper than 1 standard deviation
    features = [] #list positions of all dips that pass below 1 stdev
    for i in range(len(dips)):
        for j in range(len(dips[i])):
            if f[dips[i][j]].value < continuum-sigma:
                features.append(dips[i])
                break
            else: pass
        
    return features, continuum, sigma, flux_avg
    
#def displayFeatures(**kwargs):
#    #kwargs: file, me, xzoom, showcurves(t/f), showspec(t/f), showlines(t/f), showerror(t/f)
#    w, f, obj_name, nistlines, linenames, xzoom = loadlines(file='apStar-r6-2M00011850+7441400.fits',me=30,xzoom=[1.52,1.53])
#    features, continuum, sigma, flux_avg = def_continuum(w,f)
    
if __name__ == '__main__':
    
    regions = [[1.514823381871448, 1.5799960455729714],[1.5862732478022437, 1.6402951470486746],[1.6502053850596359, 1.689981692712261]] #1.669075568137888
    w, f, obj_name, nistlines, linenames, xzoom = loadlines(file='apStar-r6-2M00011850+7441400.fits',me=30,xzoom=regions[1])
    
#    fn = [x for x in f if str(x.value) != 'nan']
#    print(fn[0], fn[-1])
#    index1 = list(f).index(fn[0])
#    index2 = list(f).index(fn[-1])
#    print(w[index1],w[index2])

    features, continuum, sigma, flux_avg = def_continuum(w,f)
    
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
        lines_in_range = np.where((np.array(nistlines) >= w[x0[i]].value) & (np.array(nistlines) <= w[x1[i]].value))[0]
        nearlines = [nistlines[i] for i in lines_in_range]
        pos_line.append(nearlines)
        fposition.append(lines_in_range)
        center, amp, fwhm, chi_sq, err = fitcurve(w, f, features[i], nearlines, show=True)
        for j in range(len(nearlines)):   
            centers.append(center)
        line_error.append(err)
        print('Number of peaks: ', len(find_extremum(w, f, features[i])))
    pos_line = list(chain(*pos_line)) #flatten list of lists into one
    fposition = list(chain(*fposition))
    line_error = list(chain(*line_error))
    
    """Throw away lines that are not within a maximum error"""
    max_error = 1e-5
    idlines = [] #nistlines that are within the max error of a feature
    unidfeat = [] #wavelengths of undefined features (no line found in NIST)
    highlines = [linenames[fposition[i]] for i in range(len(fposition))] 
                
    """Plot Spectrum and NIST lines"""
#    print('\nDisplaying all NIST lines with energy < '+str(max_energy)+',000')
    fig=plt.figure(figsize=(9, 5))
    #Plot continuum and cutoff lines
    plt.plot(w, [continuum for i in range(len(w))], color='k')
    plt.plot(w, [flux_avg-sigma for i in range(len(w))], color='m')
    #Plot all nist lines in light red
    for i in range(len(nistlines)):
        plt.axvline(x=nistlines[i], ymin=0, ymax=1, linewidth=.5, color='r', alpha=.2)
    for i in range(len(pos_line)):
        #Hightlight lines GREEN if they are within a max error
        if abs(line_error[i]) < max_error:
            plt.axvline(x=pos_line[i], ymin=0, ymax=1, linewidth=.5, color='g')
            plt.text(pos_line[i], flux_avg+25, str(highlines[i]))
        #Highlight lines RED if the line is nearby, but not within max error
        else:
            plt.axvline(x=pos_line[i], ymin=0, ymax=1, linewidth=.5, color='r')
            unidfeat.append(centers[i])
    plt.plot(w, f, color= 'b',label=obj_name) #plot smoothed spectrum
#    plt.plot(sp.combined.wave, sp.combined.flux, color='c', label='Original')
    plt.legend(loc='upper right')
    plt.xlim(xzoom)
    plt.tight_layout()

    unidfeat = list(set(unidfeat)) #remove repeated elements
    print('Undefined features around', unidfeat)
    
    plt.show()
    
    t = Table()
    t['NIST Line'], t['Wavelength'], t['Error'] = highlines, pos_line, line_error
    print(t)