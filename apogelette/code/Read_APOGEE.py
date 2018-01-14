# -*- coding: utf-8 -*-
"""
Simple code for reading APOGEE spectra, donwloading new spectra and searching the database

"""
import numpy as np
from astropy.table import Table, vstack, join
from astropy.io import fits, ascii
import splat
import os
import urllib
import glob
from astropy import units as u
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
from scipy.optimize import curve_fit
import scipy
import emcee
import corner
import random
from lmfit.models import LorentzianModel
from itertools import groupby, chain
import collections
from operator import itemgetter

DR13_HTML = 'https://data.sdss.org/sas/dr13/'
APOGEE_REDUX = 'apogee/spectro/redux/'
DR13_DATABASE= 'https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/allStar-130e.2.fits'
LOCAL_DATABASE='allStarsAPOGEE.fits'
HOME= 'C:\\Users\\caganze\\Desktop\\Research\\APOGEE\\SPECTRA\\'

class Spectrum():
  #apogee spectrum object (made of all combined spectra)
  def __init__(self, **kwargs):
      self.combined= kwargs.get("combined", [])
      self.sky= kwargs.get("sky", [])
      self.wave= kwargs.get("wave", [])
      self.visits=kwargs.get("visits", [])
      self.nvisits=kwargs.get("nvisits", 0)

      # add parameters such as magnitudes, RV etc...
# some presets


def readSpectrum(**kwargs):
    """ Simple demo for plotting Apogee spectra, including visits

    HDU0: master header with target information
    HDU1: spectra: combined and individual
    (Each spectrum has 8575 columns with the spectra, and 2+NVISITS rows.
    The first two rows are both combined spectra, with different weighting:
    first row uses pixel-based weighting, while second uses a more "global" weighting)
    HDU2: error spectra
    HDU3: mask spectra
    HDU4: sky spectra
    HDU5: sky error spectra
    HDU6: telluric spectra
    HDU7: telluric error spectra
    HDU8: table with LSF coefficients
    HDU9: table with RV/binary information

    According to https://goo.gl/rLXEuc
    """

    #not sure
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
    sp= Spectrum(wave=combined.wave, combined=combined, noise=noises, sky=skys, visits= spectras, \
    nvisits = len(spectra.data))
    print

    return sp



def test_catalog():

    """" a bunch of scripts """
    table=ascii.read('apogee_reduced.csv')
    for ind, locid, coords in zip(table['ASPCAP_ID'], table['LOCATION_ID'], table['APOGEE_ID']):
        downloadSpectra(ind,locid, coords)
    smaller= table['CATALOG','CATALOG_NAME', 'SPTYPE',  'APOGEE_ID', 'GLAT', 'J', 'TEFF']
#    for row in smaller:
#        print row

    l2= readSpectrum(filename=HOME+'apStar-r6-2M00452143+1634446.fits')
    m7=readSpectrum(filename=HOME+'apStar-r6-'+'2M07140394+3702459.fits')
    m6=readSpectrum(filename=HOME+'apStar-r6-'+'2M22400144+0532162.fits')
#    m4=readSpectrum(filename=HOME+'apStar-r6-'+'2M05420897+1229252.fits')
    #m3=readSpectrum(filename=HOME+'apStar-r6-'+'2M10121768-0344441.fits')
#    m2=readSpectrum(filename=HOME+'apStar-r6-'+'2M11052903+4331357.fits')
    l1=readSpectrum(filename=HOME+'apStar-r6-'+'2M06154934-0100415.fits')
    #f=plt.figure()
    #ax= f.add_subplot(111)
    #ax.set_xlim([1.5, 1.7])
    #ax.plot(l2.combined.wave, l2.combined.flux, label='$L2$)
    #sp=smooth(l2.combined, l2.sky)

    #splat.plotSpectrum(sp, xrange=[1.5, 1.55])
    spectra=[l2.combined,l1.combined, m7.combined, m6.combined]
    labels=['$L2 \gamma $ J00452143+1634446', '$L1$ J06154934-0100415',
            r'$M7 \beta $ J07140394+3702459', '$M6$ J22400144+0532162']
    colors= ['r', 'g', 'b', 'k']
    counter=0
    for f in glob.glob(HOME+'/*'):
        if counter<8:
            try:
                sp=readSpectrum(filename=f)
                c='c'
                spectra.append(sp.combined)
                colors.append(c)
                labels.append('unknown')
            except:
                continue
        counter=counter+1

    splat.plotSpectrum(spectra, filename='apogee_lines.pdf',\
    labels=labels, colors=colors,xrange=[1.603, 1.608], yrange=[0, 0],
    labelLocation='outside', ylabel='Normalized Flux+c', figsize=(12, 6))


    fig=plt.figure(figsize=(10,6))
    ax1= fig.add_subplot(111)
    #ax2 =fig.add_subplot(222)
    plot=ax1.scatter(table['J']-table['H'], table['J'], c=table['GLAT'], s=50, cmap='rainbow' )
    ax1.set_xlabel('J-H')
    ax1.set_ylabel('J')
    ax1.set_xlim(0.0, 1.5)
    ax1.set_ylim(4, 17)
    ax = plt.gca()
    ax.invert_yaxis()
    clb = fig.colorbar(plot)
    clb.ax.set_title('GLAT')
    plt.title('SAMPLE')
    #ax2.scatter(table['GLAT'], table['GLON'], s=50)
    plt.show()

    #smaller.more()
#    for ind, locid, coords in zip(table['ASPCAP_ID'], table['LOCATION_ID'], table['APOGEE_ID']):
#        downloadSpectra(ind,locid, coords)

    splat_names=['J03444306+3137338', 'J05420897+1229252', 'J10121791-0344435', 'J10285555+0050275', 'J11052903+4331357']
    #short_names=[createShortname(s) for s in splat_names]
    #print short_names
    tables=vstack([splat.searchLibrary(designation=s)[0] for s in splat_names])
    spectra=[splat.Spectrum(s) for s in tables['DATA_FILE']]
    print tables['SPEX_TYPE']
    #=[splat.Spectrum(s) for s in spectra]
    #splat.plotSpectrum(spcs)
   # print short_names
   # print spectra

    return


if __name__ == '__main__':
    #test_catalog()
    #createSample()
    #filenames =['2M00452143+1634446', '2M06154934-0100415']#,'2M07140394+3702459', '2M22400144+0532162']
    #ews=[]
   #for f in glob.glob(HOME+'\*'):
      #try:
        #filename=HOME+'apStar-r6-'+f+'.fits'
    regions = [[1.514823381871448, 1.5799960455729714],[1.5862732478022437, 1.6402951470486746],[1.6502053850596359, 1.689981692712261]]
    gaps = [[1.5,1.514823381871448],[1.5799960455729714,1.5862732478022437],[1.6402951470486746,1.6502053850596359],[1.689981692712261,1.7]]
    f= '2M00452143+1634446'
    sp= readSpectrum(filename= HOME+'apStar-r6-'+f+'.fits')
    w=np.array(sp.combined.wave.value )
    f=np.array(sp.combined.flux.value)
    n=np.array(sp.combined.noise.value)
    select= np.where((w>1.5167) & (w<1.5168))[0]

    w=w[select]
    f=f[select]
    n=n[select]


    slope= np.mean((f[:3]-f[-3:])/(w[:3]-w[-3:]))
    y_off= f[0]
    cont= [slope*x+ y_off for x in w]
    scale= f[0]/cont
    cont= cont*scale

    cont=[f[0] for i in w]
    initial_guess=[cont[0]*7, 0.00175, 1.6523]
    print initial_guess, 'initial guess.............'
    eq, er=measureEW(w, f, cont, noise=n, show_plots=True,
                     guess=initial_guess,nsamples=10000, nwalkers=6)
    print 'equivalent width', eq, er
    print 'area under continuum',  np.trapz(cont, w, dx=0.1)
    #ews.append(eq)
      #except:
       # continue
    #print ews





#    for ind, locid, coords in zip(table['ASPCAP_ID'], table['LOCATION_ID'], table['APOGEE_ID']):
#        print ind
#        downloadSpectra(ind,locid, coords)