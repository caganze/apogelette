# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:56:55 2016

@author: caganze
"""
from astropy.table import Table, vstack, join
from astropy.io import fits, ascii
import urllib
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
import numpy as np
import os

import splat
import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("ticks")

from .apogee_main import *
from .initialize import *

import matplotlib.patches as patches
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

class Coordinate(object):
    
    """
    own coordinate object
    """
    def __init__(self, **kwargs):
        self.ra=kwargs.get('ra', np.nan)
        self.dec=kwargs.get('dec', np.nan)
    
    @property
    def astropy(self):
        return SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree)
    
    @property
    def designation(self):
        return splat.coordinateToDesignation(self.astropy)
    
    @property 
    def shortname(self):
        s=''
        if self.designation[:2]=='2M':
            self.designation=self.designation.replace('2M', 'J' )
            first=4
        elif self.designation[0]=='J':
            first=5
        else:
            first=5
        for d in ['-', '+']:
            if d in self.designation:
                s=self.designation.split(d)[0][:first]+d+self.designation.split(d)[1][:4]
        
        return s
        
    
def download_spectra(indx,locid, coords, **kwargs):
    """
    The allStar database contains all the stars oberseved including Main survey targets,
    Calibration targets,  Ancillary, Sgr, and Kepler Special Targets and
    Telluric Correction and Sky Targets
    Information on targets for DR12 is found at http://www.sdss.org/dr12/irspec/targets/
    We assume the same methodolgy is used for DR13 data
    """
    #allStar = fits.open(LOCAL_DATABASE)[1].data
    #k=allStar['APOGEE_SHORT'] #redefine your sample based on flags
    APRED_VERS = kwargs.get('r_version','r6')
    APSTAR_VERS = kwargs.get('star_version','stars')
    ASPCAP_VERS = kwargs.get('aspcap_version', 'apo25m')
    RESULTS_VERS =kwargs.get('folder',  'Mdwarfs')

    "Formatting the url"
    """ The link should look at
    https://data.sdss.org/sas/dr13/apogee/spectro/redux/r6/stars/apo25m/location_id """
    url_directory = os.path.join(ap.DR13_HTML,ap.APOGEE_REDUX,APRED_VERS,APSTAR_VERS,ASPCAP_VERS,str(locid))
    spectrum_filename = '/apStar-{0}-{1}.fits'.format(APRED_VERS, coords)
    url_location = os.path.join(url_directory,spectrum_filename.replace('/', ''))
    local_spectrum_path = os.path.join(ap.HOME,spectrum_filename.replace('/', ''))
    url_location=url_location.replace("\\", "/")
    "save the files if they don't exist already"
    if not os.path.exists(local_spectrum_path):
         print(url_location)
         urllib.urlretrieve(url_location, filename=local_spectrum_path)


    return

def search_database(**kwargs):

    """
    Searching DR13 or DR12 Database for stars
    Need more kwargs
    Keywords are explained in the Datamodel file
    https://goo.gl/5hcygL
    """
    database= kwargs.get('database', os.environ['ALL_STAR_FILE'])
    search_parameter=kwargs.get('parameter', 'TEFF')
    select=kwargs.get('range', [0, 3000])
    data=Table(fits.open(database)[1].data)
    p=data[search_parameter]
    if isinstance(select[0], str):
        #matching by strings
        print("Matching by ", search_parameter)
        condition=[ i for i in range(0, len(data[search_parameter]))  if data[search_parameter][i] in select]

    else:
        print( " ALL stars with ", search_parameter, "between", select)
        condition = np.where((p>=select[0] ) & (p<=select[1])) [0]
   
    t=data[condition]
    #ignore multicolumns
    new_t=Table(t[DESIRED_ALL_STAR_COLUMNS])
    return new_t.to_pandas()


def create_shortname_from_coords(coords):
    """
    coords can be a dictionary or table
    """
    if isinstance(coords, list):
        c=Coordinate(ra=coords[0], dec=coords[1]).shortname
        
    if isinstance(coords, dict) or (isinstance (coords, pd.core.series.Series)):
        if 'RA' not in coords.keys() or 'DEC' not in coords.keys():
            c=Coordinate(ra=coords.keys[0], dec=coords.keys[1]).shortname
        else:
            c=Coordinate(ra=coords['RA'], dec=coords['DEC']).shortname
   
    return c
    
def cross_match(cats, keys, **kwargs):
  
    """
    cross-matching routine
    input: APOGEE catalog: ascii table, the other catalog is  a pandas table 
    
    """
    c1, c2= cats
    #create an astropy skycoord object
    ra_dec1=SkyCoord(ra= c1[keys['RA'][0]]*u.degree, dec=c1[keys['DEC'][0]]*u.degree) 
    ra_dec2=SkyCoord(ra= np.asarray(c2[keys['RA'][1]], dtype=float)*u.degree, dec=np.asarray(c2[keys['DEC'][1]], dtype=float)*u.degree) 
      
    c1=c1.rename(columns={keys['RA'][0]: 'RA', keys['DEC'][0]:'DEC'})
    c2=c2.rename(columns={keys['RA'][1]: 'RA', keys['DEC'][1]:'DEC'})
    #match catalogs and select where angular seperations are below max sep
    
    idx2, angseps2, ds2 = ra_dec1.match_to_catalog_sky(ra_dec2)
    idx1, angseps1, ds1= ra_dec2.match_to_catalog_sky(ra_dec1)
    
    angseps22= np.array(angseps2.to(u.arcsec).value)
    angseps11= np.array(angseps1.to(u.arcsec).value)
    
    sep= kwargs.get('separation', 10.0)
    #pick indices of objects  with separations below 10 arcsecs
    select1= np.where(angseps11 < sep)[0]
    select2=np.where(angseps22 < sep)[0]
    
    #select those objects in the catalog
    
    print ('catalog {} selects {} {}'.format(keys['CATALOG'],
                                             (len(select2), c1.shape), (len(select1), c2.shape)))

 
    c1['Sep1']=angseps22
    c2['Sep2']=angseps11
    
    t1=c1.ix[select2]
    t2=c2.ix[select1]
    
   
    
    print (t1, t2)
    
    t1['DESIGNATION']=t1[['RA', 'DEC']].apply(lambda x: splat.coordinateToDesignation([x[0], x[1]]), axis=1)
    t2['DESIGNATION']=t2[['RA', 'DEC']].apply(lambda x: splat.coordinateToDesignation([x[0], x[1]]), axis=1)
    
    #create a unique id based on ra and dec
    t2['CATALOG']=keys['CATALOG']
    t1['CATALOG']=keys['CATALOG']
    
    #print (t1.SHORTNAME, t2.SHORTNAME)
    t1=t1.reset_index()
    t2=t2.reset_index()


    return t1.join(t2, on='index',  lsuffix='lsuffix', rsuffix='rsuffix')
    
def create_sample(**kwargs):
    
    """
    Create a sample based on
    Sources Gagne et al, SPL and Bardalez-Galiufi et al, Schmidt et al
    Match them to APOGEE catalog
    C.Theissen's sample isa list of sources found in APOGEE and SDSS
    Steps:
    Step 1. Load all catalogs
    Step 2. Macth to catalogs of known brown dwarfs (splat, daniella, gagne, dwarfarchives.org), obtain names,
    spectral types, parallaxes, sdss, 2 mass and wise magnituddes
    step 3. cross-match with SDSS obtain I-J colors then compute absMag and distances
    step 4. Match with a sample of all apogee stars with Teff<3500K
    Step 5. Obtain additional information from Simbad
    """
    print (CATALOGS)
    splatbds=(splat.searchLibrary(jmag=[1,15.0]))
    apbds=search_database(Parameter='H', range_=[0, 15.0])
    speculous=pd.read_excel(CATALOGS+'SPECULOOS targets.xlsx')
    latemvrs=pd.read_excel(CATALOGS+'NearbyLateMovers.xlsx')
    daniella_catalog=pd.read_csv(CATALOGS+'spexsample_488_short.csv')
    dwarfarchives_lt=pd.read_html('http://ldwarf.ipac.caltech.edu/archive/version5/viewlist.php?table=ltdwarf&format=html')[0]
    dwarfarchives_m=pd.read_html('http://ldwarf.ipac.caltech.edu/archive/version5/viewlist.php?table=mdwarf&format=html')[0]
    
    #some re-formatting for dwarfarchives: rename columns then combine them
    dwarfarchives_lt.columns=dwarfarchives_lt.ix[0]
    dwarfarchives_m.columns=dwarfarchives_m.ix[0]
    #remove the first line since it's the name of columns
    dwarfarchives_lt= dwarfarchives_lt.ix[2:]
    dwarfarchives_m= dwarfarchives_lt.ix[2:]
    
    dwarfarchives=dwarfarchives_lt.append(dwarfarchives_m)
    print(dwarfarchives.columns)
    
    shortnames1=apbds[['RA','DEC']].apply(create_shortname_from_coords, axis=1) 
    apbds['DESIGNATION']=shortnames1
 
    sdss_bud=Table((fits.open(CATALOGS+'BUD_v2_4.fits')[1]).data)
    banyan7=(Vizier.get_catalogs('J/ApJS/219/33/table4')[0]).to_pandas()
    
    pcoors0=speculous['DESIGNATION'].apply(proper_cs)
    pcoors1=latemvrs['Coordinate'].apply(proper_cs)

    speculous['RA']=[x.ra.deg for x in pcoors0]
    speculous['DEC']=[x.dec.deg for x in pcoors0]
    latemvrs['RA']=[x.ra.deg for x in pcoors1]
    latemvrs['DEC']=[x.dec.deg for x in pcoors1]

    bud=pd.DataFrame()
    for k in sdss_bud.keys():
        try:
            bud[k]=sdss_bud[k]
        except:
            continue
    Vizier.ROW_LIMIT=-1
    
    #make sure each catalog has a shortname
    print (daniella_catalog.columns, banyan7.columns, splatbds.columns, bud.columns, dwarfarchives.columns)
    
    print ('starting cross-match')
	
    t1=cross_match([apbds, daniella_catalog], {'RA':['RA','RA (deg)'], 'DEC':['DEC', 'DEC (deg)'],'CATALOG':'Daniella'}, separation=8.5)
    t2= cross_match([apbds, banyan7], {'RA':['RA','_RAJ2000'], 'DEC':['DEC', '_DEJ2000'], 'CATALOG':'BANYAN VII'}, separation=10 )
    t3= cross_match([apbds, splatbds], {'RA':['RA','RA'], 'DEC':['DEC', 'DEC'], 'CATALOG':'Splat'}, separation=10)
    t6=cross_match([apbds, bud], {'RA':['RA','RA'], 'DEC':['DEC', 'DEC'], 'CATALOG':'EBOSS'}, separation=10 )
    t7=cross_match([apbds, dwarfarchives], {'RA':['RA','ra'], 'DEC':['DEC', 'decl'], 'CATALOG':'DWARFARCHIVES'}, separation=10.0 )
   
    t4= cross_match([apbds, speculous], {'RA':['RA','RA'], 'DEC':['DEC', 'DEC'], 'CATALOG':'SPECULOUS'}, separation=6)
    t5= cross_match([apbds, latemvrs], {'RA':['RA','RA'], 'DEC':['DEC', 'DEC'], 'CATALOG':'LateMovers'}, separation=2)
    
    #create an empty table then populate it with values from all the tables
    
    ftable=pd.DataFrame()
    ts=[t1, t2, t3, t4, t5, t6, t7]
    keys=np.concatenate([t.keys() for t in ts])
    tables=[]
    for k in keys:
        ftable[k]= np.nan
    for t in ts:
        tables.append(ftable.append(t))
    combined_table=pd.concat(tables)
    spts=[]
    #spt_keys=['Opt SpT', 'SpT', 'SPEX_TYPE', 'simbad_spt', 'SIMBAD_SPT', 'SPEX_TYPE']
    spt_keys=['NIR SpT', 'NIR_TYPE', 'OPT_TYPE', 'Opt SpT', 'SIMBAD_SPT', 'SPEX_TYPE', 
    'SPT', 'SpT', 'SpT-H2O', 'SpT-NL', 'SpT-OL', 'SpTn', 'r_SpT-NL', 'r_SpT-OL', 'simbad_spt',
     'spectral_type_ir', 'spectral_type_opt']
    
    #find the spectral type
    
    for i, row in combined_table.iterrows():
        spt= ' '
        for k in spt_keys:
          try:
            if (row['CATALOG']=='Splat')&(not row[k]) :
                #print row['CATALOG']
                sp=splat.Spectrum(row['DATA_FILE'])
                #sp.plot()
                spt=splat.classifyByStandard(sp)[0]
            else:
                if not pd.isnull(row[k]):
                    spt=row[k]
          except NameError:
               spt=np.nan
               
        spts.append(spt)
    
    combined_table['SPECTTYPE']=spts
    combined_table['COMPUTED_DISTANCE']= combined_table['DISTANCE']
    combined_table['COMPUTED_DISTANCE_UNC']=combined_table['DISTANCE_E']
    #ascii.write(ap_teff, 'teff_less_than_3000K.csv')
    combined_table.to_csv('longlist.csv')

    return combined_table


def shortlist():

    data=pd.read_csv(CATALOGS+'longlist.csv')
    short=pd.DataFrame()
    otherkeys=['APOGEE_ID', 'CATALOG', 'RA', 'DEC', 'SPECTTYPE', 'SIMBAD_PARALLAX', 
    'PARALLAX', 'PARALLAX_E', 'LOGG', 'LOGG_ERR',   'TEFF', 'TEFF_ERR', 'Sep (arcsec)',
    'wise_w1', 'wise_w1_e', 'wise_w2', 'wise_w2_e']
    short[otherkeys]=data[otherkeys]
    
    hmag=[ 'H',  'HMAG', 'H_2MASS',  'hmag']
    hmage=['HMAG_E',  'H_2MASS_E', 'H_ERR','hmag_e','hmag_error']
    
    short['H']=[np.round(np.nanmean(x), 1) for x in np.array(data[hmag])]
    short['H_ER']=[np.nanmean(x) for x in np.array(data[hmage])]
  
   
    imag=['IMAG', 'sdss_imag']
    image=['IMAG_E',  'sdss_imag_e']
    short['i']=[np.round(np.nanmean(x), 1) for x in np.array(data[imag])]
    short['i_ER']=[np.nanmean(x) for x in np.array(data[image])]
    
    rmag=['sdss_rmag']
    rmage=['sdss_rmag_e']
    short['r']=[np.round(np.nanmean(x), 1) for x in np.array(data[rmag])]
    short['r_ER']=[np.nanmean(x) for x in np.array(data[rmage])]
    
    jmag=['J', 'JMAG', 'JMAGN',  'J_2MASS','Jmag', 'jmag']
    jmage=['JMAG_E', 'J_2MASS_E', 'J_ERR',  'jmag_e', 'jmag_error' ]
    short['J']=[np.round(np.nanmean(x), 1) for x in np.array(data[jmag])]
    short['J_ER']=[np.nanmean(x) for x in np.array(data[jmage])]

    kmag=['K', 'KMAG', 'KS_2MASS',   'kmag' ]
    kmage=[ 'KMAG_E', 'KS_2MASS_E','K_ERR', 'kmag_e', 'kmag_error']
    short['K']=[np.round(np.nanmean(x), 1)for x in np.array(data[kmag])]
    short['K_ER']=[np.round(np.nanmean(x), 1) for x in np.array(data[kmage])]

    zmag=['ZMAG', 'sdss_zmag']
    zmage=['ZMAG_E', 'sdss_zmag_e']
    short['z']=[np.round(np.nanmean(x), 1) for x in np.array(data[zmag])]
    short['z_ER']=[np.round(np.nanmean(x), 1) for x in np.array(data[zmage])]


    
    for i, row in short.iterrows():
        distance=ap.estimateDistance(r=(row['r'], row['r_ER']),\
                                     j=(row['J'], row['J_ER']),\
                                     i=(row['i'], row['i_ER']),\
                                     z=(row['z'], row['z_ER']),\
                                     k=(row['K'], row['K_ER']), \
                                     relation=['bochanski2010', 'Schmidt2016'])
        for dk in distance.keys():
            short.ix[i, dk]=distance[dk]
            
    short.round(5)
    print(short)
    #print short['i']-short['K']
    short.columns = map(str.lower, short.columns)
    short.to_excel('apogee_short_list.xlsx')
    short.to_csv('short_list.csv')
    
    return 
    
def select1(keys, data):
    
    column=[]
    for i, r in data.iterrows():
        good=np.nan
        for k in keys:
            if not r[k]==' ':
                good=r[k]
        column.append(good)
    return column

def spectral_sequence(**kwargs):

    """returns a figure of spectral sequence"""

    table=pd.read_csv('no_duplicates.csv')
    
    table=table.sort('SPECTTYPE', ascending=True)
    
    print( (list(table['SPECTTYPE'])), (list(table.index.values)))
    #print 
    
    new_index=[  6,  7, 9,  24.0,    6, 10,  0, 12]
    

    
    
    table=table.ix[new_index]
    table['SPECTTYPE']=[ splat.typeToNum(x) for x in table['SPECTTYPE']]
    #table=table.sort_values(['SPTYPE'],  ascending=False)
    #table=table[:-2]
    print( "sample.............."  )  
    #print table
    wrng=kwargs.get('xrange', [1.52, 1.58])
    bands= kwargs.get('bands', { 'a':[1.529, 1.535], 'b': [1.553, 1.559], 'c':[1.563, 1.567]})
    
    #sub_bands=kwargs.get('sub_bands', )
    #for ind, locid, coords in zip(table['ASPCAP_ID'], table['LOCATION_ID'], table['APOGEE_ID']):
    #    downloadSpectra(ind,locid, coords)
    #filenames=[ap.HOME+'apStar-r6-'+ap_id+'.fits' for ap_id in table['APOGEE_ID']]
    sptypes= [splat.typeToNum(s) for s in table['SPECTTYPE']]
    fig = plt.figure(figsize=(40, 10))
    ax= fig.add_subplot(111)
    cmap = plt.get_cmap('YlOrRd')
    ymaxes=[]
    #print sptypes
    #ja
    
    
    #extra_masks specified by spectrum in correct order
    extra_masks={'L2.5':[(1.6729, 1.6737),  (1.638,1.640),],
    'L3.5':[(1.6729, 1.6737), (1.638,1.640)],
                 'L1.0':[(1.6729, 1.6735),(1.6540, 1.6580)],
                 'L8.0':[(1.5186, 1.5189),(1.564,1.565 ), (1.5285, 1.5291), (1.5391, 1.5397),(1.6685, 1.6693), (1.5537, 1.5545), (1.5595, 1.5600)]
                }
    for i, id_, spt in zip(np.arange(len(sptypes)),table['APOGEE_ID'], sptypes) :
    #try:
         if ((i !=0) & (i !=1)) :
            label= spt
            #print(id_,  spt, i)
            
            sp=apogee.readSpectrum(apogee_id=str(id_))
            #rint sp
            sp.zoom(wrng)
            #remove high sigmna values
            
            #if spt in extra_masks.keys():
            sls= np.concatenate([np.where((sp.wave >= x[0]) & (sp.wave <=x[1]))[0] for x in extra_masks['L8.0']])
            sp.flux[sls]=np.nan
            if spt in extra_masks.keys():
                sls2= np.concatenate([np.where((sp.wave >= x[0]) & (sp.wave <=x[1]))[0] for x in extra_masks[spt]])
                sp.flux[sls2]=np.nan
            scl= kwargs.get('scale', 15.0)
            #choose normalization range
            sp.smooth(sigma=kwargs.get('sigma', 5.0))
            norm_range=kwargs.get('norm_range', [1.616, 1.617])
            sp.normalize(xrange= norm_range, scale=scl)
            
            offset=kwargs.get('offset', 2.0)
            #sp.plot()
           # print sp.flux, sp.wave
            color =i**3/10
            #offset=kwargs.get('offset', i*100)
            #sp.plot()
            flx= sp.flux
            #flx[np.isnan(flx)]=0.0
            #ax2 = ax.twiny()
            ax.plot(sp.wave, flx+i*offset,  c='k', linestyle='-')
            
            ax.set_xticks(np.arange(1.5, 1.7, 0.001), minor=True)
            ax.set_xticks(np.arange(1.5, 1.7, 0.01), minor=False)
            
            
            ax.text(wrng[-1]+0.0005, np.nanmean(sp.flux+i*offset), label, fontsize=22, color='k')
            if kwargs.get('show_bands', False):
                for b in bands.keys():
                    ax.axvspan(bands[b][0], bands[b][1], color='blue', alpha=0.02)
                    #ax.text(np.mean(bands[b]), 16.5, b, fontsize=22)
                    
            #ax2.cla()
            ymaxes.append(np.nanmean(sp.flux+i)+2*np.nanstd(sp.flux+i))
            ax.tick_params('both', length=14, width=2, which='major')
            # ax.tick_params('both', length=10, width=1, which='minor')
            #for tick in ax.xaxis.get_major_ticks():
            #    tick.label.set_fontsize(18)
           # plt.show()
    #except :
    #   continue
    plt.xlabel('Wavelength (micron)', fontsize=28)
    plt.ylabel('Normalized Flux + offset', fontsize=28)
    plt.xlim(wrng)
    plt.ylim(kwargs.get('ylim', (100,250)))
    #plt.legend(loc=(1.03, 0.0))
    plt.show()
    fig.show()
    fig.savefig('spectral_sequence.pdf')
    
def proper_cs(des):
    des=str(des)
    #print des
    try:
        coord=splat.properCoordinates(des) 
        return coord
    except:
        return np.nan
   
def master_list(**kwargs):
    
    #master list of list of known M,L, T brighter than APOGEE cutoff (H= 12.2)
    splatbds=(splat.searchLibrary(jmag=[1,20.0]))
    speculous=pd.read_excel(CATALOGS+'SPECULOOS targets.xlsx')
    latemvrs=pd.read_excel(CATALOGS+'NearbyLateMovers.xlsx')
    daniella_catalog=pd.read_csv(CATALOGS+'spexsample_488_short.csv')
    banyan7=(Vizier.get_catalogs('J/ApJS/219/33/table4')[0]).to_pandas()
    #sdss_bud=Table((fits.open('BUD_v2_4.fits')[1]).data)
    #change this into dataframe format
    #bud=pd.DataFrame()
    #for k in sdss_bud.keys():
    #    try:
    #        bud[k]=sdss_bud[k]
    #    except:
    #        continue
    
    splatbds['CATALOG']='Splat'
    banyan7['CATALOG']='banyan7'
    
    print (splatbds.keys())
    print (speculous.keys())
    print (latemvrs.keys())
    print (daniella_catalog.keys())
    print (banyan7.keys())

    
    
    combined_table=pd.concat([splatbds, speculous, latemvrs, daniella_catalog, banyan7])
    spts=[]
    #spt_keys=['Opt SpT', 'SpT', 'SPEX_TYPE', 'simbad_spt', 'SIMBAD_SPT', 'SPEX_TYPE']
    spt_keys=['NIR SpT', 'NIR_TYPE', 'OPT_TYPE', 'Opt SpT', 'SIMBAD_SPT', 'SPEX_TYPE',
     'SPT', 'SpT', 'SpT-H2O', 'SpT-NL', 'SpT-OL', 'SpTn', 'r_SpT-NL', 'r_SpT-OL', 'simbad_spt']
    id_keys=['DESIGNATION', 'Coordinate', 'Designation', '_RAJ2000', '_DEJ2000']
    
    spts=[]
    ids=[]
    for i, row in combined_table.iterrows():
        spt= ' '
        id_=' '
        for k in spt_keys:
         try:
            if (row['CATALOG']=='Splat')&(not row[k]) :
                #print row['CATALOG']
                sp=splat.Spectrum(row['DATA_FILE'])
                #sp.plot()
                spt=splat.classifyByStandard(sp)[0]
            else:
                if not pd.isnull(row[k]):
                    spt=row[k]
         except:
             spt=np.nan
        
        for k2 in id_keys:
         try:
            print (str(k2))
            print (row['CATALOG'])
            if k2 in ['_RAJ2000', '_DEJ2000']:
                cor=splat.properCoordinates([row['_RAJ2000'], row['_DECJ2000']])
                id_=splat.coordinateToDesignation(cor)
            else:
                cor=splat.properCoordinates(row[k2])
                id_=splat.coordinateToDesignation(cor)
         except:
            continue
                
        spts.append(spt)
        ids.append(id_)
                
            
    
    combined_table['SPECTTYPE']=spts
    combined_table['PROPER_COORD']=ids
    
    #remove duplicates
    uniques, indices = np.unique(combined_table['PROPER_COORD'], return_index=True)
    combined_table=combined_table.ix[indices]
    combined_table.to_csv('master_list.csv')
    
    return combined_table
    
def format_spectral_type(spt_list):
    #format a list of spectral type list to a uniform notation 
    x=[]
    for s in (spt_list):
        #print ('s0', s)
        try:
            if (s[0] == 'M') or (s[0] == 'L') or (s[0] == 'T' ):
                s=splat.typeToNum(s[:4])
            else:
                s=float(s)+10
        except:
            s=np.nan
        x.append(s)
    
    return x
    
def diagnostics(**kwargs):
    
    l0= pd.read_csv('apogee_short_list.csv')
    uniques, indices1 = np.unique(l0['APOGEE_ID'], return_index=True)
    l0=(l0.ix[indices1])
    
    l1=pd.read_csv('longlist.csv')
    uniques2, indices2 = np.unique(l1['APOGEE_ID'], return_index=True)
    l1=l1.ix[indices2]
    
    l0_ids= np.array(l0['APOGEE_ID'])
    
    indices=[]
    for i, row in l1.iterrows():
        if row['APOGEE_ID'] in l0_ids:
            indices.append(i)
  
    l=l1.ix[indices]
    #l['SPECTTYPE']=np.nan
    l['SPECTTYPE']=np.array(l0['SPECTTYPE'])
    print( l['SPECTTYPE'])
    
        
    l.to_csv('no_duplicates.csv')
   
    master_list=pd.read_csv('master_list.csv')
    
    #print (master_list['SPECTTYPE'])

    #x2=format_spectral_type( master_list['SPECTTYPE'])
    
    #remove duplicates 
    
   
   
    sample=l
    
    print ("sample size......", len(sample))
   
    
    #print( list(master_list.keys()))
    
    combined_mags=[[x, y, z] for x, y, z in zip(master_list['HMAG'], master_list['H_2MASS'], master_list['hmag']) ]
    
    #if none of the mags are fainter than cutoff then select 
    select=[i for i, t in enumerate(combined_mags) if np.any(np.less(t, 13.0)[0])]
    #print (list(np.array(combined_mags)[select]))

    #print( master_list['SPECTTYPE'])
    dan=pd.read_csv('spexsample_488_short.csv')
   
    x1=format_spectral_type(sample['SPECTTYPE'])
    x2=format_spectral_type( master_list['SPECTTYPE'])
    x3=format_spectral_type( (master_list.ix[select])['SPECTTYPE'])
  
    
    x1=np.array(x1)
    x1= x1[~np.isnan(x1)]
    
    x2=np.array(x2)
    x2= x2[~np.isnan(x2)]
    
    #print (x2)
    #bn
    x3=np.array(x3)
    x3= x3[~np.isnan(x3)]
    
    bins= np.arange(0, 30, 1)
    #bins=np.logspace(0.1, 1.0, 30)
    
    #seaborn.distplot(x, bins=bins, axlabel ='Spectral Type')
    
    
    ax=plt.gca()
    ax.set_xticks(np.arange(15,25, 2)+0.5, minor=False)
    ax.set_xticklabels( [ 'M5.0', 'M7.0',  'M9.0',  'L1.0',  'L3.0'])
    ax.set_xlim([14, 25])
    ax.set_ylim([0.5, 1000])
    ax.tick_params('both', length=14, width=2, which='major')
       # ax.tick_params('both', length=10, width=1, which='minor')
    #for tick in ax.xaxis.get_major_ticks():
    #           tick.label.set_fontsize(14)
    seaborn.distplot(x1, bins=bins, kde=False,
                      kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                     hist_kws={"histtype": "stepfilled", "linewidth": 3,
                             "alpha": 1, "color": "g", "label":"Sample"}, ax=ax)
    
    
    #seaborn.distplot(x2, bins=bins, kde=False,
    #                 kde_kws={"color": "k", "lw": 3, "label": "KDE"},
    #                hist_kws={"histtype": "stepfilled", "linewidth": 3,
    #                        "alpha": 0.1, "color": "b", "label":"Known M,L,T dwarfs"}, ax=ax)
  
    seaborn.distplot(x3, bins=bins, kde=False,
                      kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                     hist_kws={"histtype": "stepfilled", "linewidth": 3,
                             "alpha": 0.1, "color": "b", "label":"Known M & L dwarfs with H <12.2"}, ax=ax)
#    #make h=22.2 cutoff 
    
    #add a histogram of known M, L, T, dwarfs
    #ax.set_xlim([10, 30])
    ax.tick_params('both', length=14, width=2, which='major')
    plt.yscale('log', nonposy='clip')
    plt.ylabel('Number', fontsize=28)
    plt.legend(loc=(0.0, 1.0), fontsize=20)
    plt.xlabel('Spectral Type', fontsize=28)
    
#    mags=['sdss_imag,sdss_imag_e,sdss_rmag,sdss_rmag_e,sdss_zmag,sdss_zmag_e']
#    dists=[]
#    dist_keys='DISTANCE,DISTANCE_PHOT'
#    dist_errs='DISTANCE_E,DISTANCE_PHOT_E'
#
#    dists= np.array([np.array(sample[k]) for k in dist_keys.split(',')])
#    dists_ers= np.array([np.array(sample[k]) for k in dist_errs.split(',')])
#    
#    
##    print dists[0]
##    print dists[1]
##    print dists[3]
#  
#    distances=[]
#    for obj in dists.T:
#        distances.append(np.mean(np.nan_to_num([float(s) for s in obj])))
#    distances=np.array(distances)  
#    
#    unkns=np.where(distances==0)[0]
#    distances[np.where(distances==0)]=np.nan
#    imags= zip(sample['sdss_imag'].ix[unkns], sample['sdss_imag_e'].ix[unkns])
#    rmags=zip(sample['sdss_rmag'].ix[unkns], sample['sdss_rmag_e'].ix[unkns])
#    zmags=zip(sample['sdss_zmag'].ix[unkns], sample['sdss_zmag_e'].ix[unkns])
#
#    ds=[]
#    ds_unc=[]
#    for i, r, z in zip(imags, rmags, zmags):
#        dist_dict=ap.estimateDistance(r=r, i=i, z=z)
#        
#        #ds.append(dist_dict)
#        
#        ds.append(np.nanmean([dist_dict[k] for k in dist_dict.keys() if 'unc' not in k]))
#        ds_unc.append([np.sqrt(dist_dict[k]**2) for k in dist_dict.keys() if 'unc'  in k])
#    
#   
#    bins=np.arange(1, 100, 5)
#    distances = distances[~np.isnan(distances)]
#    #print( distances)
#    simbad_plx=np.array([float(x) for x in sample['SIMBAD_PARALLAX']])
#    other_plxs= np.array([float(x) for x in  sample['PARALLAX']])
#    
#    
#    #    print (simbad_plx)
#    #selcts= ~np.isnan(other_plxs)
#
#    #other_plxs[selcts]=simbad_plx[selcts]
#    #combined_parallaxes=other_plxs[~np.isnan(other_plxs)]
#    
#    #simbad parallaxes has all parallaxes 
#    combined_parallaxes=np.array([1000.0/i for i in simbad_plx])
#    #remove nans
#    combined_parallaxes=combined_parallaxes[~np.isnan(combined_parallaxes)]
#    
#    
#    plt.figure()
#    ax=plt.gca()
#    ax.tick_params('both', length=14, width=2, which='major')
#       # ax.tick_params('both', length=10, width=1, which='minor')
#    for tick in ax.xaxis.get_major_ticks():
#                tick.label.set_fontsize(14)
#    #seaborn.distplot(ds, bins=bins, kde=False,  hist_kws={"histtype": "step", "linewidth": 3,
#    #                         "alpha": 1, "color": "b"}, label='From SDSS Colors')
#    
#    seaborn.distplot(distances, bins=bins, kde=False,  hist_kws={"histtype": "stepfilled", "linewidth": 3,
#                             "alpha": 1, "color": "c"}, label='Photomteric Distances', ax=ax)
#    seaborn.distplot(combined_parallaxes, bins=bins, kde=False,  hist_kws={"histtype": "stepfilled", "linewidth": 3,
#                             "alpha":0.5, "color": "y"}, label='Parallaxes', ax=ax)
#    plt.legend()
#
#    plt.xlabel('Distance (pc)', fontsize=28)
#    plt.ylabel('Number', fontsize=28)
#   
    plt.figure()
    print (len(sample['J']))
    print (sample['H'])
    
    #hj
    j_h= sample['J']-sample['H']
    j_h_er= np.sqrt(np.array(sample['J_ERR'])**2+np.array(sample['H_ERR'])**2)
    h_k=sample['H']-sample['K']
    h_k_er= np.sqrt(np.array(sample['H_ERR'])**2+np.array(sample['K_ERR'])**2)
    
   
    j_h_er[np.where(j_h_er > 20.0)[0]]=np.nan
    #print([(i,x) for i, x in j_h_er if x>1.0])
    print(j_h)
    #spl= splat.searchLibrary()
    #splat_x= [float(x)-float(y) for x, y in zip(spl['J_2MASS'], spl['H_2MASS'])]
    #splat_y= [float(x)-float(y) for x, y in zip(spl['H_2MASS'], spl['K_2MASS'])]
    #print spl.keys()
    ax=plt.gca()
    #ax.plot(j_k, h_k, 'o')
    ax.plot( dan['hmag']-dan['kmag'],dan['jmag']-dan['hmag'], 'o',alpha=0.5, color='y', label='Bardalez-Gagliuffi et al')
    ax.errorbar( h_k, j_h, yerr=j_h_er, xerr=h_k_er, fmt='o', c='k', label='Sample')
    ax.set_ylabel('J-H', fontsize=28)
    ax.set_xlabel('H-K', fontsize=28)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 2.0])
    ax.legend(loc=(0.0, 1.0), fontsize=20)
    ax.tick_params('both', length=14, width=2, which='major')
       # ax.tick_params('both', length=10, width=1, which='minor')
    #for tick in ax.xaxis.get_major_ticks():
    #            tick.label.set_fontsize(14)
    # histogram of known sources 
def clean_sample():
    #remove these sources:
    r=[' 2M03424919+3150110 ', '2M03431371+3200451', '2M03440291+3152277' ,
    '2M03442663+3203583', '2M03444256+3210025', '2M03444306+3137338', 
    '2M03445205+3158252', '2M03453563+3159544' , '2M13121982+1731016']
    r.extend(['2M03424919+3150110', '2M03430679+3148204', '2M03455824+3226475', 
    '2M03445200+3226253', '2M03442789+3227189', '2M03425596+3158419', 
    '2M03451634+3206199', '2M03432820+3201591', '2M03443379+3158302', 
    '2M03445205+3158252', '2M03435953+3215551', '2M03443470+3215544', 
    '2M03434788+3217567', '2M03440216+3219399', '2M03435907+3214213', 
    '2M03453563+3159544', '2M03443588+3215028', '2M03431371+3200451',
    '2M03442555+3211307', '2M03442555+3211307', '2M11052903+4331357',
    '2M03444306+3137338', '2M03442663+3203583', '2M03443878+3219056', 
    '2M03444256+3210025'])
    
    print( r)
    #short_names_to_add=[0344+3154, 0344+3215, 0345+3212]

    l=pd.read_csv('longlist.csv')
    #print (list(l.keys()))

    apogee_ids=np.array(l['APOGEE_ID'])
    #print (len(apogee_ids))
    indices_to_keep=[i for i, x in enumerate(apogee_ids) if x not in r]
    l=l.ix[indices_to_keep]
    l.to_csv('new_longlist.csv')
    search_simbad()
   
    #search_simbad()
  
    
    #shortlist()
   # splat_sequence()
    
    #spectralSequence( xrange=[1.514, 1.69],bands=  { 'a':[1.529, 1.535], 'b': [1.6719, 1.677], 'c': [1.556, 1.559], \
     #      'd':[1.562, 1.564], 'e':[1.571, 1.578], 'f':[1.597, 1.601], 'g': [1.615, 1.622], 'h':[1.651, 1.657]}, \
      #                       norm_range= [1.673, 1.674], offset=6.0, scale=12.0, show_bands=True, sigma=10.0, ylim=(19, 100))
      # spectralSequence( xrange=[1.60, 1.690], sigma=15.0)
    
    
    return 
def search_simbad():
    "use splat to query simbad "
    data=pd.read_csv('apogee_short_list.csv')
    print (data['CATALOG'])
    #J
    #simbad query for optical  colors or parallax if they exist
    parallaxes=[]
    spts=[]
    for indx, id_ in enumerate(data['APOGEE_ID']):
     try:
        c = splat.properCoordinates(id_.replace('2M', 'J'))
        q = splat.querySimbad(c,radius=10.*u.arcsec,reject_type='**')
        #print (q['NAME', 'LIT_SPT','PARALLAX'][0])
        print (indx, float(q['PARALLAX'][0]))
        print (q['LIT_SPT'][0], (data.ix[indx])['CATALOG'])
        parallaxes.append(float(q['PARALLAX'][0]))
        if (data.ix[indx])['CATALOG'] =='Splat':
            spts.append(q['LIT_SPT'][0])
        else:
            spts.append((data.ix[indx])['SPECTTYPE'])
     except:
        # parallaxes.append(np.nan)
         spts.append(np.nan)
         
   # parallaxes=np.array(parallaxes)[np.arange(len(data))]
    print (parallaxes, len(data), len(spts))
    data['SIMBAD_PARALLAX']=np.nan
    data['SPECTTYPE']=spts

    data.to_csv('new_longlist.csv')
    
def splat_sequence():
    
    s=pd.read_csv('no_duplicates.csv')
    s=s.sort('SPECTTYPE', ascending=False)

    good_files=s['DATA_FILE']
    shortnames=s['APOGEE_ID']
    sp_types=s['SPECTTYPE']
    print (s['SPECTTYPE'], s['APOGEE_ID'])
   
    
    #m[p]
    plt.figure(figsize=(8, 12))
    ax=plt.gca()
    mult=1.0
    for i, des, s, spt in  zip( np.arange(len(good_files)), good_files, shortnames,   sp_types):
      #try: 
         
          if ((type(des) != type(np.nan)) & (s!='2M08501918+1056436') & (s!='2M03442721+3220288') & (s!='2M03441583+3159367')):
            sp=splat.Spectrum(des)
            std_spt=splat.classifyByStandard(sp)[0]
            #print( std_spt)
            #lhk
            print (mult, des, spt, s)
            std=splat.Spectrum(splat.SPEX_STDFILES[std_spt])
            
            sp.normalize(range=[1.1, 2.0])
            std.normalize(range=[1.1, 2.0])
            
            ax.plot(sp.wave.value, sp.flux.value*14.0+5.0*mult, 'k')
            #ax.plot(std.wave.value, std.flux.value*14.0+5.0*mult, 'b')
            label=str(createShortname(s))+' '+str(spt)
            if '034' in createShortname(s):
                label= str(createShortname(s)) +' '+'young ' +str(spt)
                
            ax.text(sp.wave.value[-1]+0.01, np.mean(sp.flux.value-0.3)*14.0+5.0*mult, label, fontsize=20, color='k')
            twv = [[1.5, 1.7]]
            for waveRng in twv:
                ax.axvspan(waveRng[0], waveRng[1], color='red', alpha=0.2)
                #ax.text(np.mean(waveRng), 50.0,r'$\oplus$',horizontalalignment='center',fontsize=18)
            #label=ax.get_xticklabels()
            #label.set_fontproperties(20.0)
            ax.set_xticks(np.arange(1.0, 2.5, 0.01), minor=True)
            ax.tick_params('both', length=14, width=2, which='major')
            
          else:
           mult=mult-1
          mult=mult+1.0
          
            
    plt.xlim([0.9, 2.5])
    #plt.ylim([5.0, 34])
    plt.xlabel('Wavelength (micron)', fontsize=28)
    plt.ylabel('Normalized Flux + offset', fontsize=28)
      
    
    return 
    
def longlist_short_list():
    short_list=pd.read_csv('apogee_short_list.xlsx - Sheet1.csv')
    
    return 
    
if __name__=="__main__":
      
      splat_sequence()
     
     