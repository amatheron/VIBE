import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as optimization
import pandas
import sys
import pickle
from PIL import Image
import glob, os

import vibe.rossendorfer_farbenliste as rofl


colors=[]
colors.append([0.5,0.5,0.5])
colors.append([0.4,0.4,0.6])
colors.append([0.5,0.6,0.8])
colors.append(rofl.g())
colors.append([0.8,0.7,0.2])
colors.append([0.9,0.55,0.2])
colors.append([1,0,0])
colors.append([0.6,0,0.2])
colors.append([1,0,1])
colors.append([0,0,1])


colors.append([0.,0.,0.])
colors.append([0.4,0.4,0.6])
colors.append([0.5,0.6,0.8])
colors.append(rofl.g())
colors.append([0.8,0.7,0.2])
colors.append([0.9,0.55,0.2])
colors.append([1,0,0])
colors.append([0.6,0,0.2])
colors.append([1,0,1])
colors.append([0,0,1])

colors.append([0.,0.,0.])
colors.append([0.4,0.4,0.6])
colors.append([0.5,0.6,0.8])
colors.append(rofl.g())
colors.append([0.8,0.7,0.2])
colors.append([0.9,0.55,0.2])
colors.append([1,0,0])
colors.append([0.6,0,0.2])
colors.append([1,0,1])
colors.append([0,0,1])


global times, time_labels,print_levels
print_levels=0

times =[]
time_labels =[]

def clear_times():
    global times, time_labels
    times=[]
    time_labels=[]

def tick(lab='',verbose=0):
    global times, time_labels
    import time
    times.append(time.time())
    time_labels.append(lab)
    if lab!='': l(verbose,lab)

def print_times():
    tick('end')
    global times, time_labels
    for i,t in enumerate (np.diff(times)):
        lab=time_labels[i]
        if lab!='':
            print("{:.0f}: {:.1f}s (".format(i+1,t)+lab+')')
        else:
            print("{:.0f}: {:.1f}s".format(i+1,t))
    tot=times[-1]-times[0]
    print("Total: {:.1f} s ".format(tot))
    return tot


def saveTiff(img,name):
    result = Image.fromarray((img))
    result.save(name)

def loadTiff(fn):
    image = Image.open(fn)
    im = np.array(image)
    return im

def shotNumberToDraco(PInumber):
    #works for day 12019-06-21, not complete!
    if PInumber<22:
        return PInumber-13+116
#don't    care about start - no hopg
    if PInumber==42:
        return 188
    if PInumber<43:
        return PInumber-36+181
    if PInumber<47:
        return PInumber-43+190
    if PInumber<53:
        return PInumber-47+196
#part of no care     - hopg misaligned
    if PInumber<97:
        return PInumber-96+251
    if PInumber<149:
        return PInumber-98+254
    if PInumber==308:
        return PInumber-149+308
    if PInumber==350:
        return 352
    if PInumber==386:
        return 387
    if PInumber==426:
        return 429
    if PInumber==431:
        return 432
    if PInumber>=470 and PInumber<=476:
        return PInumber+2
    if PInumber==536:
        return 537
    if PInumber==681:
        return 682
    if PInumber<740:
        return PInumber
    d=PInumber
    return d


def shotNumberToHopg(DI):
    for i in np.arange(0,800):
        dd=shotNumberToDraco(i)
        if dd==DI:
            return i
    return 0

def mask_nans(img):
    w=np.shape(img)[0]
    h=np.shape(img)[1]
    nans=np.argwhere(np.isnan(img))
    for i,x in enumerate(nans[:,0]):
        y=nans[i,1]
        area=img[x-1:x+1,y-1:y+1]
        mean=np.nanmean(area)
        img[x,y]=mean
    return(img)

def polylogfit(x,y,deg,xax,debug=0):
    """
    Fits a function (log(x),y) with polynomial of given degree onto a grid in xax

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    degree: TYPE
        DESCRIPTION.
    xax : TYPE
        DESCRIPTION.

    Returns
    -------
    The fitted curve

    """
    sel=np.isfinite(y)
    x=x[sel]
    y=y[sel]
    logx=np.log(x)
    p=np.polyfit(logx,y,deg)
    logxax=np.log(xax)
    vals=np.polyval(p, logxax)
    if debug:
        figure()
        plt.plot(logx,y)
        plt.plot(logxax,vals)
    return vals

def loglogfit(x,y,deg,xax,debug=0):
    """
    Fits a polynomial function in a loglog space

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    degree: TYPE
        DESCRIPTION.
    xax : TYPE
        DESCRIPTION.

    Returns
    -------
    The fitted curve

    """
    sel=np.isfinite(y)
    x=x[sel]
    y=y[sel]
    logx=np.log(x)
    logy=np.log(y)
    p=np.polyfit(logx,logy,deg)
    logxax=np.log(xax)
    vals=np.polyval(p, logxax)
    if debug:
        figure()
        plt.plot(logx,y)
        plt.plot(logxax,vals)
    return vals


def smooth2d(matrixIn, Nr, Nc=-1, flag='g'):
    if Nr==Nc==0:
        return matrixIn
    if Nc==-1:
        Nc=Nr
    from scipy import ndimage
    if flag=='m':
        matrixOut = ndimage.median_filter(matrixIn, size=(Nr, Nc))
    elif flag=='g':
        matrixOut = ndimage.gaussian_filter(matrixIn, sigma=(Nr, Nc))
    else:
        print('Please choose the smooth method:\n  m: median_filter,  g: gaussian_filter')

    return matrixOut

def printv(text='',value=0,units='',formats=':'):
    stri=text+('  : {'+formats+'} ').format(value)+units
    print(stri)


def getFilterTransmission(binseV,draw,thicknesses):
        # Filter loading and functions
    from astropy.io import ascii
    filterdir='/home/michal/hzdr/codes/python/filters/'
#    filterdir='./files/'
    fia=ascii.read(filterdir+"filterAl100")
    fiax=fia['col1'];
    fiay=fia['col2'];
    fic=ascii.read(filterdir+"filterCu1")
    ficx=fic['col1'];
    ficy=fic['col2'];
    fib=ascii.read(filterdir+"filterBe200")
    fibx=fib['col1'];
    fiby=fib['col2'];
    fib=ascii.read(filterdir+"filterCH100")
    fichx=fib['col1'];
    fichy=fib['col2'];
    fib=ascii.read(filterdir+"filterTi5")
    fiTix=fib['col1'];
    fiTiy=fib['col2'];

    fib=ascii.read(filterdir+"PixisDDQE")
    fiPixisx=fib['col1'];
    fiPixisy=fib['col2']/100;


    fiay[fiay<1e-30]=1e-30
    ficy[ficy<1e-30]=1e-30
    fiby[fiby<1e-30]=1e-30
    fichy[fichy<1e-30]=1e-30

    binseVcen=binseV[0:binseV.shape[0]-1];
    binseVcen=np.append(0,binseVcen);
    filterAl100=np.interp(binseVcen,fiax,fiay)
    filterCu1=np.interp(binseVcen,ficx,ficy)
    filterBe200=np.interp(binseVcen,fibx,fiby)
    filterCH100=np.interp(binseVcen,fichx,fichy)
    filterTi5=np.interp(binseVcen,fiTix,fiTiy)

    PixisDDQE=np.interp(binseVcen,fiPixisx,fiPixisy)


    filters = np.zeros(binseV.shape[0])
    filters[:]= filterBe200**(thicknesses[2]/200) * filterAl100**(thicknesses[1]/100)* filterCH100**(thicknesses[0]/100) *filterCu1**(thicknesses[3]/1)*filterTi5**(thicknesses[4]/5)

    if draw:
        plt.rcParams['figure.figsize'] = (16,5)
        ax=plt.subplot(1, 2, 1)

        plt.plot(fiax,fiay,label='Al 100')
        plt.plot(ficx,ficy,label='Cu 1')
        plt.plot(fibx,fiby,label='Be 200')
        plt.plot(fichx,fichy,label='CH 100')
        plt.plot(fiPixisx,fiPixisy,label='Pixis QE')

        plt.grid()
        plt.legend()

        #ax.set_yscale('log')
        plt.xlim(0,10e3)

        plt.subplot(1, 2, 2)
        #plt.plot(binseVcen,filterAl100,label='Al 100μm')
        plt.plot(binseVcen,filterCu1**2,label='Cu 1μm')
        plt.plot(binseVcen,filters[:],label='filter 0: fixed filters', linewidth=4)
        plt.xlim(5,15e3)
#        plt.ylim(0,0.3)
        plt.legend()
        plt.grid()

    return filters



def loadFilters(binseV,draw):
        # Filter loading and functions
    from astropy.io import ascii
    filterdir='/home/michal/hzdr/codes/python/filters/'
#    filterdir='./files/'
    fia=ascii.read(filterdir+"filterAl100")
    fiax=fia['col1'];
    fiay=fia['col2'];
    fic=ascii.read(filterdir+"filterCu1")
    ficx=fic['col1'];
    ficy=fic['col2'];
    fib=ascii.read(filterdir+"filterBe200")
    fibx=fib['col1'];
    fiby=fib['col2'];
    fib=ascii.read(filterdir+"filterCH100")
    fichx=fib['col1'];
    fichy=fib['col2'];
    fib=ascii.read(filterdir+"PixisDDQE")
    fiPixisx=fib['col1'];
    fiPixisy=fib['col2']/100;


    fiay[fiay<1e-30]=1e-30
    ficy[ficy<1e-30]=1e-30
    fiby[fiby<1e-30]=1e-30
    fichy[fichy<1e-30]=1e-30

    binseVcen=binseV[0:binseV.shape[0]-1];
    binseVcen=np.append(0,binseVcen);
    filterAl100=np.interp(binseVcen,fiax,fiay)
    filterCu1=np.interp(binseVcen,ficx,ficy)
    filterBe200=np.interp(binseVcen,fibx,fiby)
    filterCH100=np.interp(binseVcen,fichx,fichy)

    PixisDDQE=np.interp(binseVcen,fiPixisx,fiPixisy)


    filters = np.zeros((3,binseV.shape[0]))
    #filters[0,:]=filterBe200 ;
    filters[0,:]=PixisDDQE * filterBe200**(75/200) * filterAl100**(21/100)* filterCH100**(21/100);#what goes into pixis, without BFF, with profiler
    filters[1,:]=filters[0,:] * filterAl100**2 * filterCu1**2; #adding BFF
    filters[1,:]=filterAl100**2 * filterCu1**2; #adding BFF
    filters[2,:]=filters[0,:] * filterAl100**2 * filterCu1**1; #experimental

    if draw:
        plt.rcParams['figure.figsize'] = (16,5)
        ax=plt.subplot(1, 2, 1)

        plt.plot(fiax,fiay,label='Al 100')
        plt.plot(ficx,ficy,label='Cu 1')
        plt.plot(fibx,fiby,label='Be 200')
        plt.plot(fichx,fichy,label='CH 100')
        plt.plot(fiPixisx,fiPixisy,label='Pixis QE')

        plt.grid()
        plt.legend()

        #ax.set_yscale('log')
        plt.xlim(0,10e3)

        plt.subplot(1, 2, 2)
        #plt.plot(binseVcen,filterAl100,label='Al 100μm')
        plt.plot(binseVcen,filterCu1**2,label='Cu 1μm')
        plt.plot(binseVcen,filters[0,:],label='filter 0: fixed filters', linewidth=4)
        plt.plot(binseVcen,filters[1,:],label='filter 1: Al+Cu', linewidth=4)
        plt.xlim(5,15e3)
        plt.ylim(0,0.3)
        plt.legend()
        plt.grid()

    return filters


def synchrotronSpectrum(Ecrit,Ex):
    # Ex [eV]
    #assuming gamma=1
    eps=Ex/2/Ecrit
    dIdE= eps**2 * scipy.special.kv(2/3,eps)**2
    return dIdE

#loadFilters(np.arange(0.,40., 0.2)*1e3,0)
if 0:
    Ex=np.arange(1,20e3,0.5)
    I=synchrotronSpectrum(18e3,Ex)
    plt.plot(Ex,I)
    plt.grid()
    plt.xlabel("eneergy [eV]")



import numpy
from scipy.signal import savgol_filter

def smooth(x,w=5,order=2,method='new',newmethod=1,num_pass=1):
    if w==0:
        return x
    if method=='cumsum':
        data=x
        window_width=w
        orig_length=np.size(data)
        cumsum_vec = np.cumsum(np.insert(data, 0, 0))
        ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
        diffl=orig_length-np.size(ma_vec)
        start=int(diffl/2)
        ret=np.zeros(orig_length)
        rest=np.size(ma_vec)-start
        end=start+np.size(ma_vec)
        ret[start:end]=ma_vec
        ret[0:start]=data[0:start]
        ret[end:]=data[end:]
        return ret
    if method=='new' or newmethod==1:
        x2=x[np.isfinite(x)]

        for i in np.arange(num_pass):
            x2 = savgol_filter(x2, w, order)
        y=np.copy(x)
        y[np.isfinite(x)]=x2
    if newmethod==0:
        yhat = savgol_filter(x, w, order)
        y=yhat
    return y

def find_closest(array,value):
    return np.nanargmin(np.abs(array-value))

def vlines_ceil(xs,floor,xax,curve,color='k',lw=2):
    for i,x in enumerate(xs):
        data_i=find_closest(xax,x)
        ceiling=curve[data_i]
        plt.plot([x,x],[floor,ceiling],'-',color=color,lw=lw)

#https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def remove_jungfrau_stripes_1D(s1,factor=2):
    s1[511:513]=s1[511:513]/factor
    s1[255:257]=s1[255:257]/factor
    s1[767:769]=s1[767:769]/factor
    s1[0]=s1[0]/factor
    s1[1023]=s1[1023]/factor
    return s1

def normalize(spec):
    nm=np.nanmax(np.abs(spec))
    if nm!=0:
        spec=spec/nm
    return spec

def meanalize(spec):
    spec=spec/np.nanmean(spec)
    return spec

def plot_mark_at_line(x,l,ysize=10,lw=1,darkening=0.5):
    xax=l[0]._x
    curve=l[0]._y
    color=l[0].get_color()
    color=adjust_lightness(color,darkening)
    data_i=find_closest(xax,x)
    ceiling=curve[data_i]
    if np.isfinite(ceiling):
        plt.plot([x,x],[ceiling+ysize,ceiling-ysize],color=color,lw=lw)


def text(x,y,text='',fs=12,yoff=0,color='k',background=[1,1,1,0.5],darkening=1.,xi=[],zorder=0,rotation=0,verbose=1):
    """
    just do plt.text, but checks if I'm within the figure

    """
    yl=plt.ylim()
    xl=plt.xlim()
    outside=0
    if y<np.min(yl) or y>np.max(yl): outside=1
    if x<np.min(xl) or x>np.max(xl): outside=1
    if outside:
        l(verbose,'This is outside ({:.2e},{:.2e}):{:}'.format(x,y,text))
        return []
    else:
        if background is None:
            te=plt.text(x,y,text,color=color,fontsize=fs,zorder=zorder,rotation=rotation)
        else:
            te=plt.text(x,y,text,color=color,fontsize=fs,backgroundcolor=background,zorder=zorder,rotation=rotation)
        return te


def text_at_plot(l,x,text='',fs=12,yoff=0,background=[1,1,1,0.5],darkening=1.,xi=[],zorder=0):
    """
    Plots a label to a line, with the same color and at y position of the line

    Parameters
    ----------
    l : line object
        get it as l=plt.plot()
    x : int
        on which 'x' position you want to draw the label
    text : string
        what you want to write
    fs : TYPE, optional
        font size. The default is 12.
    yoff : TYPE, optional
        vertical offset above line. The default is 0.

    Returns
    -------
    None.

    """
    if x is None: return
    xax=l[0]._x
    curve=l[0]._y
    color=l[0].get_color()
    color=adjust_lightness(color,darkening)
    if text=='':
        text=l[0]._label
    if np.size(xi)==0:
        data_i=find_closest(xax,x)
        ceiling=curve[data_i]
    else:
        x=xax[xi]
        ceiling=curve[xi]
    yl=plt.ylim()
    if ceiling+yoff<yl[0]:
        ceiling=yl[0]+yoff
    if ceiling+yoff>yl[1]:
        ceiling=yl[1]+yoff
    if np.isfinite(ceiling):
        plt.text(x,ceiling+yoff,text,color=color,fontsize=fs,backgroundcolor=background,zorder=zorder)


def vline(l,x,y0=0,color='-', linewidth=1,darkening=1,label=''):
    xl=plt.xlim()
    if color=='-':
        color=l[0].get_color()
    color=adjust_lightness(color,darkening)
    if x>xl[1]: return 0, color
    xax=l[0]._x
    curve=l[0]._y

    ytop=np.interp(x,xax,curve)
    plt.plot([x,x],[y0,ytop],color=color,linewidth=linewidth)
    if label!='':
        plt.text(x+5,ytop*1.2,label,fontsize=12)
    return ytop,color

def convolve_gauss(vals,fwhm,unitsperpoint=1,debug=0,nan_ends=1):
#assume the input has equal spacing points
    #units per points: conversion from 'FWHM' units to spacing of points of 'vals'
    if fwhm==0:
        return vals
    fwhm_points=1.0*fwhm/unitsperpoint
    if fwhm_points<1:
        return vals
    sigma=fwhm_points/2.355
    w=2*int(fwhm_points)
    xg=np.arange(2*w+1)
    x0=2*fwhm_points
    gauss=np.exp(-((xg-x0)**2)/(2*sigma**2))
    gauss=gauss/np.sum(gauss)
    if debug:
        xax=np.arange(np.size(gauss))*unitsperpoint
        plt.plot(xax,gauss/np.max(gauss),label='gauss')
    res=np.convolve(vals,gauss)
    res=res[w:np.size(res)-w]
    if nan_ends:
        res[0:w]=np.nan
        res[-w:]=np.nan
    else:
        res[0:w]=0
        res[-w:]=0
    return res

def convolve_gauss_range(xax,spec,minE,maxE,fwhm):
    sel=(xax>=minE)*(xax<=maxE)
    xs2=convolve_gauss(spec,fwhm)
    spec[sel]=xs2[sel]
    return spec


def convolve_PSF(vals,PSF,PSF_scaling):
#assume the input has equal spacing points
    #PSF scaling = bin_width of PSF / bin_width of vals
    PSF=PSF[~np.isnan(PSF)]
    oldsize=np.size(PSF)
    newsize=int(oldsize*PSF_scaling/2)*2
    oldx=np.arange(oldsize)
    newx=np.arange(newsize)*(oldsize/newsize)
    newpsf=np.interp(newx,oldx,PSF)
    newpsf=newpsf/np.sum(newpsf)

    sp = int(schwerpunkt(np.arange(newsize),newpsf))
    res=np.convolve(newpsf,vals)
#    w=int(newsize/2)
 #   res=res[w:np.size(res)-w+1]
    ns=np.size(vals)
    res=res[sp:ns+sp]
    return res

def convolve_lorentz(vals,fwhm,unitsperpoint):
#assume the input has equal spacing points
    fwhm_points=fwhm/unitsperpoint
    gamma=fwhm_points/2
    w=12*int(fwhm_points)
    xg=np.arange(2*w+1)
    x0=w
    lor=1/(1+((xg-x0)/gamma)**2)
    lor=lor/np.sum(lor)
    res=np.convolve(vals,lor)
    print(gamma)
    print(np.shape(vals))
    print(np.shape(res))
    res=res[w:np.size(res)-w]
    print(np.shape(res))
#    plt.plot(xg,lor)
    return res


def pamfit(xax,data,fitmin=8970,fitmax=9100,debug=0,initpars=[],plotoffset=-100):
    sel=(xax>fitmin) *(xax<fitmax)
    x2=xax[sel]
    d2=data[sel]
    ipars=initpars
    if debug: print(ipars)
    f=pamline(x2,ipars[0],ipars[1],ipars[2],ipars[3],ipars[4],ipars[5])
    sigma=d2*0+1
    if debug:
        plt.figure(figsize=(10,6))
        plt.plot(x2,d2,label='data')
        plt.plot(x2,f+0.01,label='init')
    p=optimization.curve_fit(pamline, x2, d2, ipars, sigma)
    pars=p[0]
    f2=pamline(x2,pars[0],pars[1],pars[2],pars[3],pars[4],pars[5])
    if debug:
        plt.plot(x2,f2,label='fit')
        plt.legend()
        print(pars)
    return pars


def pamline(xax,peakx,peaky,e1,e2,basesignal,sm):
    m1=peaky/np.exp(e1*peakx)
    m2=peaky/np.exp(e2*peakx)
    exp1=np.exp(e1*xax)*m1
    exp2=np.exp(e2*xax)*m2
    exps=np.min(np.stack([exp1,exp2]),0)
  #  plt.semilogy(xax,exp1)
    exps[np.logical_not(np.isfinite(exps))]=0
    sm2=int(sm)#50.3
    exps=smooth(exps,sm2,order=1,method='cumsum')
    #exps=convolve_gauss(exps,sm)
    exps=basesignal-exps
    return exps


def bin_spectrum(xax,prof,binsize):
    # % #downsample assume equidistant xax
    leng=np.size(prof)
    newlen=int(np.floor(leng/binsize))
    newsize=newlen*binsize
    prof2=np.zeros(newsize)*np.nan
    prof2=prof[0:newsize]
    xax2=np.zeros(newsize)*np.nan
    xax2=xax[0:newsize]
    newprof=np.nansum(np.reshape(prof2,(newlen,binsize)),1)
    newxax=np.nanmean(np.reshape(xax2,(newlen,binsize)),1)
    return newxax,newprof

def bin_irregular_ibu(xax,prof,edges,draw=0,use_centers=0):
    # % #downsample
    #bin variable:
    #return also uncertainity sigma
    if use_centers:
        centers=edges*1.0
        dch=np.diff(centers)/2
        edges=centers[:-1]+dch
        e0=centers[0]-dch[0]
        el=centers[-1]+dch[-1]
        edges=np.append([e0],edges)
        edges=np.append(edges,[el])
    else:
        centers=edges[:-1]+np.diff(edges)/2
    bined=centers*0
    unc=centers*0
    for ij,cen in enumerate(centers):
        xmin=edges[ij]
        xmax=edges[ij+1]
        sel=np.logical_and(xax>=xmin,xax<=xmax)
        bined[ij]= np.nanmean(prof[sel])
        photons=np.sum(sel)*bined[ij]
        u=np.sqrt(photons)/np.sum(sel)
#        m,u = mean_uncertainty(prof[sel])
        unc[ij]=u

    bined[np.isinf(bined)]=np.nan
    if draw:
        plt.plot(edges,edges*0+0.45,'*')
        plt.plot(centers,centers*0+0.46,'*')
        plt.plot(centers,bined,'*-')
    return centers,bined,unc


def make_lineout_ibu(xax,img,edges): #irregular bin with undertainties
#this is actually merge of bin_irregular and make_lineout_uncertainty
    centers=edges[:-1]+np.diff(edges)/2
    lineout=centers*0
    unc=centers*0
    for ij,cen in enumerate(centers):
        xmin=edges[ij]
        xmax=edges[ij+1]
        sel=np.logical_and(xax>=xmin,xax<=xmax)
        region=img[:,sel] #region of img where I'm averaging
        m,u=mean_uncertainty(region) # u is sigma, value is in m+-u with 68%..
        lineout[ij]=m
        unc[ij]=u
    lineout[np.isinf(lineout)]=np.nan
    return centers,lineout,unc


def draw_poisson(xax,data,multiplier=1,yoff=0,binsize=21,sm2=-1,color=[0.3,0.3,0.9],alpha=.33,sigma=1,smooth_error=1):
    multiplierIn=multiplier
    dataIn=data
    xaxb,data=bin_spectrum(xax,dataIn,binsize)

    if np.size(multiplier)>1:
        xax2,multiplier=bin_spectrum(xax,multiplierIn,binsize)

    #data is to be in units of photons per bin
    #dApD=smooth(data,sm)
    error=np.sqrt(data)*sigma
    if sm2==-1:
        sm2=2*binsize+1
    #dataIn=smooth(dataIn,binsize)
    dataIn=convolve_gauss(dataIn,binsize,1)
    plt.plot(xax,dataIn*multiplierIn+yoff,'-',color=color)
    if smooth_error:
        errorsm=np.interp(xax,np.flip(xaxb),np.flip(error))/binsize
        dLow=dataIn + errorsm
        dHigh=dataIn - errorsm
        dLow=dLow*multiplierIn+yoff
        dHigh=dHigh*multiplierIn+yoff
        plt.fill_between(xax,dLow,dHigh,alpha=alpha,color=color,step='mid')
    else:
        dLow=data + error
        dHigh=data - error
        multiplier=multiplier/binsize/binsize
        dLow=dLow*multiplier+yoff
        dHigh=dHigh*multiplier+yoff
        plt.fill_between(xaxb,dLow,dHigh,alpha=alpha,color=color,step='mid')
    #plt.fill_between(xax,dLowsm,dHighsm,alpha=alpha,color=color)

    if 0: #plot relative error
        plt.plot(xaxb,np.sqrt(data)*sigma/data,'-',color=color)

def get_FWHM(data):
    h=data
    maxi=np.max(h)
    mini=np.min(h)
    hm=(maxi-mini)/2+mini
#floor
    datf=np.copy(h)
    datf[datf>=mini+hm/2]=np.nan
    floor=np.nanmean(datf)

    hm=(maxi-floor)/2+floor
    dat2=np.copy(h)
    dat2[dat2<=hm]=0
    dat2[dat2>hm]=1
    FWHM=np.nansum(dat2)

    if True:
        plt.plot(dat2*(maxi-floor)/2+floor)

    FWHM=np.nansum(dat2)
    return FWHM

def get_FWHM_xax(xax, data,region=[],fixedfloor=0,debug=0):
    if region!=[]:
        sel=np.logical_and(xax>=region[0],xax<=region[1])
        xax=xax[sel]
        data=data[sel]

    h=data
    maxi=np.nanmax(h)
    mini=np.nanmin(h)
    hm=(maxi-mini)/2+mini
#floor
    if fixedfloor:
        floor=0
    else:
        print(datf)
        datf=np.copy(h)
        datf[datf>=mini+hm/2]=np.nan
        floor=np.nanmean(datf)

    hm=(maxi-floor)/2+floor
    dat2=np.copy(h)
    dat2[dat2<=hm]=0
    dat2[dat2>hm]=1
    #FWHM=np.nansum(dat2)

    if debug:
        #print(FWHM)
        print(dat2)
        plt.plot(xax,data)
        plt.plot(xax,dat2*(maxi-floor)/2+floor,label='fwhm')

#    FWHM=np.nansum(dat2)
    sel=dat2>0
    minx=np.min(xax[sel])
    maxx=np.max(xax[sel])
    FWHM=maxx-minx
    return FWHM


def schwerpunkt(xax,vals,region=[]):
    if region!=[]:
        sel=np.logical_and(xax>=region[0],xax<=region[1])
        xax=xax[sel]
        vals=vals[sel]
    sw=np.sum(xax*vals)
    s=np.sum(vals)
    sp=sw/s
    return sp
#    sp=mmmUtils.schwerpunkt(xax,ema,region=(8580,8680))

def stddev(xax,vals,region=[],x0=0):
    if region!=[]:
        sel=np.logical_and(xax>=region[0],xax<=region[1])
        xax=xax[sel]
        vals=vals[sel]
    x0x=np.abs(xax-x0)
    sw=np.sum(x0x*vals)
    s=np.sum(vals)
    sig=sw/s
    return sig


def plt_zero_origin(skipY=0,skipX=0):
    if not skipX:
        xl=plt.xlim()
        plt.xlim(0,xl[1])
    if not skipY:
        yl=plt.ylim()
        plt.ylim(0,yl[1])

def plt_zero_origin_ax(ax,skipY=0,skipX=0):
    if not skipX:
        xl=ax.get_xlim()
        ax.set_xlim(0,xl[1])
    if not skipY:
        yl=ax.get_ylim()
        ax.set_ylim(0,yl[1])


def cutRect(rect,img):
    #print(rect)
    #print(np.size(rect))
    if np.size(rect)!=4 :#or (rect==None).any():
        return img
    r=rect
    if np.ndim(img)==2:
        imgc=img[r[2]:r[3],r[0]:r[1]]
    elif np.ndim(img)==3:
        imgc=img[:,r[2]:r[3],r[0]:r[1]]
    else:
        assert 0, "Wrong dimensionality in cutRect."
    return imgc

def drawRect(rect,color=[1,1,1],lw=1,label='',fs=24):
    r=rect
    plt.plot([r[0], r[0], r[1], r[1], r[0]], [r[2], r[3], r[3], r[2], r[2]], '-',linewidth=lw,color=color)
    if label!='':
        tx=(r[1]-r[0])/2+r[0]
        ty=(r[3]-r[2])/2+r[2]
        plt.text(tx,ty,label,color=color,fontsize=fs,backgroundcolor=[1,1,1,0.5])



def loadPickle(fn,debug=0,safe=1,strict=0):
    if fn[-7:]!='.pickle':
        fn=fn+'.pickle'
    if os.path.isfile(fn):
        if debug: print('Loading pickle: '+fn)
        return pickle.load(open(fn,"rb"))
    else:
        if strict:
            assert 0, ('Pickle not found: '+fn)
        elif safe:
            print('Pickle not found: '+fn)

        return 0

def dumpPickle(variable,fn):
    if fn[-7:]!='.pickle':
        fn=fn+'.pickle'
    pickle.dump(variable, open(fn, "wb" ) )
    grant_all(fn)

def figure(figx=12,figy=8,safe=0):
    if safe:
        #following : https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots
        #..this make it memory effecitve if using many figures
        #attention: do NOT use plt.close.
        fig=plt.figure(figsize=(figx,figy),num=1,clear=True)
        figg = plt.gcf()
        figg.set_size_inches(figx,figy)
    else:
        fig=plt.figure(figsize=(figx,figy))
    return fig

def figure2(ratio=1/3,width=12):
    figx=width
    figy=figx*ratio
    fig=plt.figure(figsize=(figx,figy))
    return fig

def unique_average(x,y):
    xu=np.unique(x)
    yu=xu*0
    for ui,x1 in enumerate(xu):
        sel3=(x==x1)
        yu[ui]=np.mean(y[sel3])
    return xu,yu


def savefig(fn,do_tight=1,dpi=150):
    if fn[-4:]!='.png' and fn[-4:]!='.jpg':
        fn=fn+'.jpg'
    if do_tight:
        plt.tight_layout()
    plt.savefig(fn,dpi=dpi)
    grant_all(fn)

def draw_cross(x=0,y=0,size=10,gap=3,color='w',lw=1):
    length=size
    c=color
    plt.plot([x-length,x-gap],[y,y],color=c,lw=lw)
    plt.plot([x+length,x+gap],[y,y],color=c,lw=lw)
    plt.plot([x,x],[y-length,y-gap],color=c,lw=lw)
    plt.plot([x,x],[y+length,y+gap],color=c,lw=lw)

def mean_uncertainty(values):
    mean=np.mean(values)
    n=np.size(values)
    unc=np.std(values)/np.sqrt(n)

    #the real value is in the inteval mean +-  unc with 68% prob.
    #the real value is in the inteval mean +-2*unc with 95% prob.
    return mean,unc


def write_table(table,filename,names=0,skip_nans=1):
    from astropy.io import ascii
    if skip_nans:
        sel=np.array(np.isfinite(np.sum(table,axis=1)))
        table=table[sel,:]
    if names!=0:
        ascii.write(table, filename, overwrite=True,format='commented_header',names=names)
    else:
        ascii.write(table, filename, overwrite=True,format='commented_header')



def weighted_mean(values,weights):
    return np.sum(values*weights) / np.sum(weights)

def get_rms_fwhm(xax,seeded,mean):
    spH=0
    scnt=0
    for x,val in enumerate(seeded):
        if np.isnan(val):continue
        spH+=val*(xax[x]-mean)**2
        scnt+=val
    rms=(spH/scnt)**(1/2)# std.dev
    fwhm=rms*2.3548#FWHM jedné poloviny spektra (continuum)[mm]
    return rms,fwhm


def get_darkvalue(data,fraction=0.2):
    mean=np.mean(data)
    maxi=np.max(data)
    hm=(maxi-mean)*fraction+mean #quarter of the maxima
    sel=(data<hm)
    mean2=np.mean(data[sel])
    return mean2




def plot_vline_till_plot(x,xkev,prof,y0=0,color='k', linewidth=1):
 # ind=find_closest(xkev,x)
#  ytop=prof[ind]
    if np.size(x)>1:
        for x1 in x:
          ytop=np.interp(x1,xkev,prof)
          plt.plot([x1,x1],[y0,ytop],color=color,linewidth=linewidth)
    else:
      ytop=np.interp(x,xkev,prof)
      plt.plot([x,x],[y0,ytop],color=color,linewidth=linewidth)



def draw_horizontal_edges(centralY,spectralHeight,x0=0,x1=2000,color='w', linewidth=0.5):
    plt.plot([x0,x1],[centralY-spectralHeight,centralY-spectralHeight],color=color,linewidth=linewidth)
    plt.plot([x0,x1],[centralY+spectralHeight,centralY+spectralHeight],color=color,linewidth=linewidth)



def norm_reg(xax,prof,reg=[]):
    if reg!=[]:
        sel=np.logical_and(xax>reg[0], xax<reg[1])
        maxi=np.nanmax(prof[sel])
    else:
        maxi=np.nanmax(prof)
    return prof/maxi


def arcsec_to_mrad(arcsec):
    deg=arcsec/60/60
    mrad=deg/180*np.pi*1e3
    return mrad



def mrad_to_arcsec(mrad):
    deg=mrad*1e-3/np.pi*180
    arcsec=deg*60*60
    return arcsec

def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

def grant_all(path,verbose=1):
#    mode = os.stat(path).st_mode
 #   mode |= (mode & 0o444) >> 2    # copy R bits to X
    #os.chmod(path, 0o0777)
    try:
        os.chmod(path, 0o0770)
    except:
        l(verbose,'couldnt change file permission')

def gauss(xa,a0, amp,x0,fwhm,power=2):
    ret=np.zeros(np.shape(xa))

    sigma=fwhm/2.355
    for i,x in enumerate(xa):
        ret[i]= a0 + + amp* np.exp(-((x-x0)**power)/(2*sigma**power))
    return ret

def gaussfit(xax,data,fitmin=8970,fitmax=9100,debug=0,initpars=[],plotoffset=-100):
    sel=(xax>fitmin) *(xax<fitmax)
    x2=xax[sel]
    d2=data[sel]
    if initpars==[]:
        ipars=np.array([0.78,8995,5,1])
    else:
        ipars=initpars
    if debug: print(ipars)
    f=gauss(x2,ipars[0],ipars[1],ipars[2],ipars[3])
    sigma=d2*0+1
    p=optimization.curve_fit(gauss, x2, d2, ipars, sigma)
    pars=p[0]
    #unc= np.sqrt(np.diag(p[1]))
    f2=gauss(x2,pars[0],pars[1],pars[2],pars[3])
    if plotoffset!=-100:
        plt.plot(x2,f2+plotoffset,color=[0.2,0.8,0.2])
    if debug:
        plt.figure(figsize=(10,6))
        plt.plot(x2,d2,label='data')
        plt.plot(x2,f,label='init')
        plt.plot(x2,f2,label='fit')
        plt.legend()
        print(ipars)
        print(pars)
    return pars

def gauss_l(xa,a0,xb, amp,x0,fwhm):
    ret=np.zeros(np.shape(xa))

    sigma=fwhm/2.355
    for i,x in enumerate(xa):
        ret[i]= a0 +xb*(x-x0) + amp* np.exp(-((x-x0)**2)/(2*sigma**2))
    return ret

def gaussfit_l(xax,data,fitmin=8970,fitmax=9100,debug=0,plotoffset=-100,initpars=[],raiseit=0):
#with linear background
    sel=(xax>fitmin) *(xax<fitmax)
    x2=xax[sel]
    d2=data[sel]

    if initpars==[]:
        ipars=np.array([0.78,0.52,8995,5,1])
    else:
        ipars=initpars
    f=gauss_l(x2,ipars[0],ipars[1],ipars[2],ipars[3],ipars[4])
    if debug:
        plt.figure(figsize=(10,6))
        plt.plot(x2,d2,label='data')
        plt.plot(x2,f,label='init')
    sigma=d2*0+1
    try:
        p=optimization.curve_fit(gauss_l, x2, d2, ipars, sigma)
    except:
        print('Gaussian fiting failed')
        plt.figure(figsize=(8,6))
        plt.plot(xax,data,'k',label='data whole',lw=1)
        plt.plot(x2,d2,label='data')
        plt.plot(x2,f,label='init')
        plt.title('Fitting failed!')
        plt.legend()
        if raiseit:
            raise Exception("Gaussina fit really failed.")
        else:
            return initpars,0
    pars=p[0]
    #unc= np.sqrt(np.diag(p[1]))
    f2=gauss_l(x2,pars[0],pars[1],pars[2],pars[3],pars[4])
    if plotoffset!=-100:
        plt.plot(x2,f2+plotoffset,color=[0.2,0.8,0.2])
    if debug:
        plt.plot(x2,f2,label='fit')
        plt.legend()
    maxi=np.argmax(f2)
    maxx=x2[maxi]
    return pars,maxx


def gauss_l_power(xa,a0,xb, amp,x0,fwhm,power):
    ret=np.zeros(np.shape(xa))

    sigma=fwhm/2.355
    for i,x in enumerate(xa):
        ret[i]= a0 +xb*(x-x0) + amp* np.exp(-(((x-x0)**2)/(2*sigma**2))**power)
    return ret

def gaussfit_l_power(xax,data,fitmin=8970,fitmax=9100,debug=0,plotoffset=-100,initpars=[],raiseit=0):
#with linear background
    sel=(xax>fitmin) *(xax<fitmax)
    x2=xax[sel]
    d2=data[sel]

    if initpars==[]:
        ipars=np.array([0.78,0.52,8995,5,1,1])
    else:
        ipars=initpars
    f=gauss_l_power(x2,ipars[0],ipars[1],ipars[2],ipars[3],ipars[4],ipars[5])
    if debug:
        plt.figure(figsize=(10,6))
        plt.plot(x2,d2,label='data')
        plt.plot(x2,f,label='init')
    sigma=d2*0+1
    try:
        p=optimization.curve_fit(gauss_l_power, x2, d2, ipars, sigma)
    except:
        print('Gaussian fiting failed')
        plt.figure(figsize=(8,6))
        plt.plot(xax,data,'k',label='data whole',lw=1)
        plt.plot(x2,d2,label='data')
        plt.plot(x2,f,label='init')
        plt.title('Fitting failed!')
        plt.legend()
        if raiseit:
            raise Exception("Gaussian fit really failed.")
        else:
            return initpars,0
    pars=p[0]
    #unc= np.sqrt(np.diag(p[1]))
    f2=gauss_l_power(x2,pars[0],pars[1],pars[2],pars[3],pars[4],pars[5])
    if plotoffset!=-100:
        plt.plot(x2,f2+plotoffset,color=[0.2,0.8,0.2])
    if debug:
        plt.plot(x2,f2,label='fit')
        plt.legend()
    maxi=np.argmax(f2)
    maxx=x2[maxi]
    return pars,maxx


def get_binned_uncertainty(xin,datain,binsize=21):
    xax=np.arange(np.min(xin),np.max(xin)+1e-10,step=binsize)
    xcen=xax+binsize/2
    xcen=xcen[:-1]
    data=xcen*0
    stds=xcen*0
    for i,x in enumerate(xax[0:-1]):
        sel=np.logical_and(xin>x, xin<=xax[i+1])
        #print(x)
        #if x==25:
         #   print(x)
          #  print(xax[i+1])
           # print(xin)
            #print(sel)
#            print(np.sum(sel))
        mean=np.mean(datain[sel])
        data[i]=mean
        std=np.std(datain[sel])
        stds[i]=std
    return xcen, data,stds
    #data is to be in units of photons per bin
    #dApD=smooth(data,sm)
    error=np.sqrt(data)*sigma
    if sm2==-1:
        sm2=2*binsize+1
    #dataIn=smooth(dataIn,binsize)
    dataIn=convolve_gauss(dataIn,binsize,1)
    plt.plot(xax,dataIn*multiplierIn+yoff,'-',color=color)
    if smooth_error:
        errorsm=np.interp(xax,np.flip(xaxb),np.flip(error))/binsize
        dLow=dataIn + errorsm
        dHigh=dataIn - errorsm
        dLow=dLow*multiplierIn+yoff
        dHigh=dHigh*multiplierIn+yoff
    else:
        dLow=data + error
        dHigh=data - error
        multiplier=multiplier/binsize/binsize
        dLow=dLow*multiplier+yoff
        dHigh=dHigh*multiplier+yoff
        plt.fill_between(xaxb,dLow,dHigh,alpha=alpha,color=color,step='mid')
    #plt.fill_between(xax,dLowsm,dHighsm,alpha=alpha,color=color)

    if 0: #plot relative error
        plt.plot(xaxb,np.sqrt(data)*sigma/data,'-',color=color)
        source1_in=runNos*0


def get_binned_uncertainty_log(xin,datain,numbins):
    sel=xin>0
    xin=xin[sel]
    datain=datain[sel]
    xl=np.log(xin)
    bs=(np.max(xl)-np.min(xl))/numbins
    xcen, data,stds=get_binned_uncertainty(xl,datain,bs)
    xout=np.exp(xcen)
    return xout, data,stds


def norm(arr):
    arr=arr/np.nanmax(arr)
    return arr

def centers(edges):
    centers=edges[:-1]+np.diff(edges)/2
    return centers

def remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def mkdir(fn,verbose=1):
    if not os.path.isdir(fn):
        l(verbose,"Directory {:s} created.".format(fn))
        os.makedirs(fn)
        return 1
    else:
        l(verbose,"Directory {:s} already exists.".format(fn))
        return 0

def remove(fn,verbose=0):
    if os.path.exists(fn):
        if verbose:
            print("file {:s} removed.".format(fn))
        os.remove(fn)

# Export animated gif
def animate(filenames='',outfile='',filelist=None,debug=0):
    import imageio
    if filelist!=None:
        writer=imageio.get_writer(outfile, mode='I')
        for fi,filet in enumerate(filelist):
            if not os.path.isfile(filet):
                continue
            if debug: print('adding ',filet)
            image = imageio.imread(filet)
            writer.append_data(image)

    else:
        files=glob.glob(filenames)
        writer=imageio.get_writer(outfile, mode='I')
        files=np.sort(files)
#        print(files)
 #       return 0
        for fi,filet in enumerate(files):
            if not os.path.isfile(filet):
                continue
            if debug: print('adding ',filet)
            image = imageio.imread(filet)
            writer.append_data(image)
    print('Animation done')

def measure_background_rects(im_main,rects,draw=0,maxclim=80,smooth=5):
    vals=np.zeros(np.shape(rects)[0])
    for ri,rect in enumerate(rects):
        vals[ri]=np.nanmean(cutRect(rect,im_main))
    back=np.mean(vals)
    if draw:
        figure()
        plt.subplot(121)
        plt.imshow(im_main)
        plt.colorbar()
        plt.clim(20,maxclim)
        plt.ylim(0,700)
        for ri,rect in enumerate(rects):
            drawRect(rects[ri],lw=2,label='{:.0f}'.format(vals[ri]),fs=20)
        plt.title('Raw image, raw mean is {:.0f} keV/px'.format(np.nanmean(im_main)))

        plt.subplot(122)
        im=im_main-back
        ims=smooth2d(im,smooth,smooth)
        plt.imshow(ims,cmap=rofl.cmap())
        plt.colorbar()
        plt.clim(0,120)
        plt.ylim(0,700)
        plt.title('Smoothed, after subtraction of {:.1f} keV/px'.format(back))
    return back





def update_shot_params(shot,p,silent=1):
    df = pandas.read_excel('shot_params.xls',sheet_name=0)
    a=df['Shot']
    rowb=(a==shot)
    rowi=np.argmax(rowb)
    row=df.loc[rowi, :]
    for ci,val  in enumerate(row):
        if ci==0:continue
        if not isinstance(val, str):
            if np.isnan(val): continue
        cnames=df.columns[ci]
        cnames=cnames.split('__')
        cname=cnames[0]
        if cname not in p.keys():
            if cname!='comment':
                if not silent: print('this column is wrong: '+cname)
        else:
            if np.size(cnames)==1:
                if not silent: print("Updating param '{:s}' to value '{:}'.".format(cname,val))
                p[cname]=val
            else:
                if not silent:  print("Updating param '{:s}'/'{:s}' to value '{:}'.".format(cname,cnames[1],val))
                p[cname][cnames[1]]=val
    return 1

def get_shot_list():
    df = pandas.read_excel('shot_params.xls',sheet_name=0)
    a=df['Shot']
    shots=np.zeros(np.size(a))*np.nan
    for i,shot in enumerate(a):
        shots[i]=shot
    shots2=shots[np.logical_not(np.isnan(shots))]
    shots3=shots2.astype(int)
    return shots3


def update_progress(progress,maxi,mini=0,style=0,barLength = 30):
    # https://stackoverflow.com/questions/3160699/python-progress-bar
     # Modify this to change the length of the progress bar
    status = ""
#    if isinstance(progress, int):
 #       progress = float(progress)
  #  if not isinstance(progress, float):
   #     progress = 0
    #    status = "error: progress var must be float\r\n"
    progress=progress*1.
    if progress >= maxi:
        progress = maxi
        status = "Done"
    pr2=progress-mini
    mx2=maxi-mini
    block = int(round(barLength*pr2/mx2))
    if style:
        yes="😋"
        no="😶"
    else:
        yes="*"
        no="-"
    text = "\r [{0}] {1:2,.0f}/{2:2,.0f} {3}".format( yes*block + no*(barLength-block), progress,maxi, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def update_progress2(progress1,progress2,maxi,style=0,barLength = 30):
        # https://stackoverflow.com/questions/3160699/python-progress-bar
         # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress1, int):
            progress1 = float(progress1)
        if progress1 >= maxi:
            progress1 = maxi
            status = "Done"
        block1 = int(round(barLength*progress1/maxi))
        block2 = int(round(barLength*progress2/maxi))
        if style:
            yes="😋"
            no="😶"
        else:
            yes="*"
            no="-"
        text1 = "\r [{0}] ".format( yes*block1 + no*(barLength-block1))
        text2 = " [{0}] ".format( yes*block2 + no*(barLength-block2))
        sys.stdout.write(text1+' '+text2)
        sys.stdout.flush()


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None:
        return False
    if element==False:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


# %% fitting composition of gaussian functions

def fit_composition(xax,spec,init_pars,xrange,maxiter=30,draw_progress=0,free_pars=[]):
    num_params=np.size(init_pars)
    if np.size(free_pars)!=np.size(init_pars):
        free_pars=np.isfinite(init_pars)
 #   num_params=np.sum(free_pars)

    init_steps=init_pars*0.
    init_steps[0,:]=2#eV position
    init_steps[0,:]=5#eV
    init_steps[1,:]=4#eV - width
    init_steps[2,:]=0.5#amplitude
    flat_shape=(1,num_params)
    real_shape=np.shape(init_pars)
    steps=np.squeeze(np.reshape(init_steps,flat_shape))
    pars=np.squeeze(np.reshape(init_pars,flat_shape))
    frees=np.squeeze(np.reshape(free_pars,flat_shape))
    numsteps=maxiter
    residuals=np.zeros(numsteps)*0

    spec[spec<0]=1e-4  #this funny line just avoid crashing of the fit
    if draw_progress:
        plt.title('progress')
    for i in np.arange(numsteps):
        for pi in np.arange(num_params):
            free=frees[pi]
            if not free: continue
            #else: print(pi)
            step_size=steps[pi]
            variations=np.arange(-4,5)*step_size
            variations=np.arange(-2,3)*step_size
            results=variations*0
            for vi,var in enumerate(variations):
                pars1=pars*1.0
                if pi>=int(2*num_params/3):
                    pars1[pi]=pars1[pi]*(2**var) #amplitude: log scale
                else:
                    pars1[pi]+=var #shift & width: lin scale
                pars1_2d=np.reshape(pars1,real_shape)
                c1=get_linecomposition(pars1_2d,xax)
                results[vi]=fit_least_square(xax,spec,c1,xrange)
                if not np.isfinite(results[vi]):
                    print('crash')
                    print(pars1_2d)
                    ret_pars=np.reshape(pars,real_shape)
                    return ret_pars,residuals
            if 0 and (pi==int(2*num_params/3)+1):
                print(i)
                print(pars[pi])
                print(pars[pi]*(2**variations))
                print(results)

            am=np.nanargmin(results)
            if pi>=int(2*num_params/3):
                pars[pi]=pars[pi]*(2**variations[am]) #amplitude: log scale
            else:
                pars[pi]+=variations[am] #adjusting given parameter
            if pi>=int(num_params/3) and pi<int(2*num_params/3): #width
                if pars[pi]<=0: pars[pi]=1
            if am>0 and am<np.size(variations)-1: #decrease step size
                steps[pi]=steps[pi]/1.2
        if draw_progress:
            fit1=get_linecomposition(np.reshape(pars,real_shape),xax,draw=0)
            plt.semilogy(xax,fit1,lw=1,label=i)
        residuals[i]=(np.min(results))
        update_progress(i*1.0,numsteps*1.)
    ret_pars=np.reshape(pars,real_shape)
    if draw_progress:
        plt.legend()
    return ret_pars,residuals


def getValues(xax,spec,initEs):
    vals=initEs*0.
    for ei,E in enumerate(initEs):
        data_i=find_closest(xax,E)
        val=spec[data_i]
        vals[ei]=val
    return vals


def get_linecomposition(fit_pars,xax,draw=0):
    #pars are [x0, fwhm, amplitude, ]
    lc=xax*0.
    for li in np.arange(np.shape(fit_pars)[1]):
        line=fit_pars[:,li]
        sigma=np.abs(line[1])/2.355
        x0=line[0]
        gauss=np.exp(-((xax-x0)**2)/(2*sigma**2))
        gauss=normalize(gauss)
        gauss*=line[2]
        lc+=gauss
        if draw:
            plt.semilogy(gauss,'-',label=li)
    if draw:
        plt.semilogy(lc,'k',label='sum')
        plt.legend()
    return lc


def fit_least_square(xax,spec,fit,xrange):
    sel=(xax>xrange[0])*(xax<xrange[1])
    lsq=np.nansum((np.log(spec[sel]) - np.log(fit[sel]))**2)
    return lsq


def rez(number): #Remove the useless Exponent Zero
    if number[-4:-1]=='e+0' :
        return number[:-3] + number[-1:]
    elif number[-4:-1]=='e-0':
        return number[:-2] + number[-1:]
    else:
        return number


def yamlval(key, ip, default=0):
    val = ip.get(key, default)
    
    # Treat both Python None and string 'None' as missing
    if val is None or val == 'None':
        return default

    # Try converting numeric-looking strings to float
    try:
        return float(val)
    except (TypeError, ValueError):
        return val  # keep strings, bools, etc. as-is


#def yamlval(key,ip,default=0):
#    if not key in ip.keys() :
#        return default
#    else:
#        val=ip[key]
#        if val=='None':
#            return 0
#        else:
#            return ip[key]

#double dict value
def ddval(ip,key1,key2,default1=0,default2=0):
    if not key1 in ip.keys() :
        return default1
    if not key2 in ip[key1].keys() :
        return default2
    else:
        return ip[key1][key2]


def title(text,ha='left'):
    ax=plt.gca()
    if ha=='left':
        ax.set_title(text, ha='left', x=0)
    else:
        ax.set_title(text)



def quadratic_equation(a,b,c):

    d = b**2-4*a*c # discriminant

    if d < 0:
        print ("This equation has no real solution")
    elif d == 0:
        x = (-b+np.sqrt(b**2-4*a*c))/2*a
        return np.array([x1])
#        return x
        print ("This equation has one solutions: "), x
    else:
        x1 = (-b+np.sqrt((b**2)-(4*(a*c))))/(2*a)
        x2 = (-b-np.sqrt((b**2)-(4*(a*c))))/(2*a)
        return np.array([x1,x2])
        print ("This equation has two solutions: ", x1, " or", x2)


def l(verbose,*messages,ticking=0):
    if verbose<=0: return
    global print_levels
    if verbose>print_levels:
        print_levels=verbose
    m=" " *((print_levels-verbose)*2)
    print(m,*messages)
    if ticking:
      for m1 in messages:
        m=m+m1
      tick(m)
  #  print(m)

def labels(xs,ys,texts,xlog=1,ylog=1):
    xl=plt.xlim()
    if xlog:
        xtot=xl[1]/xl[0]
        print(xtot)
        xoff=1+xtot/5e7
        print(xoff)
    for i,x in enumerate(xs):
        xt=x
        yt=ys[i]
        if xlog:
            xt=xt*xoff
        text(xt,yt,texts[i])


"""
 Always assuming that the coordintaes are [X,Y], aka [horizontal, vertical]
 xc and yc are coordinates of centers of the points
 There are essentialy 3 consitent ways of visualistaion:
     only data are provided: labeled from zero
     center-arrays arep provided: used as cooridntaes, either:
         linearize=0 - each data point taking same width
         linearize=1 - each datapoint's center at given location, edges fit between.

"""
def pcolor(data=None,xc=None,yc=None,centers=None,log=0,ticks=1,linearize=0,cmap=None,colorbar=1,xl=None,yl=None,cl=None,decor=None,qclim=None,labels=None,label_format="m0",label_size=12,label_color='auto',xticks=None,yticks=None,background='mud',aspect='auto'):
    import matplotlib.cm as cm
    dt=np.transpose(data)  #This transpotion is the weirdes thing ever; Basically I made this routine mainly to have this consistently transposed
    if decor is not None:
        xc=decor[0]
        yc=decor[1]
        xl=decor[2]
        yl=decor[3]
    if centers is not None:
        xc=centers[0]
        yc=centers[1]
    linear_dimensions=1
    if xc is None:
        tickspacing=np.floor(np.shape(data)[0]/5)
        xed =np.arange(0,np.shape(data)[0]+1,tickspacing)-0.5
        xc=xed[:-1]+0.5
        xcpos=xed[:-1]+0.5
    else:
        if not linearize:
            meanstep=np.mean(np.diff(xc))
            #xed=np.append(xc[0]-meanstep,xc)+meanstep/2 naive approcah for homogeneous fields
            xle=np.append(xc[0]-meanstep,xc)
            xri=np.append(xc,xc[-1]+meanstep)
            xed=(xle+xri)/2
            if np.std(np.diff(xc))>1e-4*meanstep:
                linear_dimensions=0
        else:
            xed =np.arange(np.shape(data)[0]+1)-0.5
            xcpos=xed[:-1]+0.5

    if yc is None:
        yed=np.arange(np.shape(data)[1]+1)-0.5
        yc=yed[:-1]+0.5
        ycpos=yed[:-1]+0.5
    else:
        if not linearize:
            meanstep=np.mean(np.diff(yc))
            yle=np.append(yc[0]-meanstep,yc)
            yri=np.append(yc,yc[-1]+meanstep)
            yed=(yle+yri)/2
            if np.std(np.diff(xc))>1e-4*meanstep:
                linear_dimensions=0
        else:
            yed =np.arange(np.shape(data)[1]+1)-0.5
            ycpos=yed[:-1]+0.5


    shading='flat'
    import matplotlib.colors
    if qclim is not None:
        dts=dt[np.isfinite(dt)]
        cli=np.nanquantile(dts,qclim)
    elif cl is not None:
        cli=cl
    else:
        dts=dt[np.isfinite(dt)]
        cli=np.nanquantile(dts,[0.05,0.95])
    if log==1:
        if cli[0]==0:
            cli[0]=1e-2*cli[1]
        norm=matplotlib.colors.LogNorm(vmin=cli[0],vmax=cli[1])
    else:
        norm=matplotlib.colors.Normalize(vmin=cli[0],vmax=cli[1])
    if cmap is None:
        #cmap = cm.get_cmap('gist_stern')
        cmap=rofl.cmap()
    if cmap=='nw':
        cmap=rofl.cmap_nw()
        #cmap = cm.get_cmap('gist_stern')
    else:
        import matplotlib.cm as cm
        #cmap = cm.get_cmap('gist_stern')
        cmap = cm.get_cmap(cmap)
    if xticks is not None:
        plt.xticks(xticks)
    elif ticks and not linearize:
        plt.xticks(xc)
        plt.yticks(yc)
    elif ticks and linearize:
        plt.xticks(xcpos,xc)
        plt.yticks(ycpos,yc)
    if yticks is not None:
        plt.yticks(yticks)
        
    if linear_dimensions:
        ex=[np.min(xed),np.max(xed),np.min(yed),np.max(yed)]
        ex=[np.min(xed),np.max(xed),yed[-1],yed[0]]
        plt.imshow(dt,norm=norm,cmap=cmap,extent=ex,aspect=aspect)
    else:
        plt.pcolormesh(xed,yed,dt,shading=shading,norm=norm,cmap=cmap)
    if xl is not None:
        plt.xlabel(xl)
    if yl is not None:
        plt.ylabel(yl)
    plt.clim(cli)
    if colorbar:
        try:
            fig = plt.gcf()
            ax = plt.gca()
            fig.colorbar(ax.images[0], ax=ax)  # works for imshow()
        except Exception as E:
            print('no color bar...', E)
    if background is not None:
        bcol=background
        if bcol=='mud':
            bcol='#413824'
            #'xkcd:salmon'
        ax=plt.gca()
        ax.set_facecolor(bcol)
    if labels is not None:
        if labels=='values':
            #cl=plt.clim()
  #          cl = plt.gci().get_clim()
#            if log: thr=cl[0]*np.sqrt(cl[1]/cl[0])
 #           else:   thr=cl[0]+(cl[1]-cl[0])*0.25
            for i in np.arange(np.size(xc)):
                for j in np.arange(np.size(yc)):
                    val=data[i,j]
              #      if label_color=='auto':
               #         col='w'
                   #     if val>thr: col='k'
                    if 1: #label_color=='invert':
                        value=dt[j,i]
                        normalized_value = norm(value)
                        col = cmap(normalized_value)
                        intens=np.sum(col[0:3])/3
                        if intens<0.5: col='w'
                        else: col='k'
                    if val==0: col='r'
                    if np.isnan(val): col='r'
                    if not np.isfinite(val): col=[0.5,0.5,0.5]
                    if label_format[0]=='m':
                        prec=int(label_format[1])
                        stri=formatm(val,prec)
                    else:
                        stri=label_format.format(val)
                    plt.text(xc[i],yc[j],stri,color=col,size=label_size,zorder=30)





def formatm(i,precision=1):
    j=np.abs(i)
    if precision==0:
        if j >1e4:
            f='{:.0e}'
        elif j>100:
            f='{:.0f}'
        elif j>0.7:
            f='{:.0f}'
        elif j>0.07:
            f='{:.1f}'
        elif j>0.007:
            f='{:.2f}'
        elif j>0.0007:
            f='{:.3f}'
        else:
            f='{:.0e}'
    elif precision==1:
        if j >1e4:
            f='{:.0e}'
        elif j>100:
            f='{:.0f}'
        elif j>0.7:
            f='{:.1f}'
        elif j>0.07:
            f='{:.1f}'
        elif j>0.007:
            f='{:.2f}'
        elif j>0.0007:
            f='{:.3f}'
        else:
            f='{:.1e}'
    elif precision==2:
        if j >1e4:
            f='{:.1e}'
        elif j>100:
            f='{:.0f}'
        elif j>0.7:
            f='{:.2f}'
        elif j>0.07:
            f='{:.2f}'
        elif j>0.007:
            f='{:.3f}'
        elif j>0.0007:
            f='{:.4f}'
        else:
            f='{:.2e}'
    st=f.format(i)
    if 'e' in st and st[-2:-1]=='0': st=st[:-2]+st[-1:]
    if j==0:st='0'

    return st


def subprocess(command,cwd='',verbose=0,print_output=0):
    import subprocess as sss

    process = sss.Popen(command, shell=True, stdout=sss.PIPE,cwd=cwd)
    l(verbose,'##### Starting process '+command)
    keywords=['FLYCHK','error','exit','output','time step']
    if 1:
        li=0
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if print_output:
                print (line)
            else:
                li+=1
                printit=0
                for w in keywords:
                    if str(line).find(w)>=0:
                        printit=1
                        break
                if printit: print(str(line)[2:])
                text = "\r written {:.0f} output lines".format( li)
                
                sys.stdout.write(text)
                sys.stdout.flush()

    code=process.returncode
    l(verbose,'      Exit process with code {:}'.format(code))
    return code