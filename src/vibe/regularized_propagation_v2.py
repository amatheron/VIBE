# -*- coding: utf-8 -*-
"""
User defined functions for LightPipes for Python

"""
import numpy as _np
from LightPipes import Field, Intensity
from LightPipes.config import _USE_PYFFTW
import matplotlib.pyplot as plt


def Forvard_reg(Fin, f, z, subtract_defocus, deregularize_after=False,
                usepyFFTW = False):
    """
    Vychazi z Jardova odvozeni pro regularizovanou propagaci.

    Parameters
    ----------
    Fin : Field
        input field.
    f : float
        radius of curvature (= focal length of ideal lens) of the parabolic phase
        that is used to regularize the wavefront. 
    z : float 
        propagation distance
    subtract_defocus : bool
        If True substract a parabolic phase from Fin.
    usepyFFTW : bool, optional
        Whether to use pyFFTW for FFT

    Returns
    -------
    Field
        Propagated field.

    """
    _using_pyfftw = False # determined if loading is successful
    if usepyFFTW or _USE_PYFFTW:
        try:
            import pyfftw as _pyfftw
            from pyfftw.interfaces.numpy_fft import fft2 as _fft2
            from pyfftw.interfaces.numpy_fft import ifft2 as _ifft2
            _fftargs = {'planner_effort': 'FFTW_ESTIMATE',
                        'overwrite_input': True,
                        'threads': -1} #<0 means use multiprocessing.cpu_count()
            _using_pyfftw = True 
        except ImportError:
            #import warnings
            #warnings.warn(_WARNING)
            _WARNING =  '\n**************************** WARNING ***********************\n'\
            +'In the Forvard command you required FFT with the pyFFTW package, \n'\
            +'or  _USE_PYFFTW = True in your config.py file.\n'\
            +'However LightPipes cannot import pyFFTW because it is not installed.\n'\
            +'Falling back to numpy.fft.\n'\
            +'(Try to) install pyFFTW on your computer for faster performance.\n'\
            +'Enter at a terminal prompt: python -m pip install pyfftw.\n'\
            +'Or reinstall LightPipes with the option pyfftw\n'\
            +'Enter: python -m pip install lightpipes[pyfftw]\n\n'\
            +'*************************************************************'
            print(_WARNING)
    if not _using_pyfftw:
        from numpy.fft import fft2 as _fft2
        from numpy.fft import ifft2 as _ifft2
        _fftargs = {}
    
    if z==0:
        Fout = Field.copy(Fin)
        return Fout #return copy to avoid hidden reference
    if f==z:
        return LensFarfield(Fin, f) #copy is created in LensFarfield
    _2pi = 2*_np.pi
    legacy = False
    if legacy:
        _2pi = 2.*3.141592654 #make comparable to Cpp version by using exact same val
    Fout = Field.shallowcopy(Fin)
    N = Fout.N
    size = Fout.siz
    lam = Fout.lam
    dtype = Fin._dtype
    
    if _using_pyfftw:
        in_out = _pyfftw.zeros_aligned((N, N),dtype=dtype)
    else:
        in_out = _np.zeros((N, N),dtype=dtype)
    in_out[:,:] = Fin.field
    k = _2pi/lam
    if subtract_defocus:
        yy, xx = Fout.mgrid_cartesian
        fi = k*(xx**2+yy**2)/(2*f) #intentionally without minus sign
        in_out *= _np.exp(1j * fi)
    
    z_reg = z*f / (f-z) #regularized propagation distance
    kz = _2pi/lam*z
    cokz = _np.cos(kz)
    sikz = _np.sin(kz)
    
    # Sign pattern effectively equals an fftshift(), see Fresnel code
    iiN = _np.ones((N,),dtype=float)
    iiN[1::2] = -1 #alternating pattern +,-,+,-,+,-,...
    iiij = _np.outer(iiN, iiN)
    in_out *= iiij
    
    z1 = abs(z_reg)*lam/2
    No2 = int(N/2)

    #faster way to create Fresnel propagator
    SW = _np.arange(-No2, N-No2)/size
    SW *= SW
    SSW = SW.reshape((-1,1)) + SW #fill NxN shape like np.outer()
    Bus = z1 * SSW
    Ir = Bus.astype(int) #truncate, not round
    Abus = _2pi*(Ir-Bus) #clip to interval [-2pi, 0]
    Cab = _np.cos(Abus)
    Sab = _np.sin(Abus)
    CC = Cab + 1j * Sab #noticably faster than writing exp(1j*Abus)
    
    if z_reg >= 0.0:
        in_out = _fft2(in_out, **_fftargs)
        in_out *= CC
        in_out = _ifft2(in_out, **_fftargs)
    else:
        in_out = _ifft2(in_out, **_fftargs)
        CCB = CC.conjugate()
        in_out *= CCB
        in_out = _fft2(in_out, **_fftargs)
    
    Fout.siz = size * abs((f-z))/abs(f) #scale size of grid
    in_out *= (cokz + 1j* sikz) #rychlejsi nez _np.exp(1j*k*z)
    in_out *= iiij #/N**2 omitted since pyfftw already normalizes (numpy too)
    in_out *= abs(f)/abs((f-z)) #to normalize energy
    if deregularize_after:
        yy_out, xx_out = Fout.mgrid_cartesian
        fi = -k*(xx_out**2+yy_out**2) / (2*(f-z))
        in_out *= _np.exp(1j * fi)
    Fout.field = in_out
    Fout._IsGauss=False
    return Fout