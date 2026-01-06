#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------- VIBE (Vacuum Interaction Birefringence Explorer) code written by Aimé MATHERON, Felix Karbstein, Michal Smid and Pooyan Khademi ------------------
#----------------------------------------- Latest updated : December 2025. All rights reserved. --------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# =========================================================
# Standard library
# =========================================================
import os
import re
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict


# =========================================================
# Core scientific stack
# =========================================================
import numpy as np
import h5py
import yaml


from scipy import signal
from scipy.signal import fftconvolve
from scipy.ndimage import (
    map_coordinates,
    gaussian_filter1d,
    shift as imshift,
)
from scipy.constants import e, m_e, epsilon_0, hbar, c, h, pi
from scipy.special import j1, wofz
from scipy.interpolate import (
    RegularGridInterpolator,
    RectBivariateSpline,
)
from numpy.polynomial.hermite import hermgauss


# =========================================================
# Optics / propagation
# =========================================================
from LightPipes import Field
from LightPipes import *


# =========================================================
# Plotting / image I/O (headless-safe)
# =========================================================
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

from PIL import Image
from astropy.io import ascii
from skimage.transform import resize


# =========================================================
# Project-local (VIBE)
# =========================================================
import vibe.rossendorfer_farbenliste as rofl
import vibe.mmmUtils_v2 as mu
import vibe.regularized_propagation_v2 as rp
import vibe.wavefront_fitting as wft


# ------------- Backend selection (for local runs) --------
def _select_backend():
    os.environ.pop("MPLBACKEND", None)
    if os.environ.get("DISPLAY", "") == "":
        print("No display detected: using non-interactive 'Agg' backend.")
        matplotlib.use("Agg")
    else:
        print("Display detected: using 'TkAgg' backend.")
        matplotlib.use("TkAgg")


# --------------- Load the input file ---------------
def load_cfg(yaml_path: str) -> dict:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


# ------------ Create the list of optical elements from the input file ---------
def elements_from_cfg(cfg: dict) -> list:
    elems = []
    for name, obj in cfg.items():
        if name in ("Xbeam", "IRLaser", "simulation", "meta"):
            continue
        obj = dict(obj)
        obj["element_name"] = name
        elems.append([obj["position"], name, obj])
    elems.sort(key=lambda e: e[0])  # sort by z-position
    return elems


# =========================================================
# YAML related functions
# =========================================================


def yamlval(key,ip,default=0):
    """
    Retrieve a value from a dictionary with a default fallback.
    """
    if not key in ip.keys() :
        return default
    else:
        return ip[key]    



def run_from_yaml(yaml_path: str, N: int):
    """
    Run a complete VIBE simulation from a YAML configuration file.

    This function:
    - loads and parses the YAML input,
    - builds the ordered list of optical elements,
    - initializes global simulation parameters,
    - runs the main VIBE engine,
    - stores result pickles and figures in ``VIBE_outputs``.

    It is the main entry point when launching VIBE from the command line
    or from batch jobs.
    """
    cfg      = load_cfg(yaml_path)
    elements = elements_from_cfg(cfg)
    yamlname = Path(yaml_path).stem

    basepath = Path(__file__).resolve().parents[2]

    projectdir = basepath / "VIBE_outputs"
    projectdir.mkdir(parents=True, exist_ok=True)
    projectdir = str(projectdir)

    print(basepath)
    print(projectdir)

    input_params = build_input_params(cfg, N=N, projectdir=projectdir, filename=yamlname)

    # ---- Run the engine exactly as before ----
    out_params, trans, figs = main_VIBE(input_params, elements)

    # ---- Optional: keep Launch's extra res pickle (cfg + params) ----
    try:
        pickles_dir = Path(projectdir) / "pickles"
        pickles_dir.mkdir(parents=True, exist_ok=True)
        mu.dumpPickle([cfg, out_params], str(pickles_dir / f"{yamlname}_res"))
    except Exception as error_p:
        print(f"[warn] Could not write extra res pickle: {error_p}")

    print("Simulation finished.")

    # --- Optional cleanup with the function "cleanup_heavy_outputs" if the option "save_fig_and_pickle" is activated  ---
    if not input_params.get("save_fig_and_pickle", 1):
        print("[INFO] save_fig_and_pickle=0 → cleaning up heavy outputs …")
        cleanup_heavy_outputs(Path(projectdir), yamlname)
    else:
        print("[INFO] save_fig_and_pickle=1 → keeping figures and pickles.")

    return out_params, trans, figs





def newobject_from_yaml(name,ip):
    """
    Create an optical object dictionary from YAML-style input parameters.
    """    
    obj=newobject()
    for k in ip.keys():
        spl=k.split('_')
        if spl[0]!=name:continue
        val=ip[k]
        if mu.is_float(val):
            val=float(val)
        obj[spl[1]]=val
    return obj



def update_object_from_yaml(obj,name,ip):
    """
    Update an existing optical object dictionary using YAML-style parameters.
    """    
    for k in ip.keys():
        spl=k.split('_')
        if spl[0]!=name:continue
        val=ip[k]
        if mu.is_float(val):
            val=float(val)
        obj[spl[1]]=val
    return obj


def simparams2str(p):
    """
    Format a list of simulation parameters into a compact string identifier.

    The returned string is typically used for labeling, file naming, or logging
    purposes, with fixed-width numeric fields for consistent sorting.
    """
    paramsstr="{:} {:03.0f} {:03.0f} {:03.0f} {:03.0f} {:} {:03.0f} {:.2f} {:.2f} {:s} {:03.0f} {:02.0f}".format(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11])
    return paramsstr


def simstr2params(simstr):
    """
    Parse a simulation identifier string into its parameter components.
    """
    pars= simstr.split('_')
    return pars


def build_input_params(cfg: dict, *, N: int, projectdir: str, filename: str) -> dict:
    """
    Construct the unified input-parameter dictionary for a VIBE simulation.

    This function extracts and combines the relevant entries from the YAML
    configuration file (X-ray beam, IR laser, and simulation sections) into a
    single flat dictionary used internally by the VIBE engine. Default values
    are applied whenever a parameter is missing from the YAML file, ensuring
    robust execution with partially specified configurations.

    The returned dictionary gathers:
    - global identifiers and output paths,
    - X-ray beam and grid parameters,
    - plotting and diagnostic configuration,
    - flow and figure-export options,
    - X-ray pulse scaling parameters,
    - IR laser parameters used for vacuum-birefringence masking.

    Parameters
    ----------
    cfg : dict
        Parsed YAML configuration dictionary.
    N : int
        Number of transverse grid points used in the simulation.
    projectdir : str
        Path to the root output directory for the simulation (e.g. VIBE_outputs).
    filename : str
        Base name of the simulation, typically derived from the YAML file name.

    Returns
    -------
    dict
        Dictionary of simulation parameters consumed by `main_VIBE`.
    """    
    
    X  = cfg.get("Xbeam", {})
    IR = cfg.get("IRLaser", {})
    S  = cfg.get("simulation", {})

    return {
        # ---- identifiers & paths ----
        "N": int(N),
        "filename": filename,
        "projectdir": projectdir,

        # ---- core simulation inputs (X-ray & grid) ----
        "photon_energy": float(yamlval("photonenergy", X, 8766)),   # fallback 8766 eV
        "beamsize":      float(yamlval("size", X, 0.002)),          # fallback 2 mm
        "gauss_x_shift": float(yamlval("offset", X, 0.0)),
        "gauss_x_tilt":  float(yamlval("tilt", X, 0.0)),
        "propsize":      float(yamlval("propsize", S, 0.0)),
        "simulation_type": 0,

        # ---- intensity units & zoom ----
        "intensity_units": yamlval("intensity_units", S, "relative"),
        "Zoom_global":     float(yamlval("Zoom_global", S, 1)),

        # ---- plotting knobs used throughout main_VIBE ----
        "fig_rows": 4, "fig_cols": 5,
        "remove_ticks": 1,
        "fig_start": 3,
        "profiles_subfig": 1,
        "ax_apertures": None,

        # ---- optional flow/figs controls ----
        "figs_to_save":   yamlval("figs_to_save", S, []),
        "figs_to_export": yamlval("figs_to_export", S, []),
        "figs_log":       yamlval("figs_log", S, 1),
        "flow_auto_save": yamlval("flow_auto_save", S, 0),
        "flow_plot_gyax": yamlval("flow_plot_gyax", S, None),
        "flow_plot_clim": yamlval("flow_plot_clim", S, None),
        "profiles_xlim":  yamlval("profiles_xlim", S, [0, 200]),
        "intensity_ylim": yamlval("intensity_ylim", S, [1e-10, 2.0]),
        "apertures_ylim": yamlval("apertures_ylim", S, [1e-10, 2.0]),
        "edge_damping":        yamlval("edge_damping", S, None),
        "edge_damping_shape":  yamlval("edge_damping_shape", S, None),
        "method":              yamlval("method", S, "FFT"),
        "subfigure_size_px":   yamlval("subfigure_size_px", S, 300),
        "flow":                yamlval("flow", S, None),
        "save_fig_and_pickle": yamlval("save_fig_and_pickle", S, 1),

        "save_individual_figures": yamlval("save_individual_figures", S, 0),
        "individual_fig_log": yamlval("individual_fig_log", S, 1),


        # ---- X-ray pulse scaling ----
        "photons_total":   float(yamlval("photons_total", X, 2e11)),
        "X_FWHM_duration": float(yamlval("X_FWHM_duration", X, 25e-15)),

        # ---- IR block (VB mask @ TCC) ----
        "f_number":          float(yamlval("f_number", IR, 1.0)),
        "IR_focal_length":   float(yamlval("IR_focal_length", IR, 0.15)),
        "fraction_blocked":   float(yamlval("fraction_blocked", IR, 0.3)),
        "IR_Energy_J":       float(yamlval("IR_Energy_J", IR, 4.8)),
        "IR_FWHM_duration":  float(yamlval("IR_FWHM_duration", IR, 30e-15)),
        "IR_wavelength":     float(yamlval("IR_wavelength", IR, 800e-9)),
        "IR_x_offset_m":     float(yamlval("IR_x_offset_m", IR, 0.0)),
        "IR_y_offset_m":     float(yamlval("IR_y_offset_m", IR, 0.0)),
        "Timing_jitter":     float(yamlval("Timing_jitter", IR, 0.0)),
        "IR_FWHM_gaussian":  float(yamlval("IR_FWHM_gaussian", IR, 1.3e-6)),
        "IR_2Dmap":          yamlval("IR_2Dmap", IR,
                                     ["gaussian", "match_integral", None, None]),
    }


def sort_elements(elements, debug=0):
    """
    Sort optical elements by their longitudinal position.

    Elements are expected to be iterable containers whose first entry
    corresponds to their position along the propagation axis. The function
    returns a new list sorted in increasing order of this position.

    Parameters
    ----------
    elements : list
        List of optical elements, each containing a position as its first item.
    debug : bool or int, optional
        If non-zero, print the element names after sorting.

    Returns
    -------
    list
        Sorted list of optical elements.
    """

    # Sort elements by their z-position (first entry)
    elements_sorted = sorted(elements, key=lambda el: el[0])

    if debug:
        print("after sorting:")
        for el in elements_sorted:
            print(el[1])

    return elements_sorted



def cleanup_heavy_outputs(projectdir: Path, filename: str):
    """
    In the case where the option "save_fig_and_pickle" is deactivated in the yaml file,
    deletes heavy figure and pickle outputs after flow_plot has run.
    Keeps only the lightweight .npz summary produced by flow_plot.
    """
    import glob, os

    # Directories
    figs_dir     = Path(projectdir) / "figures"
    flow_dir     = Path(projectdir) / "flows"
    lens_dir     = Path(projectdir) / "Lens_diags"
    vb_dir       = Path(projectdir) / "VB_figures"
    pickles_dir  = Path(projectdir) / "pickles"

    # Patterns to remove
    patterns = [
        f"{figs_dir}/{filename}_main.jpg",
        f"{figs_dir}/{filename}_summary.jpg",
        f"{figs_dir}/{filename}_VB_parr.jpg",
        f"{figs_dir}/{filename}_VB_perp.jpg",
        f"{flow_dir}/{filename}_flowplot_VB_parr.jpg",
        f"{flow_dir}/{filename}_flowplot_VB_perp.jpg",
        f"{lens_dir}/{filename}*",       # remove all lens diagnostics with this stem
        f"{vb_dir}/{filename}*",         # remove all VB diagnostics with this stem
        f"{pickles_dir}/{filename}_figs.pickle",
        f"{pickles_dir}/{filename}_res.pickle",
    ]

    removed = []
    for pat in patterns:
        for file in glob.glob(pat):
            try:
                os.remove(file)
                removed.append(file)
            except Exception:
                pass

    if removed:
        print(f"[CLEANUP] Removed {len(removed)} heavy files for {filename}:")
        for r in removed:
            print("   -", r)
    else:
        print(f"[CLEANUP] No files found to remove for {filename}.")



# =========================================================
# Defining the field bundle
# =========================================================

@dataclass
class FieldBundle:
    """
    Holds one or more LightPipes `Field` objects that propagate together.
    """
    fields: Dict[str, Field]   # Channel name {"main": F}, {"VB_perp": F} or {"VB_parr": F}
    z_pos: float               # current longitudinal position [m]
    reg: Dict                  # options for regularised propagation






def propagate_bundle(bundle: 'FieldBundle',
                     dz: float,
                     method: str = 'FFT') -> 'FieldBundle':
    """
    Move *all* fields contained in *bundle* forward by *dz* using the same
    rules that `main_VIBE` loop currently applies to `F`.
    """
    if dz == 0:
        return bundle  # nothing to do

    for key, F in bundle.fields.items():
        if not bundle.reg.get("regularized_propagation", False):
            if method.lower() == 'fresnel':
                F = Fresnel(dz, F)
            elif method.upper() == 'FFT':
                F = Forvard(dz, F)
            else:
                raise ValueError(f"Unknown propagation method: {method}")
        else:
            F = rp.Forvard_reg(
                    F,
                    bundle.reg.get("reg_parabola_focus"),
                    dz,
                    False
                 )
     
        bundle.fields[key] = F   # write back the propagated field

    bundle.z_pos += dz
    return bundle



# =========================================================
# Physics utility functions
# =========================================================

def elem2Z(elem):
    """
    Map an element or material label to an effective atomic number Z.

    For compound or polymer-like materials, the returned value corresponds
    to an approximate effective atomic number.
    """
    if elem=='Be':   return 4
    if elem=='C':   return 6
    if elem=='CH':   return 5
    if elem=='CH6':   return 4.5
    if elem=='polymer':   return 4.15
    if elem=='O':   return 8
    if elem=='Al': return 13
    if elem=='SiO2': return 10
    if elem=='Si': return 14
    if elem=='Ti':  return 22
    if elem=='Cr':  return 24
    if elem=='Fe':   return 26
    if elem=='Ni':   return 28
    if elem=='Cu':   return 29
    if elem=='Zn':   return 30
    if elem=='Ge':   return 32
    if elem=='Zr':   return 40
    if elem=='Ag':   return 47
    if elem=='W':  return 74
    if elem=='Pt':   return 78
    if elem=='Au':   return 79
    if elem=='Pb':   return 82

    assert 1, "element not found"


_index_cache = {}     # Index module cache, to be defined before the function get_index.

def get_index(elem, E, table_dir=None):
    """
    Return delta and beta by interpolating the Henke data for given element and energy.
    If table_dir is not specified, it is assumed to be next to VIBE.py
    """
    global _index_cache

    # Special case for Hafnium
    if elem == 'Hf':
        return 3.2887e-6 , 1.988e-5
    
    if elem=='W':  
        return 2.85704482E-06 , 3.86332977E-05
    
    if elem=='Au':   #Gold (79)
        return 3.55916109E-06 , 3.95017742E-05
    
    # Locate optical_constants folder relative to the file location of VIBE.py
    if table_dir is None:
        table_dir = Path(__file__).parent / "optical_constants"

    filepath = table_dir / f"{elem}.txt"

    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. Set table_dir explicitly if needed."
        )

    if elem not in _index_cache:
        data = np.genfromtxt(filepath, comments='#', skip_header=1)
        if data.shape[1] < 3:
            raise ValueError(f"Expected ≥3 columns (E, delta, beta) in {filepath}")
        energies = data[:, 0]
        delta = data[:, 1]
        beta = data[:, 2]
        _index_cache[elem] = (energies, delta, beta)

    energies, delta_vals, beta_vals = _index_cache[elem]
    delta = np.interp(E, energies, delta_vals)
    beta  = np.interp(E, energies, beta_vals)
    return beta, delta



def newobject(shape='',size=0.0*mm,rot=0,smooth=0,invert=0,offset=0,thickness=0,elem='',profile=[],typ='aperture',num=0,defect=''):
    """
    Create and return a dictionary describing an optical object or element.

    The returned dictionary stores geometric, material, and configuration
    parameters used to define apertures, lenses, or other beamline elements.
    """    
    obj={}
    obj['shape']=shape
    obj['size']=size
    obj['smooth']=smooth
    obj['rot']=rot
    obj['invert']=invert
    obj['offset']=offset
    obj['thickness']=thickness
    obj['elem']=elem
    obj['profile']=profile
    obj['type']=typ
    obj['num']=num
    obj['defect']=defect

    return obj



# =========================================================
# Phase and transmission objects, apertures definitions
# =========================================================


def thickness_to_phase_and_transmission(E_eV, delta, beta):
    """
    Compute phase shift and absorption coefficient per meter thickness
    for given photon energy and refractive index components.

    Parameters
    ----------
    E_eV : float
        Photon energy in eV
    delta : float
        Refractive index decrement (n = 1 - delta + i beta)
    beta : float
        Absorption index

    Returns
    -------
    thickness_to_phase : float
        Phase shift per meter thickness [rad/m]
    thickness_to_transmission : float
        Absorption coefficient per meter thickness [1/m]
        (so that transmission = exp(-thickness_to_transmission * thickness))
    """
    # Convert energy in eV to wavelength in meters
    E_J = E_eV * e
    lam = h * c / E_J

    # Phase shift per meter
    thickness_to_phase = -(2 * pi * delta) / lam

    # Transmission decay coefficient per meter
    k_transmission = (4 * pi * beta) / lam

    return thickness_to_phase, k_transmission




def parabolic_lens_profile(xax, r, r0, minr0=0, plot=0):
    """
    Compute a 1D parabolic lens thickness profile with optional truncation.

    The lens thickness follows a parabolic profile within a clear aperture
    of radius r0 and is capped beyond this radius to mimic a finite lens
    thickness. An optional inner truncation (minr0) can be applied to remove
    a central thickness offset. The function returns the final 1D thickness
    profile evaluated along the transverse coordinate xax.

    Parameters
    ----------
    xax : ndarray
        Transverse coordinate array.
    r : float
        Radius of curvature of the parabolic lens.
    r0 : float
        Maximum aperture radius defining the parabolic region.
    minr0 : float, optional
        Inner radius used to subtract a minimal thickness offset (default: 0).
    plot : bool, optional
        If True, plot the parabolic profile, circular reference, and final lens
        thickness (default: False).

    Returns
    -------
    ndarray
        1D array of lens thickness values.
    """
    
    x = xax

    # Ideal parabolic thickness profile: t(x) = x^2 / (2R)
    parabolic = x**2 / (2 * r)

    # Reference circular profile (for comparison / diagnostics)
    circular = r - np.sqrt(np.maximum(r**2 - x**2, 0.0))

    # Truncate thickness outside the clear aperture r0
    max_thickness = np.max(parabolic[np.abs(x) < r0])
    thickness = np.minimum(parabolic, max_thickness)

    # Optional removal of a central thickness offset
    if minr0 > 0:
        min_thickness = np.max(parabolic[np.abs(x) < minr0])
        thickness = np.maximum(thickness - min_thickness, 0.0)

    # Optional diagnostic plots
    if plot:
        mu.figure()
        plt.plot(x, parabolic, label="Parabolic profile")
        plt.plot(x, circular, label="Circular profile")
        plt.plot(x, thickness, label="Final lens profile")
        plt.ylim(0, 2 * max_thickness)
        plt.xlabel("radius [mm]")
        plt.ylabel("thickness [mm]")
        plt.legend()
        mu.figure()

    return thickness




def do_phaseplate(el_dict, params, debug=0):
    """
    Generate the phase-shift map associated with a phase plate element.

    This function builds a 2D thickness map corresponding to a phase plate
    defect (e.g. Seiboth or Celestre profiles), converts it into a phase-shift
    map using the material refractive index at the photon energy, and returns
    the resulting phase modulation. The phase defect can be scaled by the
    number of lenses specified in the element dictionary.

    Parameters
    ----------
    el_dict : dict
        Element definition dictionary. Must have type 'phaseplate' and may
        specify a defect profile and number of lenses.
    params : dict
        Global simulation parameters containing grid, material, and energy
        information.
    debug : int, optional
        Enable diagnostic plots if non-zero (default: 0).

    Returns
    -------
    ndarray
        2D array containing the phase-shift map [rad].
    """

    from pathlib import Path

    assert el_dict["type"] == "phaseplate"

    # --------------------------------------------------
    # Resolve paths (cluster / local compatible)
    # --------------------------------------------------
    basepath = Path(params["projectdir"]).parent

    # --------------------------------------------------
    # Extract parameters
    # --------------------------------------------------
    defect = el_dict.get("defect", "")
    num_lenses = el_dict.get("num", 1)

    E = params["photon_energy"]
    N = params["N"]
    pxsize = params["pxsize"]

    # --------------------------------------------------
    # Build transverse grid
    # --------------------------------------------------
    N2 = N // 2
    axis = np.arange(-N2, N2) * pxsize
    xm, ym = np.meshgrid(axis, axis)
    r = np.sqrt(xm**2 + ym**2)

    thickness = np.zeros((N, N))

    # --------------------------------------------------
    # Seiboth phase defect (radial profile from ASCII)
    # --------------------------------------------------
    if "seiboth" in defect:
        fia = ascii.read(basepath / "Seiboth_Fig4")
        r_um = fia["col1"]
        deformation_um = fia["col2"]

        img = np.zeros((N, N))
        for ix in range(N):
            for iy in range(N):
                rh_um = r[ix, iy] / um
                img[ix, iy] = np.interp(rh_um, r_um, deformation_um)

        thickness += img * um

        if debug:
            mu.figure()
            plt.plot(r_um, deformation_um)
            plt.xlabel("radial position [µm]")
            plt.ylabel("deformation [µm]")

    # --------------------------------------------------
    # Celestre phase defect (image-based map)
    # --------------------------------------------------
    if "celestre" in defect:
        image = Image.open(basepath / "Celestre_Fig8.png")
        image = image.resize((N, N))

        im = np.asarray(image)[:, :, 0]
        im = im / 255 * 24     # map figure values to µm
        im /= 11               # normalize per lens

        thickness += im * um

        if debug:
            mu.figure()
            plt.imshow(im)
            plt.colorbar(label="Thickness [µm]")

    # --------------------------------------------------
    # Scale by number of lenses
    # --------------------------------------------------
    thickness *= num_lenses

    # --------------------------------------------------
    # Convert thickness → phase shift
    # --------------------------------------------------
    elem = params.get("lens_material", "Be")
    beta, delta = get_index(elem, E)

    thickness_to_phase, _ = thickness_to_phase_and_transmission(E, delta, beta)
    phaseshift_map = thickness * thickness_to_phase

    # --------------------------------------------------
    # Optional diagnostics
    # --------------------------------------------------
    if debug:
        extent = [-N2 * pxsize / um, N2 * pxsize / um,
                  -N2 * pxsize / um, N2 * pxsize / um]

        mu.figure(10, 5)
        ax = plt.subplot(121)
        ax.set_facecolor("black")
        plt.title("Thickness [µm]")
        plt.imshow(thickness / um, extent=extent)
        plt.colorbar()

        ax = plt.subplot(122)
        ax.set_facecolor("black")
        plt.title("Phase shift [rad]")
        plt.imshow(phaseshift_map, extent=extent, cmap=rofl.cmap())
        plt.colorbar()

    return phaseshift_map



def make_sphere(radius, pxsize):
    """
    Generate a 2D thickness map of a spherical object.

    The returned array represents the optical thickness of a sphere
    of given radius, sampled on a square Cartesian grid with spacing
    pxsize. Outside the sphere, the thickness is set to zero.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    pxsize : float
        Transverse grid spacing.

    Returns
    -------
    ndarray
        2D array containing the spherical thickness profile.
    """

    # Grid size covering the full diameter
    Ns = int(np.ceil(2 * radius / pxsize))
    Ns2 = Ns // 2

    axis = np.arange(-Ns2, Ns2) * pxsize
    x, y = np.meshgrid(axis, axis)

    # Radial coordinate squared
    r2 = x**2 + y**2

    # Spherical thickness: 2 * sqrt(R^2 - r^2)
    thickness = np.zeros((Ns, Ns))
    inside = r2 <= radius**2
    thickness[inside] = 2 * np.sqrt(radius**2 - r2[inside])

    # Optional diagnostic plot
    if 0:
        extent = [-Ns2 * pxsize / um, Ns2 * pxsize / um,
                  -Ns2 * pxsize / um, Ns2 * pxsize / um]
        plt.imshow(thickness / um, extent=extent, cmap=rofl.cmap())
        plt.colorbar(label="Thickness [µm]")

    return thickness



def add_sphere(radius, xr, yr, img, pxsize, positive):
    """
    Add a spherical thickness profile to an existing 2D image.

    A sphere of given radius is inserted into the image at the position
    (xr, yr). Depending on the `positive` flag, the spherical profile is
    either added in quadrature (positive object) or subtracted (negative
    object). Outside the sphere, the image remains unchanged.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    xr, yr : float
        Transverse position of the sphere center.
    img : ndarray
        2D image to be modified in place.
    pxsize : float
        Transverse grid spacing.
    positive : bool
        If True, add the sphere thickness in quadrature.
        If False, subtract the sphere thickness.

    Returns
    -------
    ndarray
        Updated image including the spherical contribution.
    """

    # Size of the local grid covering the sphere
    size = int(np.ceil(2 * radius / pxsize))
    if size % 2 == 1:
        size -= 1

    # Pixel coordinates of the insertion point
    ix = int(xr / pxsize)
    iy = int(yr / pxsize)

    # Skip if reference point is empty or already too thick
    ref_value = img[ix + size // 2, iy + size // 2]
    if ref_value == 0 or ref_value >= 4 * radius:
        return img

    # Extract local region and build spherical profile
    local_img = img[ix:ix + size, iy:iy + size]
    sphere = make_sphere(radius, pxsize)

    # Combine profiles
    if positive:
        new_img = np.sqrt(local_img**2 + sphere**2)
    else:
        new_img = local_img - sphere
        new_img[new_img < 0] = 0

    # Insert back into the global image
    img[ix:ix + size, iy:iy + size] = new_img
    return img





def do_edge_damping_aperture(params):
    """
    Build an aperture transmission map with smooth edge damping.

    This function generates a 2D transmission array that smoothly attenuates
    the field near the grid boundaries to suppress edge effects. The damping
    can be applied with either a square or circular geometry and supports
    sinusoidal or explicit pixel-wise damping profiles.
    """
    N=params['N']
    edge_damping_shape=yamlval('edge_damping_shape',params,'square')
    trans=np.zeros([N,N])+1
    edge_damping_pixels=params['edge_damping']
    debug=0
    if np.size(edge_damping_pixels)==1: #doing sine damping #first number is fraction of N where the damping starts
        N_edge=int(N*edge_damping_pixels[0])
        x=np.arange(N_edge)/N_edge*(np.pi/2)
        y=np.sin(x)
        if edge_damping_shape=='square':
            for ri,mult in enumerate(y):
                trans[ri,:]*=mult
                trans[:,ri]*=mult
                trans[-1-ri,:]*=mult
                trans[:,-1-ri]*=mult

        if edge_damping_shape=='circular':
            mu.figure()
            N_through=(N/2)-N_edge
            rax=np.arange(N*0.8)
            prof=rax*0+0.5
            prof[rax<N_through]=1
            prof[rax>=N/2]=0
            xm=np.arange(N)
            prof2=prof*1.
            prof2[(rax>=N_through)*(rax<N/2)]=np.flip(y)

            if debug:
                mu.figure()
                x2=N_through+np.flip(np.arange(N_edge))
                plt.plot(rax,prof2,lw=3,alpha=0.5)

            N2=int(N/2)
            Na=(np.arange(N)-N2)*1
            xm,ym=np.meshgrid(Na,Na)
            r=((xm**2)+(ym**2))**0.5

            if debug:
                mu.figure()
                plt.imshow(r)
                plt.colorbar()
            for xi,x in enumerate(xm):
                for yi,y in enumerate(xm):
                    val=np.interp(r[xi,yi],rax,prof2)
                    trans[xi,yi]=val

    else: #doing silly pixel damping
        for ri,mult in enumerate(edge_damping_pixels):
            trans[ri,:]*=mult
            trans[:,ri]*=mult
            trans[-1-ri,:]*=mult
            trans[:,-1-ri]*=mult
  #  print(N_edge)
    if debug:
        mu.figure()
        plt.imshow(trans)
        plt.colorbar()
        plt.title('damping aperture')
    return trans




def get_aperture_transmission_map(pars, params={}, debug=0):
    """
    Generate a 2D aperture transmission map from geometric parameters.

    The aperture shape can be square, rectangular, circular, wire-like,
    or Gaussian (including super-Gaussian profiles). The returned map
    defines the spatial transmission applied to the optical field.
    """
    typ = pars['shape']
    pxsize = params['pxsize']  # pixel size in meters
    N = params['N']
    N2 = int(N / 2)

    # 2D coordinate grid centered at 0
    Na = (np.arange(N) - N2) * pxsize
    xm, ym = np.meshgrid(Na, Na)

    # Default map is fully transmissive
    trmap = np.ones((N, N))

    if typ == 'square':
        hs = pars['size'] / 2  # half side length
        sel = (np.abs(xm) <= hs) & (np.abs(ym) <= hs)
        trmap[sel] = 0

    elif typ == 'rectangle':
        hs = pars['size'] / 2
        vs = pars['sizevert'] / 2
        sel = (np.abs(xm) <= hs) & (np.abs(ym) <= vs)
        trmap[sel] = 0

    elif typ == 'wire':
        hs = pars['size'] / 2
        sel = (np.abs(xm) <= hs)
        trmap[sel] = 0

    elif typ == 'circle':
        r = np.sqrt(xm**2 + ym**2)
        rad = pars['size'] / 2
        trmap[r < rad] = 0

    elif typ == 'gaussian':
        r2 = xm**2 + ym**2
        fwhm = float(pars['size'])      # 'size' is FWHM
        P = float(pars.get('power', 2)) # order of super-Gaussian
        sigma = fwhm / (2 * np.sqrt(2) * (np.log(2))**(1 / (2 * P))) # Wikipedia definition of Super Gaussian profile
        trmap = np.exp( - ( (r2 / (2 * sigma**2)) ** P ) )
    else:
        raise ValueError(f"Unknown aperture shape: {typ}")
    # Invert transmission if needed
    if yamlval('invert', pars):
        trmap = 1 - trmap

    return trmap



def get_aperture_thickness_map(pars,params=[],debug=0):
    """
    Generate a 2D thickness map for an aperture or optical element.

    The thickness profile is constructed from geometric parameters and may
    include circular, serrated, parabolic, wire-like, or custom shapes, as
    well as optional defects or periodic modulations.
    """
    typ = pars['shape']
    pxsize = params['pxsize']
    N = params['N']
    N2 = int(N/2)

    Na=(np.arange(N)-N2)*pxsize
    thicknessmap=np.zeros([N,N])+1  #that mean default is 1 m thick. To be updated.

    xm,ym = np.meshgrid(Na,Na)

    if typ=='circle':
        r = ((xm**2)+(ym**2))**0.5
        rad = pars['size']/2
        thicknessmap = thicknessmap*0
        thicknessmap[r<rad] = pars['thickness']
        if yamlval('invert',pars):
            maxi = np.max(thicknessmap)
            thicknessmap = maxi-thicknessmap

    if typ == "serrated_circle":

        R0 = pars['size'] / 2         # base radius
        amplitude = float(yamlval('defect_amplitude', pars))
        n_teeth = int(yamlval('defect_n_teeth', pars))  # number of serration periods

        # ----- coordinates ----
        theta = np.arctan2(ym, xm)          # angle in radians
        r = np.sqrt(xm**2 + ym**2)

        # build angular modulation
        defect_type = yamlval('defect_type', pars, default="triangle")

        if defect_type == "sine":
            dR = amplitude * np.sin(n_teeth * theta)

        elif defect_type == "sawtooth":
            dR = amplitude * signal.sawtooth(n_teeth * theta)

        elif defect_type == "triangle":
            dR = amplitude * signal.sawtooth(n_teeth * theta, width=0.5)

        # final effective radius
        R = R0 + dR

        # build thickness mask
        thicknessmap = np.zeros_like(thicknessmap)
        thicknessmap[r < R] = pars['thickness']

    if typ in ['parabolic_lens','streichlens']:  #realistic 2D-depth maps
        if typ in ['parabolic_lens','streichlens']:
            r=((xm**2)+(ym**2))**0.5
            rax=np.arange(0,2*N2)*pxsize
            r0=pars['size']/2
            roc=pars['roc']
            prof=parabolic_lens_profile(rax,roc,r0,pars['minr0'],plot=0)
            for xi,x in enumerate(xm):
                for yi,y in enumerate(ym):
                    val=np.interp(r[xi,yi],rax,prof)
                    thicknessmap[xi,yi]=val
            if pars['double_sided']:
                thicknessmap=thicknessmap*2
            thicknessmap=thicknessmap*pars['num_lenses']

    if typ=='streichlens':

        half_gap=pars['gap_size']/2
        sel=np.abs(xm)<half_gap
        if pars['gap_fill']=='empty':
            thicknessmap[sel]=0
        if pars['gap_fill']=='flat':
            iedge=np.argmin(np.abs(Na-half_gap))
            edgeprof=thicknessmap[iedge,:]
            for i,x in enumerate(Na):
                if np.abs(x)<=half_gap:
                    thicknessmap[i,:]=edgeprof
        if pars['gap_fill']=='blade1':
            iedge=np.argmin(np.abs(Na+half_gap))
            iedge2=np.argmin(np.abs(Na-half_gap))
            edgeprof=thicknessmap[iedge,:]
            for i,x in enumerate(Na):
                if i==250:
                    print('asdf')
                horprof=thicknessmap[:,i]
                x1=Na[iedge]
                y1=horprof[iedge]
                x3=Na[iedge2]
                x2=0
                y2=horprof[0]
                blade=np.interp(Na,[x1,x2,x3],[y1,y2,y1])
                sel=blade>horprof
                horprof[sel]=blade[sel]
                thicknessmap[:,i]=horprof
    wls=['realwire','trapez','tent','customwire','pooyan','invpoo','invpar','par','wireslit','linearslit','wire_grating','wireslitup','wireslitdown']

    if typ in wls :
        wireprof = get_wire_like_profile(pars,params,debug) #get the 1D transmission profile

        wireprof[np.isnan(wireprof)] = 0
        if yamlval('smooth',pars)!=0:
            smpx=pars['smooth']/pxsize
            wireprof=mu.convolve_gauss(wireprof,smpx,1)
            ee=int(smpx*2)
            wireprof[0:ee]=wireprof[ee+1]
            wireprof[-ee:]=wireprof[-(ee+1)]

        if yamlval('invert',pars):
            maxi=np.max(wireprof)
            wireprof=maxi-wireprof

        ones = np.ones(N)
        orientate = pars.get('orientate_horizontal', 0)

        if orientate == 1:
            thicknessmap = np.outer(ones, wireprof)  # Horizontal structure (profile along y)
        else:
            thicknessmap = np.outer(wireprof, ones)  # Vertical structure (profile along x)

    defect_type=yamlval('defect_type',pars)

    # ---------------- DEFECT / SERRATION WAVEFORMS --------------------
    if defect_type in ['sine', 'sawtooth', 'triangle', 'real_triangle'] and typ != "serrated_circle":

        wavelength = float(yamlval('defect_lambda', pars))
        amplitude  = float(yamlval('defect_amplitude', pars))

        # Coordinate (in meters then in radians)
        x = Na * 2 * np.pi / wavelength

        # --- 1) Base waveforms ---
        if defect_type == 'sine':
            base = np.sin(x) * amplitude

        elif defect_type == 'sawtooth':
            base = signal.sawtooth(x) * amplitude

        elif defect_type == 'triangle':
            base = signal.sawtooth(x, width=0.5) * amplitude

        elif defect_type == 'real_triangle':

            # Step 1: ideal triangle wave
            tri = signal.sawtooth(x, width=0.5) * amplitude

            # Step 2: round the peaks
            radius = float(yamlval('defect_radius', pars))   # meters
            if radius <= 0:
                base = tri
            else:
                # radius in pixels
                r_px = abs(radius / pxsize)
                r_px = max(1.0, r_px)

                # disk kernel for smoothing
                k = int(np.ceil(r_px))
                X = np.arange(-k, k + 1)
                ker = np.zeros_like(X, dtype=float)

                # circular cap (1D slice of a disk)
                for i, xx in enumerate(X):
                    if abs(xx) <= r_px:
                        ker[i] = np.sqrt(r_px**2 - xx**2)
                    else:
                        ker[i] = 0

                # normalize (so amplitude stays consistent)
                ker = ker / np.sum(ker)

                # filtered triangle wave
                base = np.convolve(tri, ker, mode="same")

        # convert offsets from meters → integer pixel shifts
        offsets_px = np.round(base / pxsize).astype(int)

        # --- Apply defect shift line by line ---
        tmp = np.zeros_like(thicknessmap)
        for yi in range(N):
            shifted_column = np.roll(thicknessmap[:, yi], offsets_px[yi])
            tmp[:, yi] = shifted_column

        thicknessmap = tmp.copy()

    return thicknessmap



def get_wire_like_profile(pars,params,debug):
    """
    Construct a 1D thickness profile for wire-, slit-, or blade-like obstacles.

    This function generates a transverse thickness profile along one axis
    based on geometric parameters specified in `pars`. It supports a wide
    range of obstacle shapes, including trapezoidal, tent, parabolic,
    Pooyan-type, wire slits, gratings, and custom user-defined profiles.
    The resulting 1D profile is typically expanded into a 2D thickness
    map by replication along the orthogonal direction.

    Parameters
    ----------
    pars : dict
        Dictionary describing the obstacle geometry and material properties.
    params : dict
        Global simulation parameters (grid size, pixel spacing, photon energy).
    debug : bool or int
        Enable diagnostic output or plots if non-zero.

    Returns
    -------
    ndarray
        One-dimensional array representing the obstacle thickness profile.
    """
    r = yamlval('size',pars,0)/2
    off = float(yamlval('offset',pars,0))
    elem = pars['elem']
    pxsize = params['pxsize']
    N = params['N']
    N2 = int(N/2)

    Na = (np.arange(N)-N2)*pxsize
    x = Na-off
    typ = pars['shape']

    if typ.find('trapez')==0:
        thickness=pars['thickness']
        edge=pars['edge']
        g=r/edge
        wireprof=g*r-g*np.abs(x)
        wireprof=thickness/edge*(r-np.abs(x))
        wireprof[wireprof<0]=0
        wireprof[wireprof>thickness]=thickness
    elif typ.find('tent')==0:
        thickness=pars['thickness']
        wireprof=thickness*(1-np.abs(x)/r)
        wireprof[wireprof<0]=0
    elif typ.find('customwire')==0:
        wireprof=pars['profile']
    elif typ.find('invpoo')==0:
        l1=pars['l1']
        d1=pars['size']
        l2=pars['l2']

        d2=pars['size']-pars['d']*2

        d=(d1-d2)/2
        two_p=(l1**2+d**2)**0.5
        p=two_p/2
        alpha=np.arctan(l1/d)
        r=p/np.cos(alpha)
        wireprof=x*0
        assert d<=l1, 'The Pooyan shape does not work like this: d>l1 (d={:.0f},l1={:.0f})'.format(d/um,l1/um)
        circ_cen=0-d1/2+r
        wireprof[np.abs(x)>=d2/2]=l2+l1

        ss=np.logical_and(x<(-d2/2),x>(-d1/2))
        circ=(r**2-(x-circ_cen)**2)**0.5
        wireprof[ss]=l1-circ[ss]

        ss=np.logical_and(x>(d2/2),x<(d1/2))
        circ=(r**2-(x+circ_cen)**2)**0.5
        wireprof[ss]=l1-circ[ss]

    elif typ == 'wireslit':
        r = pars['r']
        wireprof = x*0
        halfsize = pars['size']/2
        off = halfsize+r
        circ1 = (r**2-(Na-off)**2)**0.5*2
        sel1 = (x>=halfsize) * (x<=off)
        wireprof[sel1] = circ1[sel1]
        circ2 = (r**2-(Na+off)**2)**0.5*2
        sel2 = (x<=halfsize) * (x>=-off) #To be updated. Wrong condition.
        wireprof[sel2] = circ2[sel2]
        wireprof[np.abs(x)>off] = 2*r
        wireprof[np.abs(x)<halfsize] = 0

    elif typ == 'wireslitup':
        r = pars['r']
        wireprof = x * 0
        halfsize = pars['size'] / 2
        off = halfsize + r
        circ1 = (r**2-(Na-off)**2)**0.5*2
        sel1 = (x>=halfsize) * (x<=off)
        wireprof[sel1] = circ1[sel1]
        wireprof[x > off] = 2 * r
        wireprof[x < halfsize] = 0

    elif typ == 'wireslitdown':
        r = pars['r']
        wireprof = x * 0
        halfsize = pars['size'] / 2
        off = halfsize + r
        circ2 = (r**2-(Na+off)**2)**0.5*2
        sel2 = (x<=halfsize) * (x>=-off) 
        wireprof[sel2] = circ2[sel2]
        wireprof[x < off] = 2 * r
        wireprof[x > halfsize] = 0

    elif typ.find('invpar')==0:
        l=pars['l']
        halfsize=pars['size']/2
        n=pars['n']
        d=pars['d']
        a=l/(d**n)
        wireprof=x*0
        wireprof[np.abs(x)>=halfsize]=l

        par=a*np.abs(x-(halfsize-d))**n
        ss=x>=halfsize-d
        wireprof[ss]=par[ss]

        par=a*np.abs(x+(halfsize-d))**n
        ss=x<=-halfsize+d
        wireprof[ss]=par[ss]

        wireprof[wireprof>l]=l

    elif typ.find('linearslit')==0:
        l=pars['l']
        d=pars['d']
        halfsize=pars['size']/2+d
        a=l/d
        angle=np.arctan(a)/np.pi*180
        print('  Angle of the {:} slit blade is {:.0f}˚'.format(pars['elem'],angle))
        wireprof=x*0
        wireprof[np.abs(x)>=halfsize]=l

        par=a*np.abs(x-(halfsize-d))
        ss=x>=halfsize-d
        wireprof[ss]=par[ss]

        par=a*np.abs(x+(halfsize-d))
        ss=x<=-halfsize+d
        wireprof[ss]=par[ss]

        wireprof[wireprof>l]=l
        if yamlval('thicksize',pars)>0:
            ss=np.abs(x)>=pars['thicksize']
            wireprof[ss]=pars['thickthickness']

    elif typ.find('par')==0:
        l=pars['l']
        halfsize=yamlval('d2',pars,0)/2
        n=pars['n']
        d=pars['d']
        a=l/(d**n)
        size=yamlval('size',pars,-1)
        if size>0:  #Estimating the d2 from effecitve size;
            par=a*np.abs(x-d)**n
            beta, delta  = get_index(elem,params['photon_energy'])
            thickness_to_phaseshift,k = thickness_to_phase_and_transmission(params['photon_energy'], delta, beta)
            par_trans=np.exp(-k*par) #transmission
            sel=(par_trans>np.exp(-1))*(x>0)
            edgex=np.min(x[sel])
            halfsize=size/2-edgex

        wireprof=x*0
        wireprof[np.abs(x)<=halfsize]=l

        if 1:
            par=a*np.abs(x-(halfsize+d))**n
            print('Parabolic obstacle parameter a={:.2f} mm-1'.format(a*1e-6))
            if 0:
                print(x)
                print(par)
                mu.figure()
                plt.plot(x*1e6,par*1e6)
                mu.figure()
            ss=x>=halfsize-d
            wireprof[ss]=par[ss]

            par=a*np.abs(x+(halfsize+d))**n
            ss=x<=-halfsize+d
            ss=x<=0
            wireprof[ss]=par[ss]
            wireprof[wireprof>l]=l
            wireprof[np.abs(x)>halfsize+d]=0

        if yamlval('edge-r',pars)>0:
                print('doing the edge')

                r=pars['edge-r']
                off=np.abs(x[np.argmin(np.abs(wireprof-2*r))])
                print(off)
                circ1=(r**2-(Na-off)**2)**0.5*2
                sel1=(x>=off) * (x<=off+r)
                wireprof[sel1]=circ1[sel1]
                circ2=(r**2-(Na+off)**2)**0.5*2
                sel2=(x<=-off) * (x>=-off-r)
                wireprof[sel2]=circ2[sel2]
                wireprof[np.abs(x)>off+r]=0

    elif typ.find('pooyan')==0:
        l1=pars['l1']
        d1=pars['d1']
        l2=pars['l2']
        d2=pars['d2']
        d=(d1-d2)/2
        two_p=(l1**2+d**2)**0.5
        p=two_p/2
        alpha=np.arctan(l1/d)
        r=p/np.cos(alpha)
        print('Pooyans shape r is {:.0f} μm'.format(r/um))
        wireprof=x*0
        assert d<=l1, 'The Pooyan shape does not work like this: d>l1 (d={:.0f},l1={:.0f})'.format(d/um,l1/um)
        circ_cen=0-d2/2-r
        circ=(r**2-(x-circ_cen)**2)**0.5
        wireprof[np.abs(x)<=d2/2]=l2+l1

        ss=np.logical_and(x<(-d2/2),x>(-d1/2))
        wireprof[ss]=l1-circ[ss]

        ss=np.logical_and(x>(d2/2),x<(d1/2))
        circ=(r**2-(x+circ_cen)**2)**0.5
        wireprof[ss]=l1-circ[ss]

        wireprof[wireprof<0]=0


    elif typ.find('realwire')==0: #round wire
            wireprof=(r**2-(Na-off)**2)**0.5*2
            
    elif typ=='mist':
        maxi=r*2*np.sqrt(2)
        wireprof=maxi-2*np.abs(x)
        wireprof[wireprof<0]=0

    elif typ.find('wire_grating')==0: #GRATING made of wires
        spacing=float(pars['spacing'])
        factor=float(pars['factor'])
        offset=float(pars['offset'])
        wireradius=spacing/factor/2
        numwires=int(np.ceil(N*pxsize/spacing))
        grating=Na*0
        for wi in np.arange(-numwires,numwires):
            wirecenter=wi*spacing+offset
            wireprof=(wireradius**2-(Na-wirecenter)**2)**0.5*2
            wireprof[wireprof<0]=0
            wireprof[np.logical_not(np.isfinite(wireprof))]=0
            grating=grating+wireprof
        wireprof=grating
    else:
        assert 1, "Obstacle type not found"
    return wireprof




def doap(pars,params=[],debug=0,return_thickness=0):
    """
    Build the optical response of an aperture or obstacle element.

    This function constructs the transmission and phase-shift maps associated
    with an aperture or material obstacle defined by `pars`. Depending on the
    element type, the aperture may be purely transmissive (geometric aperture)
    or described by a physical thickness profile that is converted into
    transmission and phase via the material refractive index.

    Optional features include random surface defects, geometric rotations,
    crossed structures, and diagnostic plotting. The resulting maps are used
    directly in the field propagation pipeline.

    Parameters
    ----------
    pars : dict
        Dictionary describing the aperture or obstacle geometry and material.
    params : dict
        Global simulation parameters (grid size, pixel spacing, photon energy).
    debug : bool or int, optional
        Enable diagnostic output and plots if non-zero.
    return_thickness : bool, optional
        If True, also return the computed thickness map.

    Returns
    -------
    transmissionmap : ndarray
        2D transmission map applied to the field amplitude.
    phaseshiftmap : ndarray
        2D phase-shift map applied to the field phase.
    thicknessmap : ndarray, optional
        2D thickness map of the obstacle (returned if `return_thickness=True`).
    """
    axap = params['ax_apertures']
    E = params['photon_energy']
    N = params['N']
    N2 = int(N/2)
    typ = pars['shape']

    if typ in ['square','rectangle','wire','gaussian']:
        transmissionmap = get_aperture_transmission_map(pars,params,debug)
        phaseshiftmap = transmissionmap * 0
        thicknessmap = transmissionmap * 0
    else:
        thicknessmap = get_aperture_thickness_map(pars,params,debug)

        if yamlval('randomizeA',pars):  # Adding random defects. 'RandomizeA' is the maximal amplitude of the noise [m]. The spatial frequency is just given by pixel size
            ra = float(yamlval('randomizeA',pars))
            rand = np.random.random((N,N))*ra - ra/2
            img2 = thicknessmap+rand
            img2[thicknessmap==0] = 0
            img2[thicknessmap<=0] = 0
            thicknessmap = img2
            print('randomized')
        if yamlval('randomizeB',pars): # Adding random defects. in better way. 'RandomizeB' is the maximal radius of sphere added to the material[m].
            maxsize=float(yamlval('randomizeB',pars))
            density=float(yamlval('density',pars,2))
            print('Density: ',density)
            print(pars)
            boxsize=params['propsize']
            numsph=density*boxsize**2/maxsize**2 
            numsph=density*pars['size']**2/maxsize**2 
            pxsize=params['pxsize']

            for i in np.arange(numsph):
                size=np.random.random()*maxsize
                xr=np.random.random()*(boxsize-2*size-2*pxsize)
                yr=np.random.random()*(boxsize-2*size-2*pxsize)
                positive=np.random.random()>0.3
                add_sphere(size,xr,yr,thicknessmap,pxsize,positive)

            print('randomized B')

        if yamlval('rot',pars)==0:
            thicknessmap=np.array(np.transpose(thicknessmap))
        elif yamlval('rot',pars)==90:
            thicknessmap=np.array(thicknessmap)
        else:
            from scipy.ndimage.interpolation import rotate
            rot=90-pars['rot']#
            thicknessmap= rotate(thicknessmap, angle=rot,reshape=0)
        if yamlval('crossed',pars,0):
            thicknessmap2=np.transpose(thicknessmap)
            thicknessmap=thicknessmap2*thicknessmap


        #CONVERTING THICKNESS MAP INTO TRANSMISSION AND PHASESHIFT MAP
        elem=pars['elem']
        beta ,delta  = get_index(elem,E)
        thickness_to_phaseshift,k = thickness_to_phase_and_transmission(E, delta, beta)
        transmissionmap = np.exp(-k*thicknessmap)
        phaseshiftmap = thicknessmap * thickness_to_phaseshift
        if debug:
            print('thickness_to_phaseshift =',thickness_to_phaseshift)
            print('max thickness = ',np.max(thicknessmap))
            print('max phaseshift = ',np.max(phaseshiftmap))
            print('min phaseshift = ',np.min(phaseshiftmap))


    if 0:
        plt.sca(axap)
        lab=pars['shape']+", {:.0f} μm".format(pars['size']/um)
        plt.semilogy(Na/um,trans,label=lab)
        plt.ylabel('Transmission [-]')
        plt.xlabel('Position [μm]')
    if debug and 0:
        mu.figure()
        plt.subplot(311)
        plt.plot(Na/um,wireprof/um)
        plt.ylabel('Thickness [μm]')
        plt.subplot(312)
        plt.semilogy(Na/um,trans)
        plt.ylabel('transmission [-]')
        plt.ylim(1e-30,1)
        plt.grid()
        plt.subplot(313)

        plt.plot(Na/um,phaseshift)
        plt.ylabel('phase shift [rad]')

        plt.xlabel('position [μm]')

    # ─── Apply transverse offsets if requested ───
    shift_x_um = float(pars.get("offset_x_um", 0.0))
    shift_y_um = float(pars.get("offset_y_um", 0.0))

    if shift_x_um != 0.0 or shift_y_um != 0.0:
        dx = shift_x_um * 1e-6 / params["pxsize"]
        dy = shift_y_um * 1e-6 / params["pxsize"]

        print(f"[APERTURE] Shifting map by dx={shift_x_um:.1f} µm, dy={shift_y_um:.1f} µm")

        transmissionmap = imshift(
            transmissionmap,
            shift=(dy, dx),
            order=1,
            mode='constant',
            cval=1.0
        )

        phaseshiftmap = imshift(
            phaseshiftmap,
            shift=(dy, dx),
            order=1,
            mode='constant',
            cval=0.0
        )



    if axap!=None:# and pars['shape']!='circle':
        plt.sca(axap)
        drawthis=1
        if drawthis:
            lab=pars['shape']+", {:.0f} μm".format(pars['size']/um)
            lab=pars['shape']
            if yamlval('invert',pars): lab=lab+', inv.'
            if typ.find('trapez')==0:
                lab=lab+", {:.0f}/{:.0f}".format(pars['thickness']/um,pars['edge']/um)
            trans1=transmissionmap[N2,:]
            plt.semilogy(Na/um,trans1,label=lab)
            plt.ylabel('Transmission [-]')
            plt.xlabel('position [μm]')

    if  debug or 0:
        mx=N2*params['pxsize']
        mu.figure()
        ax=plt.subplot(121)
        plt.title('Transmission')
        ax.set_facecolor("black")
        ex=[-mx/um,mx/um,-mx/um,mx/um]
        plt.imshow(transmissionmap,extent=ex,cmap=rofl.cmap())
        plt.colorbar()
        plt.clim(0,1)
        prof =transmissionmap[N2,:]
        prof=mu.normalize(prof)
        Na=(np.arange(N)-N2)*params['pxsize']
        plt.plot(Na*1e6,prof*mx/um,'w')

        ax=plt.subplot(122)
        if 1:
            plt.title('Thickness')
            ax.set_facecolor("black")
            ex=[-mx/um,mx/um,-mx/um,mx/um]
            plt.imshow(thicknessmap,extent=ex)
            plt.colorbar()
            prof =thicknessmap[N2,:]
            prof=mu.normalize(prof)
            Na=(np.arange(N)-N2)*params['pxsize']
            plt.plot(Na*1e6,prof*mx/um,'w')
        else:
            plt.title('phase shift')
            ax.set_facecolor("black")
            ex=[-mx/um,mx/um,-mx/um,mx/um]
            plt.imshow(phaseshiftmap,extent=ex)
            plt.colorbar()
    if return_thickness:
        return transmissionmap,phaseshiftmap,thicknessmap
    else:
        return transmissionmap,phaseshiftmap




def build_gazjet_maps(el_dict, params, F=None):
    """
    Build the gas-jet phase map (and ~unity transmission).
    Treats the plasma as n = 1 - δ (β≈0), so:
        phase(x,y) = -k * δ(x,y) * L
        trans(x,y) ≈ 1

    Supports optional transverse sinusoidal modulation.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    # --- Debug / plotting flag ---
    DEBUG_PLOT = True  # 🔹 Turn off later if not needed

    # --- INTERNAL OPTION: Orientation of fringes ---
    #    "horizontal" → stripes stacked horizontally → modulation along Y
    #    "vertical"   → stripes stacked vertically   → modulation along X
    FRINGE_ORIENTATION = "vertical"   # or: "horizontal"

    # --- read gazjet params ---
    dens_cc = float(el_dict.get("density", 1e18))            # [cm^-3]
    profile = str(el_dict.get("profile", "gaussian")).lower()
    size_t  = float(el_dict.get("Size_transv", 500e-6))      # [m]
    size_l  = float(el_dict.get("Size_long",  500e-6))       # [m]
    offset_t = float(el_dict.get("Offset_transv", 0.0))      # [m]
    lam     = float(params.get("wavelength", 1e-10))         # [m]
    k0      = 2.0 * np.pi / lam

    # --- modulation options ---
    modulate      = int(el_dict.get("modulate", 0))
    mod_period_um = float(el_dict.get("mod_period_um", 0.0))  # [µm]

    # --- grid (N, dx) from field or params ---
    if F is not None:
        N = int(getattr(F, "N"))
        if hasattr(F, "ps"):
            dx = float(F.ps)
        elif hasattr(F, "grid_size") and hasattr(F, "N"):
            dx = float(F.grid_size) / float(F.N)
        else:
            propsize = float(params["propsize"])
            dx = propsize / N
    else:
        N = int(params["N"])
        propsize = float(params["propsize"])
        dx = propsize / N

    # --- electron density in m^-3 ---
    n_e = dens_cc * 1e6

    # --- plasma index decrement: δ = n_e e² / (2 ε0 m_e ω²) ---
    omega  = 2.0 * np.pi * c / lam
    delta0 = n_e * e**2 / (2.0 * epsilon_0 * m_e * omega**2)

    # --- coordinate grid ---
    x = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")

    # --- transverse offsets (both possible) ---
    Y_shifted = Y - offset_t
    X_shifted = X  # currently no X offset, but symmetry kept for future use

    # --- base density profile ---
    if profile == "gaussian":
        sigma = size_t / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        density_map = np.exp(-(X_shifted**2 + Y_shifted**2) / (2.0 * sigma**2))
    else:  # rectangle
        half = size_t / 2.0
        density_map = np.where((np.abs(X_shifted) < half) &
                               (np.abs(Y_shifted) < half), 1.0, 0.0)

    # --- apply modulation if requested ---
    if modulate and mod_period_um > 0:
        mod_period_m = mod_period_um * 1e-6

        if FRINGE_ORIENTATION == "horizontal":
            # Modulation varies in Y → horizontal fringes
            modulation = 0.5 * (1 + np.sin(2.0 * np.pi * Y_shifted / mod_period_m))
            print(f" → Gazjet modulation ON (horizontal fringes, period = {mod_period_um} µm)")

        elif FRINGE_ORIENTATION == "vertical":
            # Modulation varies in X → vertical fringes
            modulation = 0.5 * (1 + np.sin(2.0 * np.pi * X_shifted / mod_period_m))
            print(f" → Gazjet modulation ON (vertical fringes, period = {mod_period_um} µm)")

        else:
            raise ValueError("FRINGE_ORIENTATION must be 'horizontal' or 'vertical'")

        density_map *= modulation

    # --- compute phase & transmission maps ---
    delta_map = delta0 * density_map
    phase_map = -k0 * delta_map * size_l         # [radians]

    # DEBUG: force phase to be in [0, π]
    #phase_map = np.pi * (phase_map - phase_map.min()) / (phase_map.max() - phase_map.min()) #TO BE REMOVED

    trans_map = np.ones_like(phase_map, float)

    # ----------------------------------------------------------------------
    # ------------------------ DEBUG PLOT SECTION --------------------------
    # ----------------------------------------------------------------------
    if DEBUG_PLOT:
        try:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            extent_um = (x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6)

            zoom_lim = 50  # [µm]

            # Density map
            im0 = ax[0].imshow(density_map, extent=extent_um, origin='lower', cmap='inferno')
            ax[0].set_title("Gazjet density map (zoom ±50 µm)")
            ax[0].set_xlabel("x [µm]")
            ax[0].set_ylabel("y [µm]")
            plt.colorbar(im0, ax=ax[0], label="Relative density")
            ax[0].set_xlim(-zoom_lim, zoom_lim)
            ax[0].set_ylim(-zoom_lim, zoom_lim)

            # Phase map
            im1 = ax[1].imshow(phase_map, extent=extent_um, origin='lower', cmap='RdBu')
            ax[1].set_title("Phase shift map (zoom ±50 µm)")
            ax[1].set_xlabel("x [µm]")
            ax[1].set_ylabel("y [µm]")
            plt.colorbar(im1, ax=ax[1], label="Phase [rad]")
            ax[1].set_xlim(-zoom_lim, zoom_lim)
            ax[1].set_ylim(-zoom_lim, zoom_lim)

            fig.tight_layout()

            outdir = Path("/home/yu79deg/darkfield_p5438/VIBE_outputs/figures")
            outdir.mkdir(parents=True, exist_ok=True)

            sim_name = Path(params.get("filename", "SimUnknown")).stem
            save_path = outdir / f"{sim_name}_gazjet.png"

            plt.savefig(save_path, dpi=300)
            plt.close(fig)

            print(f"[Gazjet] Saved density/phase debug plot to {save_path}")
        except Exception as err_disp:
            print(f"[Gazjet] Plotting failed: {err_disp}")

    return phase_map, trans_map



# =========================================================
# PLOT related functions
# =========================================================



def prepare_image(
    img,
    ps=750,
    max_pixels=300,
    ZoomFactor=1,
    log=1,
    norms=None,
    el_dict=None,
    normalize=True,
):
    """
    Prepare a 2D image for visualization or export.

    This function optionally normalizes the image, crops its central region
    according to a zoom factor, and resamples it to a maximum pixel size.
    Integrated intensity and total sum are tracked and returned for consistent
    normalization across multiple images.

    Parameters
    ----------
    img : ndarray
        Input 2D image.
    ps : float, optional
        Pixel size used to compute integrated quantities.
    max_pixels : int, optional
        Maximum number of pixels per dimension in the output image.
    ZoomFactor : float, optional
        Factor by which to crop the central region (ZoomFactor > 1 zooms in).
    log : bool or int, optional
        Kept for compatibility; not used internally.
    norms : list or None, optional
        Normalization reference [integrated_max, integrated_sum].
        If None or zero, values are initialized from the current image.
    el_dict : dict or None, optional
        Unused placeholder for backward compatibility.
    normalize : bool, optional
        If True, normalize integrated quantities using `norms`.

    Returns
    -------
    img_out : ndarray
        Processed image (cropped and/or resampled).
    norms : list
        Normalization reference values.
    stats : list
        Current normalized [integrated_max, integrated_sum].
    """

    import cv2

    if norms is None:
        norms = [0.0, 0.0]

    # --------------------------------------------------
    # Compute integrated quantities
    # --------------------------------------------------
    integrated_max = np.max(img) * ps**2
    integrated_sum = np.sum(img) * ps**2

    if normalize:
        if norms[0] == 0 and norms[1] == 0:
            norms[0] = integrated_max
            norms[1] = integrated_sum

        integrated_max /= norms[0]
        integrated_sum /= norms[1]

    # --------------------------------------------------
    # Central crop according to zoom factor
    # --------------------------------------------------
    img_out = img
    if ZoomFactor > 1:
        n = img.shape[0]
        half_size = int(n / (2 * ZoomFactor))
        center = n // 2
        img_out = img[
            center - half_size : center + half_size,
            center - half_size : center + half_size,
        ]

    # --------------------------------------------------
    # Downsample if image is too large
    # --------------------------------------------------
    if img_out.shape[0] > max_pixels:
        img_out = cv2.resize(
            img_out,
            dsize=(max_pixels, max_pixels),
            interpolation=cv2.INTER_CUBIC,
        )

    return img_out, norms, [integrated_max, integrated_sum]




def imshow(imgC, ps=750, ZoomFactor=1, log=1, measures=[0, 0], el_dict=None):
    """
    Display a 2D image with physical scaling and diagnostic annotations.

    This function visualizes a 2D intensity or phase map using a linear or
    logarithmic color scale, sets the spatial extent in micrometers, and
    overlays basic diagnostic information such as image size, integrated
    measures, and optional element metadata.

    Parameters
    ----------
    imgC : ndarray
        2D image to display.
    ps : float, optional
        Physical size of the displayed region (meters).
    ZoomFactor : float, optional
        Zoom factor applied to the image (affects spatial scaling).
    log : bool or int, optional
        If non-zero, use logarithmic color normalization.
    measures : list, optional
        Diagnostic values [integrated_max, integrated_sum] displayed on the plot.
    el_dict : dict or None, optional
        Dictionary describing the optical element; selected parameters are
        displayed as text annotations.

    Returns
    -------
    ndarray
        The displayed image (unchanged).
    """

    # --------------------------------------------------
    # Color normalization
    # --------------------------------------------------
    norm = colors.LogNorm() if log else colors.Normalize()

    # Adjust physical scale for zoom
    if ZoomFactor > 1:
        ps = ps / ZoomFactor

    half_size_um = ps / (2 * um)
    extent = (-half_size_um, half_size_um, -half_size_um, half_size_um)

    # --------------------------------------------------
    # Display image
    # --------------------------------------------------
    plt.imshow(imgC, norm=norm, cmap=rofl.cmap(), extent=extent)
    ax = plt.gca()

    # --------------------------------------------------
    # Image size annotation (top-left)
    # --------------------------------------------------
    if ps / um >= 10:
        size_label = f"{ps / um:.0f} μm"
    else:
        size_label = f"{ps / um:.1f} μm"

    plt.text(
        0.01, 0.99, size_label,
        ha="left", va="top",
        transform=ax.transAxes, color="w"
    )

    # --------------------------------------------------
    # Optional element metadata annotation
    # --------------------------------------------------
    if el_dict is not None:
        keys_to_show = ["size", "f", "shape", "roc"]
        units = {"size": "μm", "roc": "μm", "f": "m"}
        formats = {"size": "{:.0f}", "roc": "{:.0f}", "shape": "{:}"}

        row = 1
        for key in keys_to_show:
            if key not in el_dict:
                continue

            unit = units.get(key, "")
            fmt = formats.get(key, "{:.1f}")

            scale = 1e6 if unit == "μm" else 1.0
            value = el_dict[key] * scale if key != "shape" else el_dict[key]

            label = f"{key}: " + fmt.format(value)
            if unit:
                label += f" {unit}"

            plt.text(
                0.01, 0.99 - row * 0.1, label,
                ha="left", va="top",
                transform=ax.transAxes, color="w"
            )
            row += 1

    # --------------------------------------------------
    # Diagnostic measures (top-right)
    # --------------------------------------------------
    plt.text(
        0.99, 0.99, f"M {measures[0]:.1e}",
        ha="right", va="top",
        transform=ax.transAxes, color="w"
    )
    plt.text(
        0.99, 0.89, f"S {measures[1]:.1e}",
        ha="right", va="top",
        transform=ax.transAxes, color="w"
    )

    return imgC



def croping_to_odd(image):
    """
    Crops the image in the case where N is even such that the corresponding image has an odd number of points. This is better for air scattering convolutions
    """
    rows, cols = image.shape
    if rows % 2 == 0:
        image = image[:-1, :]
    if cols % 2 == 0:
        image = image[:, :-1]
    return image



def restore_even_shape_by_duplication(image, target_shape):
    """
    Add one row and/or column to the image by duplicating the last row/column,
    to restore original shape after cropping.
    """
    current_rows, current_cols = image.shape
    target_rows, target_cols = target_shape
    assert target_rows >= current_rows and target_cols >= current_cols

    # Start with the cropped image
    restored = image.copy()

    # If we need to add a row
    if target_rows > current_rows:
        last_row = restored[-1:, :]
        restored = np.vstack([restored, last_row])

    # If we need to add a column
    if target_cols > current_cols:
        last_col = restored[:, -1:]
        restored = np.hstack([restored, last_col])

    return restored
    



def zoom_window_with_interp(F: Field, zoom: float) -> Field:
    """
    Return a *new* LightPipes Field whose window size is ``zoom`` × the
    original one, with the optical wavefront re-sampled so that the
    physical beam is unchanged.

    Parameters
    ----------
    F     : LightPipes Field
        The input field.
    zoom  : float
        Scale factor for the window.  ``zoom < 1`` zooms *in* (smaller
        window, finer sampling); ``zoom > 1`` zooms *out*.

    Returns
    -------
    Field
        A freshly allocated LightPipes Field (input is left untouched).
    """
    if zoom <= 0:
        raise ValueError("zoom must be a positive number.")

    # --- original grid ---------------------------------------------------
    N        : int   = F.N
    lam      : float = F.lam
    size_old : float = F.siz
    size_new : float = size_old * zoom          # new physical window [m]

    # --- physical coordinates of the target grid ------------------------
    x_new = np.linspace(-0.5 * size_new, 0.5 * size_new, N, endpoint=False)
    Xn, Yn = np.meshgrid(x_new, x_new, indexing="xy")

    # --- map these coordinates onto indices of the *old* grid -----------
    dx_old   = size_old / N                      # sample pitch of old grid
    coords_x = (Xn + 0.5 * size_old) / dx_old
    coords_y = (Yn + 0.5 * size_old) / dx_old
    # keep inside array bounds for cubic interpolation
    eps = 1e-3
    coords_x = np.clip(coords_x, 0, N - 1 - eps)
    coords_y = np.clip(coords_y, 0, N - 1 - eps)

    # --- cubic interpolation of real & imag parts -----------------------
    real_i = map_coordinates(np.real(F.field), [coords_y, coords_x],
                             order=3, mode="nearest")
    imag_i = map_coordinates(np.imag(F.field), [coords_y, coords_x],
                             order=3, mode="nearest")

    # --- allocate the new LightPipes field ------------------------------
    Fnew = Begin(size_new, lam, N, dtype=F._dtype)   # safer than raw Field()
    Fnew.field    = real_i + 1j * imag_i
    Fnew._IsGauss = False        # interpolated, no longer analytic Gaussian

    return Fnew





def _save_center_crop_debug(I, params, fname_tag="debug", title_tag="", outdir=None, crop_um=4.0):
    """
    Save a 2x2 debug figure (2D linear/log + central 1D linear/log)
    for a ±crop_um (µm) window around the center. Expects I to be an intensity map (relative units).
    Scales to photons/m² if params['scale_phot'] is present.
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    # Optional scaling to photons / m²
    scale_ph = params.get("scale_phot", None)
    if scale_ph is None:
        print(f"[TCC] scale_phot not available → skipping {fname_tag} debug figure.")
        return
    I_ph = I * scale_ph  # photons / m²

    # Window half-size in pixels
    pxsize = float(params["pxsize"])            # [m/px]
    half_win_m  = float(crop_um) * 1e-6         # [m]
    half_win_px = max(1, int(round(half_win_m / pxsize)))

    # Centered crop
    Ny, Nx = I_ph.shape
    cy, cx = Ny // 2, Nx // 2
    y0, y1 = max(0, cy - half_win_px), min(Ny, cy + half_win_px)
    x0, x1 = max(0, cx - half_win_px), min(Nx, cx + half_win_px)
    Iw = I_ph[y0:y1, x0:x1]

    # Axes extent in µm (centered)
    win_x_um = Iw.shape[1] * pxsize * 1e6
    win_y_um = Iw.shape[0] * pxsize * 1e6
    ext = (-0.5 * win_x_um, 0.5 * win_x_um, -0.5 * win_y_um, 0.5 * win_y_um)
    x_um = np.linspace(ext[0], ext[1], Iw.shape[1])
    profile = Iw[Iw.shape[0] // 2, :]

    # Colormap (fallback if rofl.cmap() not available)
    try:
        cmap = rofl.cmap()
    except Exception:
        cmap = None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    use_log = bool(params.get("simulation", {}).get("figs_log", 0))
    clim    = params.get("simulation", {}).get("flow_plot_clim", [None, None])

    if title_tag:
        fig.suptitle(title_tag, y=0.98)

    # 2D linear
    im0 = axes[0, 0].imshow(
        Iw, origin="lower", extent=ext, interpolation="nearest",
        aspect="equal", cmap=(cmap or "viridis")
    )
    axes[0, 0].set_title("2D (linear) — photons/m²")
    axes[0, 0].set_xlabel("x [µm]"); axes[0, 0].set_ylabel("y [µm]")
    c0 = plt.colorbar(im0, ax=axes[0, 0], shrink=0.9); c0.set_label("photons / m²")

    # 2D log
    A = np.asarray(Iw, float)
    if use_log:
        pos = A[np.isfinite(A) & (A > 0)]
        if pos.size:
            vmin = float(pos.min())
            vmax = float(pos.max())
            # Apply YAML clim if provided
            if clim:
                lo, hi = clim
                if lo is not None:
                    vmin = max(vmin, float(lo))
                if hi is not None:
                    vmax = min(vmax, float(hi))
            # Enforce strictly positive + ordered bounds
            vmin = max(vmin, np.finfo(float).tiny)
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = np.nextafter(vmin, np.inf)
            im1 = axes[0, 1].imshow(
                np.where(A > 0, A, np.nan),
                origin="lower", extent=ext, interpolation="nearest",
                aspect="equal", cmap=(cmap or "viridis"),
                norm=LogNorm(vmin=vmin, vmax=vmax)
            )
            axes[0, 1].set_title("2D (log) — photons/m²")
        else:
            # no positive data → linear fallback
            im1 = axes[0, 1].imshow(
                A, origin="lower", extent=ext, interpolation="nearest",
                aspect="equal", cmap=(cmap or "viridis")
            )
            axes[0, 1].set_title("2D (linear fallback) — photons/m²")
    else:
        # YAML requested linear plotting
        im1 = axes[0, 1].imshow(
            A, origin="lower", extent=ext, interpolation="nearest",
            aspect="equal", cmap=(cmap or "viridis")
        )
        axes[0, 1].set_title("2D (linear) — photons/m²")

    axes[0, 1].set_xlabel("x [µm]"); axes[0, 1].set_ylabel("y [µm]")
    c1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.9); c1.set_label("photons / m²")


    # 1D central cuts
    axes[1, 0].plot(x_um, profile)
    axes[1, 0].set_title("Central cut (linear)")
    axes[1, 0].set_xlabel("x [µm]"); axes[1, 0].set_ylabel("photons / m²")

    axes[1, 1].semilogy(x_um, np.clip(profile, 1e-300, None))
    axes[1, 1].set_title("Central cut (log)")
    axes[1, 1].set_xlabel("x [µm]"); axes[1, 1].set_ylabel("photons / m²")

    # Save
    save_dir = Path(outdir) if outdir is not None else Path(params["projectdir"])
    sim = params.get("filename", "sim")
    outfile = save_dir / f"{sim}_{fname_tag}_at_TCC.png"

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[TCC] Saved {title_tag} 2D map + central cut to {outfile}")



def _sanitize_intensity_map(img: np.ndarray) -> np.ndarray:
    """
    Build a map with no negative values, no NaNs, no infinites, 
    to be able to be used as an external map for laser and xray profiles
    """
    # Convert to float, collapse RGB if needed
    arr = img.astype(float)
    if arr.ndim == 3:  # e.g., RGB
        # luminance or simple mean; choose what you prefer
        arr = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]

    # Replace NaN/Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Shift if negative baseline (e.g. high-pass artifacts)
    minv = arr.min()
    if minv < 0:
        arr = arr - minv  # makes min 0

    # Clip negative crumbs
    arr = np.clip(arr, 0.0, None)

    return arr





def flow_plot(project_dir, file, cl=[1e-11,50], gyax_def=[-1000,1000,1], vertical_type='center', log=1, xl=None, flow_figs=0, flow_plot_crange=1e-5, channel="main", include_flow=True, unit=None):
    """
    Visualizes 2D flow maps along z, with optional export for movie making.

    Parameters
    ----------
    project_dir : str         – Path to simulation folder.
    file        : str         – Base name of the pickle files.
    cl          : list        – Color limits for plotting.
    gyax_def    : list        – Vertical axis definition [start, end, step] in µm.
    vertical_type : str       – How to reduce 2D → 1D (center, average_horiz, etc.)
    log         : bool        – Use log scale in plot.
    xl          : list or None – Horizontal axis limits in meters.
    flow_figs   : bool        – If True, saves each flow slice individually.
    flow_plot_crange : float  – Color scale fraction for flow slice plots.
    channel     : str         – Channel to plot ("main", "VB_parr", ...).
    include_flow : bool       – Whether to include "flow" auto-slices.
    unit        : str or None – Intensity unit ("relative", "photons", "Wcm2").

    Returns
    -------
    params : dict
    res    : dict
    fixedfall : np.ndarray – final interpolated waterfall image
    """
    import os
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    # ─── Load Pickles ────────────────────────────────────────────────

    fn_figs = f"{file}_figs"
    fn_res  = fn_figs.replace('figs', 'res').replace('export', 'res')

    pic_path = Path(project_dir) / 'pickles' / f'{fn_figs}.pickle'
    res_path = Path(project_dir) / 'pickles' / f'{fn_res}.pickle'

    pic = mu.loadPickle(str(pic_path), strict=1)
    res = mu.loadPickle(str(res_path))
    partial = (res == 0)
    
    # ─── Safety checks ───────────────────────────────────────────────

    assert len(pic.keys()) > 0, 'No images found in the pickle!'
    if not partial:
        params = res[1]
    else:
        params = {}


    # ─── Setup core variables ────────────────────────────────────────
    first_key = sorted(pic.keys())[0]
    picc, _, _, _ = pic[first_key]
    N = np.shape(picc)[0]

    gyax = np.arange(*gyax_def)  # y-axis in µm

    numfigs = sum(1 for k in pic if k.endswith(f"_{channel}"))

    assert numfigs > 0, f"No flow slices found for channel '{channel}'"

    # ─── Filter relevant flow slices ────────────────────────────────
    ffigs = []
    figs = pic.keys()

    for fig in figs:
        if not fig.endswith(f"_{channel}"):
            continue

        if channel == "main":
            if fig.startswith("flow"):
                ffigs.append(fig)
        else:
            el_name = fig.split('_')[0]
            wanted  = params.get("figs_to_save", [])
            if (el_name == "flow" and include_flow) or (el_name in wanted and el_name != "flow"):
                ffigs.append(fig)

    # --- Apply x-limits filtering at slice level ---
    if xl is not None:
        zmin, zmax = xl
        ffigs_filtered = []

        for fig in ffigs:
            _, _, _, position = pic[fig]
            if (zmin is None or position >= zmin) and (zmax is None or position <= zmax):
                ffigs_filtered.append(fig)

        ffigs = ffigs_filtered

    print(f"[DEBUG] ffigs for channel '{channel}': {len(ffigs)} found")

    # ─── Intensity unit handling ─────────────────────────────────────
    unit_sel = unit or params.get('intensity_units', 'relative')
    scale_ph  = params.get('scale_phot',  1.0)
    scale_Wcm = params.get('scale_Wcm2', 1.0)

    if unit_sel == 'photons':
        scale = scale_ph
        y_label = "photons / px"
    elif unit_sel == 'Wcm2':
        scale = scale_Wcm
        y_label = "W cm⁻²"
    else:
        scale = 1.0
        y_label = "Normalized intensity"


    try:
        cl = [float(c) * scale for c in cl]
    except Exception as e:
        raise ValueError(f"Invalid color limits (cl): {cl}. Make sure it's a list of two floats.") from e
    
    # --- Id of the run ------------------------------------------------
    run_id = file if channel in (None, "", "main") else f"{file}_{channel}"

    # ─── Initialize arrays for waterfall plot ────────────────────────
    numfigs = len(ffigs)
    waterfall  = np.zeros((numfigs, N))
    fixedfall  = np.zeros((numfigs, len(gyax)))
    propsizes  = np.zeros((numfigs,))
    zax        = np.zeros((numfigs,))

    scatterer_L2_position=1e9
    scatterer_L1_position=1e9
    skip_existing=1

    if not partial:
    #extracting scatterers and theirloses
        if 'L1' in res[0]:
            scatterer_L1_position=res[0]['L1']['position']
            scatterer_L1_loss=yamlval('transmission_of_scatterer_L1',params,1)
        if 'L2' in res[0]:
            scatterer_L2_position=res[0]['L2']['position']
            scatterer_L2_loss=yamlval('transmission_of_scatterer_L2',params,1)
        N=res[1]['subfigure_size_px']
    else: params=[]
    assert len(pic.keys())>0, 'There are no pictures in the pickle!'

    if flow_figs:
        ffdir = Path(project_dir) / 'flow_figs' / run_id
        #mu.mkdir(ffdir,0)
        mu.mkdir(str(ffdir), 0)


    # ─── Loop over flow slices ───────────────────────────────────────
    for fi, fig in enumerate(ffigs):
        picc, elemi, propsize, position = pic[fig]
        #print(f"[DEBUG] {fig} – propsize = {propsize:.3e} m")

        imsize  = picc.shape[0]
        pxsize  = propsize * 1e6 / imsize  # µm per pixel
        half_px = imsize // 2
        ps2     = propsize / 2             # half size in meters

        # Compute horizontal axis in μm
        xax = (np.arange(imsize) - 0.5 * imsize) * pxsize  # μm

        # Compute vertical lineout based on `vertical_type`
        if vertical_type == 'center':
            w = 2
            start = max(0, half_px - w)
            end   = min(imsize, half_px + w + 1)
            lineout = np.mean(picc[start:end, :], axis=0)
        elif vertical_type == 'average_horiz':
            lineout = np.mean(picc, axis=0)
        elif vertical_type == 'vert-center':
            lineout = picc[:, half_px]
        elif vertical_type == 'vert-integral':
            lineout = np.mean(picc, axis=1)
        else:
            raise ValueError(f"Unknown vertical_type: {vertical_type}")

        # Apply scatterer correction if needed
        if position > scatterer_L2_position:
            lineout /= (scatterer_L2_loss * scatterer_L1_loss)
        elif position >= scatterer_L1_position:
            lineout /= scatterer_L1_loss

        # Interpolate onto fixed gyax grid
        interp_line = np.full(len(gyax), np.nan, dtype=np.float64)
        valid = (gyax >= np.min(xax)) & (gyax <= np.max(xax))
        interp_line[valid] = np.interp(gyax[valid], xax, lineout)
        fixedfall[fi, :] = interp_line * scale


        # Store
        waterfall[fi, :] = lineout * scale
        fixedfall[fi, :] = interp_line * scale
        propsizes[fi] = propsize
        zax[fi] = position

        # ─── Optional: export movie frame if requested ───
        if flow_figs:
            ffdir = Path(project_dir) / 'flow_figs' / run_id
            ffdir.mkdir(parents=True, exist_ok=True)

            ff_fn = ffdir / f"fixed_{fi:04d}.jpg"

            # Skip existing if flag is set
            if skip_existing and ff_fn.exists():
                print(f"[SKIP] Frame {fi:04d} already exists.")
            else:
                print(f"[MOVIE] Exporting frame {fi:04d} at z = {position:.2f} m")

                # Plot the full image slice
                fig_movie, ax_movie = plt.subplots(figsize=(10, 10))
                npix = picc.shape[0]
                xc = (np.arange(npix) - npix / 2) * pxsize
                cmax = np.max(picc)
                cl1 = [cmax * flow_plot_crange, cmax]

                picc_T = picc.T  # transpose for correct orientation

                mu.pcolor(picc_T, xc=xc, yc=xc, ticks=0, log=1, cl=cl1, background=[0, 0, 0])
                plt.axis('equal')

                h = 100 / 2  # box size = 100 μm
                plt.plot([-h, -h, h, h, -h], [-h, h, h, -h, -h], 'r.', alpha=1, markersize=7)

                plt.xlabel('X [μm]')
                plt.ylabel('Y [μm]')
                plt.title(f"{file}, z = {position * 100:.0f} cm")

                # zoom into 100 μm × 100 μm box
                plt.xlim(-h, h)
                plt.ylim(-h, h)

                plt.tight_layout()
                plt.savefig(ff_fn)
                plt.close(fig_movie)




    fig, ax = plt.subplots(figsize=(14, 8))

    mu.pcolor(
        xc=zax,
        yc=gyax,
        data=fixedfall,
        log=log,
        ticks=None,
        cl=cl,
        colorbar=False
    )

    # Add colorbar with label
    cb = plt.colorbar()
    if vertical_type in ("average_horiz", "center"):
        cb_label = {
            "photons": "photons / m²",
            "Wcm2": "W / m²",
            "relative": "relative units"
        }.get(unit_sel, "relative units")
    else:
        cb_label = y_label
    cb.set_label(cb_label)

    # Overlay propagation box profile (normalized to gyax)
    profile = mu.normalize(propsizes) * np.max(gyax)
    plt.plot(zax, profile, 'r-')

    # Draw optical element markers (only within visible z-range)
    if not partial:
        maxy = np.min(gyax)
        row = 0
        zmin_vis = np.nanmin(zax)
        zmax_vis = np.nanmax(zax)

        for el_name, el in res[0].items():
            if (
                'position' not in el or
                not mu.yamlval('in', el, 1) or
                el_name.startswith("flow_")
            ):
                continue

            pos = el['position']
            # --- skip annotations outside visible region ---
            if pos < zmin_vis or pos > zmax_vis:
                continue

            if len(el_name) == 2:  # short names like L1, L2
                yline = maxy * (0.8 if 'L' in el_name else 0.72)
                col = [1, 0.5, 0.9] if 'L' in el_name else [1, 0.9, 0.7]
            else:
                yline = maxy * (0.95 - row * 0.05)
                col = 'w'
                row = (row + 1) % 4

            plt.plot([pos, pos], [maxy, yline], color=col)
            mu.text(pos + 0.05, yline, el_name, color=col, fs=16, zorder=50, background=None)


    # Detector marker (white vertical line)
    elements_dict = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else {}
    det_dict = elements_dict.get("Det", {})
    det_pos = det_dict.get("position", None)
    roi_um  = det_dict.get("roi", 13)  # microns, consistent with gyax units

    if det_pos is not None:
        plt.plot([det_pos, det_pos], [-roi_um/2, roi_um/2], 'w-', lw=5)

    # Axes labels and limits
    plt.xlabel('Position [m]')
    plt.ylabel('Horizontal position [μm]')
    plt.xlim(xl if xl else [np.min(zax), np.max(zax)])
    plt.ylim(np.min(gyax), np.max(gyax))  # ← enforce correct y-range
    plt.title(f"{file} cut: {vertical_type}")

    plt.tight_layout()

    # ─── Total photon count at beam shaper plane ───
    elements_dict = res[0]
    beam_shaper_pos = None
    if "beam_shaper" in elements_dict:
        if yamlval("in", elements_dict["beam_shaper"], 1):
            beam_shaper_pos = elements_dict["beam_shaper"]["position"]

    # Perform only for MAIN channel
    if (beam_shaper_pos is not None) and (channel == "main") and (len(ffigs) > 0):
        # find the closest slice to the beam shaper position
        idx = int(np.argmin(np.abs(zax - beam_shaper_pos)))
        idx = max(0, min(idx, len(ffigs) - 1))  # clamp to valid range

        fig_key = ffigs[idx]
        print(f"\n✅ Beam shaper at z = {beam_shaper_pos:.2f} m → closest slice z = {zax[idx]:.2f} m")

        # get the data from that slice
        raw_img, _, propsize, _ = pic[fig_key]
        img_N = raw_img.shape[0]
        dx = dy = propsize / img_N

        img_scaled = {
            "photons": raw_img * scale_ph,
            "Wcm2": raw_img * scale_Wcm,
            "relative": raw_img
        }.get(unit_sel, raw_img)

        total_photons = np.nansum(img_scaled) * dx * dy
        photons_target = params.get("photons_total", None)

        if unit_sel == "photons":
            print(f"→ Total photons from flow = {total_photons:.3e}")
            if photons_target:
                rel_err = (total_photons - photons_target) / photons_target
                print(f"→ Target photons_total = {photons_target:.3e}")
                print(f"→ Relative error = {rel_err:.2%}")

    if not partial:
        res[1]['propsizes'] = propsizes

    # --------- Shadow Factors and Detector Marker -------
    centralelement = "TCC"
    if f"{centralelement}_{channel}" not in params.get("intensities", {}):
        centralelement = "PH"
    key_central = f"{centralelement}_{channel}"

    intens = params.get("intensities", {})
    if key_central in intens:
        t1 = intens[key_central] / intens.get("start", 1.0)
        tr_scat = yamlval("transmission_of_scatterer_L2", params, 1)

        if "roi" in intens and "roi2" in intens:
            t13 = intens["roi"] / intens[key_central] / tr_scat
            t75 = intens["roi2"] / intens[key_central] / tr_scat
            print(f"SFA13 = {t13:.1e}, SFA75 = {t75:.1e}, Ratio = {t75/t13:.2f}")

            ax = plt.gca()

    # ─── VB SIGNAL (Detector photon count) ───────────────────────────
    VB_photons = {}  # will store nested dicts like {'VB_parr': {13e-6: val, 75e-6: val}}

    try:
        tcc_dict = res[0].get("TCC", {})
        if int(tcc_dict.get("VB_signal", 0)) == 1:
            print("\n[VB SIGNAL] Computing VB_perp and VB_parr photon counts at detector...")

            for pol in ["VB_parr", "VB_perp"]:
                key = f"Det_{pol}"
                if key not in pic:
                    print(f"[VB SIGNAL] → {key} not found in pickle.")
                    continue

                img, _, propsize, _ = pic[key]
                N = img.shape[0]
                dx_m = propsize / N

                # Convert to requested physical units
                if unit_sel == "photons":
                    img_scaled = img * scale_ph
                elif unit_sel == "Wcm2":
                    img_scaled = img * scale_Wcm
                else:
                    img_scaled = img

                VB_photons[pol] = {}  # initialize sub-dict for this polarization

                # Compute for two ROI sizes
                for roi_um in [13e-6, 75e-6]:
                    roi_half_px = int((roi_um / dx_m) / 2)
                    center_px = N // 2
                    x0, x1 = center_px - roi_half_px, center_px + roi_half_px
                    y0, y1 = center_px - roi_half_px, center_px + roi_half_px

                    sub_img = img_scaled[y0:y1, x0:x1]
                    photons_total = np.nansum(sub_img) * (dx_m ** 2)
                    VB_photons[pol][roi_um] = photons_total

                    print(f"→ {pol}: {photons_total:.3e} photons in {roi_um*1e6:.0f} µm × {roi_um*1e6:.0f} µm ROI")

    except Exception as e:
        import traceback
        print("[VB SIGNAL] CRASH inside photon computation!")
        traceback.print_exc()



    # ─── BACKGROUND PHOTONS (main channel at detector) ──────────────────
    BG_photons = {}
    try:
        if "Det_main" in pic:
            print("\n[BG PHOTONS] Computing main channel photons at detector...")
            img_main, _, propsize_main, _ = pic["Det_main"]
            N_main = img_main.shape[0]
            dx_m_main = propsize_main / N_main

            # Compute for two ROI sizes: 13 µm and 75 µm
            for roi_um in [13e-6, 75e-6]:
                roi_half_px = int((roi_um / dx_m_main) / 2)
                center_px = N_main // 2
                x0, x1 = center_px - roi_half_px, center_px + roi_half_px
                y0, y1 = center_px - roi_half_px, center_px + roi_half_px

                sub_img = img_main[y0:y1, x0:x1]
                if unit_sel == "photons":
                    img_scaled_main = img_main * scale_ph
                elif unit_sel == "Wcm2":
                    img_scaled_main = img_main * scale_Wcm
                else:
                    img_scaled_main = img_main

                photons_total = np.nansum(img_scaled_main[y0:y1, x0:x1]) * (dx_m_main ** 2)
                BG_photons[roi_um] = photons_total
                print(f"→ BG photons ({roi_um*1e6:.0f} µm box): {photons_total:.3e}")
        else:
            print("[BG PHOTONS] No Det_main found in pickle.")
    except Exception as e:
        import traceback
        print("[BG PHOTONS] CRASH during background photon computation!")
        traceback.print_exc()

    SNR = {}
    if VB_photons and BG_photons:
        for pol in ["VB_parr", "VB_perp"]:
            for roi_um in [13e-6, 75e-6]:
                vb_val = VB_photons.get(pol, {}).get(roi_um, np.nan)
                bg_val = BG_photons.get(roi_um, np.nan)
                if np.isfinite(vb_val) and np.isfinite(bg_val) and bg_val != 0:
                    SNR[(pol, roi_um)] = vb_val / bg_val



    # ─── TABLE OF RESULTS ON THE PLOT ───────────────────────────────────
    if True:  # always draw if we have at least SF or VB or BG
        ax = plt.gca()
        text_lines = []

        # headers
        text_lines.append(f"{'Quantity':<10} | {'13µm':>10} | {'75µm':>10}")
        text_lines.append("-" * 35)

        # Shadow factors (always available)
        text_lines.append(f"{'SF':<10} | {t13:>10.2e} | {t75:>10.2e}")

        # --- VB photon counts (if available) ---
        if 'VB_parr' in VB_photons or 'VB_perp' in VB_photons:
            VB_parr_13 = VB_photons.get('VB_parr', {}).get(13e-6, np.nan)
            VB_parr_75 = VB_photons.get('VB_parr', {}).get(75e-6, np.nan)
            VB_perp_13 = VB_photons.get('VB_perp', {}).get(13e-6, np.nan)
            VB_perp_75 = VB_photons.get('VB_perp', {}).get(75e-6, np.nan)

            text_lines.append(f"{'VB||':<10} | {VB_parr_13:>10.2e} | {VB_parr_75:>10.2e}")
            text_lines.append(f"{'VB⊥':<10} | {VB_perp_13:>10.2e} | {VB_perp_75:>10.2e}")
        else:
            text_lines.append(f"{'VB||':<10} | {'n/a':>10} | {'n/a':>10}")
            text_lines.append(f"{'VB⊥':<10} | {'n/a':>10} | {'n/a':>10}")

        # --- Background photons (always computed) ---
        BG_13 = BG_photons.get(13e-6, np.nan)
        BG_75 = BG_photons.get(75e-6, np.nan)
        text_lines.append(f"{'BG':<10} | {BG_13:>10.2e} | {BG_75:>10.2e}")

        # --- SNR (only if VB present) ---
        if SNR:
            SNR_parr_13 = SNR.get(('VB_parr', 13e-6), np.nan)
            SNR_parr_75 = SNR.get(('VB_parr', 75e-6), np.nan)
            SNR_perp_13 = SNR.get(('VB_perp', 13e-6), np.nan)
            SNR_perp_75 = SNR.get(('VB_perp', 75e-6), np.nan)

            text_lines.append(f"{'SNR||':<10} | {SNR_parr_13:>10.2e} | {SNR_parr_75:>10.2e}")
            text_lines.append(f"{'SNR⊥':<10} | {SNR_perp_13:>10.2e} | {SNR_perp_75:>10.2e}")
        else:
            text_lines.append(f"{'SNR||':<10} | {'n/a':>10} | {'n/a':>10}")
            text_lines.append(f"{'SNR⊥':<10} | {'n/a':>10} | {'n/a':>10}")

        # Draw the text box
        table_text = "\n".join(text_lines)
        ax.text(
            0.98, 0.95, table_text,
            transform=ax.transAxes,
            fontsize=10,
            family="monospace",
            va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )



    # ─── Save main flow plot ───
    outdir = Path(project_dir) / "flows"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{file}_flowplot_{channel}.jpg"
    plt.savefig(outfile)
    print(f"[FLOW] Saved main flow plot to {outfile}")


    # ─── Save lightweight summary (.npz) ───────────────────────────────
    summary_path = Path(project_dir) / "pickles" / f"{file}_{channel}_summary.npz"

    try:
        shadow_data = {
            "SFA13": t13 if "t13" in locals() else np.nan,
            "SFA75": t75 if "t75" in locals() else np.nan,
            "Ratio": (t75/t13) if ("t13" in locals() and "t75" in locals()) else np.nan,
        }
    except Exception:
        shadow_data = {}

    # --- reset fixedfall to raw (unitless relative intensity) before saving ---
    # divide out the scale so the stored map is always in 'relative' units
    if 'scale' in locals() and scale != 1.0:
        fixedfall_save = fixedfall / scale
    else:
        fixedfall_save = fixedfall.copy()

    # --- metadata contains scaling factors for later use ---
    meta = {
        "file": file,
        "channel": channel,
        "unit": "relative",             # stored data is always relative
        "scale_ph": scale_ph,           # multiplier → photons / px or /m²
        "scale_Wcm2": scale_Wcm,        # multiplier → W/cm²
        "vertical_type": vertical_type,
        "title": f"{file} cut: {vertical_type}",
        "shadow_factors": shadow_data,
        "original_unit_selected": unit_sel,  # what user requested during this run
        "VB_photons": VB_photons,
        "BG_photons": BG_photons,
        "SNR": SNR,
    }

    np.savez_compressed(
        summary_path,
        zax=zax,
        gyax=gyax,
        fixedfall=fixedfall_save,
        meta=np.array(meta, dtype=object)
    )
    print(f"[FLOW] Saved lightweight summary → {summary_path} (stored in relative units)")


    return params, res, fixedfall







def overlay_flowplots(
    project_dir,
    file_stem,
    requested_channels=("main", "VB_perp"),
    unit_mode="photons",
    log_mode=True,
    figsize=(14, 8),
    custom_clims=None,
    shared_scale=False,
    alpha_min=1e2,
    alpha_max=1e4,
):
    """
    Create an overlay of flow plots from stored summary .npz files.

    Parameters
    ----------
    project_dir : str or Path
        Path to the project folder (same used for flow_plot outputs).
    file_stem : str
        Base name of the simulation (params['filename']).
    requested_channels : tuple
        Channels to overlay, e.g. ("main", "VB_perp").
    unit_mode : str
        Display unit ("relative", "photons", or "Wcm2").
    log_mode : bool
        Log scale for color maps.
    figsize : tuple
        Figure size.
    custom_clims : dict
        Optional manual color limits per channel, e.g. {"main": [1,1e20]}.
    shared_scale : bool
        Use one common color scale for all channels.
    alpha_min, alpha_max : float
        Range for alpha transparency (in data units).

    Saves
    -----
    A .jpg overlay figure to the `flows/` folder.
    """
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from pathlib import Path

    # --- defaults ---
    project_dir = Path(project_dir)
    custom_clims = custom_clims or {"main": [1, 1e20], "VB_perp": [10, 1e7]}

    # --- Colormaps ---
    _VIBE_SPECTRA = LinearSegmentedColormap.from_list(
        "VIBE_SPECTRA",
        ["#000000", "#2b0b3f", "#2043a8", "#00bcd4",
         "#7bd100", "#ffe800", "#ff8a00", "#e53935", "#ffffff"], N=256)
    CMAP_COOL = LinearSegmentedColormap.from_list(
        "VIBE_COOL", ["#000000", "#001f5c", "#00bcd4", "#4caf50", "#ffffff"], N=256)
    colormap_main, colormap_overlay = CMAP_COOL, _VIBE_SPECTRA

    def safe_limits(arr, desired, is_log, label=""):
        if desired and len(desired) == 2:
            vmin, vmax = map(float, desired)
        else:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return (1e-12, 1.0) if is_log else (0.0, 1.0)
            lo, hi = np.nanpercentile(finite, 1.0), np.nanpercentile(finite, 99.9)
            vmin, vmax = lo, hi
        if is_log:
            pos = arr[(arr > 0) & np.isfinite(arr)]
            if pos.size:
                vmin = max(vmin, np.nanmin(pos), 1e-16)
            else:
                vmin, vmax = 1e-12, 1.0
        if vmax <= vmin:
            vmax = vmin * (10 if is_log else 1.1)
        return vmin, vmax

    def build_alpha_map(data, a_min_lin, a_max_lin, is_log):
        tiny = 1e-30
        arr = np.array(data, dtype=float)
        if is_log:
            arr_eff = np.log10(np.clip(arr, tiny, None))
            a_min_eff = np.log10(max(a_min_lin, tiny))
            a_max_eff = np.log10(max(a_max_lin, a_min_lin * 1.000001))
        else:
            arr_eff, a_min_eff, a_max_eff = arr, a_min_lin, a_max_lin
        alpha = np.clip((arr_eff - a_min_eff) / (a_max_eff - a_min_eff), 0, 1)
        alpha[~np.isfinite(arr)] = 0
        if is_log:
            alpha[arr <= 0] = 0
        return alpha

    # --- Load summaries ---
    pickles_dir = project_dir / "pickles"
    found = sorted(glob.glob(str(pickles_dir / f"{file_stem}_*_summary.npz")))
    if not found:
        print(f"[OVERLAY] No summary files found for {file_stem}")
        return

    available = {Path(f).name[len(file_stem)+1:-len("_summary.npz")]: f for f in found}
    channels = [ch for ch in requested_channels if ch in available]

    summaries = {}
    for ch in channels:
        d = np.load(available[ch], allow_pickle=True)
        meta = d["meta"].item()
        if isinstance(meta, np.ndarray):
            meta = meta.item()
        stored_map = d["fixedfall"]
        stored_unit = meta.get("unit", "relative")
        scale_ph = meta.get("scale_ph", 1.0)
        conv = scale_ph if unit_mode == "photons" and stored_unit == "relative" else 1.0

        summaries[ch] = {
            "zax": d["zax"],
            "gyax": d["gyax"],
            "map": stored_map * conv,
            "unit_effective": unit_mode,
            "shadow_factors": meta.get("shadow_factors", {}),
            "VB_photons": meta.get("VB_photons", {}),
            "BG_photons": meta.get("BG_photons", {}),
            "SNR": meta.get("SNR", {}),
        }

    # --- Interpolate on shared z grid ---
    gyax = summaries[channels[0]]["gyax"]
    zmin = min(s["zax"].min() for s in summaries.values())
    zmax = max(s["zax"].max() for s in summaries.values())
    z_common = np.linspace(zmin, zmax, 1000)

    interp_maps = {}
    for ch in channels:
        zax, fmap = summaries[ch]["zax"], summaries[ch]["map"]
        interp_maps[ch] = np.array([np.interp(z_common, zax, col, left=np.nan, right=np.nan) for col in fmap.T]).T

    # --- Color limits ---
    per_channel_limits = {ch: safe_limits(interp_maps[ch], custom_clims.get(ch), log_mode, ch)
                          for ch in channels}
    if shared_scale:
        allvmin = min(v for v, _ in per_channel_limits.values())
        allvmax = max(v for _, v in per_channel_limits.values())
        per_channel_limits = {ch: (allvmin, allvmax) for ch in channels}

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    extent = [z_common.min(), z_common.max(), gyax.min(), gyax.max()]
    ims = []

    for ch in channels:
        vmin, vmax = per_channel_limits[ch]
        cmap = colormap_main if ch == "main" else colormap_overlay
        norm = LogNorm(vmin=vmin, vmax=vmax) if log_mode else None

        if ch == "main":
            im = ax.imshow(interp_maps[ch].T, extent=extent, aspect="auto",
                           origin="lower", cmap=cmap, norm=norm)
        else:
            alpha_img = build_alpha_map(interp_maps[ch], alpha_min, alpha_max, log_mode).T
            im = ax.imshow(interp_maps[ch].T, extent=extent, aspect="auto",
                           origin="lower", cmap=cmap, norm=norm, alpha=alpha_img)
        ims.append((ch, im))

    valid_mask = np.isfinite(interp_maps["main"])
    y_indices = np.any(valid_mask, axis=0)
    ymin_valid = gyax[np.where(y_indices)[0][0]]
    ymax_valid = gyax[np.where(y_indices)[0][-1]]
    ax.set_ylim(ymin_valid, ymax_valid)

    ax.set_xlabel("Propagation axis z [m]")
    ax.set_ylabel("Transverse axis y [μm]")
    ax.set_title(f"{file_stem}: Overlay of {' + '.join(channels)}")

    for ch, im in ims:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        label = f"{ch} intensity ({unit_mode})" + (" log₁₀" if log_mode else "")
        cb.set_label(label)

    # --- Compact table from metadata ---
    meta_main = summaries.get("main", {}).get("shadow_factors", {})
    VB_meta   = summaries.get("main", {}).get("VB_photons", {})
    BG_meta   = summaries.get("main", {}).get("BG_photons", {})
    SNR_meta  = summaries.get("main", {}).get("SNR", {})

    if any([meta_main, VB_meta, BG_meta, SNR_meta]):
        text_lines = [
            f"{'Quantity':<10} | {'13 µm':>10} | {'75 µm':>10}",
            "-" * 35,
        ]
        t13, t75 = meta_main.get("SFA13", np.nan), meta_main.get("SFA75", np.nan)
        text_lines.append(f"{'SF':<10} | {t13:>10.2e} | {t75:>10.2e}")

        vb_parr_13 = VB_meta.get("VB_parr", {}).get(13e-6, np.nan)
        vb_parr_75 = VB_meta.get("VB_parr", {}).get(75e-6, np.nan)
        vb_perp_13 = VB_meta.get("VB_perp", {}).get(13e-6, np.nan)
        vb_perp_75 = VB_meta.get("VB_perp", {}).get(75e-6, np.nan)
        text_lines += [
            f"{'VB||':<10} | {vb_parr_13:>10.2e} | {vb_parr_75:>10.2e}",
            f"{'VB⊥':<10} | {vb_perp_13:>10.2e} | {vb_perp_75:>10.2e}",
        ]
        bg_13, bg_75 = BG_meta.get(13e-6, np.nan), BG_meta.get(75e-6, np.nan)
        text_lines.append(f"{'BG':<10} | {bg_13:>10.2e} | {bg_75:>10.2e}")
        snr_parr_13 = SNR_meta.get(("VB_parr", 13e-6), np.nan)
        snr_parr_75 = SNR_meta.get(("VB_parr", 75e-6), np.nan)
        snr_perp_13 = SNR_meta.get(("VB_perp", 13e-6), np.nan)
        snr_perp_75 = SNR_meta.get(("VB_perp", 75e-6), np.nan)
        text_lines += [
            f"{'SNR||':<10} | {snr_parr_13:>10.2e} | {snr_parr_75:>10.2e}",
            f"{'SNR⊥':<10} | {snr_perp_13:>10.2e} | {snr_perp_75:>10.2e}",
        ]
        table_text = "\n".join(text_lines)
        ax.text(
            0.98, 0.95, table_text,
            transform=ax.transAxes,
            fontsize=10,
            family="monospace",
            va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
        )

    plt.tight_layout()
    outdir = project_dir / "flows"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{file_stem}_overlay.jpg"
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    print(f"[OVERLAY] Saved overlay → {outfile}")
    plt.close(fig)




# ======================================================================
#  SAVE ALL NON-FLOW PLANE FIGURES → project_dir/planes/<yaml_tag>/
# ======================================================================
def save_individual_plane_figures(project_dir, yaml_tag, *, save_units="photons"):
    """
    Load <yaml_tag>_figs.pickle and <yaml_tag>_res.pickle,
    loop over all non-flow planes, and save standalone images
    for each plane & channel.

    Output directory:
        <project_dir>/planes/<yaml_tag>/
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.colors import LogNorm

    # --------------------------------------------------------------
    # 1. Construct paths (pickle names follow VIBE conventions)
    # --------------------------------------------------------------
    pickle_figs = Path(project_dir) / "pickles" / f"{yaml_tag}_figs.pickle"
    pickle_res  = Path(project_dir) / "pickles" / f"{yaml_tag}_res.pickle"

    if not pickle_figs.exists():
        print(f"[save_planes] ERROR: missing figs pickle: {pickle_figs}")
        return
    if not pickle_res.exists():
        print(f"[save_planes] ERROR: missing res pickle: {pickle_res}")
        return

    # mu.loadPickle requires strings, not Path objects
    figs  = mu.loadPickle(str(pickle_figs))
    res   = mu.loadPickle(str(pickle_res))

    # res pickle is (elements_dict, params)
    try:
        params = res[1]
    except Exception:
        print("[save_planes] ERROR: invalid res pickle format")
        return

    individual_fig_log = float(params.get("individual_fig_log", 1))

    # intensity scaling factors
    scale_ph   = params.get("scale_phot", 1.0)
    scale_Wcm2 = params.get("scale_Wcm2", 1.0)

    if save_units == "photons":
        scale = scale_ph
        unit_label = "photons / m²"
    elif save_units == "Wcm2":
        scale = scale_Wcm2
        unit_label = "W / m²"
    else:
        scale = 1.0
        unit_label = "a.u."

    # --------------------------------------------------------------
    # 2. Create output directory
    # --------------------------------------------------------------
    out_dir = Path(project_dir) / "planes" / yaml_tag
    mu.mkdir(str(out_dir), 0)
    print(f"[save_planes] Saving plane images into:\n   {out_dir}")

    # --------------------------------------------------------------
    # 3. Loop over all stored planes
    # --------------------------------------------------------------
    for key, data in figs.items():

        # skip flow planes
        if key.startswith("flow"):
            continue

        # key = "O1_main", "CRL4c_VB_perp", etc.
        plane_name = key

        # retrieve data = [im, element_index, propsize/ZoomFactor, z]
        try:
            img      = data[0].astype(float)
            propsize = float(data[2])        # physical box size [m]
            z_pos    = float(data[3])
        except Exception:
            print(f"[save_planes] WARNING: malformed entry for key '{key}' → skipping")
            continue

        # apply scaling
        img_scaled = img * scale

        # build extent in microns
        extent = (-propsize/2*1e6, propsize/2*1e6,
                  -propsize/2*1e6, propsize/2*1e6)

        # ----------------------------------------------------------
        # Plot
        # ----------------------------------------------------------
        plt.figure(figsize=(9, 7))

        # Select normalization: log (default) or linear
        norm = None

        if individual_fig_log:  
            # ---- LOG SCALE ----
            pos = img_scaled[img_scaled > 0]
            if pos.size > 0:
                vmin = max(np.nanmin(pos), 1e-30)
                vmax = np.nanmax(pos)
                norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            # ---- LINEAR SCALE ----
            # Clip negative values to 0 for display (optional)
            img_scaled = np.clip(img_scaled, a_min=0, a_max=None)
            norm = None


        im_plot = plt.imshow(
            img_scaled,
            cmap=rofl.cmap(),
            origin="lower",
            extent=extent,
            aspect="equal",
            norm=norm
        )

        plt.colorbar(im_plot, label=f"Intensity [{unit_label}]")
        plt.xlabel("x [µm]")
        plt.ylabel("y [µm]")
        plt.title(f"{yaml_tag} — {plane_name}")

        plt.tight_layout()

        # ----------------------------------------------------------
        # Save image
        # ----------------------------------------------------------
        out_file = out_dir / f"{plane_name}.png"
        plt.savefig(out_file, dpi=200)
        plt.close()

        print(f"[save_planes] saved: {out_file}")

    print("[save_planes] DONE exporting all individual planes.")





def flow_savefig(I, out_dir, idx, propsize, label, position,
                 flow_plot_crange=1e-5):
    """
    Save a log-scaled flow map image with fixed zoom and color range.

    The function generates two images:
    - a full view with a reference box overlay
    - a zoomed-in view centered on the beam

    Parameters
    ----------
    I : 2D ndarray
        Flow intensity map.
    out_dir : str or Path
        Output directory for saved figures.
    idx : int
        Frame index used in filenames.
    propsize : float
        Physical size of the simulation box [m].
    label : str
        Text label shown in the figure title.
    position : float
        Longitudinal position (used for annotation) [m].
    flow_plot_crange : float, optional
        Relative lower bound of the log color scale.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    out_dir = str(out_dir)
    out_full = f"{out_dir}/ff_{idx:04d}.png"
    out_zoom = f"{out_dir}/fixed_{idx:04d}.png"

    # --- Prepare coordinates ---
    npix = I.shape[0]
    pxsize_um = propsize * 1e6 / npix
    x_um = (np.arange(npix) - npix / 2) * pxsize_um

    # --- Color scale ---
    Imax = np.max(I)
    clim = [Imax * flow_plot_crange, Imax]

    # --- Plot ---
    mu.figure(14, 10, safe=1)

    I_plot = I.T  # match plotting convention
    mu.pcolor(
        I_plot,
        xc=x_um,
        yc=x_um,
        ticks=0,
        log=1,
        cl=clim,
        background=[0, 0, 0]
    )

    plt.axis("equal")
    plt.xlabel("X [µm]")
    plt.ylabel("Y [µm]")
    plt.title(f"{label}, {position*100:.0f} cm")

    # --- Reference box (100 µm) ---
    box_half = 50
    plt.plot(
        [-box_half, -box_half, box_half, box_half, -box_half],
        [-box_half,  box_half, box_half, -box_half, -box_half],
        "r.", markersize=7
    )

    # --- Save full view ---
    plt.savefig(out_full)

    # --- Zoomed-in view ---
    zoom_half = 50
    plt.xlim(-zoom_half, zoom_half)
    plt.ylim(-zoom_half, zoom_half)
    plt.savefig(out_zoom)



# =========================================================
# Air scattering related functions
# =========================================================

def build_symmetric_kernel_from_particles(x_particles, y_particles, e_particles, Initial_energy_Geant4, N, propsize, nbins=401, smooth_sigma=4.0, plot_debug=False):
    
    """
    Build a smooth, rotationally symmetric scattering kernel from Geant4 particle hits.

    Parameters:
    - x_particles, y_particles: arrays of particle hit positions [µm]
    - N: output resolution (LightPipes grid)
    - propsize: LightPipes physical window size [m]
    - nbins: number of radial bins
    - smooth_sigma: Gaussian filter sigma (in bins)
    - log_bins: if True, use logarithmic binning to better resolve the center
    - plot_debug: plot radial profile and final kernel if True

    Returns:
    - kernel_2D: (N x N) normalized scattering kernel
    """
    
    # Compute the radius of each particle
    r_particles = np.sqrt(x_particles**2 + y_particles**2)   # in [um]
    
    # Convert propsize [m] to micrometers
    half_size_um = propsize * 1e6 / 2
    r_max = 2**0.5 * half_size_um

    # Filter particles within the simulation window
    valid = r_particles <= r_max
    r_particles = r_particles[valid]
    e_particles = e_particles[valid]
    
    # Step 2: define bins
    r_bins = np.linspace(0, r_max, nbins + 1) # in [um]

    Energy_weights_radial = e_particles / Initial_energy_Geant4  # only keep weights for selected particles
    radial_hist, _ = np.histogram(r_particles, bins = r_bins, weights = Energy_weights_radial)

    bin_areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2) #in [um^2]
    radial_density = radial_hist / bin_areas  # [particles / µm²]
    
    # Smoothing part
    if smooth_sigma is None or smooth_sigma == 0:
        radial_density_smooth = radial_density.copy()
    else:
        n_unsmoothed = 1 # Number of points to exclude from smoothing
        tail_smoothed = gaussian_filter1d(radial_density[n_unsmoothed:], sigma=smooth_sigma) # Create a smoothed version of the tail of the profile
        radial_density_smooth = np.concatenate([ radial_density[:n_unsmoothed], tail_smoothed]) # Concatenate unsmoothed + smoothed parts


    pixel_size_um = 2 * half_size_um / N
    linspace_um = pixel_size_um * (np.arange(N) - N // 2)

    X, Y = np.meshgrid(linspace_um, linspace_um)
    R_grid = np.sqrt(X**2 + Y**2)
    kernel_2D = np.interp(R_grid, r_bins[:-1], radial_density_smooth, left=0, right=0)

    return kernel_2D, radial_hist, r_bins, radial_density , radial_density_smooth




def apply_air_scattering_and_debug_plot(F, params, propsize, N, plot_debug = False, use_symmetric_kernel = False, compute_transmission = False, test_identity_kernel = False , crop_to_odd = True):

    """
    Applies Geant4 air scattering to the LightPipes field via convolution.
    
    Parameters:
    - F: LightPipes field object
    - params: dict containing at least 'projectdir'
    - propsize: field size in meters (LightPipes simulation window)
    - N: number of pixels in LightPipes grid
    - plot_debug: if True, show and save a comparison figure
    - use_symmetric_kernel: if True, build rotationally symmetric smoothed kernel
    
    Returns:
    - I_after_air: convolved intensity (2D numpy array)
    """
    import time
    start_time = time.time() #starts the timer
    
    basepath = Path(params["projectdir"]).parent
    data = np.load(basepath / "Air_scattering/xray_Primaries2e9_50umKapton_Air_stats.npz")
    x_particles = data["x"] # in um
    y_particles = data["y"] # in um
    e_particles = data["e"] # in keV
    Initial_energy_Geant4 = 8.8 # in keV
    
    half_size_um = propsize * 1e6 / 2

    # Step 1: Force identity kernel first if requested
    if test_identity_kernel:
        kernel_2D = np.zeros((N, N))
        kernel_2D[N // 2, N // 2] = 1.0
        use_symmetric_kernel = 0  # Prevent rebuilding later
        compute_transmission = 0  # Prevent scaling

        
    if use_symmetric_kernel:
        kernel_2D, radial_hist, r_bins, radial_density , radial_density_smooth = build_symmetric_kernel_from_particles(
            x_particles, y_particles, e_particles, Initial_energy_Geant4,  N = N, propsize = propsize,
            nbins=1001, smooth_sigma=4.0, plot_debug=False)
    else:
        kernel_2D, _, _ = np.histogram2d(x_particles, y_particles, bins=N, range=[[-half_size_um, half_size_um], [-half_size_um, half_size_um]], 
                                         weights = e_particles / Initial_energy_Geant4 )
        radial_hist = radial_density = radial_density_smooth = None

    I_lp = F    

    # Step 2: Crop both kernel and image together, if needed
    if crop_to_odd:
        kernel_2D = croping_to_odd(kernel_2D)
        I_lp = croping_to_odd(I_lp)


    if not test_identity_kernel:
        kernel_2D /= np.sum(kernel_2D) #normalizing such that the sum is 1
    
        nb_particles_after_scattering = len(x_particles) # total number of particles ending on the screen after scattering
        nb_primaries = 2e9 #number of primary particles (nb of particles intialised in the simulation)

        transmission_factor = nb_particles_after_scattering / nb_primaries #percentage of particles making it through
    
        if compute_transmission:
            kernel_2D *= transmission_factor  # Take into account that some particles are absorbed by air

    I_after_air = fftconvolve(I_lp, kernel_2D, mode='same') #convolve the image with the Kernel

    I_after_air = restore_even_shape_by_duplication(I_after_air, (N, N)) # add back a line and a row to make it (NxN) again.
    #I_lp = restore_even_shape_by_duplication(I_lp, (N, N)) # add back a line and a row to make it (NxN) again.

    # Set the values outside the disk to 0 after the convolution
    Ymask, Xmask = np.indices(I_after_air.shape)
    rmask = np.sqrt((Xmask - N//2)**2 + (Ymask - N//2)**2)
    mask = rmask <= (N//2)  # or a more precise radius
    I_after_air[~mask] = 0

    if plot_debug:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].scatter(x_particles, y_particles, s=1, alpha=0.3)
        axes[0, 0].set_xlim(-half_size_um, half_size_um)
        axes[0, 0].set_ylim(-half_size_um, half_size_um)
        axes[0, 0].set_title("Geant4 raw scatter (scatter plot)")
        axes[0, 0].set_xlabel("x [µm]")
        axes[0, 0].set_ylabel("y [µm]")
        axes[0, 0].grid(True)
        axes[0, 0].set_aspect("equal")

        if use_symmetric_kernel and r_bins is not None:
            axes[0, 1].plot(r_bins[:-1], radial_density)
            axes[0, 1].set_title("Radial histogram (nb particles at a given r / area)")
            axes[0, 1].set_xlabel("r [µm]")
            axes[0, 1].set_ylabel("Photons/area")
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)

            axes[0, 2].plot(r_bins[:-1], radial_density_smooth)
            axes[0, 2].set_title("Radial density smoothed (nb particles / area at a given r)")
            axes[0, 2].set_xlabel("r [µm]")
            axes[0, 2].set_ylabel("Photons/area")
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)

        extent_um = [-half_size_um, half_size_um, -half_size_um, half_size_um]  # [µm]
        
        im3 = axes[1, 0].imshow(kernel_2D, cmap='inferno', norm=LogNorm(), extent=extent_um, origin='lower')
        axes[1, 0].set_title("2D Geant4 kernel")
        fig.colorbar(im3, ax=axes[1, 0])
        
        im4 = axes[1, 1].imshow(I_lp, cmap='inferno', norm=LogNorm(), extent=extent_um, origin='lower')
        axes[1, 1].set_title("LightPipes Intensity")
        fig.colorbar(im4, ax=axes[1, 1])
        
        im5 = axes[1, 2].imshow(I_after_air, cmap='inferno', norm=LogNorm(), extent=extent_um, origin='lower')
        axes[1, 2].set_title("After convolution")
        fig.colorbar(im5, ax=axes[1, 2])
        
        for ax in axes[1, :]:
            ax.set_xlabel("x [µm]")
            ax.set_ylabel("y [µm]")

        plt.tight_layout()
        plt.savefig(basepath / "AirScattering_DebugPanel_6plots.png", dpi=300)
        plt.show()

    elapsed_time = time.time() - start_time  # End timer
    print(f"Air scattering + convolution took {elapsed_time/60:.2f} minutes.")
    
    return I_after_air




# =========================================================
# Pump laser creation related functions
# =========================================================


def airy_disk_map(L, N,  P_peak, lam=800e-9, f=0.10, D=0.10, return_grid=False, debug=False):
    """
    Build a 2-D Airy-disk intensity map on a square grid of side-length L.

    The pixel count (N x N) and the physical window size (L x L) are
    supplied directly, so the pattern is binned identically to any other
    field defined on the same grid.

    Parameters
    ----------
    L : float
        Physical width of the square window [m].
    N : int
        Number of samples along each axis (array is NxN).
    lam : float, default 800e-9
        Wavelength [m].
    f : float,  default 0.10
        Focal length [m].
    D : float,  default 0.10
        Aperture diameter [m].
    return_grid : bool, default False
        If True, also return the centred 1-D coordinate vector `x` [m].
    P_peak : float
        Power of the IR Relax laser. Nominal is 200 TW. Maximum is 300 TW.

    Returns
    -------
    airy : ndarray  (N x N)
        Airy disk intensity map (normalised so peak = 1).
    x : ndarray, optional
        Coordinate vector (metres), length N, centred on 0.
    """
    dx = L / N
    x  = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    r   = np.hypot(X, Y)

    kr  = (np.pi * D * r) / (lam * f)
    A   = np.ones_like(r)
    mask = r != 0
    A[mask] = 2.0 * j1(kr[mask]) / kr[mask]
    airy = A**2  # PSF

    # Normalize so that integral ≈ 1 (finite window approximation)
    norm = np.sum(airy) * dx * dx
    if norm <= 0:
        raise RuntimeError("Airy normalization failed.")
    airy /= norm

    # Convert to intensity [W/cm^2]; treat airy as spatial distribution of power
    I_W_cm2 = (P_peak * airy) / 1e4

    I_max = float(I_W_cm2.max())
    a0_pk = a0_from_I_lambda(I_max, lam)

    if debug:
        plt.figure()
        plt.imshow(I_W_cm2, extent=[x[0]*1e6, x[-1]*1e6, x[0]*1e6, x[-1]*1e6], origin='lower')
        plt.xlabel("x [μm]"); plt.ylabel("y [μm]")
        plt.title("Airy: I [W/cm²]"); plt.colorbar(label="Intensity [W/cm²]")
        plt.show()
        print(f"[Airy] ∫ airy dx dy ≈ {np.sum(airy)*dx*dx:.6f}")
        print(f"[Airy] Peak intensity: {I_max:.3e} W/cm² ; a0_peak(λ={lam*1e9:.0f} nm) ≈ {a0_pk:.3f}")

    return (I_W_cm2, x) if return_grid else I_W_cm2







def gaussian_spot_map(
    L, N, fwhm_diameter,                # FWHM of the spot *diameter* [m]
    P_peak, x_offset=0.0, y_offset=0.0,
    return_grid=False, debug=False,
    debug_outdir=None, sim_label=None
):
    """
    2-D circular Gaussian intensity map on an LxL window, NxN samples.
    Normalized to integrate to 1, then scaled by P_peak. Returns I in W/cm^2.
    Uses I(r) ∝ exp(-2 r^2 / w0^2) with w0 = FWHM / sqrt(2 ln 2).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Grid
    dx = L / N
    x  = (np.arange(N) - N/2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')

    if fwhm_diameter <= 0:
        raise ValueError("fwhm_diameter must be > 0.")
    w0 = fwhm_diameter / np.sqrt(2*np.log(2))  # 1/e^2 radius

    # Center shift
    Xs = X - float(x_offset)
    Ys = Y - float(y_offset)
    r  = np.hypot(Xs, Ys)

    # Properly normalized 2D Gaussian that integrates to 1
    G = (2.0 / (np.pi * w0**2)) * np.exp(-2.0 * (r**2) / (w0**2))

    # Intensity in W/cm^2
    I_W_cm2 = (P_peak * G) / 1e4

    if debug:
        # --------- Debug figure (2D + lineout) ----------
        lambda_IR = 800e-9  # only for setting a reasonable zoom
        x_um = x * 1e6
        extent_um = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

        fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(10, 4.2))
        im = ax2d.imshow(I_W_cm2, extent=extent_um, origin='lower')
        ax2d.set_xlabel("x [µm]"); ax2d.set_ylabel("y [µm]")
        ax2d.set_title("Gaussian: I [W/cm²]")
        cb = plt.colorbar(im, ax=ax2d); cb.set_label("W/cm²")

        # Plot requested center
        ax2d.plot([x_offset*1e6], [y_offset*1e6], 'wo', ms=5, mec='k', mew=0.8)

        # Zoom window (±4 µm)
        lim_um = 4.0
        ax2d.set_xlim(-lim_um, lim_um)
        ax2d.set_ylim(-lim_um, lim_um)

        # 1D lineout and FWHM
        j = N//2
        x_line = x.copy()
        I_line = I_W_cm2[:, j]
        Imax   = float(I_line.max())
        half   = 0.5 * Imax
        i0     = int(np.argmax(I_line))

        # Left crossing
        il = np.where(I_line[:i0] < half)[0]
        if il.size:
            k  = il[-1]
            xL = x_line[k] + (half - I_line[k]) * (x_line[k+1]-x_line[k]) / (I_line[k+1]-I_line[k])
        else:
            xL = x_line[0]
        # Right crossing
        ir = np.where(I_line[i0:] < half)[0]
        if ir.size:
            k2 = i0 + ir[0] - 1
            xR = x_line[k2] + (half - I_line[k2]) * (x_line[k2+1]-x_line[k2]) / (I_line[k2+1]-I_line[k2])
        else:
            xR = x_line[-1]

        fwhm_meas = (xR - xL)
        fwhm_theo = float(fwhm_diameter)

        ax1d.plot(x_line*1e6, I_line, lw=1.6)
        ax1d.axhline(half, color='gray', ls='--', lw=1)
        ax1d.axvline(xL*1e6, color='gray', ls=':', lw=1)
        ax1d.axvline(xR*1e6, color='gray', ls=':', lw=1)
        ax1d.set_xlabel("x [µm]"); ax1d.set_ylabel("I [W/cm²]")
        ax1d.set_title("Central lineout & FWHM check")
        ax1d.text(
            0.05, 0.95,
            f"Peak = {Imax:.3e} W/cm²\n"
            f"FWHM (meas) = {fwhm_meas*1e6:.3f} µm\n"
            f"FWHM (theo) = {fwhm_theo*1e6:.3f} µm",
            transform=ax1d.transAxes, ha='left', va='top'
        )
        ax1d.grid(True)
        ax1d.set_xlim(-lim_um, lim_um)

        # Title + save path
        if sim_label:
            fig.suptitle(f"{sim_label} — Gaussian spot debug", y=0.98)

        # Where to save: VB_figures by default, unless debug_outdir is provided
        outdir = Path(debug_outdir) if debug_outdir else Path("/home/yu79deg/darkfield_p5438/VIBE_outputs/VB_figures")
        outdir.mkdir(parents=True, exist_ok=True)

        label = sim_label or "gaussian"
        fig.tight_layout()
        plt.savefig(outdir / f"{label}_gaussian_debug.png", dpi=300)
        plt.close(fig)

        # Console checks
        print(f"[Gauss] ∫G dxdy (numeric) ≈ {np.sum(G)*dx*dx:.6f}")
        print(f"[Gauss] Peak intensity: {float(I_W_cm2.max()):.3e} W/cm²")
        print(f"[Gauss] FWHM (meas) = {fwhm_meas*1e6:.3f} µm ; FWHM (theo) = {fwhm_theo*1e6:.3f} µm")
        print(f"[Gauss] Saved: {outdir / f'{label}_gaussian_debug.png'}")

    return (I_W_cm2, x) if return_grid else I_W_cm2




def peak_power(P_peak=None, E=None, tau_FWHM=None):
    """
    Return peak power [W]. If P_peak is given, use it.
    Otherwise compute it from pulse energy E [J] and FWHM duration tau_FWHM [s]
    assuming a Gaussian temporal envelope:
        E = P0 * tau_FWHM * sqrt(pi / (4 ln 2))
      => P0 = E * sqrt(4 ln 2 / pi) / tau_FWHM
    """
    if P_peak is not None:
        return float(P_peak)
    if (E is not None) and (tau_FWHM is not None):
        return float(E * np.sqrt(4*np.log(2)/np.pi) / tau_FWHM)
    raise ValueError("Provide P_peak or (E and tau_FWHM).")




def a0_from_I_lambda(I_W_cm2, lam_m):
    """
    a0 ≈ 0.855 * sqrt(I[10^18 W/cm^2]) * λ[μm]
    """
    lam_um = lam_m * 1e6
    return 0.855 * np.sqrt(I_W_cm2 * 1e-18) * lam_um







def Interference_IR_map(F, x, Xg, Yg, params, P_peak):
    """
    Build a full IR laser focus map (W/cm²) using a 2D Fourier transform of a
    partially blocked circular near-field, then resample it to the simulation grid.

    REQUIREMENT:
        params['IR_2Dmap'][1] == "match_integral"
        Otherwise this raises an error.

    Parameters
    ----------
    F   : Field object with F.N and F.grid_size
    x   : 1D simulation grid array [m]
    Xg, Yg : 2D simulation grid [m]
    params : parameter dictionary
    P_peak : full-disk peak power [W]

    Returns
    -------
    I_sim_W_cm2 : (F.N, F.N) intensity map on simulation grid [W/cm²]
    x            : 1D simulation grid [m]
    """

    # ---------------------------------------------------------
    # 0) Check normalization requirement
    # ---------------------------------------------------------
    IRsp = params.get("IR_2Dmap", [])
    if len(IRsp) < 2 or str(IRsp[1]).lower() != "match_integral":
        raise ValueError("Interference_IR_map requires IR_2Dmap norm_policy == 'match_integral'.")

    # ---------------------------------------------------------
    # 1) Physical parameters
    # ---------------------------------------------------------
    lam  = float(params.get("IR_wavelength", 800e-9))
    fnum = float(params.get("f_number", 1.0))
    frac = float(params.get("fraction_blocked", 0.0))
    f    = float(params.get("IR_focal_length", params.get("IR_focal_length_m", 0.2)))

    D = f / fnum        # disk diameter [m]
    R = D / 2.0
    d_wire = frac * D        # blocker width

    print(f"[InterfIR] λ={lam:.4g}, f={f:.4g}, f/#={fnum}, D={D:.4g}, dwire={d_wire:.4g}")

    # ---------------------------------------------------------
    # 2) Build near-field (high-res grid for FFT)
    # ---------------------------------------------------------
    N2D  = 2048
    L_nf = 5 * D
    dx_nf = L_nf / N2D

    x_nf = (np.arange(N2D) - N2D/2) * dx_nf
    y_nf = (np.arange(N2D) - N2D/2) * dx_nf
    Xm_nf, Ym_nf = np.meshgrid(x_nf, y_nf)

    # Masks
    mask_disk  = (Xm_nf**2 + Ym_nf**2 <= R**2)
    mask_block = (np.abs(Xm_nf) <= d_wire/2)
    mask_trans = mask_disk & (~mask_block)

    # Pixel area in cm²
    dx_nf_cm2 = (dx_nf * 100)**2

    area_disk_cm2  = mask_disk.sum()  * dx_nf_cm2
    area_trans_cm2 = mask_trans.sum() * dx_nf_cm2
    eta = area_trans_cm2 / area_disk_cm2
    print(f"[InterfIR] transmitted fraction η = {eta:.4f}")

    # Field amplitude so full disk carries P_peak
    E0 = np.sqrt(P_peak / area_disk_cm2)

    # Complex near field
    E_nf = np.zeros_like(Xm_nf, dtype=complex)
    E_nf[mask_trans] = E0

    I_nf = np.abs(E_nf)**2
    P_trans = I_nf.sum() * dx_nf_cm2
    print(f"[InterfIR] P_transmitted = {P_trans:.4e} W")

    # ---------------------------------------------------------
    # 3) FAR FIELD via 2D FFT
    # ---------------------------------------------------------
    E_far = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(E_nf))
    ) * dx_nf**2

    I_far_raw = np.abs(E_far)**2

    fx = np.fft.fftshift(np.fft.fftfreq(N2D, d=dx_nf))
    fy = np.fft.fftshift(np.fft.fftfreq(N2D, d=dx_nf))
    FX, FY = np.meshgrid(fx, fy)

    x_foc = lam * f * FX
    y_foc = lam * f * FY

    # pixel area in cm²
    dx_foc = x_foc[0,1] - x_foc[0,0]
    dy_foc = y_foc[1,0] - y_foc[0,0]
    dx_foc_cm2 = (dx_foc * 100) * (dy_foc * 100)

    P_far_raw = I_far_raw.sum() * dx_foc_cm2
    I_far = I_far_raw * (P_trans / P_far_raw)

    # Axes in µm for interpolation
    x_um = x_foc[0, :] * 1e6
    y_um = y_foc[:, 0] * 1e6

    # ---------------------------------------------------------
    # 4) Super-resolution of far field (×4 or ×5)
    # ---------------------------------------------------------
    up = 5   # leads to ~10240 × 10240 grid

    interp2 = RectBivariateSpline(y_um, x_um, I_far)

    x_um_hi = np.linspace(x_um.min(), x_um.max(), len(x_um)*up)
    y_um_hi = np.linspace(y_um.min(), y_um.max(), len(y_um)*up)

    I_far_hi = interp2(y_um_hi, x_um_hi)

    # Renormalize after interpolation
    dx_hi = (x_um_hi[1] - x_um_hi[0]) * 1e-4
    dy_hi = (y_um_hi[1] - y_um_hi[0]) * 1e-4
    A_hi = dx_hi * dy_hi

    P_hi_raw = I_far_hi.sum() * A_hi
    I_far_hi *= (P_trans / P_hi_raw)

    print(f"[InterfIR] super-resolution grid: {I_far_hi.shape}")
    print(f"[InterfIR] P_far(super-res) = {I_far_hi.sum()*A_hi:.4e} W")

    # Convert super-resolution axes back to m
    x_hi_m = x_um_hi * 1e-6
    y_hi_m = y_um_hi * 1e-6

    # ---------------------------------------------------------
    # 5) RESAMPLE super-res far field onto simulation grid
    # ---------------------------------------------------------
    print("[InterfIR] Resampling to simulation grid...")

    interp_sim = RegularGridInterpolator(
        (y_hi_m, x_hi_m), I_far_hi,
        bounds_error=False, fill_value=0.0
    )

    pts = np.column_stack([Yg.ravel(), Xg.ravel()])
    I_sim = interp_sim(pts).reshape(F.N, F.N)

    # Power normalization (ensure P_sim = P_trans)
    dx_sim = F.grid_size / F.N   # [m]
    A_sim = (dx_sim * 100)**2    # cm² per pixel

    P_sim_raw = I_sim.sum() * A_sim
    I_sim *= (P_trans / P_sim_raw)

    print(f"[InterfIR] P_sim(after resample) = {I_sim.sum()*A_sim:.4e} W")

    # ---------------------------------------------------------
    # RETURN SIMULATION-GRID MAP
    # ---------------------------------------------------------
    return I_sim, x





def build_ir_focus_map(IR_spatial_params: list,
                       F,
                       params: dict,
                       P_peak: float,
                       x_off: float = 0.0,
                       y_off: float = 0.0):
    """
    Build IR focus intensity map on the simulation grid, from either a
    Gaussian descriptor or an external image.

    IR_spatial_params (params['IR_2Dmap']):
        ["gaussian" | "external", norm_policy, path, calib_um_per_px]
        - gaussian: ignores path/calib, uses params['IR_FWHM_gaussian'].
        - external: 'norm_policy' controls scaling. Currently supported:
            * "match_integral" → ∫I dA = P_peak (spatial PDF × P_peak)
              using provided calibration (μm/px) or the image extent
              inferred from calib.
    Returns:
        I_W_cm2 : (N,N) array, intensity at focus in W/cm^2
        x       : (N,) array, physical x-grid [m]
        fwhm_diam_used : float, effective FWHM diameter [m]
    """
    mode = str(IR_spatial_params[0]).lower() if IR_spatial_params else "gaussian"

    # Target grid (meters), centered
    x = np.linspace(-F.grid_size/2, F.grid_size/2, F.N)
    Xg, Yg = np.meshgrid(x, x, indexing="xy")
    fwhm_diam_used = None

    def _fwhm_diameter(A: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """
        Estimate effective FWHM *diameter* of a spot in A on a rect. grid (x_coords, y_coords),
        using a peak-centered radial profile with a 1D fallback.
        Returns diameter in the same units as x_coords/y_coords.
        """
        A = np.asarray(A, dtype=np.float64)
        if A.size == 0 or not np.isfinite(A).any() or A.max() <= 0:
            return float("nan")

        # Peak location and center coordinates
        iy, ix = np.unravel_index(np.argmax(A), A.shape)
        xc, yc = float(x_coords[ix]), float(y_coords[iy])

        # Radial profile (vectorized binning)
        Xg, Yg = np.meshgrid(x_coords, y_coords, indexing="xy")
        r = np.hypot(Xg - xc, Yg - yc)
        In = A / (A.max() + 1e-300)

        n_bins = 200
        rb = np.linspace(0.0, float(r.max()), n_bins + 1)
        bins = np.digitize(r.ravel(), rb) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        sums = np.bincount(bins, weights=In.ravel(), minlength=n_bins)
        cnts = np.bincount(bins, minlength=n_bins)
        prof = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
        rmid = 0.5 * (rb[:-1] + rb[1:])

        hit = np.where((prof[:-1] >= 0.5) & (prof[1:] < 0.5))[0]
        if hit.size:
            k = int(hit[0])
            y1, y2 = prof[k], prof[k+1]
            x1, x2 = rmid[k], rmid[k+1]
            r_half = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
            if np.isfinite(r_half) and r_half > 0:
                return float(2.0 * r_half)

        # Fallback: 1D FWHM along row/col through the peak → geometric mean
        def _fwhm_1d(y, coord):
            y = np.asarray(y, float)
            if y.max() <= 0: return float("nan")
            y = y / (y.max() + 1e-300)
            i0 = int(np.argmax(y)); half = 0.5
            L = np.where(y[:i0] < half)[0]
            if L.size:
                kL = L[-1]
                xL = coord[kL] + (half - y[kL]) * (coord[kL+1] - coord[kL]) / (y[kL+1] - y[kL] + 1e-300)
            else:
                xL = coord[0]
            R = np.where(y[i0:] < half)[0]
            if R.size:
                kR = i0 + R[0] - 1
                xR = coord[kR] + (half - y[kR]) * (coord[kR+1] - coord[kR]) / (y[kR+1] - y[kR] + 1e-300)
            else:
                xR = coord[-1]
            return float(xR - xL)

        fx = _fwhm_1d(A[iy, :], x_coords)
        fy = _fwhm_1d(A[:, ix], y_coords)
        if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
            return float(np.sqrt(fx * fy))
        return float("nan")

    # ----------------- GAUSSIAN -----------------
    if mode == "gaussian":
        FWHM_diam = float(params.get("IR_FWHM_gaussian", 1.3e-6))

        I_W_cm2, _ = gaussian_spot_map(
            F.grid_size, F.N,
            fwhm_diameter=FWHM_diam,
            P_peak=P_peak,
            return_grid=True,
            debug=True,
            debug_outdir=Path(params["projectdir"]) / "VB_figures",
            sim_label=params.get("filename", "")
        )
        # optional offsets via resampling (kept as before)
        if abs(x_off) > 0.0 or abs(y_off) > 0.0:
            interp = RegularGridInterpolator((x, x), I_W_cm2, bounds_error=False, fill_value=0.0)
            pts = np.column_stack([(Yg - y_off).ravel(), (Xg - x_off).ravel()])
            I_W_cm2 = interp(pts).reshape(F.N, F.N)


        fwhm_diam_used = FWHM_diam

    # ----------------- FRINGES (analytical interference) -----------------
    elif mode == "fringes":

        # --- Call the 2D FFT computation (returns simulation-grid map) ---
        I_W_cm2, x = Interference_IR_map(
            F = F,
            x = x,
            Xg = Xg,
            Yg = Yg,
            params = params,
            P_peak = P_peak
        )

        # --- No offsets applied here (handled inside the function) ---

        # --- Compute FWHM from existing helper ---
        try:
            fwhm_diam_used = _fwhm_diameter(I_W_cm2, x, x)
        except Exception:
            fwhm_diam_used = float(params.get("IR_FWHM_gaussian", 1.3e-6))

    elif mode == "external":
        # ----------------- EXTERNAL IMAGE -----------------
        # IR_spatial_params = ["external", norm_policy, path, calib_um_per_px]
        norm_policy    = str(IR_spatial_params[1]).lower() if len(IR_spatial_params) > 1 and IR_spatial_params[1] else "match_integral"
        path_img       = IR_spatial_params[2] if len(IR_spatial_params) > 2 else None
        calib_um_per_px = IR_spatial_params[3] if len(IR_spatial_params) > 3 else None

        if not path_img:
            raise ValueError("IR_2Dmap 'external' requires a valid image path at IR_2Dmap[2].")

        # --- Load grayscale ---
        try:
            from imageio.v2 import imread
            img = imread(path_img)
        except Exception:
            from PIL import Image
            img = np.array(Image.open(path_img).convert("L"))

        # --- Sanitize external image (no NaN/Inf/negatives) ---
        img = _sanitize_intensity_map(img)

        Ny, Nx = img.shape

        # --- Pixel size from calibration (μm/px) OR scale-to-Gaussian-FWHM when == 0 ---
        if calib_um_per_px is None:
            raise ValueError("External IR map: provide calibration (IR_2Dmap[3] in μm/px) or 0 to scale to IR_FWHM_gaussian.")

        calib_val = float(calib_um_per_px)
        if calib_val > 0.0:
            dpx = calib_val * 1e-6  # [m/px]
            dx_m = dy_m = dpx
        else:
            # Measure FWHM on the *raw image* in pixel units and choose dpx to match IR_FWHM_gaussian
            fwhm_target_m = float(params.get("IR_FWHM_gaussian", 1.3e-6))
            # pixel coordinate axes (units: px)
            xs_px = np.arange(Nx, dtype=float)
            ys_px = np.arange(Ny, dtype=float)
            fwhm_img_px = _fwhm_diameter(img, xs_px, ys_px)  # [px]
            if not np.isfinite(fwhm_img_px) or fwhm_img_px <= 0:
                raise ValueError("Could not measure FWHM on the external IR image to scale-to-Gaussian.")
            dpx = fwhm_target_m / fwhm_img_px  # [m/px] so that FWHM_px * dpx = target
            dx_m = dy_m = dpx
            print(f"[IR-map] calib=0 → scale-to-Gaussian-FWHM: FWHM_img≈{fwhm_img_px:.2f} px → "
                f"dpx={dpx*1e6:.3f} µm/px (target {fwhm_target_m*1e6:.3f} µm)")

        # Build source physical axes for interpolation (meters), centered on image center
        xs = (np.arange(Nx) - (Nx - 1)/2.0) * dx_m
        ys = (np.arange(Ny) - (Ny - 1)/2.0) * dy_m
        
        # Interpolate onto simulation grid with requested offsets
        interp = RegularGridInterpolator((ys, xs), img, bounds_error=False, fill_value=0.0, method="linear")
        pts = np.column_stack([(Yg - y_off).ravel(), (Xg - x_off).ravel()])
        img_resampled = interp(pts).reshape(F.N, F.N)

        # ---- Recenter peak to (x_off, y_off) (kept as you fixed) ----
        center_peak = bool(params.get("IR_center_peak", True))
        if center_peak:
            iy, ix = np.unravel_index(np.argmax(img_resampled), img_resampled.shape)
            x_peak, y_peak = float(x[ix]), float(x[iy])
            dx_shift, dy_shift = (x_peak - x_off), (y_peak - y_off)
            pts = np.column_stack([(Yg - y_off + dy_shift).ravel(),
                                (Xg - x_off + dx_shift).ravel()])
            img_resampled = interp(pts).reshape(F.N, F.N)

        print(f"Normalisation of the IR laser map = {norm_policy}")
        # ---- Normalization policy ----
        if norm_policy == "match_integral":
            # Spatial density s(x,y) with ∫ s dA = 1 using ORIGINAL pixel area
            total_counts = img.sum()
            if total_counts <= 0:
                raise ValueError(f"External IR map '{path_img}' has zero/negative sum.")
            px_area = dx_m * dy_m
            s_density = img_resampled / (total_counts * px_area)   # [1/m^2]
            I_W_m2  = P_peak * s_density
            I_W_cm2 = I_W_m2 / 1e4

        elif norm_policy == "match_peak":
            # Peak-match to the Gaussian spot with IR_FWHM_gaussian
            fwhm_gauss = float(params.get("IR_FWHM_gaussian", 1.3e-6))
            w0 = fwhm_gauss / np.sqrt(2.0*np.log(2.0))              # [m]
            I_target_peak_Wcm2 = (P_peak * (2.0 / (np.pi*w0*w0))) / 1e4  # [W/cm^2]

            vmax = float(np.max(img_resampled))
            if vmax <= 0.0 or not np.isfinite(vmax):
                raise ValueError("External IR map peak is zero/invalid after resampling; cannot match peak.")

            shape_peak_norm = img_resampled / vmax                  # unitless, peak=1
            I_W_cm2 = I_target_peak_Wcm2 * shape_peak_norm          # enforce desired peak

            # (optional sanity print)
            dx_sim = F.grid_size / F.N
            P_on_grid = np.nansum(I_W_cm2) * dx_sim * dx_sim * 1e4  # back to W
            print(f"[IR-map] match_peak: I_peak={I_target_peak_Wcm2:.3e} W/cm²; "
                f"∫I dA on grid = {P_on_grid:.3e} W (P_peak={P_peak:.3e} W)")


        else:
            # Future options land here (e.g., "match_fwhm", "match_energy_and_peak" via compromise, etc.)
            raise NotImplementedError(f"Unknown IR external norm_policy: '{norm_policy}'")


        # --- Effective FWHM diameter (on the simulation grid) ---
        try:
            fwhm_diam_used = _fwhm_diameter(I_W_cm2, x, x)
            if not np.isfinite(fwhm_diam_used) or fwhm_diam_used <= 0:
                # Fallback: 1/4 of the smallest full span covered by mapping
                full_span_x = Nx * dx_m
                full_span_y = Ny * dy_m
                fwhm_diam_used = float(0.25 * min(full_span_x, full_span_y))
        except Exception:
            full_span_x = Nx * dx_m
            full_span_y = Ny * dy_m
            fwhm_diam_used = float(0.25 * min(full_span_x, full_span_y))


    # ---------- Debug figure (2D + lineout) ----------
    try:
        dbg_dir = Path(params["projectdir"]) / "VB_figures"
        dbg_dir.mkdir(parents=True, exist_ok=True)

        # Axes in microns for display
        x_um = x * 1e6
        extent_um = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

        # Find peak (after resampling + offsets)
        iy, ix = np.unravel_index(np.argmax(I_W_cm2), I_W_cm2.shape)
        x_peak, y_peak = x[ix], x[iy]

        # Prepare figure
        fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(10, 4.2))

        # --- 2D map
        im = ax2d.imshow(I_W_cm2, extent=extent_um, origin="lower")
        ax2d.set_xlabel("x [µm]"); ax2d.set_ylabel("y [µm]")
        ax2d.set_title("IR focus: I [W/cm²]")
        cb = plt.colorbar(im, ax=ax2d); cb.set_label("W/cm²")

        # Mark requested offsets and measured peak
        ax2d.plot([x_off*1e6], [y_off*1e6], 'wo', ms=5, mec='k', mew=0.8, label="requested offset")
        ax2d.plot([x_peak*1e6], [y_peak*1e6], 'rx', ms=6, mew=1.2, label="peak")
        ax2d.legend(loc="upper right", frameon=True)

        # Zoom to a ±4 µm window around the **peak** (robust for off-center beams)
        lim_um = 4.0
        ax2d.set_xlim(x_peak*1e6 - lim_um, x_peak*1e6 + lim_um)
        ax2d.set_ylim(y_peak*1e6 - lim_um, y_peak*1e6 + lim_um)

        # --- 1D lineout through the peak row (horizontal profile)
        x_line = x.copy()
        I_line = I_W_cm2[iy, :]  # horizontal line at the peak y
        Imax   = float(I_line.max())
        half   = 0.5 * Imax

        # Locate half-maximum crossings around the peak index
        i0 = int(np.argmax(I_line))

        # Left crossing
        il = np.where(I_line[:i0] < half)[0]
        if il.size:
            kL  = il[-1]
            xL  = x_line[kL] + (half - I_line[kL]) * (x_line[kL+1] - x_line[kL]) / (I_line[kL+1] - I_line[kL] + 1e-300)
        else:
            xL  = x_line[0]

        # Right crossing
        ir = np.where(I_line[i0:] < half)[0]
        if ir.size:
            kR  = i0 + ir[0] - 1
            xR  = x_line[kR] + (half - I_line[kR]) * (x_line[kR+1] - x_line[kR]) / (I_line[kR+1] - I_line[kR] + 1e-300)
        else:
            xR  = x_line[-1]

        fwhm_meas = (xR - xL)  # [m]

        # Plot the lineout
        ax1d.plot(x_line*1e6, I_line, lw=1.6)
        ax1d.axhline(half, color='gray', ls='--', lw=1)
        ax1d.axvline(xL*1e6, color='gray', ls=':', lw=1)
        ax1d.axvline(xR*1e6, color='gray', ls=':', lw=1)
        ax1d.set_xlabel("x [µm]"); ax1d.set_ylabel("I [W/cm²]")
        ax1d.set_title("Lineout through peak (y = y_peak)")
        ax1d.grid(True)

        # Make the x-limits match the 2D zoom window (centered on peak)
        ax1d.set_xlim(x_peak*1e6 - lim_um, x_peak*1e6 + lim_um)

        # Annotate
        ax1d.text(
            0.05, 0.95,
            f"Peak = {Imax:.3e} W/cm²\n"
            f"FWHM (meas) = {fwhm_meas*1e6:.3f} µm",
            transform=ax1d.transAxes, ha='left', va='top'
        )

        # Title + save
        label = params.get("filename", "run")
        fig.suptitle(f"{label} — IR spot debug", y=0.98)
        fig.tight_layout()
        out_png = dbg_dir / f"{label}_IRmap_resampled.png"
        plt.savefig(out_png, dpi=300)
        plt.close(fig)

        # Console checks (use the original pixel-area normalization)
        print(f"[IR-map] Peak intensity: {Imax:.3e} W/cm²")
        print(f"[IR-map] FWHM (meas along x@peak) = {fwhm_meas*1e6:.3f} µm")
        print(f"[IR-map] Saved: {out_png}")

    except Exception as e:
        # keep silent in batch but don’t block the run
        print(f"[IR-map] Debug plotting failed: {e}")

    # =========================================================
    # SAVE IR LASER MAP TO PICKLE FOLDER (for later plotting)
    # =========================================================
    try:
        # Prepare data for saving
        out_dir = Path(params["projectdir"]) / "pickles"
        out_dir.mkdir(parents=True, exist_ok=True)

        label = params.get("filename", "IR_focus")
        out_path = out_dir / f"{label}_IRlaser_focus_map.npz"

        # Axes and extent
        x_m = x
        y_m = x
        extent_um = [x_m[0]*1e6, x_m[-1]*1e6, y_m[0]*1e6, y_m[-1]*1e6]

        np.savez_compressed(
            out_path,
            I_W_cm2=I_W_cm2,
            x=x_m,
            y=y_m,
            extent_um=extent_um,
            fwhm_diam_used=fwhm_diam_used,
            mode=mode,
            P_peak=P_peak
        )

        print(f"[IR-map] Saved IR laser focus map → {out_path.name}")
    except Exception as e:
        print(f"[IR-map] Warning: could not save IR map to pickle folder: {e}")


    return I_W_cm2, x, fwhm_diam_used





def _fwhm_diameter_generic(A: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> float:
    """
    Effective FWHM *diameter* of a spot in A on rect. grid (x_coords, y_coords).
    Peak-centered radial profile with 1D fallback. Returns same units as coords.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0 or not np.isfinite(A).any() or A.max() <= 0:
        return float("nan")
    iy, ix = np.unravel_index(np.argmax(A), A.shape)
    xc, yc = float(x_coords[ix]), float(y_coords[iy])

    Xg, Yg = np.meshgrid(x_coords, y_coords, indexing="xy")
    r = np.hypot(Xg - xc, Yg - yc)
    In = A / (A.max() + 1e-300)

    n_bins = 200
    rb = np.linspace(0.0, float(r.max()), n_bins + 1)
    bins = np.digitize(r.ravel(), rb) - 1
    bins = np.clip(bins, 0, n_bins - 1)
    sums = np.bincount(bins, weights=In.ravel(), minlength=n_bins)
    cnts = np.bincount(bins, minlength=n_bins)
    prof = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
    rmid = 0.5 * (rb[:-1] + rb[1:])

    hit = np.where((prof[:-1] >= 0.5) & (prof[1:] < 0.5))[0]
    if hit.size:
        k = int(hit[0])
        y1, y2 = prof[k], prof[k+1]
        x1, x2 = rmid[k], rmid[k+1]
        r_half = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1 + 1e-300)
        if np.isfinite(r_half) and r_half > 0:
            return float(2.0 * r_half)

    # Fallback: geometric mean of horizontal / vertical FWHM through the peak
    def _fwhm_1d(y, coord):
        y = np.asarray(y, float)
        if y.max() <= 0: return float("nan")
        y = y / (y.max() + 1e-300)
        i0 = int(np.argmax(y)); half = 0.5
        L = np.where(y[:i0] < half)[0]
        if L.size:
            kL = L[-1]
            xL = coord[kL] + (half - y[kL]) * (coord[kL+1] - coord[kL]) / (y[kL+1] - y[kL] + 1e-300)
        else:
            xL = coord[0]
        R = np.where(y[i0:] < half)[0]
        if R.size:
            kR = i0 + R[0] - 1
            xR = coord[kR] + (half - y[kR]) * (coord[kR+1] - coord[kR]) / (y[kR+1] - y[kR] + 1e-300)
        else:
            xR = coord[-1]
        return float(xR - xL)

    fx = _fwhm_1d(A[iy, :], x_coords)
    fy = _fwhm_1d(A[:, ix], y_coords)
    if np.isfinite(fx) and np.isfinite(fy) and fx > 0 and fy > 0:
        return float(np.sqrt(fx * fy))
    return float("nan")




def build_external_shaper_mask(el_dict: dict, F, params: dict) -> np.ndarray:
    """
    Build a transmission mask (0..1) from an external image to shape the
    *intensity* after the mask to match the image (up to a global factor).

    YAML (under a 'type: aperture' element):
      shape: external_map
      path: "/abs/path/to/Xray_initial_beam.png"
      calibration: <μm/px>      # if 0, scale the image to FWHM='size'
      size: <FWHM_diameter_m>   # used only when calibration == 0
      center: "barycenter" | "max" | "no"
      center_sigma_px: 2.0       # (only for center="max") smoothing in px
      x_offset_m: 0.0            # desired final beam center (meters)
      y_offset_m: 0.0
      # or in microns:
      # x_offset_um: 0.0
      # y_offset_um: 0.0
    """
    path = el_dict.get("path", None)
    calib_um_per_px = el_dict.get("calibration", None)
    if not path:
        raise ValueError("beam_shaper.external_map needs a 'path' to an image.")
    if calib_um_per_px is None:
        raise ValueError("beam_shaper.external_map needs 'calibration' (μm/px), or 0 to use 'size' as FWHM.")

    # -- load + sanitize (make grayscale, drop NaN/Inf, clamp negatives)
    try:
        from imageio.v2 import imread
        img = imread(path)
    except Exception:
        from PIL import Image
        img = np.array(Image.open(path))  # color or L, both fine

    img = _sanitize_intensity_map(img) 
    Ny, Nx = img.shape

    # -- pixel scale
    calib_val = float(calib_um_per_px)
    if calib_val > 0.0:
        dpx = calib_val * 1e-6  # [m/px]
    else:
        target_fwhm = float(el_dict.get("size", 0.0))
        if target_fwhm <= 0:
            raise ValueError("With calibration=0 you must provide a positive 'size' (FWHM diameter in meters).")
        xs_px = np.arange(Nx, dtype=float)
        ys_px = np.arange(Ny, dtype=float)
        fwhm_px = _fwhm_diameter_generic(img, xs_px, ys_px)
        if not np.isfinite(fwhm_px) or fwhm_px <= 0:
            raise ValueError("Failed to measure FWHM on external map for calibration=0 scaling.")
        dpx = target_fwhm / fwhm_px
        print(f"[beam_shaper] calib=0 → scale-to-FWHM: img≈{fwhm_px:.2f}px → dpx={dpx*1e6:.3f} µm/px (target {target_fwhm*1e6:.3f} µm)")

    # -- source axes (meters) centered on image center
    xs = (np.arange(Nx) - (Nx - 1) / 2.0) * dpx
    ys = (np.arange(Ny) - (Ny - 1) / 2.0) * dpx

    # -- destination (simulation) grid
    dx = F.siz / F.N
    x  = (np.arange(F.N) - F.N / 2) * dx
    Xg, Yg = np.meshgrid(x, x, indexing="xy")

    # -------- centering & offsets ---------------------------------
    center_mode = str(el_dict.get("center", "no")).lower()

    # desired final center (default 0)
    x_off_m = float(el_dict.get("x_offset_m", 0.0))
    y_off_m = float(el_dict.get("y_offset_m", 0.0))
    if "x_offset_um" in el_dict: x_off_m = float(el_dict["x_offset_um"]) * 1e-6
    if "y_offset_um" in el_dict: y_off_m = float(el_dict["y_offset_um"]) * 1e-6

    center_x = 0.0
    center_y = 0.0

    if center_mode == "barycenter":
        Xs, Ys = np.meshgrid(xs, ys, indexing="xy")
        w = img
        tot = float(w.sum())
        if tot > 0.0:
            center_x = float((w * Xs).sum() / tot)
            center_y = float((w * Ys).sum() / tot)
    elif center_mode == "max":
        try:
            from scipy.ndimage import gaussian_filter
            sigma_px = float(el_dict.get("center_sigma_px", 2.0))
            G = gaussian_filter(img, sigma=sigma_px) if sigma_px > 0 else img
        except Exception:
            G = img
        iy, ix = np.unravel_index(np.argmax(G), G.shape)
        center_x = float(xs[ix]); center_y = float(ys[iy])
    elif center_mode in ("no", "none", "false", "0"):
        pass
    else:
        raise ValueError("beam_shaper.center must be 'barycenter', 'max' or 'no'.")

    # Shift needed in source coordinates so that final center = (x_off_m, y_off_m)
    dx_center = x_off_m - center_x
    dy_center = y_off_m - center_y
    # --------------------------------------------------------------

    # -- resample the *shifted* map to the simulation grid
    interp = RegularGridInterpolator((ys, xs), img, bounds_error=False, fill_value=0.0)
    T = interp(np.column_stack([(Yg - dy_center).ravel(), (Xg - dx_center).ravel()])).reshape(F.N, F.N)
    T = np.maximum(T, 0.0)

    # -- build intensity transmission so that (M * I_in) ∝ T (shape match)
    I_in = Intensity(0, F)
    eps = 1e-300
    M0 = T / (I_in + eps)                 # desired ratio in intensity units
    mmax = float(np.nanmax(M0))
    if not np.isfinite(mmax) or mmax <= 0:
        return np.zeros_like(I_in)
    M = np.clip(M0 / mmax, 0.0, 1.0)      # cap so it's a *transmission* (≤1)

    return M



# =========================================================
# Compound Refractive Lenses (CRLs) related functions
# =========================================================



def CRL4_get_length(number_of_lenses,Energy):
    """
    Compute the focal length of a CRL4 stack for a given number of lenses and photon energy.
    """
    f=CRL_get_length(0.05,number_of_lenses,Energy)
    return f



def CRL_get_length(radius_mm,number,Energy):
    """
    Estimate the focal length of a compound refractive lens (CRL).

    Uses a simple refractive-index model to compute the effective focal length
    from the lens radius, number of lenses, and photon energy.
    """
    dn=340/Energy**2
    n=1+dn
    f1=radius_mm/2/(n-1)
    f_calc=f1/number*1e-3
    return f_calc #focal length [m]




def handle_custom_CRL(F: Field,
                      el_dict: dict,
                      params: dict,
                      projectdir: Path) -> Field:
    """
    Build transmission & phase maps for a ‘Custom_CRL’ element
    and apply them to *F*.  Also creates the 2-D / 1-D diagnostic
    plots (optional aperture included).
    Returns the modified field.
    """
    # ────────────────────────────────────────────────
    # 1. Read geometry from YAML ---------------------
    # ────────────────────────────────────────────────
    ROC       = yamlval('ROC',   el_dict, None)                           # Radius of curvature of the parabolic surface          [m]
    L         = yamlval('L',     el_dict, None)                           # Total mechanical lens thickness                       [m]
    A         = yamlval('A',     el_dict, None)                           # Geometric aperture radius                             [m]
    t_wall    = yamlval('twall', el_dict, None)                           # Minimal lens thickness (apex-to-apex)                 [m]
    nb_lenses = yamlval('nb_lenses',   el_dict, None)                     # Number of lenses in the CRL stack
    focal_lenght_stack = yamlval('focal_lenght_stack',   el_dict, None)   # Focal length of the stack                             [m]
    add_aperture = yamlval('add_aperture',   el_dict, 0)                  # If we want to add a circular aperture around the lens [Boolean]


    # ────────────────────────────────────────────────
    #  Derive missing geometric parameters
    # ────────────────────────────────────────────────
    if A is None and t_wall is not None:
        # Case 1: given wall thickness, derive aperture
        A = 2.0 * np.sqrt(ROC * (L - t_wall))

    elif t_wall is None and A is not None:
        # Case 2: given aperture, assume default t_wall = 30 µm and recompute L (to match the parabollic equation)
        t_wall = 30e-6  # [m] default wall thickness
        L = t_wall + (A**2) / (4.0 * ROC)
        print(f"[INFO] t_wall not provided → defaulting to 30 µm, recomputed L = {L*1e3:.3f} mm")

    elif A is not None and t_wall is not None:
        # Case 3: both given → verify consistency
        L_expected = t_wall + (A**2) / (4.0 * ROC)
        if abs(L_expected - L) > 0.05 * L:
            print(f"[WARN] Given A={A*1e6:.0f} µm and t_wall={t_wall*1e6:.0f} µm "
                f"imply L≈{L_expected*1e3:.3f} mm, differs from input L={L*1e3:.3f} mm")
    else:
        raise ValueError("Custom_CRL: must provide either (A) or (twall), and ROC, L.")

        
    print(f"[CRL parameters] : ROC = {ROC*1e6:.0f} um, L = {L*1e3:.1f} mm, Diameter Aperture A = {A*1e6:.0f} um, Apex thickness t_wall = {t_wall*1e6:.0f} um")


    # ────────────────────────────────────────────────
    # 2. Optical constants ---------------------------
    # ────────────────────────────────────────────────
    E_eV         = params['photon_energy']                   # photon energy [eV]
    wavelength_m = h * c / (E_eV * e)                        # λ = h·c / E

    lens_material     = yamlval('lens_material', el_dict, 'Be')   # default material is set to Be
    beta, delta       = get_index(lens_material, E_eV)            # from Henke tables (https://henke.lbl.gov/optical_constants/getdb2.html)
    print(f"delta = {delta}, beta={beta}")

    phase_per_m       = -2.0 * np.pi * delta / wavelength_m       # radians of phase delay per meter
    absorption_factor = 4.0 * np.pi * beta / wavelength_m         # absorption exponent scale

    if focal_lenght_stack is None and nb_lenses is not None:
        focal_lenght_stack = ROC / (2 * nb_lenses * delta)       # focal length calculated from the number of lenses in the stack [m]
    elif focal_lenght_stack is not None and nb_lenses is None:
        nb_lenses = ROC / (2 * delta * focal_lenght_stack)      # number of lenses (could be a not integer number) needed to achieve a given focal length.
    else:
        raise ValueError("Custom_CRL: provide *either* focal_lenght_stack or nb_lenses (not both).")

    print(f"Number of lenses = {nb_lenses}, focal lenght of the stack = {focal_lenght_stack} m")
    
    # ────────────────────────────────────────────────
    # 3. Mesh & thickness map ------------------------
    # ────────────────────────────────────────────────
    N   = params['N']
    Na  = (np.arange(N) -  N // 2) * params['pxsize']       # axis in meters
    
    xm, ym = np.meshgrid(Na, Na)                        # xm, ym in [m]
    r = np.sqrt(xm**2 + ym**2)

    lens_half_thickness = np.full_like(r, L / 2)                # default to max thickness (outside aperture)
    core_mask = r < A / 2 
    lens_half_thickness[core_mask] = (xm[core_mask]**2 + ym[core_mask]**2) / (2 * ROC) + t_wall / 2
    lens_thickness = 2 * lens_half_thickness

    # ────────────────────────────────────────────────
    # 4) Build aperture (optional) -------------------
    # ────────────────────────────────────────────────
    if add_aperture == 1:
        custom_ap = {'elem': 'Hf', 'thickness': 0.0001, 'shape': 'circle',
                    'size': A, 'invert': 1}
        Aperture_transmission, _ = doap(custom_ap, params)
        #F = MultIntensity(Aperture_transmission, F)  # aperture first

    # ────────────────────────────────────────────────
    # 5) Ideal transmission & phase ------------------
    # ────────────────────────────────────────────────
    transmission_map = np.exp(-nb_lenses * absorption_factor * lens_thickness)   # intensity attenuation
    ideal_phase_map  = nb_lenses * phase_per_m * lens_thickness                  # phase delay [rad]

    # ────────────────────────────────────────────────
    # 6) Defect phase (DABAM), then combine ----------
    # ────────────────────────────────────────────────
    total_phase_map = ideal_phase_map  # start with ideal
    if int(el_dict.get("defects", 0)) == 1:
        Z_def = build_dabam_defect_thickness_map(
            el_dict=el_dict, params=params, xm=xm, ym=ym, A_m=A
        )  # thickness delta [m] on (xm, ym) within aperture

        defect_phase_map = phase_per_m * Z_def  # convert to phase [rad]
        total_phase_map  = ideal_phase_map + defect_phase_map

    # ────────────────────────────────────────────────
    # 7) Apply exactly once --------------------------
    # ────────────────────────────────────────────────
    shift_x_um = float(el_dict.get("offset_x_um", 0.0))
    shift_y_um = float(el_dict.get("offset_y_um", 0.0))

    if shift_x_um != 0.0 or shift_y_um != 0.0:
        shift_x = shift_x_um * 1e-6 / params["pxsize"]
        shift_y = shift_y_um * 1e-6 / params["pxsize"]

        print(f"[Custom_CRL] Shifting full lens map by dx={shift_x_um:.1f} µm, dy={shift_y_um:.1f} µm")

        transmission_map = imshift(
            transmission_map, shift=(shift_y, shift_x),
            order=1, mode='constant', cval=1.0
        )
        total_phase_map = imshift(
            total_phase_map, shift=(shift_y, shift_x),
            order=1, mode='constant', cval=0.0
        )

        if add_aperture == 1:
            Aperture_transmission = imshift(
                Aperture_transmission,
                shift=(shift_y, shift_x),
                order=1, mode='constant', cval=1.0
            )

    # ────────────────────────────────────────────────
    # 8. NOW apply the shifted maps to the field
    # ────────────────────────────────────────────────
    if add_aperture == 1:
        F = MultIntensity(Aperture_transmission, F)    

    F = MultIntensity(transmission_map, F)
    F = MultPhase(total_phase_map, F)

    # ------------ PLOTS ------------
    # ---- build filename stem -------
    lens_name = el_dict.get("element_name") or el_dict.get("name") or el_dict.get("label") or "Lens"
    sim_name  = params.get("filename") or params.get("sim_name") or params.get("name") or Path(projectdir).stem
    stem = f"{sim_name}_{lens_name}"   # e.g. LP_610_CRL4a
    save_dir = Path(projectdir)        # plotting functions save under /Lens_diags



    # Image 1: ideal CRL (no defects)
    plot_custom_crl_ideal(
        projectdir=projectdir,
        Na=Na,
        nb_lenses=nb_lenses,
        lens_thickness=lens_thickness,
        transmission_map=transmission_map,
        total_phase_map_ideal=ideal_phase_map,  # note: ideal only
        add_aperture=add_aperture,
        Aperture_transmission=(Aperture_transmission if add_aperture else None),
        lens_material=lens_material,
        E_eV=E_eV,
        wavelength_m=wavelength_m,
        filename_stem=stem,
        save_dir=save_dir,
        sim_name=sim_name,
        lens_name=lens_name,
    )


    # Image 2: DABAM diagnostics (only if defects requested)
    if int(el_dict.get("defects", 0)) == 1:
        import vibe.wavefront_fitting as wft

        # Try to use the cached arrays produced by build_dabam_defect_thickness_map
        X      = el_dict.get("dabam_X", None)
        Y      = el_dict.get("dabam_Y", None)
        Zraw   = el_dict.get("Z_raw_native", None)
        Zfit_p = el_dict.get("Z_fit_panel_map", None)    # custom when Custom_zernike=on
        Zres   = el_dict.get("Z_residues_native", None)

        # If any are missing, fall back to recomputing (rare)
        if X is None or Y is None or Zraw is None or Zfit_p is None or Zres is None:
            def _parse_one_idx(tok):
                if isinstance(tok, (int, np.integer)):
                    return int(tok)
                if isinstance(tok, str):
                    s = tok.strip()
                    m = re.search(r"dabam2d-([0-9]+)$", s, flags=re.IGNORECASE)
                    if m:
                        return int(m.group(1))
                    nums = re.findall(r"([0-9]+)", s)
                    if nums:
                        return int(nums[-1])
                return None

            dabam_sel = el_dict.get("experimental_data", None)
            idx = None
            if isinstance(dabam_sel, (list, tuple)):
                for item in dabam_sel:
                    idx = _parse_one_idx(item)
                    if idx is not None:
                        print(f"[DABAM][PLOT] Fallback recompute: using first index from list → {idx}")
                        break
            else:
                idx = _parse_one_idx(dabam_sel)

            if idx is None:
                print("[DABAM][PLOT][SKIP] Could not parse an index from 'experimental_data' "
                      f"(value={dabam_sel!r}). Skipping DABAM plot for this lens.")
                return F

            # We have an idx → recompute a standalone fit for display
            X, Y, Zraw, _ = load_dabam2d(idx)
            nmodes    = int(el_dict.get("nmodes", 37))
            startmode = int(el_dict.get("startmode", 1))

            Zcoeffs, Zfit_LSQ, Zres = wft.fit_zernike_circ(
                Zraw, nmodes=nmodes, startmode=startmode, rec_zern=True
            )
            Zres = -Zres

            # Optional removal (for display)
            remove_avg_flag = int(el_dict.get("remove_avg_profile", 0))
            if remove_avg_flag == 1:
                I_thick_res, R = wft.average_azimuthal(Zres, X, Y)
                _, Zres = wft.remove_avg_profile(Zres, None, X, Y, I_thick_res, R, 'b')
            el_dict["remove_avg_applied"] = remove_avg_flag

            # The “fit” panel falls back to LSQ fit when we don’t have the cache
            Zfit_p = Zfit_LSQ


        # Build label state for the bar plot
        custom_active = (el_dict.get("Zfit_is_custom", 0) == 1)
        custom_pairs  = list(el_dict.get("Custom_zernike", [])[1:]) if custom_active else []

        # Zernike coeffs for the decomposition bars (LSQ of DABAM map)
        Zcoeffs = el_dict.get("Z_coeffs_native", None)
        # If somehow missing, fall back to the recompute path above (keeps behavior robust)
        if Zcoeffs is None:
            nmodes    = int(el_dict.get("nmodes", 37))
            startmode = int(el_dict.get("startmode", 1))
            Zcoeffs, _, _ = wft.fit_zernike_circ(Zraw, nmodes=nmodes, startmode=startmode, rec_zern=True)


        # --- robustly retrieve dabam_sets and always call the stack plot ---
        dabam_sets = el_dict.get("dabam_sets", [])
        def _looks_like_dabam_list(v):
            try:
                return (isinstance(v, list)
                        and len(v) >= 1
                        and isinstance(v[0], dict)
                        and {"Z_raw", "Zfit", "Zres"}.issubset(set(v[0].keys())))
            except Exception:
                return False

        # If empty, scan other keys (in case the builder stored under another name)
        if not dabam_sets:
            for k, v in el_dict.items():
                if _looks_like_dabam_list(v):
                    print(f"[DABAM][RESOLVE] Found dabam_sets under el_dict['{k}'] (len={len(v)})")
                    dabam_sets = v
                    break

        # Log what we found
        try:
            lens_name = el_dict.get("element_name") or el_dict.get("name") or el_dict.get("label") or "Lens"
            print(f"[DABAM][DIAG] lens={lens_name} | type(dabam_sets)={type(dabam_sets)} "
                f"| len={len(dabam_sets) if hasattr(dabam_sets,'__len__') else 'n/a'}")
            if isinstance(dabam_sets, list) and dabam_sets:
                print(f"[DABAM][DIAG] first-set keys: {list(dabam_sets[0].keys())}")
        except Exception as exc:   # ← renamed here
            print(f"[DABAM][DIAG][WARN] logging dabam_sets failed: {exc}")

        # Always call the stack plot (it prints how many maps it got and still saves totals even if 0/1)
        try:
            print(f"[PLOT][CALL] Invoking plot_dabam_stack_diagnostics for lens={lens_name}")
            plot_dabam_stack_diagnostics(
                projectdir=projectdir,
                el_dict=el_dict,
                dabam_sets=dabam_sets,
                filename_stem=stem,
                save_dir=save_dir,
                sim_name=sim_name,
                lens_name=lens_name,
                totals_vmin=None,
                totals_vmax=None,
            )
            print(f"[PLOT][DONE] plot_dabam_stack_diagnostics finished for {lens_name}")
        except Exception as exc:   # ← and here
            print(f"[PLOT][EXC] plot call failed for {lens_name}: {exc}")

        # 2b) Bars-only Zernike figure
        try:
            plot_zernike_bars(
                projectdir=projectdir,
                el_dict=el_dict,
                filename_stem=stem,      # gives: LP_610_CRL4b_ZernikePolynoms.png
                save_dir=save_dir,
                sim_name=sim_name,
                lens_name=lens_name,
                nmodes=int(el_dict.get("nmodes", 37)),
                startmode=int(el_dict.get("startmode", 1)),
            )
        except Exception as exc:
            print(f"[ZERNIKE][EXC] plot_zernike_bars failed: {exc}")


    return F







def plot_dabam_stack_diagnostics(projectdir, el_dict, dabam_sets,
                                 filename_stem, save_dir, sim_name, lens_name,
                                 totals_vmin=None, totals_vmax=None):
    """
    Per-map rows (Raw / Zernike fit / Residues) +
    totals row (Total raw / Total fit / Total residues pre-remove) +
    last row: Low frequency / High frequency / Total (Low+High).

    Highlight panel that feeds the simulation with the new semantics:
      defect_type=1 → Low; 2 → High; 3 → Total.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from pathlib import Path

    print(f"[DEBUG] Entering plot_dabam_stack_diagnostics for {lens_name}")
    print(f"[DEBUG] → {len(dabam_sets)} map(s)")

    defect_type = int(el_dict.get("defect_type", 3))         # 1=LF, 2=HF, 3=LF+HF
    remove_avg  = int(el_dict.get("remove_avg_profile", 0))
    custom_list = el_dict.get("Custom_zernike", [])
    custom_active = (isinstance(custom_list, (list, tuple))
                     and len(custom_list) >= 1 and int(custom_list[0]) == 1)

    # cached totals
    Zraw_tot  = el_dict["Z_raw_native"]
    Zfit_tot  = el_dict["Z_fit_panel_map"]
    Zres_post = el_dict["Z_residues_native"]
    Zres_pre  = el_dict.get("Z_residues_native_unproc", None)
    if Zres_pre is None:
        if Zraw_tot.shape == Zfit_tot.shape:
            Zres_pre = Zraw_tot - Zfit_tot
        else:
            print("[DABAM][plot] Z_residues_native_unproc missing; using zeros.")
            Zres_pre = np.zeros_like(Zfit_tot)

    # NEW LF/HF/Total (ROI)
    Z_low  = el_dict.get("Z_lowfreq_roi",  Zfit_tot)
    Z_high = el_dict.get("Z_highfreq_roi", Zres_post)
    Z_totL = el_dict.get("Z_total_lf_hf_roi", Z_low + Z_high)

    # extents
    extent_fit     = el_dict.get("Z_fit_axes_um", None)
    ext_total_full = el_dict.get("Z_interp_extent_um", None)

    # layout
    nrows = len(dabam_sets) + 2  # per-map + totals + last (LF/HF/Total)
    fig, axes = plt.subplots(nrows, 3, figsize=(12, 3.5*nrows))
    cmap = "GnBu"

    def emphasize(ax, extra=" (used for simulation)"):
        ax.set_title(ax.get_title() + extra, color="red", fontweight="bold")
        for s in ax.spines.values():
            s.set_edgecolor("red")
            s.set_linewidth(2)

    # per-map rows (native extents if available)
    for i, d in enumerate(dabam_sets):
        ext = d.get("extent_um", None)
        for j, (lbl, arr) in enumerate(zip(
            ["Raw", "Zernike fit", "Residues"],
            [d["Z_raw"], d["Zfit"], d["Zres"]]
        )):
            ax = axes[i, j]
            im = ax.imshow(arr * 1e6, origin="lower", cmap=cmap,
                           extent=(ext if ext else None))
            ax.set_title(f"{lbl} — dabam2d-{d['idx']:04d}")
            if ext:
                ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
            fig.colorbar(im, ax=ax, fraction=0.046)

    # totals row
    row_tot = len(dabam_sets)
    tot_defs = [
        ("Total raw (interpolated) [µm]", Zraw_tot, ext_total_full),
        ("Total fit",                      Zfit_tot, extent_fit),
        ("Total residues (pre-remove)",    Zres_pre, extent_fit),
    ]
    tot_axes = []
    for j, (lbl, arr, ext) in enumerate(tot_defs):
        ax = axes[row_tot, j]
        im = ax.imshow(arr * 1e6, origin="lower", cmap=cmap,
                       extent=(ext if ext else None),
                       vmin=(totals_vmin if totals_vmin is not None else None),
                       vmax=(totals_vmax if totals_vmax is not None else None))
        ax.set_title(lbl)
        if ext:
            ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]")
        fig.colorbar(im, ax=ax, fraction=0.046)
        tot_axes.append(ax)

    # annotate totals[0]
    ax0 = tot_axes[0]
    res_info = el_dict.get("Z_resample_info", None)
    Rdst_um  = (res_info or {}).get("Rdst_um")
    if Rdst_um is None and el_dict.get("A") is not None:
        Rdst_um = el_dict["A"] * 1e6 / 2.0
    if Rdst_um:
        ax0.add_patch(Circle((0, 0), Rdst_um, edgecolor='white', facecolor='none',
                             linewidth=1.2, linestyle='--', alpha=0.9))
        ax0.text(0.05, 0.06, f"Lens Ø ≈ {2*Rdst_um:.1f} µm",
                 transform=ax0.transAxes, ha='left', va='bottom', fontsize=9,
                 color='white', bbox=dict(boxstyle='round', facecolor='black',
                 alpha=0.35, linewidth=0))
    if res_info:
        ax0.text(0.02, 0.98,
                 (f"{res_info.get('policy','')}\n"
                  f"Rsrc={res_info.get('Rsrc_um',0):.1f} µm → "
                  f"Rdst={res_info.get('Rdst_um',0):.1f} µm\n"
                  f"{res_info.get('note','')}"),
                 transform=ax0.transAxes, ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, linewidth=0))

    # bottom row: LF / HF / Total(LF+HF)
    row_bot = row_tot + 1

    ax_low = axes[row_bot, 0]
    imL = ax_low.imshow(Z_low * 1e6, origin="lower", cmap=cmap,
                        extent=(extent_fit if extent_fit else None))
    ax_low.set_title("Low frequency [µm]")
    fig.colorbar(imL, ax=ax_low, fraction=0.046)

    ax_high = axes[row_bot, 1]
    imH = ax_high.imshow(Z_high * 1e6, origin="lower", cmap=cmap,
                         extent=(extent_fit if extent_fit else None))
    ax_high.set_title("High frequency [µm]")  # (renamed from 'residues (avg-removed)')
    fig.colorbar(imH, ax=ax_high, fraction=0.046)

    ax_tot = axes[row_bot, 2]
    imT = ax_tot.imshow(Z_totL * 1e6, origin="lower", cmap=cmap,
                        extent=(extent_fit if extent_fit else None))
    ax_tot.set_title("Total map = Low + High [µm]")
    fig.colorbar(imT, ax=ax_tot, fraction=0.046)

    # highlight according to new semantics
    if defect_type == 1:
        emphasize(ax_low)
    elif defect_type == 2:
        emphasize(ax_high)
    else:
        emphasize(ax_tot)

    # title
    summary = (f"defects={int(el_dict.get('defects',0))} | "
               f"type={defect_type} (1=LF, 2=HF, 3=LF+HF) | "
               f"remove_avg_profile={remove_avg} | "
               f"custom_zernike={'on' if custom_active else 'off'}")
    fig.suptitle(f"{sim_name}, {lens_name}, Defects — {summary}", fontsize=13)

    # save
    try:
        plt.tight_layout()
        Path(save_dir, "Lens_diags").mkdir(parents=True, exist_ok=True)
        out = Path(save_dir, "Lens_diags", f"{filename_stem}_defects_stack.png")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print(f"[PLOT] Saved defects stack → {out}")
    finally:
        plt.close(fig)








def plot_zernike_bars(projectdir,
                      el_dict,
                      filename_stem: str,
                      save_dir,
                      sim_name: str,
                      lens_name: str,
                      nmodes: int | None = None,
                      startmode: int | None = None):
    """
    Bars-only Zernike figure.
    If Custom_zernike is ON, plots ONLY the custom pairs (overwriting the fit in the sim logic).
    Otherwise, plots the LSQ-fit coefficients stored in el_dict['Z_coeffs_native'].
    Saves to .../Lens_diags/{filename_stem}_ZernikePolynoms.png
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Zernike name map (Noll) — keep in sync with your module-level dict
    ZERN_NOLL = {
        1:(0,0,"Piston"),2:(1,-1,"Tilt X (horizontal)"),3:(1,1,"Tilt Y (vertical)"),
        4:(2,0,"Defocus"),5:(2,-2,"Primary astigmatism (oblique)"),
        6:(2,2,"Primary astigmatism (vertical)"),7:(3,-1,"Primary coma X (horizontal)"),
        8:(3,1,"Primary coma Y (vertical)"),9:(3,-3,"Trefoil (vertical)"),
        10:(3,3,"Trefoil (oblique)"),11:(4,0,"Primary spherical"),
        12:(4,-2,"Secondary astigmatism (vertical)"),13:(4,2,"Secondary astigmatism (oblique)"),
        14:(4,-4,"Quadrafoil (vertical)"),15:(4,4,"Quadrafoil (oblique)"),
        16:(5,-1,"Secondary coma X (horizontal)"),17:(5,1,"Secondary coma Y (vertical)"),
        18:(5,-3,"Secondary trefoil (oblique)"),19:(5,3,"Secondary trefoil (vertical)"),
        20:(5,-5,"Pentafoil (oblique)"),21:(5,5,"Pentafoil (vertical)"),
        22:(6,0,"Secondary spherical"),23:(6,-2,"Tertiary astigmatism (vertical)"),
        24:(6,2,"Tertiary astigmatism (oblique)"),25:(6,-4,"Secondary quadrafoil (vertical)"),
        26:(6,4,"Secondary quadrafoil (oblique)"),27:(6,-6,"Hexafoil (vertical)"),
        28:(6,6,"Hexafoil (oblique)"),29:(7,-1,"Tertiary coma X (horizontal)"),
        30:(7,1,"Tertiary coma Y (vertical)"),31:(7,-3,"Tertiary trefoil (oblique)"),
        32:(7,3,"Tertiary trefoil (vertical)"),33:(7,-5,"Secondary pentafoil (oblique)"),
        34:(7,5,"Secondary pentafoil (vertical)"),35:(7,-7,"Heptafoil (oblique)"),
        36:(7,7,"Heptafoil (vertical)"),37:(8,0,"Tertiary spherical"),
    }

    # Determine what to plot
    custom_list = el_dict.get("Custom_zernike", [])
    custom_active = (isinstance(custom_list, (list, tuple))
                     and len(custom_list) >= 1 and int(custom_list[0]) == 1)

    if custom_active:
        pairs = list(custom_list[1:])
        if len(pairs) % 2 != 0:
            raise ValueError("Custom_zernike must contain pairs: [1, n1, v1, n2, v2, ...]")
        js = [int(pairs[i])   for i in range(0, len(pairs), 2)]
        vals_m = [float(pairs[i]) for i in range(1, len(pairs), 2)]
        title_bar = "Custom Zernike amplitudes (applied to total-fit map)"
    else:
        Zcoeffs = el_dict.get("Z_coeffs_native", None)
        if Zcoeffs is None:
            print("[ZERNIKE][WARN] Z_coeffs_native not cached; skipping bars.")
            return
        Zcoeffs = np.asarray(Zcoeffs).ravel()
        nmodes = int(nmodes if nmodes is not None else el_dict.get("nmodes", 37))
        startmode = int(startmode if startmode is not None else el_dict.get("startmode", 1))
        jmax = min(startmode + nmodes - 1, len(Zcoeffs))
        js = list(range(startmode, jmax + 1))
        vals_m = [float(Zcoeffs[j-1]) for j in js]
        title_bar = f"Zernike decomposition (modes {startmode}–{jmax})"

    # Prepare labels and values in µm
    vals_um = [v * 1e6 for v in vals_m]
    labels = []
    for j in js:
        nm = ZERN_NOLL.get(j)
        if nm is None:
            labels.append(f"$Z_{{{j}}}$")
        else:
            n, m, name = nm
            labels.append(f"$Z_{{{j}}}$ $Z_{{{n}}}^{{{m}}}$\n{name}")

    # Figure sizing
    n = len(js)
    width = max(10.0, 0.38 * n + 4.0)   # scale with number of bars
    fig, ax = plt.subplots(figsize=(width, 5.0), dpi=140)

    # grid lines behind bars
    for x in range(n):
        ax.axvline(x, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)

    ax.bar(range(n), vals_um, zorder=2)
    ax.set_xticks(range(n), labels, rotation=90, ha='center', va='top')
    ax.set_ylabel("Thickness defect [µm]")
    ax.set_title(title_bar)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.25)

    # Figure title & save
    summary = (
        f"defects={int(el_dict.get('defects',0))} | "
        f"type={int(el_dict.get('defect_type',3))} | "
        f"remove_avg={int(el_dict.get('remove_avg_profile',0))} | "
        f"custom_zernike={'on' if custom_active else 'off'}"
    )
    fig.suptitle(f"{sim_name}, {lens_name} — {summary}", fontsize=11, y=1.02)

    outdir = Path(save_dir or projectdir) / "Lens_diags"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{filename_stem}_ZernikePolynoms.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[ZERNIKE] Saved bars → {out}")




# ----------------- Zernike name map (Noll) -----------------
# Noll → (n, m, canonical name)
ZERN_NOLL = {
    1:  (0, 0,  "Piston"),
    2:  (1,-1,  "Tilt X (horizontal)"),
    3:  (1, 1,  "Tilt Y (vertical)"),
    4:  (2, 0,  "Defocus"),
    5:  (2,-2,  "Primary astigmatism (oblique)"),
    6:  (2, 2,  "Primary astigmatism (vertical)"),
    7:  (3,-1,  "Primary coma X (horizontal)"),
    8:  (3, 1,  "Primary coma Y (vertical)"),
    9:  (3,-3,  "Trefoil (vertical)"),
    10: (3, 3,  "Trefoil (oblique)"),
    11: (4, 0,  "Primary spherical"),
    12: (4,-2,  "Secondary astigmatism (vertical)"),
    13: (4, 2,  "Secondary astigmatism (oblique)"),
    14: (4,-4,  "Quadrafoil (vertical)"),
    15: (4, 4,  "Quadrafoil (oblique)"),
    16: (5,-1,  "Secondary coma X (horizontal)"),
    17: (5, 1,  "Secondary coma Y (vertical)"),
    18: (5,-3,  "Secondary trefoil (oblique)"),
    19: (5, 3,  "Secondary trefoil (vertical)"),
    20: (5,-5,  "Pentafoil (oblique)"),
    21: (5, 5,  "Pentafoil (vertical)"),
    22: (6, 0,  "Secondary spherical"),
    23: (6,-2,  "Tertiary astigmatism (vertical)"),
    24: (6, 2,  "Tertiary astigmatism (oblique)"),
    25: (6,-4,  "Secondary quadrafoil (vertical)"),
    26: (6, 4,  "Secondary quadrafoil (oblique)"),
    27: (6,-6,  "Hexafoil (vertical)"),
    28: (6, 6,  "Hexafoil (oblique)"),
    29: (7,-1,  "Tertiary coma X (horizontal)"),
    30: (7, 1,  "Tertiary coma Y (vertical)"),
    31: (7,-3,  "Tertiary trefoil (oblique)"),
    32: (7, 3,  "Tertiary trefoil (vertical)"),
    33: (7,-5,  "Secondary pentafoil (oblique)"),
    34: (7, 5,  "Secondary pentafoil (vertical)"),
    35: (7,-7,  "Heptafoil (oblique)"),
    36: (7, 7,  "Heptafoil (vertical)"),
    37: (8, 0,  "Tertiary spherical"),
}



def _defect_type_label(v: int) -> str:
    if v == 1: return "Zernike only"
    if v == 2: return "Residues only"
    if v == 3: return "Zernike + residues"
    return f"Unknown ({v})"



def plot_custom_crl_ideal(projectdir: Path,
                          Na: np.ndarray,
                          nb_lenses: float,
                          lens_thickness: np.ndarray,
                          transmission_map: np.ndarray,
                          total_phase_map_ideal: np.ndarray,
                          add_aperture: int,
                          Aperture_transmission: np.ndarray | None,
                          lens_material: str,
                          E_eV: float,
                          wavelength_m: float,
                          filename_stem: str | None = None,
                          save_dir: Path | None = None,
                          sim_name: str | None = None,
                          lens_name: str | None = None):
    """
    Plot and save diagnostic figures for an ideal Custom CRL (no defects).

    Produces 2-D maps (thickness, transmission, phase) and 1-D center cuts,
    with an optional aperture transmission. The figure is saved in the
    ``Lens_diags`` directory for validation and documentation.
    """

    import matplotlib.pyplot as plt

    N = lens_thickness.shape[0]
    center_idx = N // 2
    x_um = Na * 1e6
    extent_um = [Na[0]*1e6, Na[-1]*1e6, Na[0]*1e6, Na[-1]*1e6]

    thickness_1d     = lens_thickness[center_idx, :] * 1e6
    transmission_1d  = transmission_map[center_idx, :]
    phase_1d         = total_phase_map_ideal[center_idx, :]

    ncols = 4 if add_aperture else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5.2*ncols, 7), dpi=120)

    im0 = axes[0,0].imshow(nb_lenses * lens_thickness * 1e6, cmap='inferno',
                           extent=extent_um, origin='lower')
    axes[0,0].set(title='2-D Thickness [µm]', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im0, ax=axes[0,0], fraction=0.046)

    im1 = axes[0,1].imshow(transmission_map, cmap='viridis',
                           extent=extent_um, origin='lower')
    axes[0,1].set(title='2-D Transmission', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im1, ax=axes[0,1], fraction=0.046)

    im2 = axes[0,2].imshow(total_phase_map_ideal, cmap='twilight',
                           extent=extent_um, origin='lower')
    axes[0,2].set(title='2-D Phase (ideal) [rad]', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(im2, ax=axes[0,2], fraction=0.046)

    if add_aperture:
        im3 = axes[0,3].imshow(Aperture_transmission, cmap='gray',
                               extent=extent_um, origin='lower')
        axes[0,3].set(title='2-D Aperture (T)', xlabel='x [µm]', ylabel='y [µm]')
        plt.colorbar(im3, ax=axes[0,3], fraction=0.046)

    axes[1,0].plot(x_um, nb_lenses * thickness_1d); axes[1,0].grid()
    axes[1,0].set(ylabel='Thickness [µm]', xlabel='x [µm]', title='1-D Thickness (centre cut)')

    axes[1,1].plot(x_um, transmission_1d, color='green'); axes[1,1].grid()
    axes[1,1].set(ylabel='Transmission', xlabel='x [µm]', title='1-D Transmission (centre cut)')

    axes[1,2].plot(x_um, phase_1d, color='purple'); axes[1,2].grid()
    axes[1,2].set(ylabel='Phase [rad]', xlabel='x [µm]', title='1-D Phase (centre cut)')

    if add_aperture:
        axes[1,3].plot(x_um, Aperture_transmission[center_idx,:], color='black'); axes[1,3].grid()
        axes[1,3].set(ylabel='Aperture T', xlabel='x [µm]', title='1-D Aperture (centre cut)')

    # --- prepend simulation and lens names ---
    sim_name  = sim_name  or Path(projectdir).stem
    lens_name = lens_name or "Lens"

    fig.suptitle(f"{sim_name}, {lens_name}, Custom CRL (ideal) — {lens_material}, E={E_eV:.0f} eV",
                fontsize=13)

    plt.tight_layout(rect=[0,0,1,0.95])

    if save_dir is None:
        save_dir = Path(projectdir)
    save_dir = save_dir / "Lens_diags"
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = (filename_stem or "Lens_CRL_cut")
    out = save_dir / f"{stem}_ideal.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)





# =========================================================
# Vacuum Birefringence related functions
# =========================================================



def F_of(chi: float, chi0: float, rho: float, n_nodes: int = 160) -> float:
    """
    Universal overflow-safe evaluator for the pulse-shape correction integrand.
    Analytically cancels the large exponentials; integrates a benign kernel:
        ∫ e^{-K^2} |S|^2 dK = ∫ e^{-2*rho^2*K^2} | w(i(z+chi)) + w(i(-z+chi)) |^2 dK
    with z = rho*K - i*chi0.  Works for any (chi, chi0, rho).
    """
    K, W = hermgauss(n_nodes)

    # Skip extreme tails where exp(-2*rho^2*K^2) underflows (no contribution).
    # 2*rho^2*K^2 ≳ 700 → exp(-...) ~ 5e-305 (below double precision relevance)
    mask = (2.0*(rho*rho)*(K*K) < 700.0)
    if not np.any(mask):
        return 0.0

    K = K[mask]; W = W[mask]
    z = rho*K - 1j*chi0

    a = 1j*( z + chi)   # i(z+chi)
    b = 1j*(-z + chi)   # i(-z+chi)
    Wsum = wofz(a) + wofz(b)

    integrand = np.exp(-2.0*(rho*rho)*(K*K)) * (Wsum.real**2 + Wsum.imag**2)
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

    I = np.sum(W * integrand)
    pref = np.sqrt((1.0 + 2.0*rho*rho)/3.0) * (chi*chi)
    return pref * I







def maybe_spawn_VB_channels(bundle: FieldBundle,
                            params: dict,
                            VB_mask_parr,
                            VB_mask_perp):
    """
    At the TCC element we branch the existing field into two vacuum-
    birefringence (VB) channels by multiplying the intensity masks.
    Called exactly once; afterwards `propagate_bundle()` carries all
    three channels through the beamline.
    """
    if "VB_parr" in bundle.fields:        # already spawned
        return bundle

    F_main = bundle.fields["main"]

    bundle.fields["VB_parr"] = MultIntensity(VB_mask_parr**2, F_main)
    bundle.fields["VB_perp"] = MultIntensity(VB_mask_perp**2, F_main)

    return bundle





# =========================================================
# MAIN SIMULATION functions
# =========================================================



def apply_element(bundle: FieldBundle,
                  el_name: str,
                  el_dict: dict,
                  params: dict,
                  reg_prop_dict: dict,
                  method: str,
                  do_edge_damping: bool,
                  edge_damping_aperture=None):
    """
    Apply *one* optical element to every field contained in *bundle*.
    """
    # ──────────────────────────────────────────────────────────────────────
    # 1. bookkeeping / folders / wavelength
    # --------------------------------------------------------------------
    projectdir  = Path(params.get("projectdir", "."))  # needed by Custom_CRL
    def_do_plot = 1                                    # If it plots the element or not. Its = 1 by default
    
    el_type    = el_dict["type"]
    N          = params['N']                       # grid size
    propsize   = params['propsize']                # current physical window
    wavelength = params['wavelength']

    for ch_name in list(bundle.fields.keys()):

        F = bundle.fields[ch_name]                # ① current field

        ######## REG and DEREG elements ########
        if el_type == "zoom_window":
            zoom = el_dict.get("zoom", 1.0)
            F = zoom_window_with_interp(F, zoom)
            bundle.fields[ch_name] = F
            def_do_plot = 0

        
        if el_type == 'reg':  # regularize propagation
            reg_prop_dict["regularized_propagation"] = True
            if 'reg-by-f' in el_dict:
                tmp = el_dict['reg-by-f']
            else:
                tmp = reg_prop_dict["reg_parabola_focus"]
            F = Lens(F, -tmp)
            bundle.fields[ch_name] = F

            def_do_plot = 0

        elif el_type == 'dereg':  # deregularize propagation
            if not reg_prop_dict["regularized_propagation"]:
                print("  You can't deregularize an already deregularized field!!!")
            else:
                reg_prop_dict["regularized_propagation"] = False
                tmp = reg_prop_dict["reg_parabola_focus"]
                F = Lens(F, reg_prop_dict["reg_parabola_focus"])
                bundle.fields[ch_name] = F
                
            def_do_plot = 0
        

        if "reg-by-f" in el_dict:
            f = el_dict['reg-by-f']
            if reg_prop_dict["reg_parabola_focus"] is None:
                reg_prop_dict["reg_parabola_focus"] = f
                reg_prop_dict["regularized_propagation"] = True
                #print(f"Regularizing by CRL in {F_pos} by value {f}")
            else:
                #for inserting second, that images the focus made by first CRL
                if reg_prop_dict["regularized_propagation"] == True:
                    f2_tmp = f
                    #thin lens formula (zobrazovaci rovnice), where focus is the object
                    reg_new_tmp = 1.0/(1.0/f2_tmp + 1.0/reg_prop_dict["reg_parabola_focus"])
                    reg_prop_dict["reg_parabola_focus"] = reg_new_tmp
                    print("Re-regularizing by CRL")
                else:
                    reg_prop_dict["reg_parabola_focus"] = f
                    reg_prop_dict["regularized_propagation"] = True
                    print("Unexpected regularizing by CRL")


            ############# LENS ELEMENT ###############
        if 'lens' in el_type:
            ideal = yamlval('ideal', el_dict, 1)
            if ideal:
                f = el_dict['f']
                if "reg" in el_type:
                    if reg_prop_dict["reg_parabola_focus"] is None:
                        reg_prop_dict["reg_parabola_focus"] = f
                        reg_prop_dict["regularized_propagation"] = True
                    else:
                        #for inserting second, that images the focus made by first CRL
                        if reg_prop_dict["regularized_propagation"] == True:
                            f2_tmp = f
                            #thin lens formula (zobrazovaci rovnice), where focus is the object
                            reg_new_tmp = 1.0/(1.0/f2_tmp + 1.0/reg_prop_dict["reg_parabola_focus"])
                            reg_prop_dict["reg_parabola_focus"] = reg_new_tmp
                            print("Re-regularizing by CRL")
                        else:
                            reg_prop_dict["reg_parabola_focus"] = f
                            reg_prop_dict["regularized_propagation"] = True
                            print("Unexpected regularizing by CRL")
                else:
                    F = Lens(f,0,0,F)
                    bundle.fields[ch_name] = F

            ####### APERTURE OF THE LENS ########
            aperture = yamlval('size',el_dict,0)
            if aperture==0 and 'CRL4' in el_type:
                aperture = 400e-6

            if aperture>0: #aperture
                ap_dict = {}
                ap_dict['elem'] = 'Hf' 
                ap_dict['thickness'] = 0.0001
                ap_dict['shape'] = 'circle'
                ap_dict['size'] = aperture
                ap_dict['invert'] = 1
                tmap,phasemap = doap(ap_dict,params)   # creating the transmission and phase map of the lens aperture
                F = MultIntensity(tmap,F)              # multiplying intensity of the field by the lens aperture
                bundle.fields[ch_name] = F

            ############# CRL4 default parameters ############
            if 'CRL4' in el_type: 
                Lroc = yamlval('roc',el_dict,5.0e-5) #radius of curvature
                ab_dict = {} #absorption dictionnary
                ab_dict['elem'] = yamlval('lens_material', el_dict, params.get('lens_material', 'Be'))
                ab_dict['minr0'] = 0
                ab_dict['shape'] = 'parabolic_lens'
                ab_dict['size'] = aperture
                ab_dict['roc'] = Lroc
                ab_dict['double_sided'] = 1 #parabolic shape on both sides
                ab_dict['num_lenses'] = yamlval('num_lenses',el_dict,1)
                tmap2,phasemap = doap(ab_dict,params,debug=0)
                F = MultIntensity(tmap*tmap2,F)
                bundle.fields[ch_name] = F

                if not ideal:  
                    F = MultPhase(phasemap,F)
                    bundle.fields[ch_name] = F
                    print('doing real lens')


                if not ideal:  
                    F = MultPhase(phasemap,F)
                    bundle.fields[ch_name] = F
                    print('doing real lens')

            if yamlval('celestre',el_dict,1):
                cel_dict = {}
                cel_dict['defect'] = 'celestre'
                cel_dict['type'] = 'phaseplate'
                cel_dict['num'] = yamlval('num_lenses',el_dict,1)
                phaseshiftmap = do_phaseplate(cel_dict,params)
                F = MultPhase(phaseshiftmap,F)
                bundle.fields[ch_name] = F
                
            if yamlval('seiboth',el_dict,1):
                seib_dict = {}
                seib_dict['defect'] = 'seiboth'
                seib_dict['type'] = 'phaseplate'
                seib_dict['num'] = yamlval('num_lenses',el_dict,1)
                phaseshiftmap = do_phaseplate(seib_dict,params)
                F = MultPhase(phaseshiftmap,F)
                bundle.fields[ch_name] = F
                
            if yamlval('scatterer',el_dict,0):
                sc_dict={}
                if 0: #first attemtp
                    sc_dict['randomizeB']=2.e-6
                    sc_dict['type']='aperture'
                    sc_dict['shape']='circle'
                    sc_dict['size']=aperture
                    sc_dict['invert']=0
                    sc_dict['thickness']=2e-6
                    sc_dict['elem']='W'
                if 1: #second one
                    sc_dict['randomizeB'] = yamlval('lens_randomize_r',params,20.e-6)
                    sc_dict['type'] = 'aperture'
                    sc_dict['shape'] = 'circle'
                    sc_dict['size'] = aperture
                    default_k = 0.02 # Arbitrary
                    k = yamlval('lens_randomize_k',params,default_k)
                    if 'scatterer_k' in el_dict:
                            k = el_dict['scatterer_k']
                    sc_dict['density'] = k*yamlval('num_lenses',el_dict,1)
                    sc_dict['thickness'] = 3*yamlval('lens_randomize_r',params,20.e-6)
                    sc_dict['elem'] = yamlval('lens_randomize_elem',params,'Ti')
                    
                Ii1 = (np.nansum(Intensity(0,F)))
                tmap3,phasemap = doap(sc_dict,params,debug=0)
                F = MultIntensity(tmap3,F)
                bundle.fields[ch_name] = F
                F = MultPhase(phasemap,F)
                bundle.fields[ch_name] = F
                Ii2 = (np.nansum(Intensity(0,F)))
                loss_on_scatterer = Ii2/Ii1
                params['transmission_of_scatterer_'+el_name] = loss_on_scatterer

        ############# CUSTOM CRL ELEMENT ###############
        if el_type == 'Custom_CRL':
            F = handle_custom_CRL(F, el_dict, params, projectdir)
            bundle.fields[ch_name] = F
            
        ############# ELEMENT : PHASE PLATE ###########
        if el_type=='phaseplate':
            phaseshiftmap=do_phaseplate(el_dict,params)
            F=MultPhase(phaseshiftmap,F)
            bundle.fields[ch_name] = F

        # ############# ELEMENT : GAZ JET (pure phase) ###############
        if el_type.lower() == "gazjet":
            if ch_name == "main":
                # Build plasma phase (ignore transmission ~ 1 for low densities)
                phase_map, _ = build_gazjet_maps(el_dict, params, F)

                # Apply phase to the main channel and persist
                F = MultPhase(phase_map, F)
                bundle.fields[ch_name] = F
                
            else:
                # Skip Gazjet for non-main channels
                pass


        ############# ELEMENT : PURE APERTURE ###########
        if 'aperture' in el_type:
            # -----External_map shaping at the beam entrance ---
            if el_dict.get('shape', '') == 'external_map':
                M = build_external_shaper_mask(el_dict, F, params)
                F = MultIntensity(M, F)
                bundle.fields[ch_name] = F
            else:
                # --- Build a regular aperture ---
                if len(el_dict)==0:
                    do_nothing = 1
                num = yamlval('num',el_dict,1)
                merged = 1
                if merged:
                    bt = np.zeros((N,N))+1.
                    ph = np.zeros((N,N))
                    for i in np.arange(num):
                        tmap,phasemap = doap(el_dict,params)
                        bt = bt*tmap
                        ph+=phasemap
                    if yamlval('do_intensity',el_dict,1):
                        F = MultIntensity(bt,F)
                        bundle.fields[ch_name] = F
                    if yamlval('do_phaseshift',el_dict,1):
                        F = MultPhase(ph,F)
                        bundle.fields[ch_name] = F
                else:
                    for i in np.arange(num):
                        tmap,phasemap=doap(el_dict,params)
                        if yamlval('do_intensity',el_dict,1):
                            F = MultIntensity(tmap,F)
                            bundle.fields[ch_name] = F
                        if yamlval('do_phaseshift',el_dict,1):
                            F = MultPhase(phasemap,F)
                            bundle.fields[ch_name] = F


        ############## Air Scattering ##########
        
        if el_name == 'Det' and el_dict.get('AirScat', 0):
            
            compute_transmission = el_dict.get('compute_transmission', 0)
            use_symmetric_kernel = el_dict.get('use_symmetric_kernel', True)
            test_identity_kernel = el_dict.get('test_identity_kernel', 0)
            crop_to_odd = el_dict.get('crop_to_odd', 1)  # crops the image if N is even such that the image has an odd number of pixels (better for convolution)
            
            I_after_air = apply_air_scattering_and_debug_plot(F,
                                                                params, 
                                                                propsize, 
                                                                N, 
                                                                plot_debug = True, 
                                                                use_symmetric_kernel = use_symmetric_kernel, compute_transmission = compute_transmission,
                                                                test_identity_kernel = test_identity_kernel , crop_to_odd = crop_to_odd)
            

        # ---------------- edge damping block ----------------------------
        if do_edge_damping:
            F = MultIntensity(edge_damping_aperture, F)
            bundle.fields[ch_name] = F
        # ----------------------------------------------------------------

    ################## CREATE VB FIELDS AT TCC #################
    if el_name == 'TCC' and el_dict.get('VB_signal', 0) and "VB_parr" not in bundle.fields:

        I_cr = 4.7e29 # Critical intensity in [W/cm^2]
        alpha_cst = e**2 / (4 * np.pi * epsilon_0 * hbar * c)

        # X and Y offset of the laser 
        x_off = float(params.get("IR_x_offset_m", 0.0))      # x offset of the laser at TCC [m]
        y_off = float(params.get("IR_y_offset_m", 0.0))    # y offset of the laser at TCC [m]

        # Calculate the intensity of IR laser :
        wavelength_IR = float(params.get("IR_wavelength", 800e-9))            # [m]
        phi_VB = np.pi / 4 #Polarisation between the pump and probe. By default set to 45 degrees for maximisation of the signal number
        phi_VB       = np.deg2rad(float(params.get("phi_VB_deg", 45.0))) #Polarisation angle between the IR laser and the Xray beam [rad]

        # Define E_pulse (OR) P_peak
        tau_FWHM = float(params.get("IR_FWHM_duration", 30e-15))             # Pump IR Laser FWHM duration in second
        E_pulse  = float(params.get("IR_Energy_J", 4.8))                      # [J]
        #P_peak_direct = 200e12 # Peak power of the laser, in Watt

        P_peak = peak_power(E=E_pulse, tau_FWHM=tau_FWHM) # In [Watt]
        #P_peak = peak_power(P_peak=P_peak_direct)

        # 1) ------- Option 1 = Airy Disk -------
        #I_W_cm2, x = airy_disk_map(F.grid_size, F.N, P_peak, lam=wavelength_IR, f=f_IR, D=D_IR,return_grid=True)

        # 2) ------- Option 2 = Gaussian --------
        IR_spatial_params = yamlval("IR_2Dmap", params, ["gaussian", "match_integral", None, None])

        I_W_cm2, x, FWHM_diam = build_ir_focus_map(
            IR_spatial_params=IR_spatial_params,
            F=F,
            params=params,
            P_peak=P_peak,
            x_off=x_off,
            y_off=y_off
        )


        # ---------------- Pulse Shape factor ---------------
        # The following factor is made to account for: 1) The finite pump Rayleigh length. 2) The probe and pump finite durations. 3) The time offset of the pump/probe focus

        T_FWHM = float(params.get("X_FWHM_duration", 25e-15))       # probe duration FWHM [s]
        t0  = float(params.get("Timing_jitter", 0.0))               # [s] pump–probe focus timing offset

        tau_Felix = tau_FWHM * np.sqrt(2 / np.log(2))   #The tau defined in the paper [VIBE].
        w0 = FWHM_diam / (np.sqrt(2 * np.log(2)))       # 1/e^2 waist radius [m]
        zR  = np.pi * w0**2 / wavelength_IR             # Rayleigh length [m]
                      
        T   = T_FWHM * 2 / (np.sqrt(2 * np.log(2)))          # probe duration (1/e^2) [s]
        tau = tau_FWHM * 2 / (np.sqrt(2 * np.log(2)))        # pump duration (1/e^2) [s]

        chi   = 4.0 * zR / (np.sqrt((c*T)**2 + (c*tau)**2 / 2.0))
        chi0  = 2.0 * (c*t0) / (np.sqrt((c*T)**2 + (c*tau)**2 / 2.0))
        rho   = T / tau

        Pulse_shape_factor  = (np.sqrt(3.0*np.pi) / 4.0) * F_of(chi, chi0, rho, n_nodes=180) # Pulse shaped factor (unitless)
        print(f"Pulse shape factor = {Pulse_shape_factor:.12g}")
        print(f"t0={t0}, FWHM_diam={FWHM_diam}, T_FWHM={T_FWHM}, tau_FWHM = {tau_FWHM}")
        # --------------- Creating the masks -------------------

        prefactor = (c * tau_Felix / wavelength) * (alpha_cst / 90) * np.sqrt(np.pi / 2) * np.sqrt(Pulse_shape_factor)
        VB_mask_parr = (I_W_cm2 / I_cr) * prefactor * (11 - 3 * np.cos(2 * phi_VB))  #mask of the intensity of IR laser at TCC (unitless)
        VB_mask_perp = (I_W_cm2 / I_cr) * prefactor * ( 3 * np.sin(2 * phi_VB))      #mask of the intensity of IR laser at TCC (unitless)

        print(f"max Intensity IR = {np.max(I_W_cm2)} W/cm^2")
        print(f"phi VB = {phi_VB}")
        print(f"tau=tau_FWHM*sqrt(2/ln(2)) ={tau_Felix} s")
        print(f"prefactor VB = {prefactor}")
        print(f"I_W_cm2 / I_cr = {I_W_cm2 / I_cr}")
        print(f"11 - 3 * np.cos(2 * phi_VB) = {11 - 3 * np.cos(2 * phi_VB)}")
        print(f"3 * np.sin(2 * phi_VB) = {3 * np.sin(2 * phi_VB)}")



        bundle = maybe_spawn_VB_channels(bundle,
                                        params,
                                        VB_mask_parr,
                                        VB_mask_perp)

    return bundle, bundle.fields["main"], def_do_plot




def main_VIBE(params,elements):
    """
    Simulate the beam line described by *elements* and *params*.

    Parameters
    ----------
    params   : dict   – global run settings (see YAML template)
    elements : list of optical elements   – [(z_pos, element name, yaml_dict), …] in *any* order

    Returns
    -------
    params   : dict   – updated with results (transmission, intensities, …)
    trans    : 1-D np.ndarray – I(z)/I(start) for each element plane
    figs     : dict   – pickled images requested via `figs_to_save`

    Others
    -------
    norms = [0,0] :  in the first passage of the loop and then norm=[integral / normalised , maximum] which gets updated at each passage in the loop.
    im, or imC    :  image prepared by the "prepare_image" function. 
    F             :  Electric field
    I             :  2D map of intensity
    measures      :  an array with the maximum and sum of the intensity map
    """

    # ──────────────────────────────────────────────────────────────────────
    # 0. Initialisation
    # --------------------------------------------------------------------
    mu.clear_times(); mu.tick()      # timing clear
    import pprint
    from pathlib import Path
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.ticker as mticker

    # ──────────────────────────────────────────────────────────────────────
    # 1. bookkeeping / folders / wavelength
    # --------------------------------------------------------------------

    pprint.pprint(params)

    projectdir = Path(params.get("projectdir", "."))
    basepath = Path(params["projectdir"]).parent
    
    N = params['N']
    propsize = params['propsize']
    params['initial_propsize'] = params['propsize']            # store the initial propsize of the simulation for the flow_plot
    params['pxsize'] = params['propsize'] / params['N']
    Na = ( np.arange(N) - 0.5 * N ) * params['pxsize'] / um
    max_pixels = yamlval('subfigure_size_px',params,300)
    if max_pixels>N: max_pixels=N
    if max_pixels == 0: max_pixels = N                        # If we put the parameter "subfigure_size_px" = 0 in the yaml, it means that we want to take the original number of cells N 

    dtype = np.complex64

    E_eV      = params['photon_energy']          # photon energy [eV]
    wavelength = h * c / (E_eV * e)             # λ = h·c / E     [m]
    params['wavelength'] = wavelength           # make it globally visible

    # ──────────────────────────────────────────────────────────────────────
    # 2. auto-flow (inject extra imager planes) & element sorting
    # --------------------------------------------------------------------
    

    auto_flow = yamlval('flow_auto_save',params)
    if 'flow' in params:
        flowdef=params['flow']
        flowposs= None
        if np.ndim(flowdef)==2:
            flowposs=np.array([])
            for r in np.arange(np.shape(flowdef)[0]):
                ff=flowdef[r]
                flowp1=np.arange(ff[0],ff[1],ff[2])
                flowposs=np.concatenate((flowposs,flowp1))

        if np.size(flowdef)==3:
            flowposs=np.arange(flowdef[0],flowdef[1],flowdef[2])
        if flowposs is not None:
            for fi,flowpos in enumerate(flowposs):
                fe={}
                fe['type']='imager'
                fe['position']=flowpos
                fe['plot']=yamlval('flow_images',params,0)
                fe['zoom']=yamlval('flow_zoom',params,1)
                flowname='flow_{:.2f}'.format(flowpos)
                flowname='flow_{:03.0f}_{:.2f}'.format(fi,flowpos)
                EE=[flowpos,flowname,fe]
                elements.append(EE)
        if auto_flow:
            ffdir = Path(params["projectdir"]) / "flow_figs" / f"{params['filename']}_auto"
            mu.mkdir(ffdir,0)

    elements = sort_elements(elements)          # sort the elements with their longitudinal position


    # ──────────────────────────────────────────────────────────────────────
    # 3. initial field  & regularisation
    # --------------------------------------------------------------------

    F = Begin(params['propsize'], wavelength, N, dtype=dtype)
    F = GaussBeam(
            F,
            params['beamsize'],
            x_shift=params['gauss_x_shift'],
            tx=params['gauss_x_tilt'])


    reg_prop_dict = {
        "regularized_propagation": params.get("regularized_propagation", False),
        "reg_parabola_focus":      params.get("reg_parabola_focus_mm", None),}

    bundle = FieldBundle(
                fields={"main": F},
                z_pos=elements[0][0],
                reg=reg_prop_dict
            )

    # ──────────────────────────────────────────────────────────────────────
    # 4. global stores
    # ----------------------------------------------------------------------

    figs_to_save = yamlval('figs_to_save',params,[])
    figs_to_export = yamlval('figs_to_export',params,[])

    figs = {}       # Definition of the pickle "figs"
    export = {}
    intensities = {}
    integ = 0.0
    profi = 0
    norms = [0,0]
    numel = len(elements)

    # --- traces for post-run normalization/plotting ---
    trace_z, trace_I, trace_names = [], [], []
    # --- channel-specific trace stores ---
    trace_z_dict = {"main": [], "VB_parr": [], "VB_perp": []}
    trace_I_dict = {"main": [], "VB_parr": [], "VB_perp": []}
    trace_names_dict = {"main": [], "VB_parr": [], "VB_perp": []}


    
    if 'edge_damping' in params:
        do_edge_damping = 1
        edge_damping_aperture = do_edge_damping_aperture(params)
    else:
        do_edge_damping = 0
    
    
    # ──────────────────────────────────────────────────────────────────────
    # 5. plotting infrastructure
    # ---------------------------------------------------------------------
    mosaic_cbar_dynamic = True    # Whether the colorbar axis change from plot to plot in the mosaic figure.

    max_panels     = params['fig_rows'] * params['fig_cols']
    fig_by_ch      = {}
    pi_by_ch       = {}          # current subplot number inside that figure
    step_x, step_y = [], []
    fig_summary    = plt.figure(figsize=(params['fig_cols'] * 3, 3))   # wide & short
    ax_prof        = fig_summary.add_subplot(1, 2, 1)    # left panel
    ax_steps       = fig_summary.add_subplot(1, 2, 2)    # right panel
    profiles_done  = False                               # helper flag
    unit_sel = params['intensity_units']                 # Intensity unit is "photons" or "relative"
    Zoom_global = params['Zoom_global']                  # Zoom global of all the outputs of the simulation
    print(f"unit read from yaml : {unit_sel}")
    print(f"Zoom global ={Zoom_global}")
    
    save_parts = yamlval('save_parts', params, 0)
    if save_parts:
        mu.mkdir('part')
        mu.savefig('part/'+params['filename']+'__start')

    # ──────────────────────────────────────────────────────────────────────
    # 5 bis. Find the active beam shape (if any)
    # --------------------------------------------------------------------
    beam_shaper_index = None
    for i, E in enumerate(elements):
        name = E[1]
        dct  = E[2]
        if name == "beam_shaper" and yamlval('in', dct, 1):
            beam_shaper_index = i
            break
    

    # ──────────────────────────────────────────────────────────────────────
    # 6. main loop over elements
    # --------------------------------------------------------------------
    method = yamlval('method',params,'FFT')
    
    for ei,E in enumerate(elements):      # ei = element index ([0,1,2,3...])   E = element dictionnary
        z       = E[0]                    # position of the element
        el_name = E[1]                    # element name
        el_dict = E[2]                    # element aperture
        el_type = el_dict['type']
        
        print('{:} (elem.n.{:.0f})   ###'.format(el_name,ei))

        # ---- skip inactive planes ---------------------------------------
        if 'off' in el_type: continue
        if not yamlval('in', el_dict, 1): continue  # Skip inactive elements

        # ---- propagate to element position ------------------------------
        delta_z = z - bundle.z_pos

        if delta_z == 0:
            print('skipping zero propagation')
        else:
            bundle = propagate_bundle(
                        bundle,
                        dz = delta_z,
                        method='Fresnel' if method.lower() == 'fresnel' else 'FFT'
                    )
            F = bundle.fields["main"]                                                 # (keep the legacy variable alive)
        
            propsize           = F.siz                                                # update physical size of grid
            params['propsize'] = propsize
            params['pxsize']   = F.siz / N                                            # update pixel size
            Na                 = (np.arange(N) - 0.5 * N) * params['pxsize'] / um     # Because the grid changes size with propagation we need to recalculate Na each time.
            
            if reg_prop_dict["reg_parabola_focus"] is not None:
                reg_prop_dict["reg_parabola_focus"] -= delta_z

        # ---------- apply optical element ---------------------------------
        bundle, F, def_do_plot = apply_element(
                bundle,
                el_name,
                el_dict,
                params,
                reg_prop_dict,
                method,
                do_edge_damping,
                edge_damping_aperture if do_edge_damping else None)
        
        all_channels = bundle.fields.items()     # [('main', F), ('VB_parr', F1) …]

        # ---- Prepare the plot variables ------------------------------
        #ZoomFactor    = yamlval('zoom',el_dict,1)
        ZoomFactor = Zoom_global if Zoom_global != 1 else yamlval('zoom', el_dict, 1) # If Zoom Global exist, it is prioritized over the Zoom_factor (local for each plane)
        do_plot       = yamlval('plot', el_dict, def_do_plot)
        plot_phase    = yamlval('plot_phase', params, 0)
        logg          = yamlval('figs_log', params, 1)

        
        if auto_flow and 'flow' in el_name:         # Decide once whether this element is an auto-flow plane
            fi       = int(el_name.split('_')[1])   # e.g. flow_005_…
            position = float(el_name.split('_')[2])
        else:
            fi = position = None                    # will be ignored below 

        # ──────────────────────────────────────────────────────────────────────
        # 4. Loop over channels (compute, store and plot for every active channel)
        # ----------------------------------------------------------------------

        for ch_name, Fch in all_channels: # Channels are : main, VB_parr, VB_perp

            # ---- decide whether to plot this channel ----
            if ch_name == "main":
                do_plot_ch = do_plot                    # unchanged
            else:                                       # VB channels
                do_plot_ch = do_plot and yamlval('plot_VB', params, 1)

            # ---- choose phase vs intensity ----
            if yamlval('plot_phase', params, 0):
                I_ch = Phase(Fch)
            elif 'I_after_air' in locals() and ch_name == "main":
                I_ch = I_after_air
            else:
                I_ch = Intensity(0, Fch)
            
            # ----- Choice of unit for intensity -----

            if ch_name == "main" and ((beam_shaper_index is not None and ei == beam_shaper_index) or (beam_shaper_index is None and ei == 0)):

                photons_tot = params.get('photons_total')
                tau         = params.get('X_FWHM_duration')

                if photons_tot:
                    params['scale_phot'] = photons_tot / np.nansum(I_ch) / ((propsize / N)**2) # Factor to scale the map to photons/m^2. Unit is [photons / m^2]
                    print(f"scale photons {params['scale_phot']}")
                    print(f"∑I_ch = {np.nansum(I_ch):.3e}")
                    print(f"∑I_ch x dx^2 = {np.nansum(I_ch)*((propsize / N)**2):.3e}")


                if photons_tot and tau:
                    E_J   = params['photon_energy'] * e
                    params['scale_Wcm2'] = (photons_tot * E_J / tau) \
                                        / np.nansum(I_ch) / propsize**2 / 1e4

                    
            # --- optional rectangular ROI integrals (for SFA labels) -------------
            if 'roi' in el_dict and ch_name == "main":
                s_um               = 0.5 * el_dict['roi']                 # half-width [µm]
                mask               = np.abs(Na) <= s_um                   # Na is 1-D μm-axis
                intensities['roi'] = (np.nansum(I_ch[np.ix_(mask, mask)])* propsize**2)

            if 'roi2' in el_dict and ch_name == "main":
                s_um                = 0.5 * el_dict['roi2']
                mask                = np.abs(Na) <= s_um
                intensities['roi2'] = (np.nansum(I_ch[np.ix_(mask, mask)]) * propsize**2)
                
            # ------ bookkeeping ---------------------------------------------------
            Iint = np.nansum(I_ch) * propsize**2            # Summed intensity
            intensities[f"{el_name}_{ch_name}"] = Iint      # e.g. "Det_main"

            # ──────────────────────────────────────────────────────────────────────
            # 4. Summary figure for the main field (Intensity as a function of propagation)
            # ----------------------------------------------------------------------
            # --- store intensity trace for all channels ---
            trace_z_dict[ch_name].append(z)
            trace_I_dict[ch_name].append(Iint)
            trace_names_dict[ch_name].append(el_name)

            # For backward compatibility, still fill the legacy "main" arrays
            if ch_name == "main":
                trace_z.append(z)
                trace_I.append(Iint)
                trace_names.append(el_name)


                # left panel: transverse profile at this plane
                prof = np.sum(I_ch, axis=0)
                if yamlval('profiles_normalize', params, 1):
                    prof = mu.normalize(prof)

                profiles_xlim = yamlval('profiles_xlim', params, [0, 200])
                mask          = (Na >= profiles_xlim[0]) & (Na <= profiles_xlim[1])
                ax_prof.semilogy(Na[mask], prof[mask], color=rofl.cmap()(ei/numel))
                profiles_done = True
            # -----------------------------------------------------------------------
            # --- DEBUG FIGURE @ TCC: 2D main field in photons/m² over ±4 µm (x,y) --
            # -----------------------------------------------------------------------
            if el_name == "TCC" and ch_name in ("main", "VB_perp"):
                vb_dir = Path(params["projectdir"]) / "VB_figures"
                sim    = params.get("filename", "")
                tag    = "Xray" if ch_name == "main" else "VB_perp"
                title  = "Main @ TCC" if ch_name == "main" else "VB_perp @ TCC"
                _save_center_crop_debug(
                    I_ch, params,
                    fname_tag=tag,
                    title_tag=f"{sim} — {title}",
                    outdir=vb_dir,
                )
            # ----------------------------------------------------------------------
            # -------------------------- end DEBUG FIGURE --------------------------
            # ----------------------------------------------------------------------

            # ---- prepare image (log, zoom, …) ----

            im, norms, measures = prepare_image(
                I_ch, ps=propsize, max_pixels=max_pixels,
                ZoomFactor=ZoomFactor, log=logg,
                norms=norms,
                el_dict=el_dict,
                normalize=(unit_sel == 'relative')  # only normalize in relative mode
            )

            # --------------------------------------------------------------
            # Always keep *every* automatic flow slice for *all* channels
            if el_name.startswith("flow") and "flow" in figs_to_save:
                figs[f"{el_name}_{ch_name}"] = [im, ei, propsize/ZoomFactor, z]
            # --------------------------------------------------------------


            # ---- auto-save figs / flow images ----
            key = f"{el_name}_{ch_name}"
            if el_name.startswith(tuple(figs_to_save)):
                figs[key] = [im, ei, propsize/ZoomFactor, z]
            if fi is not None:
                flow_savefig(I_ch, ffdir, fi, propsize, f"{params['filename']}_{ch_name}", position)

            # ---- plotting (subfigures) -------------------------------------
            if do_plot_ch:

                # 1. get or create the figure for this channel
                if ch_name not in fig_by_ch:
                    fig_size           = (params['fig_cols'] * 3, params['fig_rows'] * 3)
                    fig_by_ch[ch_name] = plt.figure(figsize=fig_size)
                    pi_by_ch[ch_name]  = 1
                fig_ch = fig_by_ch[ch_name]
                pi_ch  = pi_by_ch[ch_name]

                # 2. open a new page if the grid is full
                if pi_ch > max_panels:
                    fig_by_ch[ch_name] = plt.figure(figsize=fig_size)
                    fig_ch             = fig_by_ch[ch_name]
                    pi_ch              = 1

                # Intensity unit selection
                if unit_sel == 'photons':
                    scale_ph = params.get('scale_phot', 1.0)      # 1.0 if not defined yet
                    vmin, vmax = [c * scale_ph for c in [1e-11, 50]]
                    label_unit = "photons / m²"
                    scale_plot = scale_ph
                elif unit_sel == 'Wcm2':
                    vmin, vmax = [c * params['scale_Wcm2'] for c in [1e-11, 50]]
                    label_unit = "W cm⁻²"
                    scale_plot = 1.0
                else:                             # relative
                    vmin, vmax = [1e-11, 50]
                    label_unit = "relative"
                    scale_plot = 1.0

                # 3. ----- Draw the Mosaic --------
                lab_ch = f"{el_name}"
                plt.figure(fig_ch.number)             # activate this channel's fig
                ax = plt.subplot(params['fig_rows'], params['fig_cols'], pi_ch)
                ax.set_facecolor("black")

                im_plot = im * scale_plot

                # --- choose color limits ---
                if mosaic_cbar_dynamic:
                    # per-image dynamic limits (robust to zeros/NaN)
                    pos = im_plot[im_plot > 0]
                    if pos.size:
                        vmin = float(np.nanmin(pos))
                        vmax = float(np.nanmax(pos))
                        if vmax <= vmin:  # edge case
                            vmax = vmin * 1.01
                    else:
                        vmin, vmax = 1.0, 1.0
                else:
                    # fixed range (your current constants)
                    base = [1e-11, 50]
                    if unit_sel == 'photons':
                        vmin, vmax = [c * scale_ph for c in base]
                    elif unit_sel == 'Wcm2':
                        vmin, vmax = [c * scale_plot for c in base]
                    else:
                        vmin, vmax = base

                # --- draw ---
                half_span_m = 0.5 * propsize / ZoomFactor   # [m]
                img = ax.imshow(
                    im_plot,
                    extent=[-half_span_m*1e6, +half_span_m*1e6,-half_span_m*1e6, +half_span_m*1e6],
                    origin='lower',
                    cmap=rofl.cmap(),
                    norm=LogNorm(vmin=max(vmin, np.finfo(float).tiny), vmax=vmax) if logg else None
                )

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(img, cax=cax)
                cbar.ax.tick_params(labelsize=6)
                cbar.set_label(label_unit, fontsize=7)
                
                ax.set_title(f"{lab_ch}", fontsize=9)
                ax.set_xlabel('x  [µm]', fontsize=8)
                ax.set_ylabel('y  [µm]', fontsize=8)
                
                # --------- Add the squares and Shadow factor on the Det pannel ----------
                if ch_name == "main" and el_name == "Det":
                    # draw the rectangles corresponding to the size of 1 camera pixel
                    if 'roi2' in el_dict:                       # SFA-75 (green, larger box)
                        s_um = 0.5 * el_dict['roi2']
                        ax.add_patch(plt.Rectangle((-s_um, -s_um), 2*s_um, 2*s_um,
                                                edgecolor='lime',  facecolor='none',
                                                lw=1.3))
                    if 'roi' in el_dict:                        # SFA-13 (red, smaller box)
                        s_um = 0.5 * el_dict['roi']
                        ax.add_patch(plt.Rectangle((-s_um, -s_um), 2*s_um, 2*s_um,
                                                edgecolor='red',  facecolor='none',
                                                lw=1.3))

                    # text at bottom-left
                    central_key = "TCC_main" if "TCC_main" in intensities else (
                                "PH_main"  if "PH_main"  in intensities else "start")
                    den    = intensities[central_key]
                    tr_scat = yamlval("transmission_of_scatterer_L2", params, 1)

                    if 'roi2' in intensities:
                        t75 = intensities['roi2'] / den / tr_scat
                        ax.text(0.02, 0.12, f"SF75 {t75:.1e}",
                                transform=ax.transAxes, color='lime', fontsize=8,
                                va='bottom', ha='left')

                    if 'roi' in intensities:
                        t13 = intensities['roi'] / den / tr_scat
                        ax.text(0.02, 0.02, f"SF13 {t13:.1e}",
                                transform=ax.transAxes, color='red', fontsize=8,
                                va='bottom', ha='left')

                # ----------------------------------------------------------------

                pi_by_ch[ch_name] = pi_ch + 1    # Advance the panel counter for this channel

        # ────────────────────────────────────────────────────────────────────
        
        if save_parts:
            for ch_name, fig_ch in fig_by_ch.items():
                fname = f"part/{params['filename']}__{ch_name}__{ei:02d}.jpg"
                plt.figure(fig_ch.number)
                mu.savefig(fname)
        
        if yamlval('end_after',params,'asdfasdfasdf')==el_name: break


    # ──────────────────────────────────────────────────────────────────────
    # 6 bis. Choose normalisation of intensity for summary plot
    # ----------------------------------------------------------------------
    if beam_shaper_index is not None:
        norm_idx = beam_shaper_index            # intensity is evaluated *after* element
    else:
        norm_idx = 0

    trace_z = np.asarray(trace_z, float)
    trace_I = np.asarray(trace_I, float)

    I0int = trace_I[norm_idx]
    trans  = trace_I / I0int

    # keep for later / pickling
    params['trace_z']       = trace_z
    params['trace_I']       = trace_I
    params['trace_trans']   = trans

    for ch_name in ("main", "VB_parr", "VB_perp"):
        if len(trace_z_dict[ch_name]) > 0:
            tz = np.asarray(trace_z_dict[ch_name], float)
            ti = np.asarray(trace_I_dict[ch_name], float)
            norm = ti[0] if ti[0] != 0 else 1.0
            params[f"trace_z_{ch_name}"] = tz
            params[f"trace_I_{ch_name}"] = ti
            params[f"trace_trans_{ch_name}"] = ti / norm
            print(f"[TRACE] Saved {ch_name} channel trace with {len(tz)} points")


    params['norm_index']    = int(norm_idx)
    params['norm_ref_name'] = trace_names[norm_idx]
    params['norm_ref_z']    = float(trace_z[norm_idx])

    # make 'start' consistent with this normalization choice
    intensities['start'] = float(I0int)

    # ──────────────────────────────────────────────────────────────────────
    # 7. stash results in params
    # ----------------------------------------------------------------------
    params['transmission'] =  trans
    params['intensities']  =  intensities
    params['integ']        =  integ

    if params['ax_apertures']!=None:
        plt.sca(params['ax_apertures'])
        plt.title('Apertures')
        plt.xlim(yamlval('profiles_xlim',params,[0,200]))
        plt.ylim(yamlval('apertures_ylim',params,[1e-10,2]))

    duration           = mu.print_times() # secondes
    params['duration'] = duration         # secondes

    if np.size(figs)>0:
        pkl_name = projectdir / "pickles" / f"{params['filename']}_figs"
        mu.dumpPickle(figs, str(pkl_name))
    if len(export)>0:
        pkl_name = projectdir / "pickles" / f"{params['filename']}_figs"
        mu.dumpPickle(figs, str(pkl_name))


    #------------ Save the .res pickle -----------
    elements_dict = {el[1]: el[2] for el in elements}

    res_name = projectdir / "pickles" / f"{params['filename']}_res"
    mu.dumpPickle((elements_dict, params), str(res_name))
    print(f"Saved RES pickle to {res_name}")

    # ─── finalise the summary figure (if anything was drawn) ───
    if profiles_done:
        # left panel cosmetics
        ax_prof.set_xlabel('Position [μm]')
        ax_prof.set_ylabel('Intensity')
        ax_prof.set_title('Intensity profiles')

        # right panel – cumulative transmission normalized at norm_ref
        ax_steps.clear()
        ax_steps.step(params['trace_z'], params['trace_trans'], where='post', color='tab:orange')
        ax_steps.set_xlabel('z  [m]')
        ax_steps.set_ylabel('∑ I  /  I(ref)')
        ax_steps.set_yscale('log')
        ax_steps.set_title('Cumulative transmission')


        # ----------------------------------------------------------
        # textual annotations
        info_lines = [f'N = {N}', f'I_ref = {I0int:.1e}', f'ref: {params["norm_ref_name"]} @ z={params["norm_ref_z"]:.2f} m']

        if 'TCC_main' in intensities and 'start_main' in intensities:
            r = intensities['TCC_main']/intensities['start_main']
            info_lines.append(f'start→TCC = {r:.1e}')
        
        I_start = params['intensities']['start']

        for i, txt in enumerate(info_lines):
            ax_steps.text(0.02, 0.98 - i*0.08, txt,
                        transform=ax_steps.transAxes,
                        va='top', ha='left',
                        fontsize=10,
                        color=('black'))
            
        # ------ Figure Summary saving ------------------------------
        fig_summary.tight_layout(pad=1.2)
        plt.figure(fig_summary.number)
        out_sum = projectdir / "figures" / f"{params['filename']}_summary.jpg"
        mu.savefig(str(out_sum))

    # ---------- Mosaic figure saving ------------------------------

    for ch_name, fig_ch in fig_by_ch.items():
        fig_ch.suptitle(f"Channel: {ch_name}, intensity unit: {unit_sel}", fontsize=14, y=0.97) # Add a title to the full figure
        out_name = projectdir / "figures" / f"{params['filename']}_{ch_name}.jpg"
        plt.figure(fig_ch.number)
        mu.savefig(str(out_name))

    # -----------------------------------------------------------------
    # 8. flow-plot for every channel that may exist
    # -----------------------------------------------------------------
    
    flow_list = params.get("flow", [])
    if not (isinstance(flow_list, (list, tuple)) and len(flow_list) > 0):
        print("[FLOW] Disabled (no windows provided).")
        return params, trans, figs

    flow_list = params.get("flow", None)
    xl = None
    if isinstance(flow_list, (list, tuple)) and len(flow_list) >= 2:
        xl = [float(flow_list[0]), float(flow_list[1])]
        print(f"[FLOW] Using x-limits from YAML flow = {xl}")


    # ─── Define vertical range for flow plot: gyax ───
    initial_propsize_m = params.get("initial_propsize", 1000)  # fallback if not set earlier
    Zoom_flow = params.get("Zoom_global", 1)
    half_box_um = (initial_propsize_m / Zoom_flow) * 1e6 / 2
    print(
        f"[FLOW] Using zoom-aware box: "
        f"propsize={initial_propsize_m*1e6:.1f} µm, "
        f"Zoom={Zoom_flow} → window={2*half_box_um:.1f} µm")
    gyax_step = 1  # resolution in µm
    gyax_def = [-half_box_um, half_box_um, gyax_step]
    params["flow_plot_gyax_def"] = gyax_def



    for ch in ("main", "VB_parr", "VB_perp"):
        try:
            print(f"[DEBUG] flow_plot() for channel={ch} → gyax_def = {gyax_def}")

            flow_plot(projectdir,
                    params['filename'],
                    vertical_type="center",
                    channel=ch,
                    gyax_def=gyax_def,xl=xl)
        except AssertionError:
            # channel wasn’t recorded – simply skip
            pass


    # ─── Overlay flow figure (main + VB_perp) ───
    try:
        overlay_flowplots(
            project_dir=projectdir,
            file_stem=params['filename'],
            requested_channels=("main", "VB_perp"),
            unit_mode=params.get("intensity_units", "photons"),
            log_mode=True,
        )
    except Exception as error_p:
        print(f"[OVERLAY] Skipped due to error: {error_p}")

    # ---------------------------------------------------------------
    # 9. OPTIONAL: Save individual plane images (YAML controlled)
    # ---------------------------------------------------------------
    if params.get("save_individual_figures", 0):
        print("[save_planes] Saving individual figures…")
        save_individual_plane_figures(
            project_dir=params["projectdir"],
            yaml_tag=params["filename"],
            save_units=params.get("intensity_units", "photons")
        )



    return params, trans, figs







# ================================================================================
# Integration of the library DABAM to simulate lenses defects with high accuracy
# ================================================================================



def _read_dabam_meta(txt_path: Path) -> dict:
    if not txt_path.exists():
        return {}
    raw = txt_path.read_text()
    # 1) try JSON
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 2) try Python-literal dict (DABAM2D README shows this style)
    try:
        meta = ast.literal_eval(raw)
        if isinstance(meta, dict):
            return meta
    except Exception:
        pass
    # 3) very simple "key: value" fallback
    meta = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta




def load_dabam2d(index: int, dabam_root: str | Path = None):
    """
    Load a DABAM2D surface by index.
    Returns: X (Nx,), Y (Ny,), Z (Ny,Nx), meta (dict)
    """
    if dabam_root is None:
        dabam_root = Path(__file__).parent / "Dabam2D" / "data"
    else:
        dabam_root = Path(dabam_root)

    stem = f"dabam2d-{index:04d}"
    h5_path  = dabam_root / f"{stem}.h5"
    txt_path = dabam_root / f"{stem}.txt"

    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found. Clone the DABAM2D repo into src/darkfield/Dabam2D.")

    with h5py.File(h5_path, "r") as f:
        X = f["surface_file/X"][()]  # horizontal
        Y = f["surface_file/Y"][()]  # vertical
        Z = f["surface_file/Z"][()]  # shape (Y.size, X.size)

    meta = _read_dabam_meta(txt_path)
    return X, Y, Z, meta





def build_dabam_defect_thickness_map(el_dict, params, xm, ym, A_m):
    """
    Interpolate all DABAM maps to the simulation grid, sum them, and compute
    Zernike fit / residues on the lens ROI (disk).

    New semantics:
      - LF (Low frequency):
          remove_avg_profile=0 → LF = Zernike fit (or custom Zernike if active)
          remove_avg_profile=1 → LF = Zernike fit + (removed radial-average from residues)
      - HF (High frequency):
          remove_avg_profile=0 → HF = raw residues
          remove_avg_profile=1 → HF = residues after avg-removal
      - defect_type (yaml): 1=LF, 2=HF, 3=LF+HF → returned full-grid map follows this.

    Caches for plotting (ROI-based):
      Z_lowfreq_roi, Z_highfreq_roi, Z_total_lf_hf_roi
      (plus existing caches: Z_fit_panel_map, Z_residues_native, etc.)
    """
    import re
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    import vibe.wavefront_fitting as wft
    from vibe.VIBE import load_dabam2d

    print("[DABAM][LF/HF] builder active")

    # ---- inputs / yaml
    defects_flag = int(el_dict.get("defects", 0))
    expdata_raw  = el_dict.get("experimental_data", None)
    defect_type  = int(el_dict.get("defect_type", 3))         # 1=LF, 2=HF, 3=LF+HF
    remove_avg   = int(el_dict.get("remove_avg_profile", 0))  # 0/1
    nmodes       = int(el_dict.get("nmodes", 37))
    startmode    = int(el_dict.get("startmode", 1))
    nb_lenses    = int(el_dict.get("nb_lenses", 1))
    rescale_flag = int(el_dict.get("rescale_dabam", 1))

    # normalize dataset list
    if isinstance(expdata_raw, (list, tuple)):
        exp_list = list(expdata_raw)
    elif isinstance(expdata_raw, (str, int, np.integer)):
        exp_list = [expdata_raw]
    else:
        exp_list = []

    if defects_flag != 1 or len(exp_list) == 0:
        return np.zeros_like(xm)

    # pad/crop to nb_lenses
    if len(exp_list) == 1:
        exp_list *= nb_lenses
    elif len(exp_list) < nb_lenses:
        exp_list += [exp_list[-1]] * (nb_lenses - len(exp_list))
    else:
        exp_list = exp_list[:nb_lenses]

    # ---- sim grid & lens mask
    X_sim = xm[0, :]
    Y_sim = ym[:, 0]
    Rdst  = A_m / 2.0
    r     = np.sqrt(xm**2 + ym**2)
    disk_mask = (r <= Rdst)

    # ---- accumulate resampled RAW maps
    Zraw_total = np.zeros_like(xm)
    dabam_sets = []
    first_Rsrc = None
    resample_note = None

    for entry in exp_list:
        # parse an index
        if isinstance(entry, (int, np.integer)):
            idx = int(entry)
        else:
            s = str(entry).strip()
            m = re.search(r"dabam2d-([0-9]+)$", s, flags=re.IGNORECASE)
            if m:
                idx = int(m.group(1))
            else:
                nums = re.findall(r"([0-9]+)", s)
                idx = int(nums[-1]) if nums else None
        if idx is None:
            print(f"[DABAM] Skipping invalid entry: {entry}")
            continue

        X, Y, Z, meta = load_dabam2d(idx)  # Z in meters

        # coord sanity (µm → m) if needed
        if np.abs(X).max() > 1e-3 or np.abs(Y).max() > 1e-3:
            print(f"[DABAM] dabam2d-{idx:04d} coords look like µm → converting to meters.")
            X = X * 1e-6
            Y = Y * 1e-6

        # native per-map fit/ress
        Zc, Zfit_i, Zres_i = wft.fit_zernike_circ(Z, nmodes=nmodes, startmode=startmode, rec_zern=True)
        Zres_i = -Zres_i
        extent_um = [float(X.min()*1e6), float(X.max()*1e6),
                     float(Y.min()*1e6), float(Y.max()*1e6)]
        dabam_sets.append({"idx": idx, "Z_raw": Z, "Zfit": Zfit_i, "Zres": Zres_i, "extent_um": extent_um})

        # resample RAW to sim grid
        Rx = 0.5 * (X.max() - X.min()); Ry = 0.5 * (Y.max() - Y.min())
        Rsrc = min(Rx, Ry) if (Rx > 0 and Ry > 0) else 1.0
        if first_Rsrc is None:
            first_Rsrc = Rsrc

        interp = RegularGridInterpolator((Y, X), Z, bounds_error=False, fill_value=0.0)

        if rescale_flag == 1:
            scale = Rdst / Rsrc if Rsrc > 0 else 1.0
            pts = np.column_stack([(ym / scale).ravel(), (xm / scale).ravel()])
            Z_res = interp(pts).reshape(xm.shape)
            resample_note = "stretch to fill" if scale > 1 else ("shrink to fill" if scale < 1 else "1:1")
        else:
            pts = np.column_stack([ym.ravel(), xm.ravel()])
            Z_res = interp(pts).reshape(xm.shape)
            if Rsrc < Rdst:   resample_note = "native smaller than lens (zeros outside)"
            elif Rsrc > Rdst: resample_note = "native larger than lens (masked at edge)"
            else:             resample_note = "1:1 size (native == lens)"

        Z_res = np.where(disk_mask, Z_res, 0.0)
        Zraw_total += Z_res

    # ---- tight ROI (box around disk)
    jj, ii = np.where(disk_mask)
    j0, j1 = jj.min(), jj.max()
    i0, i1 = ii.min(), ii.max()
    Zraw_roi = Zraw_total[j0:j1+1, i0:i1+1]
    X_roi    = X_sim[i0:i1+1]
    Y_roi    = Y_sim[j0:j1+1]
    disk_roi = disk_mask[j0:j1+1, i0:i1+1]
    Zraw_roi = np.where(disk_roi, Zraw_roi, 0.0)

    # ---- Zernike on ROI
    Zc_tot, Zfit_roi, Zres_roi = wft.fit_zernike_circ(
        Zraw_roi, nmodes=nmodes, startmode=startmode, rec_zern=True
    )
    Zres_roi = -Zres_roi
    Zres_roi_pre = Zres_roi.copy()

    # ---- Custom Zernike overwrites the fit if active
    custom_list   = el_dict.get("Custom_zernike", [])
    custom_active = (isinstance(custom_list, (list, tuple))
                     and len(custom_list) >= 1 and int(custom_list[0]) == 1)

    def _center_fit_to_size(arr, out_shape):
        import numpy as _np
        ay, ax = arr.shape
        ty, tx = out_shape
        if ay > ty:
            y0 = (ay - ty) // 2
            arr = arr[y0:y0 + ty, :]
        if ax > tx:
            x0 = (ax - tx) // 2
            arr = arr[:, x0:x0 + tx]
        ay, ax = arr.shape
        py_top  = (ty - ay) // 2
        py_bot  = (ty - ay) - py_top
        px_left = (tx - ax) // 2
        px_right= (tx - ax) - px_left
        if py_top or py_bot or px_left or px_right:
            arr = _np.pad(arr, ((py_top, py_bot), (px_left, px_right)),
                          mode='constant', constant_values=0.0)
        return arr

    if custom_active:
        pairs = list(custom_list[1:])
        if len(pairs) % 2 != 0:
            raise ValueError("Custom_zernike must be [1, n1, v1, n2, v2, ...]")
        nolls = [int(pairs[i])   for i in range(0, len(pairs), 2)]
        vals  = [float(pairs[i]) for i in range(1, len(pairs), 2)]
        max_n = max(nolls) if nolls else 0

        zvec = np.zeros(max(1, max_n), dtype=float)
        for n, v in zip(nolls, vals):
            zvec[n-1] = v

        pxsize   = params['pxsize']
        Rdst_px  = int(round((A_m / 2.0) / pxsize))
        diam_px  = 2 * Rdst_px + 1
        Zfit_custom_sq = wft.calc_zernike_circ(zvec, rad=Rdst_px, mask=True)

        ret_diam = Zfit_custom_sq.shape[0]
        if ret_diam != Zfit_custom_sq.shape[1]:
            raise RuntimeError(f"[CustomZernike] Non-square map returned: {Zfit_custom_sq.shape}")
        if ret_diam < (2*Rdst_px + 1):
            from scipy.ndimage import zoom as _zoom
            scale = (2*Rdst_px + 1) / float(ret_diam)
            print(f"[CustomZernike] Upscaling from {ret_diam}px to {2*Rdst_px + 1}px ×{scale:.3f}")
            Zfit_custom_sq = _zoom(Zfit_custom_sq, zoom=scale, order=1, mode="nearest")

        Zfit_custom_sq  = _center_fit_to_size(Zfit_custom_sq, (diam_px, diam_px))
        Zfit_custom_roi = _center_fit_to_size(Zfit_custom_sq, Zraw_roi.shape)
        Zfit_roi = np.where(disk_roi, Zfit_custom_roi, 0.0)
        el_dict["Zfit_is_custom"] = 1
    else:
        el_dict["Zfit_is_custom"] = 0

    # ---- Build LF/HF
    Xm_roi, Ym_roi = np.meshgrid(X_roi, Y_roi)
    R_roi = np.sqrt(Xm_roi**2 + Ym_roi**2)

    if remove_avg == 1:
        # radial average of residues
        I_thick_res, Rprof = wft.average_azimuthal(Zres_roi, X_roi, Y_roi)
        # remove-average (for HF)
        _, Zres_roi_post = wft.remove_avg_profile(Zres_roi, None, X_roi, Y_roi,
                                                  I_thick_res, Rprof, 'b')
        # reconstruct removed part to 2-D
        avg_map = np.interp(R_roi, Rprof, I_thick_res, left=I_thick_res[0], right=0.0)
        avg_map = np.where(disk_roi, avg_map, 0.0)

        Z_low_roi  = Zfit_roi + avg_map     # LF
        Z_high_roi = Zres_roi_post          # HF
    else:
        Z_low_roi  = Zfit_roi               # LF
        Z_high_roi = Zres_roi               # HF
        Zres_roi_post = Zres_roi            # for caching

    Z_total_lf_hf_roi = Z_low_roi + Z_high_roi

    # ---- Choose the map FOR THE SIMULATION 
    if defect_type == 1:        # LF only
        Z_final_roi = Z_low_roi
    elif defect_type == 2:      # HF only
        Z_final_roi = Z_high_roi
    else:                       # 3 → LF + HF
        Z_final_roi = Z_total_lf_hf_roi

    # embed to full grid
    Z_final_full = np.zeros_like(xm)
    Z_final_full[j0:j1+1, i0:i1+1] = np.where(disk_roi, Z_final_roi, 0.0)

    # ---- caches for plotting
    el_dict["dabam_sets"] = dabam_sets
    el_dict["Z_raw_native"] = Zraw_total
    el_dict["Z_interp_extent_um"] = [X_sim.min()*1e6, X_sim.max()*1e6,
                                     Y_sim.min()*1e6, Y_sim.max()*1e6]

    el_dict["Z_fit_panel_map"]          = Zfit_roi
    el_dict["Z_residues_native_unproc"] = Zres_roi_pre
    el_dict["Z_residues_native"]        = Zres_roi_post
    el_dict["Z_coeffs_native"]          = Zc_tot
    el_dict["Z_fit_axes_um"] = [X_roi.min()*1e6, X_roi.max()*1e6,
                                Y_roi.min()*1e6, Y_roi.max()*1e6]

    # NEW LF/HF caches (ROI)
    el_dict["Z_lowfreq_roi"]     = Z_low_roi
    el_dict["Z_highfreq_roi"]    = Z_high_roi
    el_dict["Z_total_lf_hf_roi"] = Z_total_lf_hf_roi

    # final applied map (full grid)
    el_dict["Z_def_resampled"]       = Z_final_full
    el_dict["Z_total_def_native"]    = Z_final_full
    el_dict["Z_total_def_extent_um"] = el_dict["Z_interp_extent_um"]

    # resample info
    el_dict["Z_resample_info"] = {
        "policy": ("rescale: fill aperture" if rescale_flag == 1 else "no-rescale: keep native size"),
        "note": (resample_note or ""),
        "rescale_flag": int(rescale_flag),
        "Rsrc_um": (float(first_Rsrc*1e6) if first_Rsrc is not None else None),
        "Rdst_um": float(Rdst*1e6),
        "scale": (float((Rdst/first_Rsrc) if (first_Rsrc and first_Rsrc > 0) else 1.0)),
    }

    print(f"[DABAM][APPLY] defect_type={defect_type} → "
          f"{'LF' if defect_type==1 else 'HF' if defect_type==2 else 'LF+HF'} map applied. "
          f"remove_avg={remove_avg}")
    return Z_final_full









# ================================================================================
# Simulation launch section
# ================================================================================


if __name__ == "__main__":
    _select_backend()
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="Run VIBE directly from a YAML.")
    ap.add_argument("--yaml", required=True, help="Path to YAML config")
    ap.add_argument("-N", type=int, default=1000, help="Number of simulation points")
    args = ap.parse_args()

    run_from_yaml(args.yaml, args.N)


