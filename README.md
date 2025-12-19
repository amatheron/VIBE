
# VIBE
Vacuum Interaction Birefringence Explorer
=======
<p align="center">
  <img src="docs/VIBE_logo.png" alt="VIBE Logo" width="300"/>
</p>


## What this project does

**VIBE (Vacuum Interaction Birefringence Explorer)** is a **diffraction-based simulation toolkit** for modelling **quantum-vacuum signals**, in particular **vacuum birefringence**, in **realistic laser–laser and laser–XFEL pump–probe experiments**.

The code extends the **diffractive beam propagation framework LightPipes** by embedding a **quantum-vacuum emission module** directly into the optical propagation pipeline. This makes it possible to simulate, within a *single and self-consistent framework*:

- the **generation** of vacuum birefringence signals at the pump–probe interaction point, and  
- the **subsequent propagation** of both the probe background and the induced quantum-vacuum signal through a **complete experimental beamline**, including lenses, apertures, masks, slits, and detectors.

### Core idea

Instead of treating vacuum birefringence as a purely far-field or analytical effect, VIBE expresses the dominant quantum-vacuum contribution at the probe frequency as a **virtual source field** defined in the interaction plane.  
This source field is constructed from the **local overlap between the pump intensity and the probe field**, and is then propagated using the **same Fresnel diffraction formalism** as any conventional optical field.

As a result, quantum-vacuum signals are handled **on exactly the same footing** as standard diffraction, imaging, and absorption effects.

### What this enables

VIBE allows one to:

- simulate **vacuum birefringence pump–probe experiments** with **experimentally realistic beam profiles**,  
- include **diffraction, absorption, clipping, and aberrations** from optical components,  
- propagate **multiple polarization channels** (probe, VB∥, VB⊥) consistently through the setup,   
- estimate **absolute signal photon numbers** at the detector with moderate numerical cost.


## Citation

If you use **VIBE** in your work, please cite the software as:

```bibtex
@software{VIBE,
  author  = {Aimé Matheron, Michal Smíd, Matt Zepf and Felix Karbstein},
  title   = {VIBE: Vacuum Interaction Birefringence Explorer},
  year    = {2025},
  version = {v1.0.0},
  doi     = {10.5281/zenodo.17979735},
  url     = {https://github.com/amatheron/VIBE}
}

```


---
## Code structure

```text
VIBE/
├── src/
│   └── vibe/
│       ├── VIBE.py                       # Main simulation engine
│       ├── utils.py                      # Generic utilities
│       ├── bash_config.py                # HPC / batch job helpers
│       ├── mmmUtils_v2.py                # Numerical & optical helper routines
│       ├── wavefront_fitting.py          # Wavefront analysis tools
│       ├── regularized_propagation_v2.py # Regularized propagation tool
│       ├── rossendorfer_farbenliste.py   # Ploting custom colormaps
│       ├── optical_constants/            # Folder containing refractive index data for all used materials, from https://henke.lbl.gov/optical_constants/getdb2.html
│       │   ├── Be.txt                    # Refractive indices for Beryllium in a certain energy range (can be updated if necessary).
│       │   └── diamond.txt               # Refractive indices for Diamond in a certain energy range (can be updated if necessary).
│       └── Dabam2D/                      # 2D surface-defect maps (mirrors / optics)
│
├── VIBE_outputs/                  # All simulation outputs and diagnostics
│   ├── figures/                   # Mosaic figures with all interest planes for all channels
│   ├── flows/                     # Flow plots = side views / waterfall like plots
│   ├── pickles/                   # Simulation pickle. Warning : can be heavy.
│   ├── planes/                    # Individual ploted planes
│   ├── Lens_diags/                # Lens diagnostics. Ideal parabollic profiles and defects (via DABAM 2D)
│   └── VB_figures/                # Interaction plane distribution of pump, probe and signal
│
├── notebooks/                     # Example notebooks
├── yamls/                         # Example YAML file = simulation input file
├── docs/                          # Documentation assets (logo, figures)
└── pyproject.toml                 # Packaging and dependencies
```
---

## Simulation workflow (how the code operates)

`main_VIBE(params, elements)` orchestrates the entire run:

1. **Initialization**  
   - Creates a LightPipes field with a given source shape (Gaussian, Super-Gaussian, Flat-top, Experimental map etc...).
   - Wraps the field in a **FieldBundle** so additional channels (VB∥, VB⊥) can be carried transparently. The bundle also holds the current **z-position** and flags for **regularized propagation**.

2. **Book-keeping & plotting setup**  
   - Prepares figures, mosaics, “flow” exports, intensity unit selection (`relative` vs `photons`), and stores traces for post-run normalization.

3. **Main loop over elements in z**  
   For each element `(z, name, dict)`:
   - **Propagate** the bundle by `Δz` via FFT Fresnel or regularized propagation. Pixel size and physical window are updated after each hop.  
   - **Apply** the element to **every active channel** in the bundle (`main`, `VB∥`, `VB⊥`) using `apply_element(...)`.  
   - **Create VB channels at Interaction plane (called TCC) (once):** if `name=='TCC'` and `VB_signal: 1`, build VB masks from the IR intensity map and spawn VB∥/VB⊥ fields via masked multiplication.  
   - **Compute intensities** per channel, in requested units, and generate per-plane plots/snapshots. For flows, the Y-axis unit and colorbar label adapt to the selected intensity unit; photon scaling is applied consistently to raw and fixed grids.

4. **Outputs**  
   - Returns updated `params`, a transmission trace, and a dict of figures for saving/export.

---

## Inside `VIBE.py` — important building blocks

### Field containers & propagation
- **`FieldBundle` dataclass** groups LightPipes fields under channel keys (currently `"main"`, later `"VB_parr"`, `"VB_perp"`), with a shared z-position and regularization flags.  
- **`propagate_bundle(bundle, dz, method)`** advances all channels by the same distance using `Forvard` (FFT) or `Fresnel`, or a **regularized** variant when enabled.

### Optical constants & phase/thickness conversions
- **`get_index(elem, E)`** loads δ, β by **interpolating Henke tables** from `optical_constants/<Elem>.txt`. Results feed the **phase shift per meter** and **absorption coefficient** used to convert thickness maps to transmission/phase maps.
- **`thickness_to_phase_and_transmission(E_eV, delta, beta)`** computes these per-meter conversion factors (phase and exponential attenuation) for a given energy. Used throughout to turn geometry into physics.

### Apertures, lenses, defects → field operators
- **`get_aperture_transmission_map(...)`** builds **binary/analytic transmission maps** for simple shapes (`square`, `rectangle`, `wire`, **`gaussian` blocker**) including super-Gaussian order.  
- **`get_aperture_thickness_map(...)`** builds **thickness maps** for **refractive elements** and complex “wire-like” obstacles (`parabolic_lens`, `streichlens`, `realwire`, `trapez`, `tent`, `par`, `invpar`, `wireslit`, `wire_grating`, etc.), plus orientation and randomization hooks.  
- **`doap(...)`** converts thickness to **transmission** and **phase** maps via δ, β and applies optional profiling, rotation, and diagnostic plotting (1D cuts and 2D views).  
- **`do_edge_damping_aperture(...)`** creates smooth apodizers to suppress wrap-around/edge artefacts when the field is close to the boundaries of the box.

### Element application 
- **`apply_element(bundle, el_name, el_dict, params, reg_prop_dict, ...)`** is the **central dispatcher** that:
  - iterates over active channels;
  - applies **apertures/phaseplates** by multiplying intensity and/or phase maps.

### VB creation at TCC
- **Pump laser profile**: Two options exist — **Direct FFT** for a Near-field profile, calculates the corresponding Far-field focal spot; **External map** to choose an experimental map to input for the pump laser; both normalized to power and converted to **I [W cm⁻²]** given `P_peak` (from pulse energy or direct).  
- **Masks & channels**: Using the analytical prefactors (critical intensity `I_cr`, fine-structure constant and geometry), the code builds **VB∥** and **VB⊥** **intensity masks** at TCC and **spawns the new channels** by multiplying the main field. From that point on, **all channels propagate together**.


### Plotting & flows
- **Per-plane figures**: phase or intensity per channel; configurable mosaic, zoom, log scaling.  
- **Flow plots**: longitudinal waterfall of vertical cuts, with **unit-aware** colorbars (`photons` → “photons/m²”, else relative/W m⁻²) and profile overlays of the propagated box size. Detector and ROI are annotated; **SNR** and **Number of signal and background photons** can be printed on the panel.


## Install VIBE from scratch and run a simulation
1. **Cloning the VIBE repository** 
   Clone the VIBE repository :
   ```bash
   git clone https://github.com/amatheron/VIBE.git
   cd VIBE
   ```
   
2. **Environment preparation** (Micromamba recommended)
   We recommend the use of a micromambda environment, a particularly clean, light and easy solution for HPC simulations. Note however that VIBE will also run on any other (recent) python environment.
   
   When micromamba is available on the machine, initialize the shell support :
   ```bash
   micromamba shell init -s bash -p ~/micromamba
   source ~/.bashrc
   ```
   Then create a dedicated VIBE environment :
   ```bash
   micromamba create -n vibe python=3.10 -y
   ```
   and activate the environment :
   ```bash
   micromamba activate vibe
   ```
3. **Installing VIBE** 
   Install the VIBE repository using the pyproject.toml file :
   ```bash
   pip install -e .
   ```
   This command will install VIBE, resolve all dependencies automatically from pyproject.toml and install optional plotting backends.
   
   You can verify that the installation worked well with :

   ```bash
   python - << EOF
   import vibe
   print("VIBE successfully installed")
   EOF
   ```
 4. **Run the first example simulation**

    Try to run the first example simulation. The corresponding yaml file is /VIBE/yaml/Simulation_example_1.yaml

    An automatic bash script has been written to help with job submission. This bash script is handled by /VIBE/src/vibe/bash_config.py.
    The file bash_config.py **must be updated to fit your HPC requirements.**

    When updated, open the template Notebook /VIBE/notebooks/RUN_vibe_example.ipynb and execute :
    ```bash
    import os
    from vibe.bash_config import write_bash
    ```
    and,
    ```bash
    upd_params = {
        'n_cpus': 24,
        'mem': '600GB',
        'yaml': 'Simulation_example_1.yaml'
    }
    
    bash_path = write_bash("bash", 300, upd_params)
    print("Generated:", bash_path)
    ```
    to run the first simulation.

    Check if the simulation logs in VIBE/bash/bash_output/Simulation_example_1.log and VIBE/bash/bash_output/Simulation_example_1.err.

    
---

## Input yaml file description :

**Two examples of yaml input files can be found in /VIBE/yaml/** :
- **"Simulation_example_1.yaml"** is a simple simulation corresponding to the geometry used in Figure 2 of the paper. It simulates a simple flat-top probe beam colliding a Gaussian like pump at z = 0.
- **"Simulation_definitions.yaml"** is a complete file containing all possible options to include into an input file. Note that this file obviously does not run on its own since some options contradict each other. This is made to serve as a reference for yaml file writing on VIBE.

**Description of some blocks of yaml file :**

- **"beam_shaper"** serves as a reference for the total number of photons of the probe, as well as the shape of the initial probe beam. The code will make sure that the number of photons defined in Xbeam/photons_total is met at the beam_shaper position (therefore scalling the full simulation field by a factor "scale_ph"). In addition, this plane can be defined as an aperture to shape the beam to whatever geometry. For flat-top and Gaussian geometry, the "Super-Gaussian" formula is used, where the lineout shape $f(x)$ is expressed as :
$f(x) = f_0 \exp \left( -\left(\frac{x^2}{2\sigma^2} \right)^P\right)$,
with $P$ the order of the gaussian, set in the code with the option "power" (power = 1 corresponds to a standard Gaussian, and power > 6 correspond to profiles very close to flat-top beams), and $\sigma = \frac{FWHM}{2\sqrt{2} \ln(2)^{1/(2P)}}$, the "rms" size of the Super-gaussian, meeting the regular definition for a Gaussian, and getting close to the full diameter of the flat-top beam for P>6. In the code, the beam size corresponds to the FWHM (in intensity) and is indicated by the variable "size".

 
- **"Custom_CRL"** allows to simulate a Compound Refractive Lens (CRL) stack using the standard equations. The CRL is defined with :
  A parabolic shape on each side such that their total thickness can be written : $\Delta_z(x,y) = \frac{x^2+y^2}{ROC}$ for $\sqrt{x^2+y^2}<A$ and $\Delta_z(x,y)=L$ for $\sqrt{x^2+y^2}>A$. The lens geometric aperture $A$ is linked to the radius of curvature $ROC$ and apex thickness $t_{wall}$ by $A=2\sqrt{ROC(L-t_{wall})}$. The total focal length of the stack is defined as : $f=\frac{ROC}{2N\delta}$ with $N$ the number of lenses in the stack and $delta$ the (x-ray defined) refractive index.
In the code the Custom_CRL block is parametrised by  : ROC (radius of curvature), L (lens thickness), A (Lens geometric aperture), nb_lenses (the number of individual lenses in the stack) and lens_material, the material for the lens (only "diamond" and "Be" are implemented at the moment).

 



