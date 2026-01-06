import numpy as np
import sys
import os
import re

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg

import darkfield.rossendorfer_farbenliste as rofl
import darkfield.mmmUtils_v2 as mu

from importlib import reload

from itertools import product

import time
import random
import yaml
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


#HOME = '/home/yu79deg/darkfield_p5438/'

# Template and output setup
yaml_template = '/home/yu79deg/darkfield_p5438/yamls/LP_154.yaml' #name of the template yaml file that we modify
outdir = '/home/yu79deg/darkfield_p5438/yamls' # folder where it will save the generate yamls
os.makedirs(outdir, exist_ok=True)

########### PARAMETERS TO SET AND SCAN #############

#param_dict = {
#    'O2in': [0, 1],
#    'wide_factor': np.array([0.5, 1, 1.25, 1.5, 2, 3, 4, 5, 6, 7, 10]),
#    'O_mult': np.array([0.5, 1, 1.3]),
#    'A_mult': np.array([0.5, 1, 1.3]),
#}

param_dict = {
    'A1_size': [100,90,80,70,60,50,40,30,20,10,0], 
}

####################################################

# Extract starting number from the template filename
match = re.search(r'LP_(\d+)', yaml_template)
if match:
    start_index = int(match.group(1)) + 1
else:
    raise ValueError("Template name must include a number like 'LP_28_template.yaml'")

# Prepare combinations of parameters
param_names = list(param_dict.keys())
param_values = list(param_dict.values())
combinations = list(product(*param_values))

# Optional: specify defect pattern
#defect = 'def-20-2'

# Loop over all parameter combinations
for i, combo in enumerate(combinations):
    with open(yaml_template) as f:
        ip = yaml.safe_load(f)  # Reload fresh template each time

    params = dict(zip(param_names, combo))

    # Apply wide_factor if present
    if 'wide_factor' in params:
        wf = params['wide_factor']
        ip['beam_shaper']['size'] *= wf
        ip['L1']['size'] *= wf
        ip['L2']['size'] *= wf
        ip['beam']['size'] *= wf
    else:
        wf = 1.0  # fallback for dependent parameters

    # Apply O_mult if present
    if 'O_mult' in params:
        Om = params['O_mult']
        ip['O1']['size'] *= wf * Om
        ip['O2']['size'] *= wf * Om
    elif 'wide_factor' in params:
        ip['O1']['size'] *= wf
        ip['O2']['size'] *= wf

    # Apply A_mult if present
    if 'A_mult' in params:
        Am = params['A_mult']
        ip['A1']['size'] *= wf * Am
        ip['A2']['size'] *= wf * Am
    elif 'wide_factor' in params:
        ip['A1']['size'] *= wf
        ip['A2']['size'] *= wf

    # Apply O2in if present
    if 'O2_size' in params:
        O2_s = params['O2_size']
        if O2_s==0:
            ip['O2']['in'] = 0 #if the size of O2 = 0 we put it out
        else:
            ip['O2']['in'] = 1
            ip['O2']['size'] = O2_s * 1e-6
            
    if 'PH_size' in params:
        PH_s = params['PH_size']
        if PH_s==0:
            ip['PH']['in'] = 0 #if the size of O2 = 0 we put it out
        else:
            ip['PH']['in'] = 1
            ip['PH']['size'] = PH_s * 1e-6

    if 'A1_size' in params:
        A1_s = params['A1_size']
        if A1_s==0:
            ip['A1']['in'] = 0 #if the size of A1 = 0 we put it out
        else:
            ip['A1']['in'] = 1
            ip['A1']['size'] = A1_s * 1e-6

            
    # Apply defect if requested
    #if defect:
    #    defective = ['O1', 'O2', 'A1', 'A2']
    #    for di, dd in enumerate(defective):
    #        ip[dd]['defect_type'] = 'sine'
    #        ip[dd]['defect_lambda'] = float((20 + di * 0.2) * 1e-6)
    #        ip[dd]['defect_amplitude'] = float(2e-6)

    ###################### Save modified YAML ########################
    filename = f'LP_{start_index + i}.yaml'
    fullpath = os.path.join(outdir, filename)
    with open(fullpath, 'w') as f_out:
        yaml.dump(ip, f_out, sort_keys=False)

    # Print the filename and parameter values for this scan
    param_str = ', '.join(f"{k}={v}" for k, v in params.items())
    print(f"Written {filename} ({param_str})")