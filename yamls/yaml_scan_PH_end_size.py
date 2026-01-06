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

def convert_np_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_to_native(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

#HOME = '/home/yu79deg/darkfield_p5438/'

# Template and output setup
yaml_template = '/home/yu79deg/darkfield_p5438/yamls/LP_621.yaml' #name of the template yaml file that we modify
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
    'PH_size': [0,1,2,5,8,10,15,20], 
}


####################################################

# Extract starting number from the template filename
template_name = os.path.basename(yaml_template)  # e.g., 'Felix_1.yaml'
match = re.match(r'([A-Za-z]+)_(\d+)\.yaml$', template_name)

if match:
    prefix = match.group(1)        # e.g., 'Felix'
    start_index = int(match.group(2)) + 1  # e.g., 2
else:
    raise ValueError("Template name must follow format like 'Felix_1.yaml' or 'LP_28.yaml'")

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


    if 'O1_size' in params:
        O1_s = params['O1_size']
        if O1_s==0:
            ip['O1']['in'] = 0 #if the size of O1 = 0 we put it out
        else:
            ip['O1']['in'] = 1
            ip['O1']['size'] = O1_s * 1e-6
            
            
    if 'wide_factor' in params:
        wf = params['wide_factor']

        ip['beam']['size'] *= wf
        ip['beam_shaper']['size'] *= wf
        ip['L1']['size'] *= wf
        ip['L2']['size'] *= wf
        ip['O1']['size'] *= wf
        ip['O2']['size'] *= wf
        ip['A1']['size'] *= wf
        ip['A2']['size'] *= wf 
        #ip['simulation']['propsize'] *= wf #If we want to adapt the size of the window.
        ip['simulation']['propsize'] *= np.max(params['wide_factor']) #Take the window size to match the largest wide factor.
        print(np.max(params['wide_factor']))
    else:
        wf = 1.0  # fallback for dependent parameters




            
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

    if 'defect_amplitude' in params:
        defect_amp = params['defect_amplitude']        
        ip['A1']['defect_amplitude'] = defect_amp * 1e-6


    if 'TCSlit_size' in params:
        TCslit_s = params['TCSlit_size']
        if TCslit_s==0:
            ip['TCSlit']['in'] = 0 #if the size of O2 = 0 we put it out
        else:
            ip['TCSlit']['in'] = 1 
            ip['TCSlit']['size'] = TCslit_s * 1e-6
        
        

            
    # Apply defect if requested
    #if defect:
    #    defective = ['O1', 'O2', 'A1', 'A2']
    #    for di, dd in enumerate(defective):
    #        ip[dd]['defect_type'] = 'sine'
    #        ip[dd]['defect_lambda'] = float((20 + di * 0.2) * 1e-6)
    #        ip[dd]['defect_amplitude'] = float(2e-6)

    ###################### Save modified YAML ########################
    #filename = f'LP_{start_index + i}.yaml'
    filename = f'{prefix}_{start_index + i}.yaml'

    fullpath = os.path.join(outdir, filename)
    
    native_ip = convert_np_to_native(ip)
    
    with open(fullpath, 'w') as f_out:
        yaml.dump(native_ip, f_out, sort_keys=False, default_flow_style=False)

    # Print the filename and parameter values for this scan
    param_str = ', '.join(f"{k}={v}" for k, v in params.items())
    print(f"Written {filename} ({param_str})")