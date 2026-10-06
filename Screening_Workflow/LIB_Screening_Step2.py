#!/usr/bin/env python3

'''
    This program performs global optimization to find the preferable arrangement
    of Li and Li-vacancy in Ni-rich layered oxide (LiNi0.8Co0.08Mn0.08(ABC)0.04O2).

    For each compound, Simulated Annealing (SA) is used.
    Please see here for more detail on SA implementation:

    T. Q. Nguyen et al., J. Mater. Chem. A (2026) 14 (56): 39136–39150
    DOI:10.1039/D6TA04623A
    
    During the search for Li/Li-vacancy arrangement, metal arrangement is fixed
    based on the structure found in previous step.

    Created on Jul 10, 2024 at RISM (Shinshu University)
    Last update: Oct 06, 2026 16:00 JST

    Copyright © 2024-2026 Quang Nguyen. All rights reserved.
'''

import time
import datetime
import math
import numpy as np
import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode, EstimatorMethodType
from ase.io import read, write
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter, ExpCellFilter, FrechetCellFilter
from ase_annealing_new import SimAnn
from pathlib import Path

# Initialize screening parameters
conc_delLi        = 0.1
Na, Nb, Nc        = 5, 5, 1
parallel_tasks    = 5 
current_task_id   = 0 
nconfig           = 10 
SA_mode           = 'unopt'
SA_free_swap      = 'on' 
SA_N_steps        = 50000
SA_PrintFrequency = 100 
PFP_type          = EstimatorMethodType.PFVM
PFP_mode          = EstimatorCalcMode.PBE_U_PLUS_D3
PFP_version       = 'v7.0.0'
logfile_main      = 'LiB_Screening_part'+str(current_task_id+1)+'.log'

# Let's go
start_time   = time.time()
with open(logfile_main, 'w') as f:
    print(f"Executed on: {datetime.datetime.now()}", file=f)

# Specify estimator and calculator for simulations
estimator  = Estimator(method_type=PFP_type, calc_mode=PFP_mode, model_version=PFP_version)
calculator = ASECalculator(estimator)
with open(logfile_main, 'a') as f:
    print(f"PFP client version: {pfp_api_client.__version__}", file=f)
    print(f"Model version: {estimator.model_version}", file=f)
    print(f"Calculation mode: {str(estimator.calc_mode).split('.')[1]}", file=f)
    print(f"Method type: {str(estimator.method_type).split('.')[1]}", file=f)
    print(f"****************************************", file=f)

# Scan through folder and partition files for parallel runs
directory          = Path("../Senary_Step1_SA_for_Metals")
filepaths          = sorted(list(directory.glob("*_Most.cif")))
filepaths_for_task = [[] for _ in range(parallel_tasks)]
for i, filepath in enumerate(filepaths):
    filepaths_for_task[i % parallel_tasks].append(filepath)
compounds_count    = len(filepaths_for_task[current_task_id])
with open(logfile_main, 'a') as f:
    print(f"Number of detected compounds: {len(filepaths)}", file=f)
    print(f"Number of screening tasks: {parallel_tasks}", file=f)
    print(f"Number of compounds for the task: {compounds_count}", file=f)

# MAIN: Screening process (Arrangement of Li atoms)
for i, filepath in enumerate(filepaths_for_task[current_task_id]):
    start_time_cp   = time.time()
    compound        = filepath.name.split(".")[0]
    compound_prefix = filepath.name.split("_")[0]
    logfile         = compound_prefix + '_SA.log'
    with open(logfile_main, 'a') as f:
        print(f"  Screening compound {i+1}/{compounds_count} : {compound}...", file=f)

    # Read stable structure from file (from 1st stage screening)
    with open(logfile, 'w') as f:
        print(f"Reading initial structure from {filepath.name}...", file=f)
    SPC = read(filepath)

    # Replace a part of Li with vacancies (use symbol X, not delete)
    np.random.seed(12345)
    with open(logfile, 'a') as f:
        print(f"Replacing {conc_delLi * 100}% Li with vacancies...", file=f)
    idx_Li_all = [atom.index for atom in SPC if atom.symbol == 'Li']
    idx_Li     = [atom.index for atom in SPC if atom.symbol == 'Li' and 
                  ((0   < atom.scaled_position[2] < 1/3) or 
                   (1/3 < atom.scaled_position[2] < 2/3) or 
                   (2/3 < atom.scaled_position[2] < 1))]
    if len(idx_Li_all) != len(idx_Li):
        N_defects = 2 * (len(idx_Li_all) - len(idx_Li))
        with open(logfile, 'a') as f:
            print(f"  {N_defects} aniti-site defects found!", file=f)
    Natm_X     = np.floor(len(idx_Li_all) * conc_delLi).astype(int)
    Natm_Li    = len(idx_Li) - Natm_X
    symbol_LiX = ['Li'] * Natm_Li + ['X'] * Natm_X
    np.random.shuffle(symbol_LiX)
    SPC.symbols[idx_Li] = symbol_LiX
    with open(logfile, 'a') as f:
        print(f"  Total number of atoms of initial structure: {len(SPC)}", file=f)
        print(f"  Total number of atoms of delithiated structure:: {len(SPC) - Natm_X}", file=f)

    # Try to find optimal cell parameters for the delithiated compound
    with open(logfile, 'a') as f:
        print(f"Searching for optimal lattice constant...", file=f)
    a, b, c = SPC.cell.lengths()[0] / Na, SPC.cell.lengths()[1] / Nb, SPC.cell.lengths()[2] / Nc
    alpha, beta, gamma = 90, 90, 120
    a_opt, b_opt, c_opt = [], [], []
    for i in range(nconfig):
        SPCx = SPC.copy()
        np.random.shuffle(symbol_LiX)
        SPCx.symbols[idx_Li] = symbol_LiX
        del SPCx[[atom.index for atom in SPCx if atom.symbol=='X']]
        SPCx.calc = calculator
        opt = LBFGS(FrechetCellFilter(SPCx, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
        opt.run(fmax=0.05, steps=1000)
        a_opt.append(SPCx.cell.lengths()[0] / Na)
        b_opt.append(SPCx.cell.lengths()[1] / Nb)
        c_opt.append(SPCx.cell.lengths()[2] / Nc)
    a_ave, b_ave, c_ave = np.mean(a_opt), np.mean(b_opt), np.mean(c_opt)
    a_new, b_new, c_new = (a_ave + b_ave) / 2, (a_ave + b_ave) / 2, c_ave
    SPC.set_cell([Na * a_new, Nb * b_new, Nc * c_new, alpha, beta, gamma], scale_atoms=True)
    with open(logfile, 'a') as f:
        print(f"  Number of random configurations: {nconfig}", file=f)
        print(f"  Old supercell parameters: {Na * a : .6f} {Nb * b : .6f} {Nc * c : .6f}", file=f)
        print(f"  New supercell parameters: {Na * a_new : .6f} {Nb * b_new : .6f} {Nc * c_new : .6f} (average)", file=f)

    # Perform global search for stable atomic arrangement
    with open(logfile, 'a') as f:
        print(f"Searching for optimal arrangement of metals...", file=f)
    N_configs_max       = math.factorial(Natm_Li + Natm_X) / (math.factorial(Natm_Li) * math.factorial(Natm_X))
    N_configs_max       = int(N_configs_max)
    struct_stable_most  = '_'.join(compound.split("_")[:2])+'_SA'+SA_mode+'_Most.cif'
    struct_stable_least = '_'.join(compound.split("_")[:2])+'_SA'+SA_mode+'_Least.cif'
    with open(logfile, 'a') as f:
        print(f"  Max number of configurations: {N_configs_max : .2E}", file=f)
        print(f"  Number of provided configurations: {SA_N_steps}", file=f)
        print(f"  Frequency to print SA status: {SA_PrintFrequency}", file=f)
        if SA_mode=='opt':
            Optimization = True
            print(f"  Optimization mode: ON", file=f)
        else:
            Optimization = False
            print(f"  Optimization mode: OFF", file=f)
        if SA_free_swap=='on':
            Layered = False
            print(f"  Free swapping mode: ON", file=f)
        else:
            Layered = True
            print(f"  Free swapping mode: OFF", file=f)
        print(f"  Most  stable configuration is saved to: {struct_stable_most}", file=f)
        print(f"  Least stable configuration is saved to: {struct_stable_least}", file=f)
    SPC_best, SPC_worst = SimAnn(Structure=SPC, Calculator=calculator, SwappingList=idx_Li, SwappingListX=None,
                                 Layered=Layered, Target='Stable', Optimization=Optimization, MaxForce=0.05,
                                 CoolingType='Exponential', T_start=1.0e4, T_stop=1.0e1,
                                 N_steps=SA_N_steps, PrintFrequency=SA_PrintFrequency, RandomSeed=123, LogFile=logfile)
    write("Senary_Step2_SA_for_Lithium/" + struct_stable_most, SPC_best, format='cif')
    write("Senary_Step2_SA_for_Lithium/" + struct_stable_least, SPC_worst, format='cif')
    with open(logfile, 'a') as f:
        print(f"Done successfully!", file=f)

    # End time for current compound
    end_time_cp   = time.time()
    with open(logfile_main, 'a') as f:
        print(f"    ---> Screening time: {end_time_cp-start_time_cp:.6f} seconds", file=f)

# Everything is ok if you can reach this point
end_time = time.time()
elapsed_time = end_time - start_time
with open(logfile_main, 'a') as f:
    print(f"****************************************", file=f)
    print(f"Done successfully!", file=f)
    print(f"Elapsed time: {elapsed_time:.6f} seconds", file=f)
    print(f"Finished on: {datetime.datetime.now()}", file=f)