#!/usr/bin/env python3

'''
    This program performs global optimization to find the preferable arrangement
    of transition metals in Ni-rich layered oxide (LiNi0.8Co0.08Mn0.08(ABC)0.04O2).

    For each compound, Simulated Annealing (SA) is used.
    Please see here for more detail on SA implementation:

    T. Q. Nguyen et al., J. Mater. Chem. A (2026) 14 (56): 39136–39150
    DOI:10.1039/D6TA04623A
    
    The final compounds will then be used in subsequent SA runs to search for 
    the most favorable arrangement of Li.

    Created on Jul 09, 2024 at RISM (Shinshu University)
    Last update: Oct 06, 2026 15:57 JST

    Copyright © 2024-2026 Quang Nguyen. All rights reserved.
'''

import os
import time
import datetime
import math
import numpy as np
import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode, EstimatorMethodType
from itertools import combinations
from ase.io import read, write
from ase.spacegroup import crystal
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter, ExpCellFilter, FrechetCellFilter
from ase_annealing_new import SimAnn

def count_unique_elements(cations):
    if all(not any(char.isdigit() or char == '+' for char in cation) for cation in cations):
        return len(cations)
    unique_elements = {cation.rstrip('1234567+') for cation in cations}
    return len(unique_elements)

# Let's go
start_time      = time.time()
parallel_tasks  = 5
current_task_id = 0
logfile_main    = 'LiB_Screening_Step1_part'+str(current_task_id+1)+'.log'
with open(logfile_main, 'w') as f:
    print(f"Executed on: {datetime.datetime.now()}", file=f)

# Specify estimator and calculator for simulations
estimator  = Estimator(method_type=EstimatorMethodType.PFVM, calc_mode=EstimatorCalcMode.PBE_U_PLUS_D3, model_version='v7.0.0')
calculator = ASECalculator(estimator)
with open(logfile_main, 'a') as f:
    print(f"PFP client version: {pfp_api_client.__version__}", file=f)
    print(f"Model version: {estimator.model_version}", file=f)
    print(f"Calculation mode: {str(estimator.calc_mode).split('.')[1]}", file=f)
    print(f"Method type: {str(estimator.method_type).split('.')[1]}", file=f)
    print(f"****************************************", file=f)

# Create a conventional cell of hexagonal LiNiO2 (R-3m)
a, b, c            = 2.87, 2.87, 14.2
alpha, beta, gamma = 90, 90, 120
LiNiO2             = crystal(['Li', 'Ni', 'O'],
                             basis=[(0, 0, 0.5), (0, 0, 0), (0, 0, 0.26)],
                             spacegroup=166,
                             cellpar=[a, b, c, alpha, beta, gamma],
                             size=(1, 1, 1),
                             pbc=(True, True, True),
                             primitive_cell=False)

# List of all investigated elements
all_cations      = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'}
unused_cations   = {'Ni', 'Mn', 'Co'}    # Exists in host or is hard to synthesize
extended_cations = {'Mg', 'Al', 'Zr', 'Nb', 'Mo'}
selected_cations = all_cations - unused_cations | extended_cations
with open(logfile_main, 'a') as f:
    print(f"Total number of elements: {count_unique_elements(selected_cations)}", file=f)

# Generate possible compounds and filter if applicable
n_component            = 3    # 6 in total, including Ni, Co, Mn
all_possible_compounds = list(combinations(sorted(selected_cations), n_component))
all_compounds_sorted   = sorted(all_possible_compounds, key=lambda compound: tuple(sorted(compound)))
compounds_for_task = [[] for _ in range(parallel_tasks)]
for i, compound in enumerate(all_compounds_sorted):
    compounds_for_task[i % parallel_tasks].append(compound)
compounds_count    = len(compounds_for_task[current_task_id])
with open(logfile_main, 'a') as f:
    print(f"Number of compounds: {len(all_possible_compounds)}", file=f)
    print(f"Number of parallel tasks: {parallel_tasks}", file=f)
    print(f"Number of compounds for the task: {compounds_count}", file=f)
    
# MAIN: Screening process
for i, compound in enumerate(compounds_for_task[current_task_id]):
    start_time_cp  = time.time()
    Elements       = [element for element in compound]
    Elements_ext   = ['Ni', 'Co', 'Mn'] + Elements
    conc_cations   = [0.04/n_component] * n_component
    compound_name  = 'Ni' + 'Co' + 'Mn' + ''.join(compound)
    logfile        = compound_name + '_SA.log'
    Na, Nb, Nc     = 5, 5, 1
    nconfig        = 10 
    mode           = 'unopt'
    free_swap      = 'on'
    PrimaryList    = Elements_ext
    SecondaryList  = ['Li']
    InterMixing    = True
    MixingList     = Elements_ext + ['Li']
    N_steps        = 250000
    PrintFrequency = 500
    if (os.path.exists(compound_name+'_5x5x1_SAunopt_Most.cif')):
        continue
    with open(logfile_main, 'a') as f:
        print(f"  Screening compound {i+1}/{len(compounds_for_task[current_task_id])} : {('Ni', 'Co', 'Mn') + compound}...", file=f)
        
    # Create a supercell to host cations (replace all Ni atoms)
    with open(logfile, 'w') as f:
        print(f"Creating a supercell from conventional cell...", file=f)
    np.random.seed(12345)
    SPC            = LiNiO2.repeat((Na, Nb, Nc))
    idx_cations    = [atom.index for atom in SPC if atom.symbol == 'Ni']
    Na_cations     = [np.floor(len(idx_cations) * conc_cations[i]).astype(int) for i in range(len(conc_cations))]
    Na_cations_ext = [np.floor(len(idx_cations) * 0.8).astype(int),
                      np.floor(len(idx_cations) * 0.08).astype(int),
                      np.floor(len(idx_cations) * 0.08).astype(int)] + Na_cations
    symbol_cations = [Element for Element, count in zip(Elements_ext, Na_cations_ext) for _ in range(count)]
    np.random.shuffle(symbol_cations)
    SPC.symbols[idx_cations] = symbol_cations
    with open(logfile, 'a') as f:
        print(f"  Supercell size: {Na} × {Nb} × {Nc}", file=f)
        print(f"  Total number of atoms: {len(SPC)}", file=f)
        print(f"  Total number of cations: {sum(Na_cations_ext)}", file=f)

    # Try to find optimal cell parameters for the compound
    with open(logfile, 'a') as f:
        print(f"Searching for optimal lattice constant...", file=f)
    a_opt, b_opt, c_opt = [], [], []
    for i in range(nconfig):
        SPCx = SPC.copy()
        np.random.shuffle(symbol_cations)
        SPCx.symbols[idx_cations] = symbol_cations
        SPCx.calc = calculator
        opt = LBFGS(UnitCellFilter(SPCx, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
        opt.run(fmax=0.05, steps=1000)
        a_opt.append(SPCx.cell.lengths()[0])
        b_opt.append(SPCx.cell.lengths()[1])
        c_opt.append(SPCx.cell.lengths()[2])
    a_new, b_new, c_new = np.mean(a_opt), np.mean(b_opt), np.mean(c_opt)
    SPC.set_cell([(a_new + b_new)/2, (a_new + b_new)/2, c_new, alpha, beta, gamma], scale_atoms=True)
    with open(logfile, 'a') as f:
        print(f"  Number of random configurations: {nconfig}", file=f)
        print(f"  Old supercell parameters: {Na * a : .6f} {Nb * b : .6f} {Nc * c : .6f}", file=f)
        print(f"  New supercell parameters: {(a_new + b_new)/2 : .6f} {(a_new + b_new)/2 : .6f} {c_new : .6f} (average)", file=f)

    # Perform global search for stable atomic arrangement
    with open(logfile, 'a') as f:
        print(f"Searching for optimal arrangement of metals...", file=f)
    N_configs_max       = math.factorial(sum(Na_cations_ext))
    for i in range(len(Elements)):
        N_configs_max  /= math.factorial(Na_cations_ext[i])  # Just simple estimation
    N_configs_max       = int(N_configs_max)
    struct_stable_most  = compound_name+'_'+str(Na)+'x'+str(Nb)+'x'+str(Nc)+'_SA'+mode+'_Most.cif'
    struct_stable_least = compound_name+'_'+str(Na)+'x'+str(Nb)+'x'+str(Nc)+'_SA'+mode+'_Least.cif'
    with open(logfile, 'a') as f:
        print(f"  Max number of configurations: {N_configs_max : .2E}", file=f)
        print(f"  Number of provided configurations: {N_steps}", file=f)
        print(f"  Frequency to print SA status: {PrintFrequency}", file=f)
        if mode=='opt':
            Optimization = True
            print(f"  Optimization mode: ON", file=f)
        else:
            Optimization = False
            print(f"  Optimization mode: OFF", file=f)
        if free_swap=='on':
            Layered = False
            print(f"  Free swapping mode: ON", file=f)
        else:
            Layered = True
            print(f"  Free swapping mode: OFF", file=f)
        print(f"  Most  stable configuration is saved to: {struct_stable_most}", file=f)
        print(f"  Least stable configuration is saved to: {struct_stable_least}", file=f)
    SPC_best, SPC_worst = SimAnn(Structure=SPC, Calculator=calculator, SwappingList=PrimaryList, SwappingListX=SecondaryList,
                                 InterMixing=InterMixing, MixingList=MixingList, 
                                 Layered=Layered, Target='Stable', Optimization=Optimization, MaxForce=0.05,
                                 CoolingType='Exponential', T_start=1.0e4, T_stop=1.0e1,
                                 N_steps=N_steps, PrintFrequency=PrintFrequency, RandomSeed=123, LogFile=logfile)
    write("Senary_Step1_SA_for_Metals/" + struct_stable_most, SPC_best, format='cif')
    write("Senary_Step1_SA_for_Metals/" + struct_stable_least, SPC_worst, format='cif')
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