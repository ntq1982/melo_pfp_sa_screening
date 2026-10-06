# !/usr/bin/env python3

'''
    This program is to screen transition metal layered oxide compounds for LIB.

    Here, we access all compounds with stable structures identified through SA searches.
    For each compound, we perform optimizations for fully Li-occupied and partly Li-occupied structures
    to calculate voltage and changes in lattice parameters. Additionally, we compute formation energy
    and theoretical capacity. The results are then saved to a CSV file for further analysis.

    Created on Jul 11, 2024 at RISM (Shinshu University)
    Last update: Oct 06, 2026 16:05 JST

    Copyright © 2024-2026 Quang Nguyen. All rights reserved.
'''

import os
import re
import csv
import shutil
import math
import time
import datetime
import pandas as pd
import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode, EstimatorMethodType
from ase.io import read, write
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter, ExpCellFilter, FrechetCellFilter
from pathlib import Path

def oxide_formula(element, ox_state):
	g = math.gcd(ox_state, 2)
	return (2 // g, ox_state // g)

# Let's go
start_time      = time.time()
directory1      = Path("../Senary_Step1_SA_for_Metals")
directory2      = Path("../Senary_Step2_SA_for_Lithium")
parallel_tasks  = 5
current_task_id = 0
output_prefix   = "LiB_Screening_"
opt_struct_dir  = "Structures_Opt"
logfile         = output_prefix+str(current_task_id+1)+".log"
os.makedirs(opt_struct_dir, exist_ok=True)
with open(logfile, 'w') as f:
    print(f"Executed on: {datetime.datetime.now()}", file=f)

# Specify estimator and calculator for simulations
estimator  = Estimator(method_type=EstimatorMethodType.PFVM, calc_mode=EstimatorCalcMode.CRYSTAL_PLUS_D3, model_version='v7.0.0')
calculator = ASECalculator(estimator)
with open(logfile, 'a') as f:
    print(f"PFP client version: {pfp_api_client.__version__}", file=f)
    print(f"Model version: {estimator.model_version}", file=f)
    print(f"Calculation mode: {str(estimator.calc_mode).split('.')[1]}", file=f)
    print(f"Method type: {str(estimator.method_type).split('.')[1]}", file=f)
    print(f"****************************************", file=f)

# Reference energies for calculating cohesive energies (per atoms)
ref_bulk_energies = {'Sc' : -4.534984, 'Ti' : -5.890736, 'V'  : -3.735721, 'Cr' : -2.688523, 'Mn' : -4.367030,
                     'Fe' : -2.630210, 'Co' : -3.963792, 'Ni' : -2.980612, 'Cu' : -3.679986, 'Zn' : -1.494744,
                     'Y'  : -4.502628, 'Zr' : -6.726096, 'Nb' : -7.654998, 'Mo' : -3.349406, 'Hf' : -6.842138,
                     'Ta' : -8.763738, 'W'  : -5.234562, 'Mg' : -1.780591, 'Ca' : -2.130833, 'Sr' : -1.804757,
                     'Ba' : -2.065923, 'Al' : -3.751751, 'Ga' : -2.903566, 'In' : -2.627424, 'Sn' : -3.451110,
                     'Pb' : -3.321407, 'Si' : -4.858629, 'Ge' : -4.012200, 'As' : -3.391421, 'Sb' : -3.161641,
                     'Te' : -2.824299, 'Ru' : -7.344923, 'Rh' : -6.396638, 'Pd' : -4.383632, 'La' : -4.640344,
                     'Ce' : -5.148569, 'Nd' : -4.590388, 'Eu' : -2.182362, 'Tm' : -4.477730, 'Yb' : -1.992633,
                     'Lu' : -4.487095, 'Li' : -1.778362}

# Reference energies for calculating formation energies (per formula)
oxidation_states   = {'Sc' : +3, 'Ti' : +4, 'V'  : +5, 'Cr' : +3, 'Mn' : +2,
                      'Fe' : +3, 'Co' : +2, 'Ni' : +2, 'Cu' : +1, 'Zn' : +2,
                      'Y'  : +3, 'Zr' : +4, 'Nb' : +5, 'Mo' : +6, 'Hf' : +4,
                      'Ta' : +5, 'W'  : +6, 'Mg' : +2, 'Ca' : +2, 'Sr' : +2,
                      'Ba' : +2, 'Al' : +3, 'Ga' : +3, 'In' : +3, 'Sn' : +4,
                      'Pb' : +2, 'Si' : +4, 'Ge' : +4, 'As' : +3, 'Sb' : +3,
                      'Te' : +4, 'Ru' : +4, 'Rh' : +3, 'Pd' : +2, 'La' : +3,
                      'Ce' : +4, 'Nd' : +3, 'Eu' : +3, 'Tm' : +3, 'Yb' : +3,
                      'Lu' : +3, 'Li' : +1}
ref_oxide_energies = {'Sc' : -36.380977, 'Ti' : -21.148676, 'V'  : -38.188684, 'Cr' : -25.680658, 'Mn' :  -9.610013,
                      'Fe' : -23.683983, 'Co' :  -9.581681, 'Ni' :  -9.408609, 'Cu' : -11.602590, 'Zn' :  -7.440717,
                      'Y'  : -36.609262, 'Zr' : -23.366382, 'Nb' : -48.118030, 'Mo' : -20.213821, 'Hf' : -23.868529,
                      'Ta' : -52.737462, 'W'  : -23.140479, 'Mg' : -10.518655, 'Ca' : -11.402921, 'Sr' : -10.582831,
                      'Ba' : -10.317149, 'Al' : -32.192442, 'Ga' : -24.492500, 'In' : -22.696985, 'Sn' : -14.791994,
                      'Pb' :  -8.672008, 'Si' : -19.429585, 'Ge' : -15.215125, 'As' : -21.891211, 'Sb' : -22.368315,
                      'Te' : -12.157224, 'Ru' : -16.687877, 'Rh' : -25.503449, 'Pd' :  -8.336678, 'La' : -36.071529,
                      'Ce' : -21.767413, 'Nd' : -35.575121, 'Eu' : -27.943265, 'Tm' : -36.894733, 'Yb' : -28.093925,
                      'Lu' : -37.108820, 'Li' : -12.650907}

# Other reference energies
other_energies = {'O2' : -5.895723, 'CO2' : -17.802047, 'Li2CO3' : -32.493953}

# Reference molar mass for calculating theoretical capacity
molar_masses = {'Sc' :  44.9559, 'Ti' :  47.8670, 'V'  :  50.9415, 'Cr' :  51.9961, 'Mn' :  54.9380,
                'Fe' :  55.8450, 'Co' :  58.9332, 'Ni' :  58.6934, 'Cu' :  63.5460, 'Zn' :  65.3900,
                'Y'  :  88.9059, 'Zr' :  91.2240, 'Nb' :  92.9064, 'Mo' :  95.9600, 'Hf' : 178.4900,
                'Ta' : 180.9479, 'W'  : 183.8400, 'Mg' :  24.3050, 'Ca' :  40.0780, 'Sr' :  87.6200,
                'Ba' : 137.3270, 'Al' :  26.9815, 'Ga' :  69.7230, 'In' : 114.8180, 'Sn' : 118.7100,
                'Pb' : 207.2000, 'Si' :  28.0855, 'Ge' :  72.6300, 'As' :  74.9216, 'Sb' : 121.7600,
                'Te' : 127.6000, 'Ru' : 101.0700, 'Rh' : 102.9055, 'Pd' : 106.4200, 'La' : 138.9055,
                'Ce' : 140.1160, 'Nd' : 144.2420, 'Eu' : 151.9640, 'Tm' : 168.9342, 'Yb' : 173.0450,
                'Lu' : 174.9668, 'Li' :   6.9410,  'O'  : 15.9994}

# CSV file to contain the information for all compounds
csvfilename = output_prefix+str(current_task_id+1)+".csv"
csv_exists  = Path(csvfilename).exists()
if csv_exists:
    df_existing = pd.read_csv(csvfilename)
    calculated_structures = set(df_existing['Compound'].values)
else:
    with open(csvfilename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Compound', 'Ni', 'Conc_Ni', 'Co', 'Conc_Co', 'Mn', 'Conc_Mn',
                         'A', 'Conc_A', 'B', 'Conc_B', 'C', 'Conc_C', 
                         'E_pot_fuli', 'E_coh_fuli', 'E_form_fuli', 'O_excess', 'a_fuli', 'b_fuli', 'c_fuli', 'V_fuli',
                         'E_pot_deli', 'a_deli', 'b_deli', 'c_deli', 'V_deli',
                         'Voltage', 'Capacity', 'delta_a', 'delta_b', 'delta_c', 'delta_V', 'N_defects'])
    calculated_structures = set()

# Scan through folder and partition files for parallel runs
filepaths          = sorted(list(directory1.glob("*_Most.cif")))
filepaths_for_task = [[] for _ in range(parallel_tasks)]
for i, filepath in enumerate(filepaths):
    filepaths_for_task[i % parallel_tasks].append(filepath)
compound_count     = len(filepaths_for_task[current_task_id])
with open(logfile, 'a') as f:
    print(f"Number of detected compounds: {len(filepaths)}", file=f)
    print(f"Number of tasks for optimization: {parallel_tasks}", file=f)
    print(f"Number of compounds for the task: {compound_count}", file=f)

# MAIN: Load all CIF files and generate necessary data
for i, filepath in enumerate(filepaths_for_task[current_task_id]):
    start_time_cp = time.time()
    compound_full = filepath.name.split(".")[0]
    compound      = compound_full.split("_")[0]
    if compound in calculated_structures:
        continue
    with open(logfile, 'a') as f:
        print(f"  Optimizing compound {i+1}/{compound_count} : {compound}...", file=f)

    # Extract information from filename
    file_prefix = filepath.name.split("_", 1)[0]
    pattern  = r'[A-Z][a-z]?'
    elements = re.findall(pattern, file_prefix)

    # Extract concentration of metals and calculate capacity
    SPC_fuli        = read(filepath)
    del SPC_fuli[[atom.index for atom in SPC_fuli if atom.symbol=='X']]
    SPC_deli        = read(directory2 / filepath.name)
    del SPC_deli[[atom.index for atom in SPC_deli if atom.symbol=='X']]
    idx_Li_all = [atom.index for atom in SPC_fuli if atom.symbol == 'Li']
    idx_Li     = [atom.index for atom in SPC_fuli if atom.symbol == 'Li' and 
                  ((0   < atom.scaled_position[2] < 1/3) or 
                   (1/3 < atom.scaled_position[2] < 2/3) or 
                   (2/3 < atom.scaled_position[2] < 1))]
    if len(idx_Li_all) != len(idx_Li):
        N_defects = 2 * (len(idx_Li_all) - len(idx_Li))
    else:
        N_defects = 0
    idx_metals      = [atom.index for atom in SPC_fuli if atom.symbol not in ['Li', 'O']]
    idx_Ni          = [atom.index for atom in SPC_fuli if atom.symbol=='Ni']
    idx_Co          = [atom.index for atom in SPC_fuli if atom.symbol=='Co']
    idx_Mn          = [atom.index for atom in SPC_fuli if atom.symbol=='Mn']
    conc_Ni         = len(idx_Ni) / len(idx_metals)
    conc_Co         = len(idx_Co) / len(idx_metals)
    conc_Mn         = len(idx_Mn) / len(idx_metals)
    conc_metals     = [conc_Ni, conc_Co, conc_Mn] + [(1 - conc_Ni - conc_Co - conc_Mn) / (len(elements) - 3)] * (len(elements) - 3)
    molar_mass      = molar_masses.get('Li') + 2 * molar_masses.get('O')
    for i, element in enumerate(elements):
        molar_mass += conc_metals[i] * molar_masses.get(element)
    faraday_const   = 96485
    mAh_to_C        = 3.6
    capacity        = 1 * faraday_const / molar_mass / mAh_to_C

    # Optimize the full-Li and delithiated structures
    SPC_fuli.calc = calculator
    opt           = LBFGS(FrechetCellFilter(SPC_fuli, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
    opt.run(fmax=0.01, steps=100000)
    energy_fuli   = SPC_fuli.get_potential_energy()
    write(f"{opt_struct_dir}/{compound}_fuli.cif", SPC_fuli)
    SPC_deli.calc = calculator
    opt           = LBFGS(FrechetCellFilter(SPC_deli, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
    opt.run(fmax=0.01, steps=100000)
    energy_deli   = SPC_deli.get_potential_energy()
    write(f"{opt_struct_dir}/{compound}_deli.cif", SPC_deli)

    # Extract structural parameters and calculate changes
    a_fuli, b_fuli, c_fuli = SPC_fuli.cell.lengths()[0], SPC_fuli.cell.lengths()[1], SPC_fuli.cell.lengths()[2]
    V_fuli                 = SPC_fuli.get_volume()
    a_deli, b_deli, c_deli = SPC_deli.cell.lengths()[0], SPC_deli.cell.lengths()[1], SPC_deli.cell.lengths()[2]
    V_deli                 = SPC_deli.get_volume()
    delta_a                = (a_deli - a_fuli ) / a_fuli
    delta_b                = (b_deli - b_fuli ) / b_fuli
    delta_c                = (c_deli - c_fuli ) / c_fuli
    delta_V                = (V_deli - V_fuli ) / V_fuli

    # Calculate voltage from delithiation reaction
    idx_Li_fuli  = [atom.index for atom in SPC_fuli if atom.symbol=='Li']
    N_Li_fuli    = len(idx_Li_fuli)
    idx_Li_deli  = [atom.index for atom in SPC_deli if atom.symbol=='Li']
    N_Li_deli    = len(idx_Li_deli)
    delta_energy = (energy_deli + (N_Li_fuli - N_Li_deli) * ref_bulk_energies.get('Li')) - energy_fuli
    if N_Li_deli == N_Li_fuli:
        voltage  = 0
    else:
        voltage      = delta_energy / (N_Li_fuli - N_Li_deli)

    # Calculate cohesive energy (per formula unit)
    bulk_energies = 0.0
    for i, element in enumerate(elements):
        bulk_energies += conc_metals[i] * ref_bulk_energies.get(element)
    cohesive_energy = energy_fuli / N_Li_fuli - ref_bulk_energies.get("Li") \
                                              - bulk_energies - other_energies.get("O2")

    # Calculate formation energy (per formula unit)
    oxides_energies = 0.0
    O_from_oxides = 0.0
    for i, element in enumerate(elements):
        m, o = oxide_formula(element, oxidation_states.get(element))
        oxides_energies += conc_metals[i] * ref_oxide_energies.get(element) / m
        O_from_oxides += conc_metals[i] * o / m
    formation_energy = energy_fuli / N_Li_fuli + 0.5 * other_energies.get("CO2") \
                                               - oxides_energies - 0.5 * other_energies.get("Li2CO3")
    delta_O = (2 + 0.5 * 2) - (O_from_oxides + 0.5 * 3)
    formation_energy_corrected = formation_energy - delta_O * other_energies.get("O2") / 2

    # Save all data to CSV file
    with open(logfile, 'a') as f:
        print(f"    ---> E_coh: {cohesive_energy : .6f} \t| E_form: {formation_energy_corrected : .6f} \t| Capacity: {capacity : .6f}",
              f"\t| Voltage: {voltage : .6f} \t| V_change: {100 * delta_V : .6f}", file=f)
    with open(csvfilename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([compound, elements[0], conc_metals[0], elements[1], conc_metals[1], elements[2], conc_metals[2],
                         elements[3], conc_metals[3], elements[4], conc_metals[4], elements[5], conc_metals[5],
                         energy_fuli, cohesive_energy, formation_energy_corrected, delta_O, a_fuli, b_fuli, c_fuli, V_fuli,
                         energy_deli, a_deli, b_deli, c_deli, V_deli,
                         voltage, capacity, delta_a, delta_b, delta_c, delta_V, N_defects])
    
    # End time for current compound
    end_time_cp   = time.time()
    with open(logfile, 'a') as f:
        print(f"    ---> Optimizing time: {end_time_cp-start_time_cp:.6f} seconds", file=f)

# Sort the CSV file alphabetically by "Compound"
if csv_exists:
    df = pd.read_csv(csvfilename)
    df_sorted = df.sort_values(by='Compound')
    df_sorted.to_csv(csvfilename, index=False)

# Everything is ok if you can reach this point
end_time = time.time()
elapsed_time = end_time - start_time
with open(logfile, 'a') as f:
    print(f"****************************************", file=f)
    print(f"Done successfully!", file=f)
    print(f"Elapsed time: {elapsed_time:.6f} seconds", file=f)
    print(f"Finished on: {datetime.datetime.now()}", file=f)