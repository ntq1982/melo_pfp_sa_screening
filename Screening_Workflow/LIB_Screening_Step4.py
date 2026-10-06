"""
    Pareto-front optimization for ranking MELOs
    
    Objectives: dc range, max abs(dV), and H2→H3 SOC

    Created on Jul 11, 2024 at RISM (Shinshu University)
    Last update: Oct 06, 2026 16:17 JST

    Copyright © 2024-2026 Quang Nguyen. All rights reserved.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def flip(items, ncol):
    return list(itertools.chain(*[items[i::ncol] for i in range(ncol)]))

# === Input and output files ===
reference   = "DelithiationNCM_updated.csv"
screen_file = "LiB_Screening_all_combined_updated.csv"
png_file_1  = "Pareto_dc_range_abs_dV_max.png"
png_file_2  = "Pareto_dc_range_h2h3.png"
png_file_3  = "Pareto_abs_dV_max_h2h3.png"
png_file_4  = "Pareto_Emix.png"
png_file_5  = "Pareto_Average_Voltage.png"
png_file_6  = "Pareto_Equilibrium_Voltage.png"
png_file_7  = "Pareto_Eform.png"  # Reaction energy
png_file_8  = "Pareto_Ecoh.png"   # Formation energy

# === Load reference data and manipulate ===
df_reference             = pd.read_csv(reference)
df_reference             = df_reference.sort_values("x")
df_reference["Compound"] = "NiCoMn"
df_reference             = df_reference.rename(columns={"x": "Conc_deLi"})
df_reference["E_mix_fu"] = df_reference["E_mix"] / 60.0  # 5×4×1 supercell = 60 f.u.
df_reference_dc          = (np.max(df_reference["delta_c"] * 100) - np.min(df_reference["delta_c"] * 100))
df_reference_dV          = np.max(np.abs(df_reference["delta_V"] * 100))
df_reference_h2_h3       = df_reference["H2_H3_cspline"].iloc[0]
print("\nNCM811 descriptors")
print(f"Δc_range = {df_reference_dc:.3f}")
print(f"|ΔV|max  = {df_reference_dV:.3f}")
print(f"H2→H3    = {df_reference_h2_h3:.3f}")

# === Build descriptor table for HELOs ===
df             = pd.read_csv(screen_file)
df             = df.sort_values(["Compound", "Conc_deLi"])
df["delta_V"]  = df["delta_V"] * 100
df["delta_c"]  = df["delta_c"] * 100
df["E_mix_fu"] = df["E_mix"] / 75.0
results        = []
for compound, group in df.groupby("Compound"):
    group      = group.sort_values("Conc_deLi")
    range_dc   = (np.max(group["delta_c"]) - np.min(group["delta_c"]))
    max_abs_dV = np.max(np.abs(group["delta_V"]))
    h2_h3      = group["H2_H3_cspline"].iloc[0]
    results.append([compound, range_dc, max_abs_dV, h2_h3])
ranking = pd.DataFrame(results, columns=["Compound", "Range_dc", "Max_abs_dV", "H2_H3"])
print(f"Number of HELO compounds: {len(ranking)}")

# === Pareto-front analysis and ranking === 
pareto_mask = np.ones(len(ranking), dtype=bool)
for i in range(len(ranking)):
    for j in range(len(ranking)):
        if i == j:
            continue
        better_or_equal = (ranking.loc[j, "Range_dc"] <= ranking.loc[i, "Range_dc"] and
                           ranking.loc[j, "Max_abs_dV"] <= ranking.loc[i, "Max_abs_dV"] and
                           ranking.loc[j, "H2_H3"] >= ranking.loc[i, "H2_H3"])
        strictly_better = (ranking.loc[j, "Range_dc"] < ranking.loc[i, "Range_dc"] or
                           ranking.loc[j, "Max_abs_dV"] < ranking.loc[i, "Max_abs_dV"] or
                           ranking.loc[j, "H2_H3"] > ranking.loc[i, "H2_H3"])
        if better_or_equal and strictly_better:
            pareto_mask[i] = False
            break
ranking["Pareto"]      = pareto_mask
pareto                 = ranking[ranking["Pareto"]]
non_pareto             = ranking[~ranking["Pareto"]]
pareto_ranked          = pareto.sort_values(["Range_dc", "Max_abs_dV", "H2_H3"], ascending=[True, True, False]).copy()
pareto_ranked["Score"] = (pareto_ranked["Range_dc"].rank(ascending=True) +
                          pareto_ranked["Max_abs_dV"].rank(ascending=True) +
                          pareto_ranked["H2_H3"].rank(ascending=False))
pareto_ranked          = pareto_ranked.sort_values("Score")
print("\nPareto-optimal compounds:")
print(pareto_ranked[["Compound", "Range_dc", "Max_abs_dV", "H2_H3", "Score"]].to_string(index=False))

# === Full ranking (all compounds) ===
print("\nTop-ranked compounds:")
ranking = ranking.sort_values(["Pareto", "Range_dc", "Max_abs_dV", "H2_H3"], ascending=[False, True, True, False])
print(ranking[["Compound", "Range_dc", "Max_abs_dV", "H2_H3", "Pareto"]].head(20))

# === Figure 1 : Pareto-front for Δc_range vs |ΔV|max ===
plt.figure(figsize=(6.5,6))
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2.0))
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(2.0))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.scatter(non_pareto["Range_dc"], non_pareto["Max_abs_dV"], s=80, facecolors="white", edgecolors="grey", linewidths=1.0, label="MELOs")
plt.scatter(pareto["Range_dc"], pareto["Max_abs_dV"], s=120, facecolors="white", edgecolors="red", linewidths=1.5, label="Pareto front", zorder=10)
plt.scatter(df_reference_dc, df_reference_dV, s=250, marker="*", facecolors="red", edgecolors="black", linewidths=1.5, label="NCM811", zorder=20)
pareto_sorted = pareto.sort_values("Range_dc")
plt.plot(pareto_sorted["Range_dc"], pareto_sorted["Max_abs_dV"], "-", linewidth=1.5, color="red")
plt.xlabel(r"Range of $\Delta c/c$ (%)", fontsize=20)
plt.ylabel(r"max $|\Delta V/V|$ (%)", fontsize=20)
plt.xlim([2.0 - 0.3, 10.0 + 0.3])
plt.ylim([4.0 - 0.3, 12.0 + 0.3])
plt.grid(True)
plt.legend(loc="upper left", facecolor='white', edgecolor='grey', framealpha=1, fontsize=18, frameon=True)
plt.tight_layout()
plt.savefig(png_file_1, dpi=300)
plt.show()
plt.close()

# === Figure 2 : Pareto-front for Δc_range vs H2-H3 ===
plt.figure(figsize=(6.5,6))
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2.0))
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.1))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.scatter(non_pareto["Range_dc"], non_pareto["H2_H3"], s=80, facecolors="white", edgecolors="grey", linewidths=1.0, label="MELOs")
plt.scatter(pareto["Range_dc"], pareto["H2_H3"], s=120, facecolors="white", edgecolors="red", linewidths=1.5, label="Pareto front", zorder=10)
plt.scatter(df_reference_dc, df_reference_h2_h3, s=250, marker="*", facecolors="red", edgecolors="black", linewidths=1.5, label="NCM811", zorder=20)
pareto_sorted = pareto.sort_values("Range_dc")
plt.plot(pareto_sorted["Range_dc"], pareto_sorted["H2_H3"], "-", linewidth=1.5, color="red")
plt.xlabel(r"Range of $\Delta c/c$ (%)", fontsize=20)
plt.ylabel(r"SOC of H2→H3 Trans.", fontsize=20)
plt.xlim([2.0 - 0.3, 10.0 + 0.3])
plt.ylim([0.5 - 0.01, 0.9 + 0.01])
plt.grid(True)
plt.legend(loc="upper left", facecolor='white', edgecolor='grey', framealpha=1, fontsize=18, frameon=True)
plt.tight_layout()
plt.savefig(png_file_2, dpi=300)
plt.show()
plt.close()

# === Figure 3 : Pareto-front for |ΔV|max vs H2-H3 ===
plt.figure(figsize=(6.5,6))
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2.0))
plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(0.1))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.tick_params(axis='both', which='major', labelsize=18)
plt.scatter(non_pareto["Max_abs_dV"], non_pareto["H2_H3"], s=80, facecolors="white", edgecolors="grey", linewidths=1.0, label="MELOs")
plt.scatter(pareto["Max_abs_dV"], pareto["H2_H3"], s=120, facecolors="white", edgecolors="red", linewidths=1.5, label="Pareto front", zorder=10)
plt.scatter(df_reference_dV, df_reference_h2_h3, s=250, marker="*", facecolors="red", edgecolors="black", linewidths=1.5, label="NCM811", zorder=20)
pareto_sorted = pareto.sort_values("Max_abs_dV")
plt.plot(pareto_sorted["Max_abs_dV"], pareto_sorted["H2_H3"], "-", linewidth=1.5, color="red")
plt.xlabel(r"max $|\Delta V/V|$ (%)", fontsize=20)
plt.ylabel(r"SOC of H2→H3 Trans.", fontsize=20)
plt.xlim([4.0 - 0.3, 12.0 + 0.3])
plt.ylim([0.5 - 0.01, 0.9 + 0.01])
plt.grid(True)
plt.legend(loc="upper left", facecolor='white', edgecolor='grey', framealpha=1, fontsize=18, frameon=True)
plt.tight_layout()
plt.savefig(png_file_3, dpi=300)
plt.show()
plt.close()

# === Figure 4 : Emix of Pareto compounds vs NCM811 ===
pareto_compounds = pareto_ranked["Compound"].tolist()
colors = plt.cm.tab10(np.arange(len(pareto_compounds)))
plt.figure(figsize=(6.5,6))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.axhline(y=0, color="black", linewidth=1, linestyle="--")
for color, compound in zip(colors, pareto_compounds):
    group = df[df["Compound"] == compound].sort_values("Conc_deLi")
    plt.plot(group["Conc_deLi"], group["E_mix_fu"], "o-", color=color, markerfacecolor="white", markersize=8, linewidth=1.0, label=compound.replace("NiCoMn", "*"),)
plt.plot(df_reference["Conc_deLi"], df_reference["E_mix_fu"], "s-", markersize=8, markerfacecolor="white", color="black", linewidth=2.0, label="NCM811",)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim([0.0 - 0.02, 1.0 + 0.02])
plt.ylim([-0.2 - 0.004, 0.05 + 0.004])
plt.xlabel(r"$x$ in Li$_{1-x}$(M)O$_2$", fontsize=20, fontweight='regular')
plt.ylabel(r"$E_{mix}$ (eV/f.u.)", fontsize=20, fontweight='regular')
plt.grid(True)
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(flip(handles, 3), flip(labels, 3), loc="lower center", ncol=3, facecolor='white', edgecolor='grey', framealpha=1, fontsize=12, frameon=True)
plt.savefig(png_file_4, dpi=300)
plt.show()
plt.close()

# === Figure 5 : Average voltage of Pareto compounds vs NCM811 ===
pareto_compounds = pareto_ranked["Compound"].tolist()
colors = plt.cm.tab10(np.arange(len(pareto_compounds)))
plt.figure(figsize=(6.5,6))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.axhline(y=0, color="black", linewidth=1, linestyle="--")
for color, compound in zip(colors, pareto_compounds):
    group = df[df["Compound"] == compound].sort_values("Conc_deLi")
    # min_voltage = group.loc[group["Voltage"] != 0, "Voltage"].min()
    # max_voltage = group["Voltage"].max()
    # print(f"Voltage range of {compound} : {min_voltage:.3f} {max_voltage:.3f}")
    plt.plot(group["Conc_deLi"], group["Voltage"], "o-", color=color, markerfacecolor="white", markersize=8, linewidth=1.0, label=compound.replace("NiCoMn", "*"),)
plt.plot(df_reference["Conc_deLi"], df_reference["Voltage"], "s-", markersize=8, markerfacecolor="white", color="black", linewidth=2.0, label="NCM811",)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim([0.0 - 0.02, 1.0 + 0.02])
plt.ylim([3.6 - 0.015, 4.6 + 0.015])
plt.xlabel(r"$x$ in Li$_{1-x}$(M)O$_2$", fontsize=20, fontweight='regular')
plt.ylabel(r"Average voltage (V)", fontsize=20, fontweight='regular')
plt.grid(True)
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(flip(handles, 3), flip(labels, 3), loc="upper center", ncol=3, facecolor='white', edgecolor='grey', framealpha=1, fontsize=12, frameon=True)
plt.savefig(png_file_5, dpi=300)
plt.show()
plt.close()

# === Figure 6 : Equilibrium voltage of Pareto compounds vs NCM811 ===
plt.figure(figsize=(6.5,6))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
plt.axhline(y=0, color="black", linewidth=1, linestyle="--")
for color, compound in zip(colors, pareto_compounds):
    group = df[df["Compound"] == compound].sort_values("Conc_deLi")
    x = group["Conc_deLi"].values
    V = group["Equilibrium_Voltage"].values
    x_step, V_step = [], []
    for i in range(len(x)-1):
        x_step += [x[i], x[i+1]]
        V_step += [V[i], V[i]]
    plt.plot(x_step, V_step, "-", color=color, linewidth=1.0, label=compound.replace("NiCoMn", "*"))
    plt.plot(x[:-1], V[:-1], "o", color=color, markerfacecolor="white", markersize=8, linewidth=1.0)
x_ref = df_reference["Conc_deLi"].values
V_ref = df_reference["Equilibrium_Voltage"].values
x_step_ref, V_step_ref = [], []
for i in range(len(x_ref)-1):
    x_step_ref += [x_ref[i], x_ref[i+1]]
    V_step_ref += [V_ref[i], V_ref[i]]
plt.plot(x_step_ref, V_step_ref, "-", color="black", linewidth=2.0, label="NCM811")
plt.plot(x_ref[:-1], V_ref[:-1], "s", markersize=8, markerfacecolor="white", color="black", linewidth=2.0)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlim([0.0 - 0.02, 1.0 + 0.02])
plt.ylim([3.0 - 0.0375, 5.5 + 0.0375])
plt.xlabel(r"$x$ in Li$_{1-x}$(M)O$_2$", fontsize=20, fontweight='regular')
plt.ylabel(r"Equilibrium voltage (V)", fontsize=20, fontweight='regular')
plt.grid(True)
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(flip(handles, 3), flip(labels, 3), loc="lower center", ncol=3, facecolor='white', edgecolor='grey', framealpha=1, fontsize=12, frameon=True)
plt.savefig(png_file_6, dpi=300)
plt.show()
plt.close()

# === Figure 7 : Reaction energy of Pareto compounds
ranking_all = df.groupby("Compound").first().reset_index()
pareto_thermo = ranking_all[ranking_all["Compound"].isin(pareto_ranked["Compound"])].copy()
pareto_thermo["Rank"] = pareto_thermo["Compound"].map(dict(zip(pareto_ranked["Compound"], range(1, len(pareto_ranked)+1))))
pareto_thermo = pareto_thermo.sort_values("Rank")
E_form_ref = df_reference["E_form_fuli"].iloc[0]
plt.figure(figsize=(6.5,6))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
plt.bar(np.arange(len(pareto_thermo)), pareto_thermo["E_form_fuli"], edgecolor="black", linewidth=1.2)
plt.axhline(E_form_ref, color="red", linestyle="--", linewidth=2.0, label="NCM811")
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xticks(np.arange(len(pareto_thermo)), np.arange(1, len(pareto_thermo)+1))
plt.ylim([0.5 - 0.002, 0.6 + 0.002])
plt.xlabel("Pareto Rank", fontsize=20, fontweight='regular')
plt.ylabel(r"$E_{rxn}$ (eV/f.u.)", fontsize=20, fontweight='regular')
plt.grid(False)
plt.tight_layout()
plt.legend(facecolor='white', edgecolor='grey', framealpha=1, fontsize=18, frameon=True)
plt.savefig(png_file_7, dpi=300)
plt.show()
plt.close()

# === Figure 8 : Form energy of Pareto compounds
ranking_all = df.groupby("Compound").first().reset_index()
pareto_thermo = ranking_all[ranking_all["Compound"].isin(pareto_ranked["Compound"])].copy()
pareto_thermo["Rank"] = pareto_thermo["Compound"].map(dict(zip(pareto_ranked["Compound"], range(1, len(pareto_ranked)+1))))
pareto_thermo = pareto_thermo.sort_values("Rank")
E_coh_ref = df_reference["E_coh_fuli"].iloc[0]
plt.figure(figsize=(6.5,6))
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
plt.bar(np.arange(len(pareto_thermo)), -pareto_thermo["E_coh_fuli"], edgecolor="black", linewidth=1.2)
plt.axhline(-E_coh_ref, color="red", linestyle="--", linewidth=2.0, label="NCM811")
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xticks(np.arange(len(pareto_thermo)), np.arange(1, len(pareto_thermo)+1))
plt.ylim([6.8 - 0.005, 7.1 + 0.005])
plt.xlabel("Pareto Rank", fontsize=20, fontweight='regular')
plt.ylabel(r"$-E_{form}$ (eV/f.u.)", fontsize=20, fontweight='regular')
plt.grid(False)
plt.tight_layout()
plt.legend(facecolor='white', edgecolor='grey', framealpha=1, fontsize=18, frameon=True)
plt.savefig(png_file_8, dpi=300)
plt.show()
plt.close()
