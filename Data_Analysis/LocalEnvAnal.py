"""
    Analyze TM@Li/Li@TM antisites and dopant-centered local environments in
    an ideal layered-oxide CIF.

    For every antisite center, the script reports:
      - 6 nearest cation neighbors in the same layer (in-plane)
      - 6 nearest cation neighbors in the two adjacent layers (3 from each side)

    For every dopant atom, the script reports:
      - 6 nearest cation neighbors in the same layer (in-plane)
      - 6 nearest cation neighbors in the two adjacent layers (3 from each side)

    Dopants are defined as TM elements other than the host elements Ni, Co, and Mn.

    Main outputs:
       layers.csv
          Crystallographic layer information.

       antisite_sites.csv
          One row per TM@Li or Li@TM antisite.

       antisite_local_environments.csv
          One row per antisite, listing all 12 cation neighbors.

       antisite_neighbors_raw.csv
          One row per antisite-neighbor relationship.

       antisite_pair_summary.csv
          Counts of antisite-centered pair types, separated into in-plane
          and adjacent-layer environments.

       dopant_local_environments.csv
          One row per dopant atom, listing all 12 cation neighbors.

    The analysis is intentionally raw/observational. It is designed as the first step
    for examining whether particular antisite or dopant-centered local configurations recur in
    low-energy structures. No energetic causality or enrichment analysis is inferred by this script.

    Created on Sep 25, 2026 at RISM (Shinshu University)
    Last update: Oct 06, 2026 16:29 JST

    Copyright © 2026 Quang Nguyen. All rights reserved.
"""

import argparse
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure
from collections import Counter, defaultdict

TM_ELEMENTS = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
               "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Hf",
               "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"}

CATION_ELEMENTS = TM_ELEMENTS | {"Li", "Al", "Mg"}
HOST_ELEMENTS = {"Li", "Ni", "Co", "Mn"}
DOPANT_ELEMENTS = CATION_ELEMENTS - HOST_ELEMENTS

def is_metal(sp):
    """Any non-Li cation (TM, Al, Mg)."""
    return sp in CATION_ELEMENTS and sp != "Li"

def layer_groups(structure, tol=0.1):
    """Group sites into crystallographic layers using fractional z."""
    zsites = sorted([(i, structure[i].frac_coords[2] % 1.0) for i in range(len(structure))],
                    key=lambda x: x[1])
    groups = []
    for i, z in zsites:
        placed = False
        for g in groups:
            dz = abs(z - g["z"])
            dz = min(dz, 1.0 - dz)
            if dz < tol:
                g["indices"].append(i)
                placed = True
                break
        if not placed:
            groups.append({"z": z, "indices": [i]})
    groups.sort(key=lambda g: g["z"])
    site_to_layer = {}
    for lid, g in enumerate(groups):
        for i in g["indices"]:
            site_to_layer[i] = lid
    return groups, site_to_layer

def layer_type(structure, indices):
    """Classify a layer as Li or TM from its cation occupancy."""
    cations = [structure[i].specie.symbol for i in indices
               if structure[i].specie.symbol in CATION_ELEMENTS]
    n_li = sum(x == "Li" for x in cations)
    n_tm = sum(is_metal(x) for x in cations)
    if n_li > n_tm:
        return "Li"
    return "TM"

def nearest_in_layer(structure, center, candidate_indices, n):
    """Return n nearest cation sites to center from candidate_indices."""
    candidates = [j for j in candidate_indices if j != center]
    candidates.sort(key=lambda j: structure.get_distance(center, j))
    return candidates[:n]

def format_neighbor(structure, j):
    return f"{structure[j].specie.symbol}@{j}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cif", help="Input CIF file")
    parser.add_argument("-o", "--output", default=None, help="Output directory "
                        "(default: <CIF stem>_antisite_analysis)")
    parser.add_argument("--layer-tol", type=float, default=0.1,
                        help="Fractional-z tolerance for grouping layers (default: 0.1)")
    args = parser.parse_args()
    cif = Path(args.cif)
    outdir = (Path(args.output) if args.output else cif.with_name(cif.stem + "_antisite_analysis"))
    outdir.mkdir(parents=True, exist_ok=True)
    structure = Structure.from_file(cif)
    cation_indices = [i for i, site in enumerate(structure) if site.specie.symbol in CATION_ELEMENTS]
    groups, site_to_layer = layer_groups(structure, args.layer_tol)

    # --------------------------------------------------------------
    # Determine whether each crystallographic layer is Li or TM.
    # --------------------------------------------------------------
    layer_info = []
    for lid, g in enumerate(groups):
        ltype = layer_type(structure, g["indices"])
        layer_info.append({"layer": lid, "z_fractional": g["z"], "layer_type": ltype,
                          "n_sites": len(g["indices"]),
                          "Li": sum(structure[i].specie.symbol == "Li" for i in g["indices"]),
                          "TM": sum(structure[i].specie.symbol in TM_ELEMENTS for i in g["indices"]),})
    cation_site_info = {}
    for i in cation_indices:
        lid = site_to_layer[i]
        cation_site_info[i] = {"layer": lid, "layer_type": layer_info[lid]["layer_type"],}

    # ==============================================================
    # 1. ANTISITE ANALYSIS
    # ==============================================================
    antisites = []
    for i in cation_indices:
        sp = structure[i].specie.symbol
        ltype = cation_site_info[i]["layer_type"]
        if ltype == "Li" and is_metal(sp):
            antisite_type = "TM@Li"
        elif ltype == "TM" and sp == "Li":
            antisite_type = "Li@TM"
        else:
            continue
        antisites.append({"site_index": i, "species": sp, "antisite_type": antisite_type,
                          "site_label": f"{sp}@{ltype}", "layer": cation_site_info[i]["layer"],
                          "z_fractional": (structure[i].frac_coords[2] % 1.0),})
    n_LiTM = sum(a["antisite_type"] == "Li@TM" for a in antisites)
    n_TMLi = len(antisites) - n_LiTM
    if n_LiTM != n_TMLi:  # Sanity check
        print(f"WARNING: unbalanced antisites: Li@TM={n_LiTM}, TM@Li={n_MLi}")
    neighbor_rows = []
    site_rows = []
    n_layers = len(groups)
    for a in antisites:
        center = a["site_index"]
        lid = a["layer"]
        same_layer = [j for j in groups[lid]["indices"] if j in cation_indices and j != center]
        lower_lid = (lid - 1) % n_layers
        upper_lid = (lid + 1) % n_layers
        lower_candidates = [j for j in groups[lower_lid]["indices"] if j in cation_indices]
        upper_candidates = [j for j in groups[upper_lid]["indices"] if j in cation_indices]
        inplane = nearest_in_layer(structure, center, same_layer, 6)
        adjacent_lower = nearest_in_layer(structure, center, lower_candidates, 3)
        adjacent_upper = nearest_in_layer(structure, center, upper_candidates, 3)
        adjacent = adjacent_lower + adjacent_upper
        if len(inplane) != 6 or len(adjacent) != 6:
            print(f"WARNING: site {center} ({a['antisite_type']}) has "
                  f"{len(inplane)} in-plane and {len(adjacent)} adjacent-layer neighbors.")
        inplane_species = [structure[j].specie.symbol for j in inplane]
        adjacent_species = [structure[j].specie.symbol for j in adjacent]

        # ----------------------------------------------------------
        # Compact one-row-per-antisite table.
        # ----------------------------------------------------------
        site_rows.append({
            "antisite_type": a["antisite_type"],
            "center": f"{a['species']}@{layer_info[lid]['layer_type']}",
            "center_index": center,
            "layer":lid,
            "in_plane_neighbors": "; ".join(format_neighbor(structure, j) for j in inplane),
            "adjacent_layer_neighbors": "; ".join(format_neighbor(structure, j) for j in adjacent),
            "in_plane_species": "; ".join(inplane_species),
            "adjacent_layer_species": "; ".join(adjacent_species),})

        # ----------------------------------------------------------
        # Long-form raw relationship table.
        # ----------------------------------------------------------
        for position, j in enumerate(inplane, start=1):
            neighbor_rows.append({
                "antisite_type": a["antisite_type"],
                "center_species": a["species"],
                "center_index": center,
                "center_layer": lid,
                "neighbor_region": "in_plane",
                "neighbor_side": "same_layer",
                "neighbor_order": position,
                "neighbor_species": structure[j].specie.symbol,
                "neighbor_index": j,
                "distance_A": structure.get_distance(center, j),})
        for position, j in enumerate(adjacent_lower, start=1):
            neighbor_rows.append({
                "antisite_type": a["antisite_type"],
                "center_species": a["species"],
                "center_index": center,
                "center_layer": lid,
                "neighbor_region": "adjacent_layer",
                "neighbor_side": "lower",
                "neighbor_order": position,
                "neighbor_species": structure[j].specie.symbol,
                "neighbor_index": j,
                "distance_A": structure.get_distance(center, j),})
        for position, j in enumerate(adjacent_upper, start=4):
            neighbor_rows.append({
                "antisite_type": a["antisite_type"],
                "center_species": a["species"],
                "center_index": center,
                "center_layer": lid,
                "neighbor_region": "adjacent_layer",
                "neighbor_side": "upper",
                "neighbor_order": position,
                "neighbor_species": structure[j].specie.symbol,
                "neighbor_index": j,
                "distance_A": structure.get_distance(center, j),})

    # --------------------------------------------------------------
    # Antisite pair summaries.
    # --------------------------------------------------------------
    pair_counts = defaultdict(int)
    for row in neighbor_rows:
        key = (row["antisite_type"], row["center_species"], row["neighbor_region"], row["neighbor_species"],)
        pair_counts[key] += 1
    pair_summary = pd.DataFrame([{"antisite_type": key[0], "center_species": key[1],
                                  "neighbor_region": key[2], "neighbor_species": key[3], "count": value,}
        for key, value in sorted(pair_counts.items())])

    # ==============================================================
    # 2. DOPANT-CENTERED ANALYSIS
    # ==============================================================
    dopants = []
    for i in cation_indices:
        sp = structure[i].specie.symbol
        if sp not in DOPANT_ELEMENTS:
            continue
        lid = cation_site_info[i]["layer"]
        ltype = cation_site_info[i]["layer_type"]
        dopants.append({"site_index": i, "dopant": sp, "site": f"{sp}@{ltype}", "layer": lid,
                       "z_fractional": (structure[i].frac_coords[2] % 1.0),})
    dopant_rows = []
    for d in dopants:
        center = d["site_index"]
        lid = d["layer"]
        same_layer = [j for j in groups[lid]["indices"] if j in cation_indices and j != center]
        lower_lid = (lid - 1) % n_layers
        upper_lid = (lid + 1) % n_layers
        lower_candidates = [j for j in groups[lower_lid]["indices"] if j in cation_indices]
        upper_candidates = [j for j in groups[upper_lid]["indices"] if j in cation_indices]
        inplane = nearest_in_layer(structure, center, same_layer, 6)
        adjacent_lower = nearest_in_layer(structure, center, lower_candidates, 3)
        adjacent_upper = nearest_in_layer(structure, center, upper_candidates, 3)
        adjacent = adjacent_lower + adjacent_upper
        if len(inplane) != 6 or len(adjacent) != 6:
            print(f"WARNING: dopant site {center} ({d['site']}) has "
                  f"{len(inplane)} in-plane and {len(adjacent)} adjacent-layer neighbors.")
        inplane_species = [structure[j].specie.symbol for j in inplane]
        adjacent_species = [structure[j].specie.symbol for j in adjacent]
        dopant_rows.append({
            "dopant": d["dopant"],
            "site": d["site"],
            "center_index": center,
            "layer": lid,
            "in_plane_neighbors": "; ".join(format_neighbor(structure, j) for j in inplane),
            "adjacent_layer_neighbors": "; ".join(format_neighbor(structure, j) for j in adjacent),
            "in_plane_species": "; ".join(inplane_species),
            "adjacent_layer_species": "; ".join(adjacent_species),})

    # ==============================================================
    # 3. SAVE OUTPUTS
    # ==============================================================
    pd.DataFrame(layer_info).to_csv(outdir / "layers.csv", index=False)
    pd.DataFrame(antisites).to_csv(outdir / "antisite_sites.csv", index=False)
    pd.DataFrame(site_rows).to_csv(outdir / "antisite_local_environments.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(outdir / "antisite_neighbors_raw.csv", index=False)
    pair_summary.to_csv(outdir / "antisite_pair_summary.csv", index=False)
    pd.DataFrame(dopant_rows).to_csv(outdir / "dopant_local_environments.csv", index=False)

    # ==============================================================
    # 4. CONSOLE SUMMARY
    # ==============================================================
    print(f"\nInput: {cif}")
    print(f"Output: {outdir}")
    print(f"Total antisites analyzed: {len(antisites)}")
    counts = Counter(a["antisite_type"] for a in antisites)
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"\nTotal dopant atoms analyzed: {len(dopants)}")
    dopant_counts = Counter(d["dopant"] for d in dopants)
    for k, v in sorted(dopant_counts.items()):
        print(f"  {k}: {v}")
    print("\nMain antisite file:")
    print("  antisite_local_environments.csv")
    print("\nNew dopant file:")
    print("  dopant_local_environments.csv")
    print("\nBoth analyses give 12 cation neighbors:")
    print("  6 in-plane + 6 adjacent-layer.")

if __name__ == "__main__":
    main()