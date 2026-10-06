#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd

os.chdir("~/Desktop/Data_Analysis/AnalyzedData")
base = Path(".")

files = ["antisite_local_environments.csv",
         "antisite_pair_summary.csv",
         "dopant_local_environments.csv",]

for filename in files:
    dfs = []

    for compound_dir in sorted(base.iterdir()):
        if not compound_dir.is_dir():
            continue

        csv_file = compound_dir / filename

        if not csv_file.exists():
            continue

        df = pd.read_csv(csv_file)

        # Compound name = directory name
        df.insert(0, "Compound", compound_dir.name)

        dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)

        output = base / filename.replace(
            ".csv", "_all.csv"
        )

        combined.to_csv(output, index=False)

        print(f"Created: {output}")
        print(f"Rows: {len(combined)}")