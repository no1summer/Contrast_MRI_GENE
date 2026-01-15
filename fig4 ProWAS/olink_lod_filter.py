import pandas as pd
import numpy as np
from pathlib import Path

# Paths
lod_path = Path("/data484_4/txia2/mocov2/proWAS/olink_limit_of_detection.dat")
proteomics_path = Path("/data484_1/kmohammed2/xie_proteomics/all_proteomics.csv")
output_path = Path("/data484_4/txia2/mocov2/proWAS/proteomics_above_mean_LOD_min.csv")

# Read LOD file
lod_df = pd.read_csv(
    lod_path,
    sep=r'\s+',
    dtype={"PlateID": str, "Assay": str, "Instance": str},
    low_memory=False
)

# Keep only Instance 0
lod_df = lod_df[lod_df["Instance"] == "0"]

# Ensure LOD column is numeric
lod_df["LOD"] = pd.to_numeric(lod_df["LOD"], errors="coerce")

# Drop rows with missing LOD or protein
lod_df = lod_df.dropna(subset=["LOD", "Assay"])

# Compute mean LOD per protein (Assay) across plates, only Instance 0
mean_lod = lod_df.groupby("Assay")["LOD"].min()

# Load proteomics file
prot_df = pd.read_csv(proteomics_path)
prot_df["eid"] = prot_df["eid"].astype(int)
prot_df = prot_df.rename(columns={c: c.upper() for c in prot_df.columns if c != "eid"})

# Filter proteomics values below mean LOD
protein_cols = [c for c in prot_df.columns if c != "eid"]
mean_lod.index = mean_lod.index.str.upper()
print("Proteins missing in mean LOD:", [c for c in protein_cols if c not in mean_lod.index])

print("Mean LOD per protein:", mean_lod.head())
print("Proteomics min values per protein:", prot_df[protein_cols].min().head())


lod_series = mean_lod.reindex(protein_cols)
filtered_prot_df = prot_df.copy()
filtered_prot_df[protein_cols] = filtered_prot_df[protein_cols].where(
    filtered_prot_df[protein_cols] >= lod_series, np.nan
)

# Save filtered matrix
filtered_prot_df.to_csv(output_path, index=False)
print(f"Filtered proteomics matrix saved to {output_path}")

total_new_nans = (filtered_prot_df[protein_cols].isna() & ~prot_df[protein_cols].isna()).sum().sum()

print(f"Total new NaNs introduced by LOD filtering: {total_new_nans}")

