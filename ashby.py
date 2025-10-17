import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the file into a DataFrame
Ni = pd.read_csv(
    "mats_data/ni_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ni.columns = ["Trade name", "UNS number", "IACS_percent", "Conductivity_10E6"]
Ni.sort_values(by=Ni.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ag = pd.read_csv(
    "mats_data/ag_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ag.columns = ["Alloy", "IACS_percent", "Conductivity_10E6"]
Ag.sort_values(by=Ag.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Al = pd.read_csv(
    "mats_data/al_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Al.columns = ["Alloy", "IACS_percent", "Conductivity_10E7"]
Al['Alloy'] = Al['Alloy'].str.extract(r'(\d{4})')
Al.sort_values(by=Al.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Al_cast = pd.read_csv(
    "mats_data/cast_al_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Al_cast.columns = ["Alloy", "IACS_percent", "Conductivity_10E7"]
Al_cast.sort_values(by=Al_cast.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Cu_cast = pd.read_csv(
    "mats_data/cast_cu_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Cu_cast.columns = ["Alloy", "IACS_percent", "Conductivity_10E7"]
Cu_cast['Alloy'] = Cu_cast['Alloy'].str.extract(r'(\d{5})')
Cu_cast.sort_values(by=Cu_cast.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
ceramics = pd.read_csv(
    "mats_data/ceramic.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
ceramics.columns = ["Name", "Class", "Conductivity"]
ceramics.sort_values(by=ceramics.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Cu = pd.read_csv(
    "mats_data/cu_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Cu.columns = ["Alloy", "IACS_percent", "Conductivity_10E7"]
Cu['Alloy'] = Cu['Alloy'].str.extract(r'(\d{5})')
Cu.sort_values(by=Cu.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
metal = pd.read_csv(
    "mats_data/metal.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
metal.columns = ["Metal", "Conductivity_10E6"]
metal.sort_values(by=metal.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Mg = pd.read_csv(
    "mats_data/mg_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Mg.columns = ["Trade name", "UNS number", "IACS_percent", "Conductivity_10E6"]
Mg.sort_values(by=Mg.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
polymer = pd.read_csv(
    "mats_data/polymer.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
polymer.columns = ["Name", "Conductivity_10E7"]
polymer.sort_values(by=polymer.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ti = pd.read_csv(
    "mats_data/ti_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ti.columns = ["Trade name", "UNS number", "IACS_percent", "Conductivity_10E6"]
Ti.sort_values(by=Ti.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Zn = pd.read_csv(
    "mats_data/zn_alloy.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Zn.columns = ["Alloy", "Composition", "IACS_percent", "Conductivity_10E6"]
Zn.sort_values(by=Zn.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ni_rho = pd.read_csv(
    "mats_data/ni_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ni_rho.sort_values(by=Ni_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ag_rho = pd.read_csv(
    "mats_data/ag_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ag_rho.sort_values(by=Ag_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Al_rho = pd.read_csv(
    "mats_data/al_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Al_rho.columns = ["Alloy", "Density"]
Al_rho.sort_values(by=Al_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Al_cast_rho = pd.read_csv(
    "mats_data/cast_al_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Al_cast_rho.columns = ["Alloy", "Density"]
Al_cast_rho.sort_values(by=Al_cast_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Cu_cast_rho = pd.read_csv(
    "mats_data/cast_cu_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Cu_cast_rho.columns = ["Alloy", "Density"]
Cu_cast_rho['Alloy'] = Cu_cast_rho['Alloy'].str.extract(r'(\d{5})')
Cu_cast_rho.sort_values(by=Cu_cast_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
ceramics_rho = pd.read_csv(
    "mats_data/ceramic_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
ceramics_rho.sort_values(by=ceramics_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Cu_rho = pd.read_csv(
    "mats_data/cu_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Cu_rho.columns = ["Alloy", "Density"]
Cu_rho['Alloy'] = Cu_rho['Alloy'].str.extract(r'(\d{5})')
Cu_rho.sort_values(by=Cu_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
metal_rho = pd.read_csv(
    "mats_data/metal_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
metal_rho.sort_values(by=metal_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Mg_rho = pd.read_csv(
    "mats_data/mg_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Mg_rho.sort_values(by=Mg_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
polymer_rho = pd.read_csv(
    "mats_data/polymer_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
polymer_rho.sort_values(by=polymer_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ti_rho = pd.read_csv(
    "mats_data/ti_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ti_rho.sort_values(by=Ti_rho.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Zn_rho = pd.read_csv(
    "mats_data/zn_density.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Zn_rho.columns = ["Alloy", "UNS number", "Composition", "Density"]
Zn_rho.sort_values(by=Zn_rho.columns[0], ascending=True, inplace=True)

Ni.columns = Ni.columns.str.strip()
Ni_rho.columns = Ni_rho.columns.str.strip()
Ni_merged = pd.merge(Ni, Ni_rho, on='Trade name', how='inner')

Ag.columns = Ag.columns.str.strip()
Ag_rho.columns = Ag_rho.columns.str.strip()
Ag_merged = pd.merge(Ag, Ag_rho, on='Alloy', how='inner')

Al.columns = Al.columns.str.strip()
Al_rho.columns = Al_rho.columns.str.strip()
Al['Alloy'] = Al['Alloy'].astype(int)
Al_rho['Alloy'] = Al_rho['Alloy'].astype(int)
Al_merged = pd.merge(Al, Al_rho, on='Alloy', how='inner')

Al_cast.columns = Al_cast.columns.str.strip()
Al_cast_rho.columns = Al_cast_rho.columns.str.strip()
Al_cast['Alloy'] = Al_cast['Alloy'].astype(int)
Al_cast_rho['Alloy'] = Al_cast_rho['Alloy'].astype(int)
Al_cast_merged = pd.merge(Al_cast, Al_cast_rho, on='Alloy', how='inner')

Cu.columns = Cu.columns.str.strip()
Cu_rho.columns = Cu_rho.columns.str.strip()
Cu_merged = pd.merge(Cu, Cu_rho, on='Alloy', how='inner')

Cu_cast.columns = Cu_cast.columns.str.strip()
Cu_cast_rho.columns = Cu_cast_rho.columns.str.strip()
Cu_cast_merged = pd.merge(Cu_cast, Cu_cast_rho, on='Alloy', how='inner')

ceramics.columns = ceramics.columns.str.strip()
ceramics_rho.columns = ceramics_rho.columns.str.strip()
ceramics_merged = pd.merge(ceramics, ceramics_rho, on='Name', how='inner')

metal.columns = metal.columns.str.strip()
metal_rho.columns = metal_rho.columns.str.strip()
metal_merged = pd.merge(metal, metal_rho, on='Metal', how='inner')

Mg.columns = Mg.columns.str.strip()
Mg_rho.columns = Mg_rho.columns.str.strip()
Mg_merged = pd.merge(Mg, Mg_rho, on='Trade name', how='inner')

polymer.columns = polymer.columns.str.strip()
polymer_rho.columns = polymer_rho.columns.str.strip()
polymer_merged = pd.merge(polymer, polymer_rho, on='Name', how='inner')

Ti.columns = Ti.columns.str.strip()
Ti_rho.columns = Ti_rho.columns.str.strip()
Ti_merged = pd.merge(Ti, Ti_rho, on='Trade name', how='inner')
Ti_merged = Ti_merged.dropna(subset=['Conductivity_10E6'])

Zn.columns = Zn.columns.str.strip()
Zn_rho.columns = Zn_rho.columns.str.strip()
Zn_merged = pd.merge(Zn, Zn_rho, on='Composition', how='inner')

# --- 1) Collect your dataframes with a friendly label -------------------------
dataframes = {
    "Nickel alloys": Ni_merged,
    "Silver alloys": Ag_merged,
    "Aluminum wrought": Al_merged,
    "Aluminum cast": Al_cast_merged,
    "Copper cast": Cu_cast_merged,
    "Ceramics": ceramics_merged,
    "Copper wrought": Cu_merged,
    "Metals (pure)": metal_merged,
    "Magnesium alloys": Mg_merged,
    "Polymers": polymer_merged,
    "Titanium alloys": Ti_merged,
    "Zinc alloys": Zn_merged,
}

# --- 2) Plot each dataframe as a single bubble on an Ashby diagram ------------

import numpy as np
import matplotlib.pyplot as plt

def get_conductivity_s_per_m(df):
    """
    Return a 1D array of conductivity in S/m, inferring the correct column
    and unit scaling from the column name.
    """
    # Candidates in order of preference
    col = None
    if 'Conductivity_10E6' in df.columns:
        col = 'Conductivity_10E6'
        scale = 1e6
    elif 'Conductivity_10E7' in df.columns:
        col = 'Conductivity_10E7'
        scale = 1e7
    elif 'Conductivity' in df.columns:
        col = 'Conductivity'
        scale = 1.0
    else:
        raise ValueError("No conductivity column found in dataframe.")

    vals = df[col].astype(float) * scale
    # Drop non-positive / NaN (log scale needs positive)
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0]
    return vals

def get_density_g_per_cm3(df):
    if 'Density' not in df.columns:
        raise ValueError("No Density column found in dataframe.")
    dens = df['Density'].astype(float)
    dens = dens.replace([np.inf, -np.inf], np.nan).dropna()
    dens = dens[dens > 0]
    return dens

bubble_points = []
for label, df in dataframes.items():
    try:
        sigma = get_conductivity_s_per_m(df)
        rho   = get_density_g_per_cm3(df)
        # Align lengths only where both present (inner-join on index if needed)
        # If different lengths (common when merged), reduce to medians independently.
        if len(sigma) == 0 or len(rho) == 0:
            continue
        sigma_med = float(np.median(sigma))
        rho_med   = float(np.median(rho))
        n = min(len(sigma), len(rho))  # conservative count
        bubble_points.append((label, rho_med, sigma_med, n))
    except Exception:
        # Skip datasets missing required fields
        continue

# Size bubbles by dataset size with a gentle scale factor
if not bubble_points:
    raise RuntimeError("No valid datasets to plot. Check columns/merges.")

labels, rho_vals, sigma_vals, counts = zip(*bubble_points)
sizes = np.array(counts, dtype=float)
sizes = 50.0 * (1.0 + np.log10(sizes))  # readable scaling

plt.figure(figsize=(9, 7))
plt.scatter(rho_vals, sigma_vals, s=sizes, alpha=0.6, edgecolor='k', linewidth=0.5)

# Annotate each bubble
for lbl, x, y, n in bubble_points:
    plt.annotate(lbl, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Density (g/cm³)')
plt.ylabel('Electrical conductivity (S/m)')
plt.title('Ashby diagram: conductivity vs density (one bubble per dataset)')

# Nice grid on log axes
plt.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)

plt.tight_layout()
plt.show()
