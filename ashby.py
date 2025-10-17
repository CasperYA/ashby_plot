import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import re
from itertools import cycle

def normalize_strengths(df, method="mean"):
    """
    Cleans and normalizes 'Yield strength' and 'Ultimate strength' columns in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        May contain columns 'Yield strength' and/or 'Ultimate strength'.
    method : str, optional
        How to handle intervals ('390 - 520'):
        - 'mean': use the average of the two values
        - 'max': use the higher value

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with numeric strength columns (if present).
    """

    def parse_strength(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)

        s = str(value).strip()
        match = re.match(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", s)
        if match:
            a, b = map(float, match.groups())
            return (a + b) / 2 if method == "mean" else max(a, b)

        try:
            return float(s)
        except ValueError:
            return np.nan

    df = df.copy()

    # Only process columns that exist
    for col in ["Yield strength", "Ultimate strength"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_strength)

    return df


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

# Load the file into a DataFrame
Ni_yield = pd.read_csv(
    "mats_data/ni_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ni_yield.sort_values(by=Ni_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ag_yield = pd.read_csv(
    "mats_data/ag_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ag_yield.sort_values(by=Ag_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Al_yield = pd.read_csv(
    "mats_data/al_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Al_yield.columns = ["Alloy", "Yield strength", "Ultimate strength"]
Al_yield['Alloy'] = Al_yield['Alloy'].str.extract(r'(\d{4})')
Al_yield.sort_values(by=Al_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Al_cast_yield = pd.read_csv(
    "mats_data/cast_al_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Al_cast_yield.columns = ["Alloy", "Yield strength", "Ultimate strength"]
Al_cast_yield.sort_values(by=Al_cast_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Cu_cast_yield = pd.read_csv(
    "mats_data/cast_cu_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Cu_cast_yield.columns = ["Alloy", "Yield strength", "Ultimate strength"]
Cu_cast_yield['Alloy'] = Cu_cast_yield['Alloy'].str.extract(r'(\d{5})')
Cu_cast_yield.sort_values(by=Cu_cast_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Cu_yield = pd.read_csv(
    "mats_data/cu_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Cu_yield.columns = ["Alloy", "Yield strength", "Ultimate strength"]
Cu_yield['Alloy'] = Cu_yield['Alloy'].str.extract(r'(\d{5})')
Cu_yield.sort_values(by=Cu_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
metal_yield = pd.read_csv(
    "mats_data/metal_yield.csv",
    sep=r",",
    engine="python",   # allows regex separator
)
metal_yield.columns = ["Metal", "Yield strength", "Ultimate strength"]
metal_yield.sort_values(by=metal_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Mg_yield = pd.read_csv(
    "mats_data/mg_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Mg_yield.sort_values(by=Mg_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Ti_yield = pd.read_csv(
    "mats_data/ti_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Ti_yield.sort_values(by=Ti_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
Zn_yield = pd.read_csv(
    "mats_data/zn_yield.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
Zn_yield.columns = ["Alloy", "UNS", "Composition", "Ultimate strength"]
Zn_yield.sort_values(by=Zn_yield.columns[0], ascending=True, inplace=True)

# Load the file into a DataFrame
scatter = pd.read_csv(
    "mats_data/scatter.txt",
    sep=r"\t+",        # split by one or more tabs
    engine="python",   # allows regex separator
)
scatter.sort_values(by=scatter.columns[0], ascending=True, inplace=True)


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

# Dataframes with strengths (Removes significant data from earlier dataframes if overwritten)
Ni_merged.columns = Ni_merged.columns.str.strip()
Ni_yield.columns = Ni_yield.columns.str.strip()
Ni_merged_strength = pd.merge(Ni_merged, Ni_yield, on='Trade name', how='inner')
Ni_merged_strength = normalize_strengths(Ni_merged_strength, method='max')

Ag_merged.columns = Ag_merged.columns.str.strip()
Ag_yield.columns = Ag_yield.columns.str.strip()
Ag_merged_strength = pd.merge(Ag_merged, Ag_yield, on='Alloy', how='inner')
Ag_merged_strength = normalize_strengths(Ag_merged_strength, method='max')

Al_merged.columns = Al_merged.columns.str.strip()
Al_yield.columns = Al_yield.columns.str.strip()
Al_yield['Alloy'] = Al_yield['Alloy'].astype(int)
Al_merged_strength = pd.merge(Al_merged, Al_yield, on='Alloy', how='inner')
Al_merged_strength = normalize_strengths(Al_merged_strength, method='max')

Al_merged.columns = Al_merged.columns.str.strip()
Al_cast_yield.columns = Al_cast_yield.columns.str.strip()
Al_cast_yield['Alloy'] = Al_cast_yield['Alloy'].astype(int)
Al_cast_merged_strength = pd.merge(Al_cast_merged, Al_cast_yield, on='Alloy', how='inner')
Al_cast_merged_strength = normalize_strengths(Al_cast_merged_strength, method='max')

Cu_merged.columns = Cu_merged.columns.str.strip()
Cu_yield.columns = Cu_yield.columns.str.strip()
Cu_merged_strength = pd.merge(Cu_merged, Cu_yield, on='Alloy', how='inner')
Cu_merged_strength = normalize_strengths(Cu_merged_strength, method='max')

Cu_cast_merged.columns = Cu_cast_merged.columns.str.strip()
Cu_cast_yield.columns = Cu_cast_yield.columns.str.strip()
Cu_cast_merged_strength = pd.merge(Cu_cast_merged, Cu_cast_yield, on='Alloy', how='inner')
Cu_cast_merged_strength = normalize_strengths(Cu_cast_merged_strength, method='max')

metal_merged.columns = metal_merged.columns.str.strip()
metal_yield.columns = metal_yield.columns.str.strip()
metal_merged_strength = pd.merge(metal_merged, metal_yield, on='Metal', how='inner')
metal_merged_strength = normalize_strengths(metal_merged_strength, method='max')

Mg_merged.columns = Mg_merged.columns.str.strip()
Mg_yield.columns = Mg_yield.columns.str.strip()
Mg_merged_strength = pd.merge(Mg_merged, Mg_yield, on='Trade name', how='inner')
Mg_merged_strength = normalize_strengths(Mg_merged_strength, method='max')

Ti_merged.columns = Ti_merged.columns.str.strip()
Ti_yield.columns = Ti_yield.columns.str.strip()
Ti_merged = pd.merge(Ti_merged, Ti_yield, on='Trade name', how='inner')
Ti_merged_strength = Ti_merged.dropna(subset=['Conductivity_10E6'])
Ti_merged_strength = normalize_strengths(Ti_merged_strength, method='mean')

Zn_merged.columns = Zn_merged.columns.str.strip()
Zn_yield.columns = Zn_yield.columns.str.strip()
Zn_merged_strength = pd.merge(Zn_merged, Zn_yield, on='Composition', how='inner')
Zn_merged_strength = normalize_strengths(Zn_merged_strength, method='max')

# --- 1) Collect your dataframes with a friendly label -------------------------
conductivities = {
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

strengths = {
    "Nickel alloys": Ni_merged_strength,
    "Silver alloys": Ag_merged_strength,
    "Aluminum wrought": Al_merged_strength,
    "Aluminum cast": Al_cast_merged_strength,
    "Copper cast": Cu_cast_merged_strength,
    "Copper wrought": Cu_merged_strength,
    "Metals (pure)": metal_merged_strength,
    "Magnesium alloys": Mg_merged_strength,
    "Titanium alloys": Ti_merged_strength,
    "Zinc alloys": Zn_merged_strength,
}

# --- 2) Plot each dataframe as a single bubble on an Ashby diagram ------------

def _to_float_series(s):
    """Coerce a pandas Series of messy numerics to float."""
    def parse_val(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if v is None:
            return np.nan
        t = str(v).strip()
        if t == "" or t.lower() in {"na", "nan", "none", "-", "—"}:
            return np.nan
        # "1,23" -> "1.23" if no dot present
        if "," in t and "." not in t:
            t = t.replace(",", ".")
        # direct float first
        try:
            return float(t)
        except Exception:
            pass
        # a * 10^b  (3*10^7, 3×10^7, 3·10^7, 3 * 10 ** 7)
        m = re.match(r'^\s*([+-]?\d*\.?\d+)\s*([*x×·])\s*10\s*(?:\^|\*\*)\s*([+-]?\d+)\s*$', t, re.I)
        if m:
            a = float(m.group(1)); b = int(m.group(3))
            return a * (10.0 ** b)
        # a*10eB  ('5*10e-22' => a * 10^(B+1))
        m = re.match(r'^\s*([+-]?\d*\.?\d+)\s*([*x×·])\s*10[eE]\s*([+-]?\d+)\s*$', t)
        if m:
            a = float(m.group(1)); b = int(m.group(3))
            return a * (10.0 ** (b + 1))
        # 10eB alone -> 10^(B+1)
        m = re.match(r'^\s*10[eE]\s*([+-]?\d+)\s*$', t)
        if m:
            b = int(m.group(1))
            return 10.0 ** (b + 1)
        return np.nan
    return s.apply(parse_val)

# ---------- Column pickers ----------
def _pick_conductivity(df):
    """Return a Series of conductivity in S/m (best available column)."""
    if 'Conductivity_10E6' in df.columns:
        s = _to_float_series(df['Conductivity_10E6']) * 1e6
    elif 'Conductivity_10E7' in df.columns:
        s = _to_float_series(df['Conductivity_10E7']) * 1e7
    elif 'Conductivity' in df.columns:
        s = _to_float_series(df['Conductivity'])
    elif 'IACS_percent' in df.columns:
        # 100% IACS ≈ 58 MS/m = 5.8e7 S/m
        s = _to_float_series(df['IACS_percent']) * (5.8e7 / 100.0)
    else:
        return None
    return s

def _density_g_cm3(df):
    """Return density in g/cm³ (auto-detect kg/m³ and convert)."""
    if 'Density' not in df.columns:
        return None
    d = _to_float_series(df['Density'])
    d = d.replace([np.inf, -np.inf], np.nan)
    # If densities look like kg/m³ (e.g., 8890), convert to g/cm³ by /1000
    if d.dropna().median() and d.dropna().median() > 100:
        d = d / 1000.0
    return d

def _ultimate_strength_mpa(df):
    """Return ultimate tensile strength in MPa."""
    col_candidates = ['Ultimate strength', 'Ultimate_strength', 'UTS', 'Tensile strength']
    for c in col_candidates:
        if c in df.columns:
            return _to_float_series(df[c]).replace([np.inf, -np.inf], np.nan)
    return None

# ---------- Geometry: convex hull in log space ----------
def _convex_hull_monotone_chain(points):
    pts = np.array(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]  # sort by x, then y
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    return np.array(lower[:-1] + upper[:-1], dtype=float)

# ---------- Envelope plotting helper ----------
def _plot_envelopes(ax, datasets, get_xy_fn, color_map, fill_alpha=0.12, line_width=1.8):
    for label, df in datasets.items():
        xy = get_xy_fn(df)
        if xy is None:
            continue
        x, y = xy
        # keep positive finite pairs only
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x, y = x[m], y[m]
        if x.size == 0:
            continue

        # Work in log space for hull
        log_pts = np.column_stack((np.log10(x), np.log10(y)))
        c = color_map.get(label, None)

        if log_pts.shape[0] == 1:
            xp, yp = 10**log_pts[0,0], 10**log_pts[0,1]
            ax.plot([xp], [yp], marker='o', markersize=6, label=label, color=c)
            continue
        if log_pts.shape[0] == 2:
            xp, yp = 10**log_pts[:,0], 10**log_pts[:,1]
            ax.plot(xp, yp, linewidth=line_width, label=label, color=c)
            continue

        hull_log = _convex_hull_monotone_chain(log_pts)
        if hull_log.shape[0] < 3:
            xp, yp = 10**hull_log[:,0], 10**hull_log[:,1]
            ax.plot(xp, yp, linewidth=line_width, label=label, color=c)
            continue

        hx, hy = 10**hull_log[:,0], 10**hull_log[:,1]
        ax.plot(list(hx) + [hx[0]], list(hy) + [hy[0]], linewidth=line_width, label=label, color=c)
        ax.fill(hx, hy, alpha=fill_alpha, color=c)

# ---------- Build consistent colors across both plots ----------
all_labels = list(conductivities.keys()) + list(set(strengths.keys()) - set(conductivities.keys()))
cmap = cm.get_cmap("tab20", len(all_labels))  # 20 distinct colors
color_map = {lbl: mcolors.to_hex(cmap(i)) for i, lbl in enumerate(all_labels)}

# ---------- XY getters for the two charts ----------
def _xy_conductivity(df):
    dens = _density_g_cm3(df)
    sigma = _pick_conductivity(df)
    if dens is None or sigma is None:
        return None
    sub = np.column_stack((dens.values, sigma.values))
    sub = sub[~np.isnan(sub).any(axis=1)]
    if sub.size == 0:
        return None
    return sub[:,0], sub[:,1]  # x=density (g/cm³), y=conductivity (S/m)

def _xy_strength(df):
    dens = _density_g_cm3(df)
    uts = _ultimate_strength_mpa(df)
    if dens is None or uts is None:
        return None
    sub = np.column_stack((dens.values, uts.values))
    sub = sub[~np.isnan(sub).any(axis=1)]
    if sub.size == 0:
        return None
    return sub[:,0], sub[:,1]  # x=density (g/cm³), y=UTS (MPa)

# ---------- Plot side-by-side ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

# Conductivity vs density
_plot_envelopes(ax1, conductivities, _xy_conductivity, color_map)
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel('Densitet (g/cm³)')
ax1.set_ylabel('Elektrisk Ledningsförmåga (S/m)')
ax1.set_title('Ashbydiagram: Elektrisk Ledningsförmåga mot Densitet')
ax1.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)

# Ultimate strength vs density
_plot_envelopes(ax2, strengths, _xy_strength, color_map)
ax2.set_xscale('log'); ax2.set_yscale('log')
ax2.set_xlabel('Densitet (g/cm³)')
ax2.set_ylabel('Brottgräns (MPa)')
ax2.set_title('Ashbydiagram: Brottgräns mot Densitet')
ax2.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)

# Legends (keep only legend, no on-plot text)
# If there are many labels, placing legends outside helps:
ax1.legend(loc='lower left', frameon=True, fontsize=9)
ax2.legend(loc='lower right', frameon=True, fontsize=9)

handles, labels = ax1.get_legend_handles_labels()
new_labels = [lbl.replace("Aluminum cast", "Cast Aluminum").replace("Copper cast", "Cast Copper")
              for lbl in labels]
ax1.legend(handles, new_labels, loc='lower left', frameon=True, fontsize=9)
ax2.legend(handles, new_labels, loc='lower right', frameon=True, fontsize=9)

def _parse_num_series(s):
    """Lightweight parser for messy numerics (e.g., '5*10e-2', '3×10^7', '1,23')."""
    def p(v):
        if isinstance(v, (int, float, np.floating)): return float(v)
        t = str(v).strip()
        if not t or t.lower() in {"na","nan","none","-","—"}: return np.nan
        if "," in t and "." not in t: t = t.replace(",", ".")
        try: return float(t)
        except: pass
        m = re.match(r'^\s*([+-]?\d*\.?\d+)\s*([*x×·])\s*10\s*(?:\^|\*\*)\s*([+-]?\d+)\s*$', t, re.I)
        if m: return float(m.group(1)) * (10.0 ** int(m.group(3)))
        m = re.match(r'^\s*([+-]?\d*\.?\d+)\s*[x×*·]\s*10[eE]\s*([+-]?\d+)\s*$', t)
        if m: return float(m.group(1)) * (10.0 ** (int(m.group(2)) + 1))
        m = re.match(r'^\s*10[eE]\s*([+-]?\d+)\s*$', t)
        if m: return 10.0 ** (int(m.group(1)) + 1)
        return np.nan
    return s.apply(p)

# Density (convert kg/m³ -> g/cm³ if needed)
dens = _parse_num_series(scatter['Density'])
if dens.dropna().median() and dens.dropna().median() > 100:
    dens = dens / 1000.0

# Conductivity in S/m (try multiple columns; fall back to IACS if present)
if 'Conductivity_10E6' in scatter.columns:
    sigma = _parse_num_series(scatter['Conductivity_10E6']) * 1e6
elif 'Conductivity_10E7' in scatter.columns:
    sigma = _parse_num_series(scatter['Conductivity_10E7']) * 1e7
elif 'Conductivity' in scatter.columns:
    sigma = _parse_num_series(scatter['Conductivity'])
elif 'IACS_percent' in scatter.columns:
    sigma = _parse_num_series(scatter['IACS_percent']) * (5.8e7 / 100.0)
else:
    sigma = pd.Series(np.nan, index=scatter.index)

# Ultimate strength (MPa)
if 'Ultimate strength' in scatter.columns:
    uts = _parse_num_series(scatter['Ultimate strength'])
else:
    uts = pd.Series(np.nan, index=scatter.index)

# Masks
m_c = np.isfinite(dens) & np.isfinite(sigma) & (dens > 0) & (sigma > 0)
m_s = np.isfinite(dens) & np.isfinite(uts)   & (dens > 0) & (uts > 0)

# Plot helper
def _plot_red_with_labels(ax, x, y, labels, dx=4, dy=4):
    ax.scatter(x, y, s=25, color='red', zorder=10)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(str(lab), (xi, yi), xytext=(dx, dy),
                    textcoords='offset points', fontsize=9,
                    color='red', zorder=11)

# Plot onto existing subplots
if m_c.any():
    _plot_red_with_labels(ax1, dens[m_c].values, sigma[m_c].values,
                          scatter.loc[m_c, 'Material'].values)
if m_s.any():
    _plot_red_with_labels(ax2, dens[m_s].values, uts[m_s].values,
                          scatter.loc[m_s, 'Material'].values)

Mg_data = [2.24*10e7, 1738, 380]
Al_data = [3.5*10e7, 2700, 495]
Ti_data = [2.38*10e6, 4506, 900]

sigma_Mg, rho_Mg, uts_Mg = Mg_data
_, rho_Al, uts_Al = Al_data
_, rho_Ti, uts_Ti = Ti_data

# Convert density to the plot's units (g/cm³) if it looks like kg/m³
rho_plot = rho_Mg / 1000.0 if rho_Mg > 100 else rho_Mg
rho_Al_plot = rho_Al / 1000.0 if rho_Al > 100 else rho_Al
rho_Ti_plot = rho_Ti / 1000.0 if rho_Ti > 100 else rho_Ti

k_cond = sigma_Mg / rho_plot  # slope for conductivity plot: y = k_cond * x
k_uts  = uts_Mg  / rho_plot   # slope for UTS plot:          y = k_uts  * x
k_uts_Al = uts_Al / rho_Al_plot
k_uts_Ti = uts_Ti / rho_Ti_plot

xmin, xmax = ax2.get_xlim()
xline = np.array([xmin, xmax], dtype=float)
yline = k_uts_Al * xline

# Helper to draw y = k*x across current x-limits of an axis
def _draw_through_point(ax, k, label):
    xmin, xmax = ax.get_xlim()
    xline = np.array([xmin, xmax], dtype=float)
    yline = k * xline
    ax.plot(xline, yline, linestyle="--", linewidth=1.5, color="red", alpha=0.9, label=label)

# Plot the lines on existing subplots
_draw_through_point(ax1, k_cond, "y = (σ/ρ)·x (Mg)")
_draw_through_point(ax2, k_uts,  "y = (UTS/ρ)·x (Mg)")

ax2.plot(xline, yline, linestyle="--", linewidth=1.5, color="green",
         alpha=0.9, label="y = (UTS/ρ)·x (Al)")

yline = k_uts_Ti * xline
ax2.plot(xline, yline, linestyle="--", linewidth=1.5, color="blue",
         alpha=0.9, label="y = (UTS/ρ)·x (Ti)")

plt.tight_layout()
plt.show()