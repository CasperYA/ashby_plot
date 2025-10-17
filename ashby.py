import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

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

def _pick_conductivity_col(df):
    """
    Choose a conductivity column and its scale to S/m from the name.
    Priority: Conductivity_10E6 (×1e6), Conductivity_10E7 (×1e7), Conductivity (S/m).
    """
    if 'Conductivity_10E6' in df.columns:
        return 'Conductivity_10E6', 1e6
    if 'Conductivity_10E7' in df.columns:
        return 'Conductivity_10E7', 1e7
    if 'Conductivity' in df.columns:
        return 'Conductivity', 1.0
    return None, None

def _to_float_series(s):
    """Coerce a pandas Series of messy numerics to float.
    Handles cases like '5*10e-22', '3×10^7', '1,23', standard '1e-3', etc."""
    def parse_val(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if v is None:
            return np.nan
        t = str(v).strip()
        if t == "" or t.lower() in {"na", "nan", "none", "-", "—"}:
            return np.nan

        # Use comma as decimal if no dot present: "1,23" -> "1.23"
        if "," in t and "." not in t:
            t = t.replace(",", ".")

        # Try standard float / scientific first
        try:
            return float(t)
        except Exception:
            pass

        # Match a * 10^b (e.g., '3*10^7', '3×10^7', '3·10^7', '3 * 10 ** 7')
        m = re.match(r'^\s*([+-]?\d*\.?\d+)\s*([*x×·])\s*10\s*(?:\^|\*\*)\s*([+-]?\d+)\s*$', t, re.I)
        if m:
            a = float(m.group(1)); b = int(m.group(3))
            return a * (10.0 ** b)

        # Match the nonstandard: a*10eB  (e.g., '5*10e-22' -> a * 10^(B+1))
        m = re.match(r'^\s*([+-]?\d*\.?\d+)\s*([*x×·])\s*10[eE]\s*([+-]?\d+)\s*$', t)
        if m:
            a = float(m.group(1)); b = int(m.group(3))
            return a * (10.0 ** (b + 1))

        # Lone "10eB" (rare): treat as 10^(B+1)
        m = re.match(r'^\s*10[eE]\s*([+-]?\d+)\s*$', t)
        if m:
            b = int(m.group(1))
            return 10.0 ** (b + 1)

        return np.nan

    return s.apply(parse_val)

def _extract_xy(df):
    ccol, scale = _pick_conductivity_col(df)
    if ccol is None or 'Density' not in df.columns:
        return None, None

    sub = df[[ccol, 'Density']].copy()

    # Parse textual numbers safely
    sub[ccol] = _to_float_series(sub[ccol]) * (scale if scale is not None else 1.0)
    sub['Density'] = _to_float_series(sub['Density'])

    # Clean and keep only positive, finite rows
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    m = (sub['Density'] > 0) & (sub[ccol] > 0)
    sub = sub[m]
    if sub.empty:
        return None, None

    x = sub['Density'].astype(float).values
    y = sub[ccol].astype(float).values
    return x, y


def _convex_hull_monotone_chain(points):
    """
    Andrew's monotone chain convex hull (works on 2D np.array of shape (n,2)).
    Returns hull points in counter-clockwise order without repeating the first point.
    """
    pts = np.array(points, dtype=float)
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]  # sort by x, then y
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull

plt.figure(figsize=(10, 8))

legend_handles = []
for label, df in dataframes.items():
    x, y = _extract_xy(df)
    if x is None:
        continue

    # Work in log10 space for a proper Ashby hull
    log_pts = np.column_stack((np.log10(x), np.log10(y)))
    if log_pts.shape[0] == 1:
        # Single point: draw a marker
        xp, yp = 10**log_pts[0,0], 10**log_pts[0,1]
        h = plt.plot([xp], [yp], marker='o', markersize=6, label=label)[0]
        legend_handles.append(h)
        continue
    if log_pts.shape[0] == 2:
        # Two points: draw a line segment
        xp, yp = 10**log_pts[:,0], 10**log_pts[:,1]
        h = plt.plot(xp, yp, linewidth=1.8, label=label)[0]
        legend_handles.append(h)
        continue

    hull_log = _convex_hull_monotone_chain(log_pts)
    if hull_log.shape[0] < 3:
        xp, yp = 10**hull_log[:,0], 10**hull_log[:,1]
        h = plt.plot(xp, yp, linewidth=1.8, label=label)[0]
        legend_handles.append(h)
        continue

    # Back to linear for plotting
    hull_x = 10**hull_log[:,0]
    hull_y = 10**hull_log[:,1]

    # Draw outline + light fill
    [h] = plt.plot(hull_x.tolist() + [hull_x[0]],
                   hull_y.tolist() + [hull_y[0]],
                   linewidth=1.8, label=label)
    plt.fill(hull_x, hull_y, alpha=0.12)

    # Label near the hull centroid (in log space, then convert back)
    cx, cy = 10**np.mean(hull_log[:,0]), 10**np.mean(hull_log[:,1])
    # plt.annotate(label, (cx, cy), xytext=(4, 4),
    #              textcoords='offset points', fontsize=9)
    legend_handles.append(h)

# Axes + cosmetics
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Density (g/cm³)')
plt.ylabel('Electrical conductivity (S/m)')
plt.title('Ashby diagram: conductivity vs density — dataset envelopes')
plt.grid(True, which='both', linestyle=':', linewidth=0.6, alpha=0.7)
if legend_handles:
    plt.legend(loc='best', frameon=True, fontsize=9)
plt.tight_layout()
plt.show()