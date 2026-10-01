"""
pipeline/utils.py
=================
Shared helper functions used across all pipeline steps.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────
# Rolling mean
# ─────────────────────────────────────────────────────────────────
def roll(arr, w):
    """Centred rolling mean, min_periods=3."""
    return pd.Series(arr).rolling(w, center=True, min_periods=3).mean()


# ─────────────────────────────────────────────────────────────────
# SPI from precipitation series
# ─────────────────────────────────────────────────────────────────
def compute_spi(precip_arr, window=30):
    """
    Compute SPI from a daily precipitation array.
    Uses rolling `window`-day accumulation, then standardises.
    Returns a pd.Series of same length as input.
    """
    r = pd.Series(precip_arr).rolling(window, min_periods=10).sum()
    return (r - r.mean()) / (r.std() + 1e-9)


# ─────────────────────────────────────────────────────────────────
# Axes shade helper
# ─────────────────────────────────────────────────────────────────
def shade(ax, onset, peak, end_d, ms, me, col):
    """
    Add drought-period shading and maize-season shading to ax.
    All date arguments should be pandas Timestamps.
    """
    ax.axvspan(ms,    me,    alpha=0.07, color='#33a02c', zorder=0)
    ax.axvspan(onset, end_d, alpha=0.13, color=col,       zorder=1)
    ax.axvline(onset, color=col, lw=1.3, ls='--', alpha=0.80, zorder=3)
    ax.axvline(peak,  color=col, lw=2.0, ls='-',  alpha=0.95, zorder=4)
    ax.axvline(end_d, color=col, lw=1.3, ls='--', alpha=0.80, zorder=3)
    ax.axhline(0, color='#444', lw=0.8, ls=':')
    ax.spines[['top', 'right']].set_visible(False)


# ─────────────────────────────────────────────────────────────────
# Shared bottom legend
# ─────────────────────────────────────────────────────────────────
def shared_legend(fig, col):
    """Attach a 4-item shared legend at the bottom of fig."""
    leg_h = [
        mpatches.Patch(color=col,      alpha=0.25, label='Drought period'),
        mpatches.Patch(color='#33a02c',alpha=0.20, label='Maize season'),
        plt.Line2D([0],[0], color=col, lw=2.0, ls='-',  label='Drought peak'),
        plt.Line2D([0],[0], color=col, lw=1.3, ls='--', label='Onset / end'),
    ]
    fig.legend(handles=leg_h, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.005), framealpha=0.95, edgecolor='#aaa')


# ─────────────────────────────────────────────────────────────────
# Save figure
# ─────────────────────────────────────────────────────────────────
def save_fig(fig, outdir, fname, dpi=300):
    """Save fig as PNG, PDF and SVG into outdir."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(outdir, f'{fname}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅  {fname}")


# ─────────────────────────────────────────────────────────────────
# LOEO cross-validation
# ─────────────────────────────────────────────────────────────────
def loeo_cv(df, feat_cols, target, years,
            n_estimators=200, random_state=42):
    """
    Leave-One-Event-Out cross-validation using Random Forest.

    Parameters
    ----------
    df          : feature matrix DataFrame with 'year' column
    feat_cols   : list of feature column names
    target      : target column name
    years       : list of event years (ints)

    Returns
    -------
    all_true, all_pred, all_years : np.ndarrays
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    avail = [c for c in feat_cols if c in df.columns]
    df2   = df.copy()
    for c in avail:
        df2[c] = df2[c].fillna(df2[c].median())

    X = df2[avail].values
    y = df2[target].values

    all_true, all_pred, all_yr = [], [], []
    for held in years:
        mask   = (df2['year'] == held).values
        X_tr, y_tr = X[~mask], y[~mask]
        X_te, y_te = X[mask],  y[mask]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        sc  = StandardScaler().fit(X_tr)
        rf  = RandomForestRegressor(n_estimators=n_estimators,
                                    random_state=random_state).fit(
                                    sc.transform(X_tr), y_tr)
        yp  = rf.predict(sc.transform(X_te))
        all_true.extend(y_te)
        all_pred.extend(yp)
        all_yr.extend([held] * len(y_te))

    return np.array(all_true), np.array(all_pred), np.array(all_yr)
