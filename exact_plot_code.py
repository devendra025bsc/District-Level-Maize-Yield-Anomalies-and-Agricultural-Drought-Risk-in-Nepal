"""
exact_plot_code.py
==================
EXACT code used to generate all clean figures (no titles, no fig numbers).
This is the verbatim script that produced the plots you see.

SETUP
-----
pip install matplotlib numpy pandas geopandas scikit-learn scipy h5py netCDF4

Set DATA_ROOT below to your folder containing all uploaded data files.

RUN
---
python exact_plot_code.py
python exact_plot_code.py --figs 1 2 6    # specific figures only
"""

import os, sys, glob, argparse, warnings
import numpy as np, pandas as pd
import netCDF4 as nc, h5py
from datetime import datetime
import warnings; warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
#  ▶▶  SET THIS TO YOUR DATA FOLDER  ◀◀
# ════════════════════════════════════════════════════════════════
DATA_ROOT = r"C:\Users\YourName\Downloads\nepal_drought_data"
# Linux/Mac: DATA_ROOT = "/home/yourname/data/nepal_drought_data"
# ════════════════════════════════════════════════════════════════

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import interp1d

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,
                     'axes.linewidth':0.8,'xtick.labelsize':8.5,'ytick.labelsize':8.5})

# ── Constants ─────────────────────────────────────────────────────────────────
DISTRICTS = ['Jhapa','Ilam','Bhojpur','Morang','Dhankuta','Sunsari']
YEARS     = [2015,2016,2018,2022,2024]
COLORS    = ['#2166ac','#d6604d','#1a9641','#7b2d8b','#e6821e']
EVENT_META = {
    2015:{'color':'#2166ac','onset':'2015-06-15','peak':'2015-08-01','end':'2015-09-30','ms':'2015-04-15','me':'2015-10-31'},
    2016:{'color':'#d6604d','onset':'2016-06-01','peak':'2016-07-15','end':'2016-09-30','ms':'2016-04-15','me':'2016-09-30'},
    2018:{'color':'#1a9641','onset':'2018-06-01','peak':'2018-07-20','end':'2018-09-30','ms':'2018-04-15','me':'2018-09-30'},
    2022:{'color':'#7b2d8b','onset':'2022-07-01','peak':'2022-08-01','end':'2022-09-30','ms':'2022-06-01','me':'2022-09-30'},
    2024:{'color':'#e6821e','onset':'2024-06-01','peak':'2024-07-15','end':'2024-09-30','ms':'2024-02-01','me':'2024-09-30'},
}
MAIZE={'Jhapa':{'area':12400,'prod':34720},'Ilam':{'area':15200,'prod':38000},
       'Bhojpur':{'area':16100,'prod':35420},'Morang':{'area':11800,'prod':33040},
       'Dhankuta':{'area':18300,'prod':47580},'Sunsari':{'area':9500,'prod':26600}}
YIELD_MOLD={
    'Jhapa':    {2015:2.52,2016:2.78,2018:2.95,2022:2.41,2024:3.10},
    'Ilam':     {2015:2.21,2016:2.48,2018:2.63,2022:2.15,2024:2.72},
    'Bhojpur':  {2015:1.98,2016:2.20,2018:2.35,2022:1.89,2024:2.44},
    'Morang':   {2015:2.45,2016:2.71,2018:2.88,2022:2.33,2024:3.02},
    'Dhankuta': {2015:2.31,2016:2.55,2018:2.69,2022:2.20,2024:2.80},
    'Sunsari':  {2015:2.48,2016:2.74,2018:2.90,2022:2.38,2024:3.05}}
GS0,GS1 = 91,273
LON_MIN,LON_MAX,LAT_MIN,LAT_MAX = 86.0,88.2,26.3,28.0
PEAK_DOY   = 200
LEAD_WEEKS = [1,2,3,4,6,8]
FEAT_COLS  = ['precip_gs','temp_gs','evap_gs','spi_min','spi_mean',
              'sm_surface','sm_rootzone','ndvi_gs','evi_gs']
TARGET = 'yield_anom_pct'

# ── Utility functions ─────────────────────────────────────────────────────────
def roll(arr,w): return pd.Series(arr).rolling(w,center=True,min_periods=3).mean()

def spi_calc(arr,w=30):
    r=pd.Series(arr).rolling(w,min_periods=10).sum()
    return (r-r.mean())/(r.std()+1e-9)

def shade(ax,onset,peak,end_d,ms,me,col):
    ax.axvspan(ms,me,      alpha=0.07,color='#33a02c',zorder=0)
    ax.axvspan(onset,end_d,alpha=0.13,color=col,      zorder=1)
    ax.axvline(onset,color=col,lw=1.3,ls='--',alpha=0.80,zorder=3)
    ax.axvline(peak, color=col,lw=2.0,ls='-', alpha=0.95,zorder=4)
    ax.axvline(end_d,color=col,lw=1.3,ls='--',alpha=0.80,zorder=3)
    ax.axhline(0,color='#444',lw=0.8,ls=':')
    ax.spines[['top','right']].set_visible(False)

def shared_leg(fig,col):
    h=[mpatches.Patch(color=col,      alpha=0.25,label='Drought period'),
       mpatches.Patch(color='#33a02c',alpha=0.20,label='Maize season'),
       plt.Line2D([0],[0],color=col,lw=2.0,ls='-', label='Drought peak'),
       plt.Line2D([0],[0],color=col,lw=1.3,ls='--',label='Onset / end')]
    fig.legend(handles=h,loc='lower center',ncol=4,fontsize=9,
               bbox_to_anchor=(0.5,0.005),framealpha=0.95,edgecolor='#aaa')

def save(fig,subdir,fname):
    d=os.path.join(OUT,subdir); os.makedirs(d,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(os.path.join(d,f'{fname}.{ext}'),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f"  ✅  {fname}")

def lbl(ax,letter):
    ax.text(0.01,0.97,f'({letter})',transform=ax.transAxes,fontsize=11,fontweight='bold',va='top')

# ── Global data containers ────────────────────────────────────────────────────
_era5=_chirps=_smap=_s2=_feat=None

# ── Data loading ──────────────────────────────────────────────────────────────
def load_all():
    global _era5,_chirps,_smap,_s2,_feat
    print("Loading data …")

    # ERA5 — build from monthly .nc files
    rows=[]
    for fpath in sorted(glob.glob(os.path.join(DATA_ROOT,'era5_land_*.nc'))):
        parts=os.path.basename(fpath).replace('.nc','').split('_')
        district=parts[2]; year=int(parts[3])
        ds=nc.Dataset(fpath)
        times=ds.variables['valid_time'][:]
        t2m=ds.variables['t2m'][:].astype(float)
        tp =ds.variables['tp'][:].astype(float)
        e  =ds.variables['e'][:].astype(float)
        ds.close()
        dates_all=[datetime.utcfromtimestamp(int(s)) for s in times]
        for udate in sorted(set(d.date() for d in dates_all)):
            idx=[i for i,d in enumerate(dates_all) if d.date()==udate]
            rows.append({'district':district,'year':year,'date':str(udate),
                         'temp_2m_C':round(np.nanmean(t2m[idx].mean(axis=(1,2)))-273.15,3),
                         'precip_mm':round(np.nansum(tp[idx].mean(axis=(1,2)))*1000,4),
                         'evap_mm':  round(abs(np.nansum(e[idx].mean(axis=(1,2))))*1000,4)})
    _era5=pd.DataFrame(rows); _era5['date']=pd.to_datetime(_era5['date'])
    print(f"  ERA5   : {len(_era5)} rows")

    # CHIRPS — merge all district CSVs
    dfs=[pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA_ROOT,'chirps_*.csv')))]
    _chirps=pd.concat(dfs,ignore_index=True); _chirps['date']=pd.to_datetime(_chirps['date'])
    print(f"  CHIRPS : {len(_chirps)} rows")

    # SMAP district daily CSV
    _smap=pd.read_csv(os.path.join(DATA_ROOT,'smap_district_daily.csv'),parse_dates=['date'])
    print(f"  SMAP   : {len(_smap)} rows")

    # Sentinel-2 timeseries CSV
    _s2=pd.read_csv(os.path.join(DATA_ROOT,'sentinel2_timeseries_district.csv'),parse_dates=['date'])
    print(f"  S2     : {len(_s2)} rows")

    # Feature matrix
    _feat=_build_features()
    print(f"  Features: {_feat.shape}")

def _build_features():
    rows=[]
    for yr in YEARS:
        for d in DISTRICTS:
            sub_e =_era5[(_era5['year']==yr)&(_era5['district']==d)].copy()
            sub_e['doy']=sub_e['date'].dt.dayofyear
            gs_e  =sub_e[(sub_e['doy']>=GS0)&(sub_e['doy']<=GS1)]
            sub_sm=_smap[(_smap['event_year']==yr)&(_smap['district']==d)].copy()
            sub_sm['doy']=sub_sm['date'].dt.dayofyear
            gs_sm =sub_sm[(sub_sm['doy']>=GS0)&(sub_sm['doy']<=GS1)]
            sub_ch=_chirps[(_chirps['event_year']==yr)&(_chirps['district']==d)].copy()
            spi_a =spi_calc(sub_ch['precip_mm'].values) if len(sub_ch)>10 else pd.Series([np.nan])
            sub_s2=_s2[(_s2['event_year']==yr)&(_s2['district']==d)].copy()
            sub_s2['doy']=sub_s2['date'].dt.dayofyear
            gs_s2 =sub_s2[(sub_s2['doy']>=GS0)&(sub_s2['doy']<=GS1)]
            rows.append({'year':yr,'district':d,
                'precip_gs': gs_e['precip_mm'].mean()  if len(gs_e)>0  else np.nan,
                'temp_gs':   gs_e['temp_2m_C'].mean()  if len(gs_e)>0  else np.nan,
                'evap_gs':   gs_e['evap_mm'].mean()    if len(gs_e)>0  else np.nan,
                'sm_surface':gs_sm['surface_soil_moisture'].mean()  if len(gs_sm)>0 else np.nan,
                'sm_rootzone':gs_sm['rootzone_soil_moisture'].mean() if len(gs_sm)>0 else np.nan,
                'spi_min':   float(np.nanmin(spi_a))   if len(spi_a)>0 else np.nan,
                'spi_mean':  float(np.nanmean(spi_a))  if len(spi_a)>0 else np.nan,
                'ndvi_gs':   gs_s2['NDVI'].mean() if len(gs_s2)>0 else np.nan,
                'evi_gs':    gs_s2['EVI'].mean()  if len(gs_s2)>0 else np.nan,
                'yield_t_ha':YIELD_MOLD[d][yr]})
    df=pd.DataFrame(rows)
    for d in DISTRICTS:
        m=df.loc[df['district']==d,'yield_t_ha'].mean()
        df.loc[df['district']==d,'yield_anom_pct']=(df.loc[df['district']==d,'yield_t_ha']-m)/m*100
    return df

def E(yr):  return _era5[_era5['year']==yr].groupby('date')[['temp_2m_C','precip_mm','evap_mm']].mean()
def SM(yr): return _smap[_smap['event_year']==yr].groupby('date')[['surface_soil_moisture','rootzone_soil_moisture']].mean()
def CH(yr): return _chirps[_chirps['event_year']==yr].groupby('date')[['precip_mm']].mean()

# ══════════════════════════════════════════════════════════════════════════════
# FIG01 — Study Area
# ══════════════════════════════════════════════════════════════════════════════
def fig01():
    print("\n── FIG01 ──")
    import geopandas as gpd
    gdf_all  =gpd.read_file(os.path.join(DATA_ROOT,'gadm41_NPL_3.json'))
    gdf_study=gpd.read_file(os.path.join(DATA_ROOT,'study_districts.geojson'))
    gdf_study['NAME_3']=gdf_study['NAME_3'].str.strip()
    gdf_all['NAME_3']  =gdf_all['NAME_3'].str.strip()
    gdf_s=gdf_study.merge(pd.DataFrame([{'NAME_3':k,'area':v['area'],'prod':v['prod']}
                                         for k,v in MAIZE.items()]),on='NAME_3',how='left')
    b=gdf_s.total_bounds; pad=0.20
    colors6=['#2166ac','#74add1','#fee090','#f46d43','#d73027','#a50026']

    fig,axes=plt.subplots(2,2,figsize=(14,10))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.32,wspace=0.28,left=0.06,right=0.97,top=0.97,bottom=0.06)

    ax=axes[0,0]
    gdf_all.plot(ax=ax,color='#f0ede8',edgecolor='#999',linewidth=0.35)
    gdf_s.plot(ax=ax,color='#c0392b',edgecolor='#7b1f1f',linewidth=1.0)
    ax.set_xlim(79.9,88.5); ax.set_ylim(26.0,30.6)
    ax.set_xlabel('Longitude (°E)',fontsize=9); ax.set_ylabel('Latitude (°N)',fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(handles=[mpatches.Patch(color='#c0392b',label='Study districts (n=6)'),
                       mpatches.Patch(color='#f0ede8',edgecolor='#999',label='Other districts')],
              fontsize=8,loc='lower right',framealpha=0.9)
    ax.annotate('N',xy=(80.3,30.2),fontsize=11,ha='center',fontweight='bold')
    ax.annotate('',xy=(80.3,30.45),xytext=(80.3,30.2),arrowprops=dict(arrowstyle='->',color='k',lw=1.5))
    ax.plot([85.5,86.6],[26.35,26.35],'k-',lw=2.5); ax.text(86.05,26.18,'~122 km',ha='center',fontsize=7.5)
    lbl(ax,'a')

    ax=axes[0,1]
    for i,(_,row) in enumerate(gdf_s.iterrows()):
        gpd.GeoDataFrame([row],geometry='geometry',crs=gdf_s.crs).plot(ax=ax,color=colors6[i%6],edgecolor='#1a1a1a',linewidth=1.0)
    for _,row in gdf_s.iterrows():
        cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
        ax.text(cx,cy,row['NAME_3'],ha='center',va='center',fontsize=8.5,fontweight='bold',
                bbox=dict(facecolor='white',alpha=0.72,edgecolor='none',pad=1.5))
    ax.set_xlim(b[0]-pad,b[2]+pad); ax.set_ylim(b[1]-pad,b[3]+pad)
    ax.set_xlabel('Longitude (°E)',fontsize=9); ax.set_ylabel('Latitude (°N)',fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
    xl2,yl2=ax.get_xlim(),ax.get_ylim()
    ax.annotate('N',xy=(xl2[0]+0.14,yl2[1]-0.18),fontsize=10,ha='center',fontweight='bold')
    ax.annotate('',xy=(xl2[0]+0.14,yl2[1]-0.06),xytext=(xl2[0]+0.14,yl2[1]-0.18),
                arrowprops=dict(arrowstyle='->',color='k',lw=1.5))
    ax.plot([xl2[0]+0.1,xl2[0]+0.65],[yl2[0]+0.08,yl2[0]+0.08],'k-',lw=2.5)
    ax.text(xl2[0]+0.375,yl2[0]+0.02,'~60 km',ha='center',fontsize=7.5)
    lbl(ax,'b')

    ax=axes[1,0]
    gdf_s.plot(ax=ax,column='area',cmap='YlGn',vmin=8000,vmax=20000,edgecolor='#1a1a1a',linewidth=1.0)
    for _,row in gdf_s.iterrows():
        if pd.notna(row.get('area')):
            cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
            ax.text(cx,cy,f"{int(row['area']):,}\nha",ha='center',va='center',fontsize=8,fontweight='bold',
                    bbox=dict(facecolor='white',alpha=0.75,edgecolor='none',pad=1.5))
    ax.set_xlim(b[0]-pad,b[2]+pad); ax.set_ylim(b[1]-pad,b[3]+pad)
    ax.set_xlabel('Longitude (°E)',fontsize=9); ax.set_ylabel('Latitude (°N)',fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
    div=make_axes_locatable(ax); cax=div.append_axes('right',size='4%',pad=0.06)
    sm_cb=plt.cm.ScalarMappable(cmap='YlGn',norm=plt.Normalize(8000,20000)); sm_cb.set_array([])
    cb=plt.colorbar(sm_cb,cax=cax); cb.set_label('Area (ha)',fontsize=8); cb.ax.tick_params(labelsize=7)
    lbl(ax,'c')

    ax=axes[1,1]
    x=np.arange(len(DISTRICTS)); w=0.35
    areas=[MAIZE[d]['area'] for d in DISTRICTS]; prods=[MAIZE[d]['prod'] for d in DISTRICTS]
    ax.bar(x-w/2,areas,w,label='Area (ha)',   color='#4575b4',edgecolor='#1a1a1a',lw=0.7)
    ax2r=ax.twinx()
    ax2r.bar(x+w/2,prods,w,label='Production (mt)',color='#d6604d',edgecolor='#1a1a1a',lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([d.capitalize() for d in DISTRICTS],fontsize=9)
    ax.set_ylabel('Cultivated Area (ha)',fontsize=9,color='#4575b4')
    ax2r.set_ylabel('Maize Production (mt)',fontsize=9,color='#d6604d')
    ax.spines[['top']].set_visible(False); ax2r.spines[['top']].set_visible(False)
    ax.legend(loc='upper left',fontsize=8,framealpha=0.9)
    ax2r.legend(loc='upper right',fontsize=8,framealpha=0.9)
    lbl(ax,'d')

    save(fig,'FIG01','FIG01_Study_Area')

# ══════════════════════════════════════════════════════════════════════════════
# FIG02 — Drought Verification (one per event)
# ══════════════════════════════════════════════════════════════════════════════
def fig02():
    print("\n── FIG02 ──")
    for yr in YEARS:
        ev=EVENT_META[yr]; col=ev['color']
        onset=pd.to_datetime(ev['onset']); peak=pd.to_datetime(ev['peak'])
        end_d=pd.to_datetime(ev['end']);   ms=pd.to_datetime(ev['ms']); me=pd.to_datetime(ev['me'])
        era5=E(yr); smap_s=SM(yr); chirps=CH(yr)
        gs_e=(era5.index.dayofyear>=GS0)&(era5.index.dayofyear<=GS1)
        p_ref=era5.loc[gs_e,'precip_mm'].mean() if gs_e.any() else era5['precip_mm'].mean()
        t_ref=era5.loc[gs_e,'temp_2m_C'].mean() if gs_e.any() else era5['temp_2m_C'].mean()
        gs_s=(smap_s.index.dayofyear>=GS0)&(smap_s.index.dayofyear<=GS1)
        sm_ref=smap_s.loc[gs_s,'surface_soil_moisture'].mean() if gs_s.any() else smap_s['surface_soil_moisture'].mean()
        p_anom=era5['precip_mm']-p_ref; t_anom=era5['temp_2m_C']-t_ref
        sm_anom=smap_s['surface_soil_moisture']-sm_ref
        spi_s=pd.Series(spi_calc(chirps['precip_mm'].values).values,index=chirps.index)

        fig,axes=plt.subplots(4,1,figsize=(13,14))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(hspace=0.44,left=0.10,right=0.97,top=0.98,bottom=0.06)

        ax=axes[0]
        ax.bar(era5.index,p_anom.values,color=np.where(p_anom.values>=0,'#4393c3','#d6604d'),
               width=1,alpha=0.55,zorder=2,label='ERA5 daily anomaly')
        ax.plot(era5.index,roll(p_anom.values,7).values,color='#08306b',lw=1.8,zorder=5,label='ERA5 7-day mean')
        ch_anom=chirps['precip_mm']-chirps['precip_mm'].mean()
        ax.plot(chirps.index,roll(ch_anom.values,7).values,color='#f4a582',lw=1.4,ls='--',zorder=4,alpha=0.85,label='CHIRPS 7-day mean')
        shade(ax,onset,peak,end_d,ms,me,col)
        ax.set_ylabel('Precip. anomaly (mm day⁻¹)',fontsize=9.5)
        ax.legend(fontsize=8,ncol=3,loc='upper right',framealpha=0.9)
        yabs=max(abs(float(p_anom.min())),abs(float(p_anom.max())))
        ax.set_ylim(-yabs*1.3,yabs*1.5); lbl(ax,'a')

        ax=axes[1]
        ax.fill_between(era5.index,t_anom.values,0,where=t_anom.values>=0,color='#d6604d',alpha=0.45,label='Warm anomaly')
        ax.fill_between(era5.index,t_anom.values,0,where=t_anom.values<0, color='#4393c3',alpha=0.45,label='Cool anomaly')
        ax.plot(era5.index,roll(t_anom.values,7).values,color='#8b0000',lw=1.8,zorder=5,label='7-day mean')
        shade(ax,onset,peak,end_d,ms,me,col)
        ax.set_ylabel('Temp. anomaly (°C)',fontsize=9.5)
        ax.legend(fontsize=8,ncol=3,loc='upper right',framealpha=0.9); lbl(ax,'b')

        ax=axes[2]
        ax.fill_between(smap_s.index,sm_anom.values,0,where=sm_anom.values>=0,color='#74add1',alpha=0.55,label='Wet anomaly')
        ax.fill_between(smap_s.index,sm_anom.values,0,where=sm_anom.values<0, color='#d73027',alpha=0.50,label='Dry anomaly')
        ax.plot(smap_s.index,roll(sm_anom.values,7).values,color='#023858',lw=1.8,zorder=5,label='7-day mean')
        shade(ax,onset,peak,end_d,ms,me,col)
        ax.set_ylabel('SM anomaly (m³ m⁻³)',fontsize=9.5)
        ax.legend(fontsize=8,ncol=3,loc='upper right',framealpha=0.9); lbl(ax,'c')

        ax=axes[3]
        ax.fill_between(chirps.index,spi_s.values,0,where=spi_s.values>=0,color='#4393c3',alpha=0.50,label='SPI > 0')
        ax.fill_between(chirps.index,spi_s.values,0,where=spi_s.values<0, color='#d6604d',alpha=0.65,label='SPI < 0')
        ax.plot(chirps.index,roll(spi_s.values,14).values,color='#1a1a1a',lw=1.8,zorder=5,label='14-day mean')
        for th,lc,lb2 in [(-1.0,'#f4a582','Moderate (−1.0)'),(-1.5,'#d6604d','Severe (−1.5)'),(-2.0,'#8b0000','Extreme (−2.0)')]:
            ax.axhline(th,color=lc,lw=1.4,ls='--',label=lb2)
        shade(ax,onset,peak,end_d,ms,me,col)
        ax.set_ylabel('SPI (–)',fontsize=9.5)
        ax.legend(fontsize=7.8,ncol=3,loc='upper left',framealpha=0.9)
        spi_min=float(spi_s.min())
        cls='Extreme' if spi_min<=-2 else 'Severe' if spi_min<=-1.5 else 'Moderate' if spi_min<=-1 else 'Near-normal'
        ax.text(0.99,0.04,f'SPI_min = {spi_min:.2f} ({cls})',transform=ax.transAxes,ha='right',
                fontsize=8.5,color='#8b0000',fontweight='bold',
                bbox=dict(facecolor='#fff0f0',edgecolor='#d6604d',pad=3,alpha=0.95))
        ax.set_ylim(min(-0.5,spi_min*1.3),max(0.5,float(spi_s.max())*1.3)); lbl(ax,'d')
        shared_leg(fig,col)
        save(fig,'FIG02',f'FIG02_{yr}_Drought_Verification')

# ══════════════════════════════════════════════════════════════════════════════
# FIG03 — Meteorological Evolution (one per event, 5 panels)
# ══════════════════════════════════════════════════════════════════════════════
def fig03():
    print("\n── FIG03 ──")
    for yr in YEARS:
        ev=EVENT_META[yr]; col=ev['color']
        onset=pd.to_datetime(ev['onset']); peak=pd.to_datetime(ev['peak'])
        end_d=pd.to_datetime(ev['end']);   ms=pd.to_datetime(ev['ms']); me=pd.to_datetime(ev['me'])
        era5=E(yr); smap_s=SM(yr); chirps=CH(yr)
        gs_e=(era5.index.dayofyear>=GS0)&(era5.index.dayofyear<=GS1)
        p_ref=era5.loc[gs_e,'precip_mm'].mean() if gs_e.any() else era5['precip_mm'].mean()
        t_ref=era5.loc[gs_e,'temp_2m_C'].mean() if gs_e.any() else era5['temp_2m_C'].mean()
        e_ref=era5.loc[gs_e,'evap_mm'].mean()   if gs_e.any() else era5['evap_mm'].mean()
        gs_s=(smap_s.index.dayofyear>=GS0)&(smap_s.index.dayofyear<=GS1)
        sm_ref=smap_s.loc[gs_s,'surface_soil_moisture'].mean() if gs_s.any() else smap_s['surface_soil_moisture'].mean()
        spi_s=pd.Series(spi_calc(chirps['precip_mm'].values).values,index=chirps.index)

        fig,axes=plt.subplots(5,1,figsize=(13,18))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(hspace=0.42,left=0.10,right=0.97,top=0.98,bottom=0.04)

        ax=axes[0]; pp=era5['precip_mm']
        ax.bar(era5.index,pp.values,color='#4393c3',width=1,alpha=0.55,zorder=2,label='ERA5 daily')
        ax.plot(era5.index,roll(pp.values,7).values,color='#08306b',lw=1.8,zorder=5,label='ERA5 7-day mean')
        if len(chirps)>2: ax.plot(chirps.index,roll(chirps['precip_mm'].values,7).values,color='#f4a582',lw=1.4,ls='--',alpha=0.85,label='CHIRPS 7-day mean')
        ax.axhline(p_ref,color='#666',lw=0.9,ls=':',label=f'GS mean = {p_ref:.1f} mm d⁻¹')
        shade(ax,onset,peak,end_d,ms,me,col); ax.set_ylabel('Precipitation (mm day⁻¹)',fontsize=9.5)
        ax.legend(fontsize=8,ncol=4,loc='upper right',framealpha=0.9); lbl(ax,'a')

        ax=axes[1]; tt=era5['temp_2m_C']
        ax.plot(era5.index,tt.values,color='#f4a582',lw=0.7,alpha=0.55)
        ax.plot(era5.index,roll(tt.values,7).values,color='#d6604d',lw=1.8,zorder=5,label='7-day mean')
        ax.axhline(t_ref,color='#666',lw=0.9,ls=':',label=f'GS mean = {t_ref:.1f} °C')
        shade(ax,onset,peak,end_d,ms,me,col); ax.set_ylabel('2-m Temperature (°C)',fontsize=9.5)
        ax.legend(fontsize=8,ncol=2,loc='upper right',framealpha=0.9); lbl(ax,'b')

        ax=axes[2]; ee=era5['evap_mm']
        ax.fill_between(era5.index,ee.values,alpha=0.40,color='#74add1')
        ax.plot(era5.index,roll(ee.values,7).values,color='#023858',lw=1.8,zorder=5,label='7-day mean')
        ax.axhline(e_ref,color='#666',lw=0.9,ls=':',label=f'GS mean = {e_ref:.1f} mm d⁻¹')
        shade(ax,onset,peak,end_d,ms,me,col); ax.set_ylabel('Evapotranspiration (mm day⁻¹)',fontsize=9.5)
        ax.legend(fontsize=8,ncol=2,loc='upper right',framealpha=0.9); lbl(ax,'c')

        ax=axes[3]
        sm_surf=smap_s['surface_soil_moisture']; sm_rz=smap_s['rootzone_soil_moisture']
        ax.fill_between(smap_s.index,sm_surf.values,sm_ref,where=sm_surf.values>=sm_ref,color='#74add1',alpha=0.50,label='Surface: above mean')
        ax.fill_between(smap_s.index,sm_surf.values,sm_ref,where=sm_surf.values<sm_ref, color='#d73027',alpha=0.45,label='Surface: below mean')
        ax.plot(smap_s.index,roll(sm_surf.values,7).values,color='#023858',lw=1.8,zorder=5,label='Surface 7-day')
        ax.plot(smap_s.index,roll(sm_rz.values,7).values,  color='#5e3c99',lw=1.4,ls='--',zorder=4,label='Root-zone 7-day')
        ax.axhline(sm_ref,color='#666',lw=0.9,ls=':',label=f'GS mean = {sm_ref:.3f}')
        shade(ax,onset,peak,end_d,ms,me,col); ax.set_ylabel('Soil moisture (m³ m⁻³)',fontsize=9.5)
        ax.legend(fontsize=7.8,ncol=3,loc='upper right',framealpha=0.9); lbl(ax,'d')

        ax=axes[4]
        ax.fill_between(chirps.index,spi_s.values,0,where=spi_s.values>=0,color='#4393c3',alpha=0.50,label='SPI > 0')
        ax.fill_between(chirps.index,spi_s.values,0,where=spi_s.values<0, color='#d6604d',alpha=0.65,label='SPI < 0')
        ax.plot(chirps.index,roll(spi_s.values,14).values,color='#1a1a1a',lw=1.8,zorder=5,label='14-day mean')
        for th,lc,lb2 in [(-1.0,'#f4a582','Moderate'),(-1.5,'#d6604d','Severe'),(-2.0,'#8b0000','Extreme')]:
            ax.axhline(th,color=lc,lw=1.4,ls='--',label=f'{lb2} (SPI={th})')
        shade(ax,onset,peak,end_d,ms,me,col); ax.set_ylabel('SPI (–)',fontsize=9.5)
        ax.legend(fontsize=7.8,ncol=4,loc='upper left',framealpha=0.9)
        ax.set_ylim(min(-0.5,float(spi_s.min())*1.3),max(0.5,float(spi_s.max())*1.3)); lbl(ax,'e')
        shared_leg(fig,col)
        save(fig,'FIG03',f'FIG03_Meteorological_{yr}')

# ══════════════════════════════════════════════════════════════════════════════
# FIG04 — Sentinel-2 Vegetation
# ══════════════════════════════════════════════════════════════════════════════
def fig04():
    print("\n── FIG04 ──")
    for yr in YEARS:
        ev=EVENT_META[yr]; col=ev['color']
        onset=pd.to_datetime(ev['onset']); peak=pd.to_datetime(ev['peak'])
        end_d=pd.to_datetime(ev['end']);   ms=pd.to_datetime(ev['ms']); me=pd.to_datetime(ev['me'])
        sub=_s2[_s2['event_year']==yr]
        if len(sub)==0: continue
        idx_d=sub.groupby('date')[['NDVI','EVI','NDRE','NDWI','SAVI']].mean()

        fig,axes=plt.subplots(5,1,figsize=(13,16))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(hspace=0.42,left=0.10,right=0.97,top=0.98,bottom=0.05)
        cfg=[('NDVI','#1a9641'),('EVI','#2166ac'),('NDRE','#7b2d8b'),('NDWI','#4393c3'),('SAVI','#e6821e')]
        for ai,(idxn,ic) in enumerate(cfg):
            ax=axes[ai]; vals=idx_d[idxn]
            pre=vals.index<onset; base=vals[pre].mean() if pre.any() else vals.mean()
            ax.scatter(vals.index,vals.values,color=ic,s=40,zorder=5,alpha=0.85,label='District mean')
            if len(vals)>=3:
                try:
                    x_n=np.array([(d-vals.index[0]).days for d in vals.index])
                    fi=interp1d(x_n,vals.values,kind='linear')
                    x_d=np.linspace(x_n.min(),x_n.max(),200)
                    dd=[vals.index[0]+pd.Timedelta(days=int(x)) for x in x_d]
                    ax.plot(dd,fi(x_d),color=ic,lw=1.4,alpha=0.55,ls='--')
                except: pass
            ax.axhline(base,color='#555',lw=1.0,ls=':',label=f'Pre-drought mean = {base:.3f}')
            ax.axvspan(ms,me,      alpha=0.07,color='#33a02c',zorder=0)
            ax.axvspan(onset,end_d,alpha=0.13,color=col,      zorder=1)
            ax.axvline(onset,color=col,lw=1.3,ls='--',alpha=0.80,zorder=3)
            ax.axvline(peak, color=col,lw=2.0,ls='-', alpha=0.95,zorder=4)
            ax.axvline(end_d,color=col,lw=1.3,ls='--',alpha=0.80,zorder=3)
            ax.set_ylabel(f'{idxn} (–)',fontsize=9.5)
            ax.legend(fontsize=8,ncol=3,loc='lower right',framealpha=0.9)
            ax.spines[['top','right']].set_visible(False); lbl(ax,chr(97+ai))
        shared_leg(fig,col)
        save(fig,'FIG04',f'FIG04A_S2_Vegetation_{yr}')

    fig,axes=plt.subplots(2,3,figsize=(15,10))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.38,wspace=0.30,left=0.07,right=0.97,top=0.97,bottom=0.10)
    for pi,idx_name in enumerate(['NDVI','EVI','NDRE','NDWI','SAVI']):
        ax=axes[pi//3,pi%3]; data_box=[]; means=[]
        for yr in YEARS:
            sub=_s2[_s2['event_year']==yr][idx_name].dropna()
            data_box.append(sub.values); means.append(sub.mean())
        bp=ax.boxplot(data_box,positions=range(5),widths=0.5,patch_artist=True,
                      medianprops={'color':'#1a1a1a','lw':2},whiskerprops={'lw':1.2},
                      capprops={'lw':1.2},flierprops={'marker':'.','ms':3,'alpha':0.4})
        for patch,c in zip(bp['boxes'],COLORS): patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.scatter(range(5),means,color='#1a1a1a',s=60,zorder=5,marker='D',label='Event mean')
        ax.plot(range(5),means,color='#1a1a1a',lw=1.5,ls='--',alpha=0.7)
        ax.set_xticks(range(5)); ax.set_xticklabels([str(y) for y in YEARS],fontsize=9)
        ax.set_xlabel('Year',fontsize=9); ax.set_ylabel(f'{idx_name} (–)',fontsize=9)
        ax.legend(fontsize=8,framealpha=0.9); ax.spines[['top','right']].set_visible(False)
        lbl(ax,chr(97+pi))
    axes[1,2].set_visible(False)
    save(fig,'FIG04','FIG04B_S2_Index_Comparison')

# ══════════════════════════════════════════════════════════════════════════════
# FIG05 — Spatial Drought Maps (one per event)
# ══════════════════════════════════════════════════════════════════════════════
def fig05():
    print("\n── FIG05 ──")
    import geopandas as gpd
    gdf_study=gpd.read_file(os.path.join(DATA_ROOT,'study_districts.geojson'))
    gdf_study['NAME_3']=gdf_study['NAME_3'].str.strip()
    ref_sm={d:_smap[_smap['district']==d]['surface_soil_moisture'].mean() for d in DISTRICTS}
    for yr in YEARS:
        rows_m=[]
        for d in DISTRICTS:
            sub_sm=_smap[(_smap['event_year']==yr)&(_smap['district']==d)].copy()
            sub_sm['doy']=sub_sm['date'].dt.dayofyear
            gs=sub_sm[(sub_sm['doy']>=GS0)&(sub_sm['doy']<=GS1)]
            sm_v=gs['surface_soil_moisture'].mean() if len(gs)>0 else sub_sm['surface_soil_moisture'].mean()
            sub_c=_chirps[(_chirps['event_year']==yr)&(_chirps['district']==d)]
            spi_a=spi_calc(sub_c['precip_mm'].values) if len(sub_c)>10 else pd.Series([np.nan])
            fv=_feat[(_feat['year']==yr)&(_feat['district']==d)]
            rows_m.append({'NAME_3':d,'sm_anom':sm_v-ref_sm[d],
                           'spi_min':float(np.nanmin(spi_a)),
                           'ndvi_gs':fv['ndvi_gs'].values[0]     if len(fv)>0 else np.nan,
                           'yield_anom':fv['yield_anom_pct'].values[0] if len(fv)>0 else np.nan})
        gdf_m=gdf_study.merge(pd.DataFrame(rows_m),on='NAME_3')
        fig,axes=plt.subplots(1,4,figsize=(20,5.5))
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(wspace=0.26,left=0.03,right=0.99,top=0.97,bottom=0.09)
        for ci,(var,ylabel,cmap,vmin,vmax) in enumerate([
            ('sm_anom',   'SM anomaly (m³ m⁻³)',   'RdBu',   -0.10,0.10),
            ('spi_min',   'SPI minimum (–)',         'RdBu',   -3,   1   ),
            ('ndvi_gs',   'NDVI growing season (–)', 'RdYlGn', 0.1,  0.7 ),
            ('yield_anom','Yield anomaly (%)',        'RdYlGn',-15,  15  )]):
            ax=axes[ci]
            gdf_m.plot(ax=ax,column=var,cmap=cmap,vmin=vmin,vmax=vmax,
                       edgecolor='#1a1a1a',linewidth=1.0,missing_kwds={'color':'lightgray'})
            for _,row in gdf_m.iterrows():
                if pd.notna(row[var]):
                    cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
                    ax.text(cx,cy,f'{row[var]:.2f}',ha='center',va='center',fontsize=8.5,fontweight='bold',
                            bbox=dict(facecolor='white',alpha=0.75,edgecolor='none',pad=1.5))
            ax.set_xlabel('Longitude (°E)',fontsize=8)
            ax.set_ylabel('Latitude (°N)' if ci==0 else '',fontsize=8)
            ax.tick_params(labelsize=7.5)
            div=make_axes_locatable(ax); cax=div.append_axes('right',size='6%',pad=0.05)
            sm_cb=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin,vmax)); sm_cb.set_array([])
            cb=plt.colorbar(sm_cb,cax=cax); cb.set_label(ylabel,fontsize=7.5); cb.ax.tick_params(labelsize=7)
            lbl(ax,chr(97+ci))
        save(fig,'FIG05',f'FIG05_Spatial_Drought_{yr}')

# ══════════════════════════════════════════════════════════════════════════════
# FIG06 — Yield Analysis
# ══════════════════════════════════════════════════════════════════════════════
def fig06():
    print("\n── FIG06 ──")
    fig,axes=plt.subplots(2,2,figsize=(13,11))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.38,wspace=0.33,left=0.09,right=0.97,top=0.97,bottom=0.08)

    ax=axes[0,0]
    for i,d in enumerate(DISTRICTS):
        sub=_feat[_feat['district']==d].sort_values('year')
        ax.plot(sub['year'].astype(str),sub['yield_t_ha'],'o-',color=COLORS[i%5],lw=1.8,ms=8,
                label=d.capitalize(),markeredgecolor='#1a1a1a',markeredgewidth=0.6)
    ax.set_xlabel('Year',fontsize=10); ax.set_ylabel('Maize yield (t ha⁻¹)',fontsize=10)
    ax.legend(fontsize=8.5,ncol=2,framealpha=0.9); ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(1.5,3.5); lbl(ax,'a')

    ax=axes[0,1]
    valid=_feat.dropna(subset=['spi_min','yield_t_ha'])
    for i,yr in enumerate(YEARS):
        sub=valid[valid['year']==yr]
        ax.scatter(sub['spi_min'],sub['yield_t_ha'],color=COLORS[i],s=90,label=str(yr),edgecolor='#1a1a1a',linewidth=0.7,zorder=5)
    r,p2=pearsonr(valid['spi_min'],valid['yield_t_ha'])
    x_l=np.linspace(valid['spi_min'].min(),valid['spi_min'].max(),50)
    ax.plot(x_l,LinearRegression().fit(valid[['spi_min']],valid['yield_t_ha']).predict(x_l.reshape(-1,1)),'k--',lw=1.5,alpha=0.7)
    ax.set_xlabel('SPI minimum',fontsize=10); ax.set_ylabel('Maize yield (t ha⁻¹)',fontsize=10)
    ax.text(0.05,0.92,f'r = {r:.3f}  p = {p2:.3f}',transform=ax.transAxes,fontsize=9,
            bbox=dict(facecolor='#f0f4fa',edgecolor='#aaaacc',pad=3,alpha=0.9))
    ax.legend(fontsize=8.5,framealpha=0.9); ax.spines[['top','right']].set_visible(False); lbl(ax,'b')

    ax=axes[1,0]
    valid2=_feat.dropna(subset=['ndvi_gs','yield_t_ha'])
    for i,yr in enumerate(YEARS):
        sub=valid2[valid2['year']==yr]
        ax.scatter(sub['ndvi_gs'],sub['yield_t_ha'],color=COLORS[i],s=90,label=str(yr),edgecolor='#1a1a1a',linewidth=0.7,zorder=5)
    if len(valid2)>3:
        r2b,p2b=pearsonr(valid2['ndvi_gs'],valid2['yield_t_ha'])
        ax.text(0.05,0.92,f'r = {r2b:.3f}  p = {p2b:.3f}',transform=ax.transAxes,fontsize=9,
                bbox=dict(facecolor='#f0f4fa',edgecolor='#aaaacc',pad=3,alpha=0.9))
    ax.set_xlabel('NDVI growing-season mean',fontsize=10); ax.set_ylabel('Maize yield (t ha⁻¹)',fontsize=10)
    ax.legend(fontsize=8.5,framealpha=0.9); ax.spines[['top','right']].set_visible(False); lbl(ax,'c')

    ax=axes[1,1]
    hm=np.zeros((len(DISTRICTS),len(YEARS)))
    for i,d in enumerate(DISTRICTS):
        for j,yr in enumerate(YEARS):
            v=_feat.loc[(_feat['district']==d)&(_feat['year']==yr),'yield_anom_pct']
            hm[i,j]=v.iloc[0] if len(v)>0 else 0.0
    im=ax.imshow(hm,cmap='RdYlGn',vmin=-15,vmax=15,aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels([str(y) for y in YEARS],fontsize=9)
    ax.set_yticks(range(6)); ax.set_yticklabels([d.capitalize() for d in DISTRICTS],fontsize=9)
    for i in range(6):
        for j in range(5):
            ax.text(j,i,f'{hm[i,j]:.1f}%',ha='center',va='center',fontsize=8.5,fontweight='bold',
                    color='white' if abs(hm[i,j])>9 else '#1a1a1a')
    ax.set_xlabel('Year',fontsize=10); ax.set_ylabel('District',fontsize=10)
    div=make_axes_locatable(ax); cax=div.append_axes('right',size='4%',pad=0.06)
    cb=plt.colorbar(im,cax=cax); cb.set_label('Yield anomaly (%)',fontsize=8.5); cb.ax.tick_params(labelsize=8)
    lbl(ax,'d')
    save(fig,'FIG06','FIG06_Yield_Analysis')

# ══════════════════════════════════════════════════════════════════════════════
# FIG07 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
def fig07():
    print("\n── FIG07 ──")
    avail=[c for c in FEAT_COLS if c in _feat.columns]
    GROUPS={'Satellite\nonly':[c for c in ['ndvi_gs','evi_gs'] if c in avail],
            'Climate\nonly': [c for c in ['precip_gs','temp_gs','evap_gs','spi_min'] if c in avail],
            'SM\nonly':      [c for c in ['sm_surface','sm_rootzone'] if c in avail],
            'Sat +\nClimate':[c for c in ['ndvi_gs','evi_gs','precip_gs','temp_gs','spi_min'] if c in avail],
            'Sat + SM':      [c for c in ['ndvi_gs','evi_gs','sm_surface','sm_rootzone'] if c in avail],
            'Climate + SM':  [c for c in ['precip_gs','temp_gs','spi_min','sm_surface','sm_rootzone'] if c in avail],
            'Full\nmultimodal':avail}
    def run(feats):
        df2=_feat.copy()
        for c in feats: df2[c]=df2[c].fillna(df2[c].median())
        X=df2[feats].values; y=df2[TARGET].values
        at,ap=[],[]
        for held in YEARS:
            mask=(df2['year']==held).values
            X_tr,y_tr=X[~mask],y[~mask]; X_te,y_te=X[mask],y[mask]
            sc=StandardScaler().fit(X_tr)
            rf=RandomForestRegressor(n_estimators=150,random_state=42).fit(sc.transform(X_tr),y_tr)
            ap.extend(rf.predict(sc.transform(X_te))); at.extend(y_te)
        at,ap=np.array(at),np.array(ap)
        return r2_score(at,ap),np.sqrt(mean_squared_error(at,ap)),mean_absolute_error(at,ap)
    names,r2s,rmses,maes=[],[],[],[]
    for gname,feats in GROUPS.items():
        if not feats: continue
        r2,rmse,mae=run(feats); names.append(gname); r2s.append(r2); rmses.append(rmse); maes.append(mae)
    bar_col=(['#d9d9d9']*3+['#74add1']*3+['#2166ac'])[:len(names)]
    fig,axes=plt.subplots(1,3,figsize=(15,6.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.33,left=0.07,right=0.97,top=0.97,bottom=0.22)
    for ci,(metric,ylabel,vals) in enumerate([('R²','R² (LOEO-CV)',r2s),('RMSE','RMSE (%)',rmses),('MAE','MAE (%)',maes)]):
        ax=axes[ci]; x=np.arange(len(names))
        bars=ax.bar(x,vals,color=bar_col,edgecolor='#333',lw=0.7,alpha=0.88)
        best=int(np.argmax(vals)) if metric=='R²' else int(np.argmin(vals))
        bars[best].set_edgecolor('#e6821e'); bars[best].set_linewidth(2.8)
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,v+0.01,f'{v:.2f}',ha='center',va='bottom',fontsize=8,rotation=40)
        ax.set_xticks(x); ax.set_xticklabels(names,rotation=45,ha='right',fontsize=8.5)
        ax.set_ylabel(ylabel,fontsize=10); ax.spines[['top','right']].set_visible(False)
        if metric=='R²': ax.axhline(0,color='gray',lw=0.8,ls=':')
        lbl(ax,chr(97+ci))
    fig.legend(handles=[mpatches.Patch(color='#d9d9d9',label='Single modality'),
                        mpatches.Patch(color='#74add1',label='Two modalities'),
                        mpatches.Patch(color='#2166ac',label='Full multimodal')],
               loc='lower center',ncol=3,fontsize=9.5,bbox_to_anchor=(0.5,0.005),framealpha=0.9)
    save(fig,'FIG07_Ablation','FIG07_Ablation_Study')

# ══════════════════════════════════════════════════════════════════════════════
# FIG08 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
def fig08():
    print("\n── FIG08 ──")
    avail=[c for c in FEAT_COLS if c in _feat.columns]
    df2=_feat.copy()
    for c in avail: df2[c]=df2[c].fillna(df2[c].median())
    X=df2[avail].values; y=df2[TARGET].values
    sc=StandardScaler().fit(X)
    rf=RandomForestRegressor(n_estimators=300,random_state=42).fit(sc.transform(X),y)
    imp=rf.feature_importances_
    corrs=[pearsonr(df2[c].values,y)[0] for c in avail]
    spears=[spearmanr(df2[c].values,y)[0] for c in avail]
    MODCOL={'precip_gs':'#2166ac','temp_gs':'#2166ac','evap_gs':'#2166ac','spi_min':'#2166ac','spi_mean':'#2166ac',
            'sm_surface':'#74add1','sm_rootzone':'#74add1','ndvi_gs':'#1a9641','evi_gs':'#1a9641'}
    sorted_idx=np.argsort(imp)
    labels_s=[avail[i] for i in sorted_idx]; colors_s=[MODCOL.get(avail[i],'#aaa') for i in sorted_idx]
    imp_s=imp[sorted_idx]; corrs_s=[corrs[i] for i in sorted_idx]; spears_s=[spears[i] for i in sorted_idx]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.42,left=0.14,right=0.97,top=0.97,bottom=0.08)
    ax=axes[0]
    ax.barh(labels_s,imp_s,color=colors_s,edgecolor='#333',lw=0.6,height=0.72)
    for i,v in enumerate(imp_s): ax.text(v+0.003,i,f'{v:.3f}',va='center',fontsize=8)
    ax.set_xlabel('RF importance (mean decrease impurity)',fontsize=9.5)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color='#1a9641',label='Optical (S2)'),
                       mpatches.Patch(color='#2166ac',label='Climate (ERA5/CHIRPS)'),
                       mpatches.Patch(color='#74add1',label='Soil Moisture (SMAP)')],
              fontsize=8,loc='lower right',framealpha=0.9); lbl(ax,'a')
    ax=axes[1]
    c_col=['#d6604d' if r<0 else '#2166ac' for r in corrs_s]
    ax.barh(labels_s,corrs_s,color=c_col,edgecolor='#333',lw=0.6,height=0.72,label='Pearson r',alpha=0.80)
    ax.scatter(spears_s,range(len(avail)),color='#e6821e',s=40,zorder=5,marker='D',label='Spearman ρ')
    ax.axvline(0,color='#333',lw=0.9)
    ax.axvline(0.3,color='#aaa',lw=0.8,ls='--',alpha=0.7); ax.axvline(-0.3,color='#aaa',lw=0.8,ls='--',alpha=0.7)
    ax.set_xlabel('Correlation with yield anomaly (%)',fontsize=9.5)
    ax.legend(fontsize=8.5,framealpha=0.9,loc='lower right')
    ax.spines[['top','right']].set_visible(False); lbl(ax,'b')
    save(fig,'FIG08_Feature_Importance','FIG08_Feature_Importance')

# ══════════════════════════════════════════════════════════════════════════════
# FIG09 — Cross-Event LOEO
# ══════════════════════════════════════════════════════════════════════════════
def fig09():
    print("\n── FIG09 ──")
    avail=[c for c in FEAT_COLS if c in _feat.columns]
    df2=_feat.copy()
    for c in avail: df2[c]=df2[c].fillna(df2[c].median())
    X=df2[avail].values; y=df2[TARGET].values
    all_t,all_p,all_y=[],[],[]
    ev_r2,ev_rmse,ev_mae=[],[],[]
    for held in YEARS:
        mask=(df2['year']==held).values
        X_tr,y_tr=X[~mask],y[~mask]; X_te,y_te=X[mask],y[mask]
        sc=StandardScaler().fit(X_tr)
        rf=RandomForestRegressor(n_estimators=200,random_state=42).fit(sc.transform(X_tr),y_tr)
        yp=rf.predict(sc.transform(X_te))
        all_t.extend(y_te); all_p.extend(yp); all_y.extend([held]*len(y_te))
        ev_r2.append(r2_score(y_te,yp)); ev_rmse.append(np.sqrt(mean_squared_error(y_te,yp))); ev_mae.append(mean_absolute_error(y_te,yp))
    at,ap,ay=np.array(all_t),np.array(all_p),np.array(all_y)
    fig,axes=plt.subplots(1,2,figsize=(13,6.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.34,left=0.09,right=0.97,top=0.97,bottom=0.13)
    ax=axes[0]; x=np.arange(5); w=0.25
    ax.bar(x-w,ev_r2,  w,label='R²',  color='#2166ac',edgecolor='#1a1a1a',lw=0.7,alpha=0.88)
    ax.bar(x,  ev_rmse,w,label='RMSE',color='#d6604d',edgecolor='#1a1a1a',lw=0.7,alpha=0.88)
    ax.bar(x+w,ev_mae, w,label='MAE', color='#1a9641',edgecolor='#1a1a1a',lw=0.7,alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in YEARS],fontsize=10)
    ax.set_xlabel('Held-out event year',fontsize=10); ax.set_ylabel('Metric value',fontsize=10)
    ax.legend(fontsize=9,framealpha=0.9); ax.axhline(0,color='gray',lw=0.8,ls=':'); ax.spines[['top','right']].set_visible(False)
    for i,v in enumerate(ev_r2):
        ax.text(i-w,v+(0.02 if v>=0 else -0.08),f'{v:.2f}',ha='center',fontsize=7.5,color='#2166ac',fontweight='bold')
    lbl(ax,'a')
    ax=axes[1]
    lo=min(at.min(),ap.min())-1.5; hi=max(at.max(),ap.max())+1.5
    for i,yr in enumerate(YEARS):
        m=ay==yr; ax.scatter(at[m],ap[m],color=COLORS[i],s=85,label=str(yr),edgecolor='#1a1a1a',linewidth=0.7,alpha=0.88,zorder=5)
    ax.plot([lo,hi],[lo,hi],'k--',lw=1.5,alpha=0.7,label='1:1',zorder=1)
    x_reg=np.linspace(lo,hi,100)
    ax.plot(x_reg,LinearRegression().fit(at.reshape(-1,1),ap).predict(x_reg.reshape(-1,1)),'k-',lw=1.3,alpha=0.45,label='Regression')
    r2p=r2_score(at,ap); rmsep=np.sqrt(mean_squared_error(at,ap)); maep=mean_absolute_error(at,ap)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
    ax.set_xlabel('Observed yield anomaly (%)',fontsize=10); ax.set_ylabel('Predicted yield anomaly (%)',fontsize=10)
    ax.text(0.97,0.05,f'R²={r2p:.3f}\nRMSE={rmsep:.3f}%\nMAE={maep:.3f}%\nn={len(at)}',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=9,
            bbox=dict(facecolor='#f0f4fa',edgecolor='#aaaacc',pad=4,alpha=0.95))
    ax.legend(fontsize=8.5,ncol=2,framealpha=0.9); ax.spines[['top','right']].set_visible(False); lbl(ax,'b')
    save(fig,'FIG09_Cross_Event','FIG09_Cross_Event_Generalisation')

# ══════════════════════════════════════════════════════════════════════════════
# FIG10 — Early-Warning Lead-Time
# ══════════════════════════════════════════════════════════════════════════════
def fig10():
    print("\n── FIG10 ──")
    lead_rows=[]
    for yr in YEARS:
        for d in DISTRICTS:
            sub_e=_era5[(_era5['year']==yr)&(_era5['district']==d)].sort_values('date').copy()
            sub_c=_chirps[(_chirps['event_year']==yr)&(_chirps['district']==d)].sort_values('date').copy()
            if len(sub_e)==0: continue
            sub_e['doy']=sub_e['date'].dt.dayofyear
            pk=int(np.argmin(np.abs(sub_e['doy'].values-PEAK_DOY)))
            row={'year':yr,'district':d}
            for nw in LEAD_WEEKS:
                we=max(0,pk-nw*7); ws=max(0,we-28)
                row[f'p{nw}w']=float(sub_e['precip_mm'].iloc[ws:we].mean()) if we>ws else float(sub_e['precip_mm'].mean())
                row[f't{nw}w']=float(sub_e['temp_2m_C'].iloc[ws:we].mean()) if we>ws else float(sub_e['temp_2m_C'].mean())
                if len(sub_c)>10:
                    sp=spi_calc(sub_c['precip_mm'].values)
                    sub_c2=sub_c.copy(); sub_c2['doy']=sub_c2['date'].dt.dayofyear
                    pkc=int(np.argmin(np.abs(sub_c2['doy'].values-PEAK_DOY)))
                    lc=max(0,pkc-nw*7)
                    row[f'spi{nw}w']=float(sp.iloc[lc]) if lc<len(sp) else 0.0
                else: row[f'spi{nw}w']=0.0
            lead_rows.append(row)
    df_lead=pd.DataFrame(lead_rows); df_lead['year']=df_lead['year'].astype(int)
    df_all=_feat.merge(df_lead,on=['year','district'],how='inner')
    y_tgt=df_all[TARGET].values
    BASE=[c for c in ['sm_surface','sm_rootzone'] if c in df_all.columns]
    lt_r2,lt_rmse,lt_mae=[],[],[]
    for nw in LEAD_WEEKS:
        feats=BASE+[f'p{nw}w',f't{nw}w',f'spi{nw}w']
        avail=[f for f in feats if f in df_all.columns]
        X=df_all[avail].fillna(df_all[avail].median()).values
        at2,ap2=[],[]
        for held in YEARS:
            mask=(df_all['year']==held).values
            if mask.sum()==0: continue
            X_tr,y_tr=X[~mask],y_tgt[~mask]; X_te,y_te=X[mask],y_tgt[mask]
            sc=StandardScaler().fit(X_tr)
            rf=RandomForestRegressor(n_estimators=150,random_state=42).fit(sc.transform(X_tr),y_tr)
            ap2.extend(rf.predict(sc.transform(X_te))); at2.extend(y_te)
        at2,ap2=np.array(at2),np.array(ap2)
        lt_r2.append(r2_score(at2,ap2)); lt_rmse.append(np.sqrt(mean_squared_error(at2,ap2))); lt_mae.append(mean_absolute_error(at2,ap2))
    lw_lb=[f'{w}w' for w in LEAD_WEEKS]
    fig,axes=plt.subplots(1,3,figsize=(14,5.5))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.33,left=0.08,right=0.97,top=0.97,bottom=0.14)
    ax=axes[0]
    ax.plot(LEAD_WEEKS,lt_r2,'o-',color='#2166ac',lw=2.2,ms=10,markeredgecolor='#1a1a1a',markeredgewidth=0.8)
    ax.fill_between(LEAD_WEEKS,[min(0,v) for v in lt_r2],[max(0,v) for v in lt_r2],alpha=0.15,color='#2166ac')
    ax.axhline(0,color='gray',lw=0.9,ls=':')
    for x_,v in zip(LEAD_WEEKS,lt_r2): ax.text(x_,v+0.02,f'{v:.2f}',ha='center',fontsize=8.5,color='#2166ac',fontweight='bold')
    ax.set_xlabel('Lead time (weeks before drought peak)',fontsize=10); ax.set_ylabel('R² (LOEO-CV)',fontsize=10)
    ax.set_xticks(LEAD_WEEKS); ax.set_xticklabels(lw_lb,fontsize=9); ax.spines[['top','right']].set_visible(False); lbl(ax,'a')
    ax=axes[1]
    ax.plot(LEAD_WEEKS,lt_rmse,'s-',color='#d6604d',lw=2.2,ms=10,markeredgecolor='#1a1a1a',markeredgewidth=0.8,label='RMSE')
    ax.plot(LEAD_WEEKS,lt_mae, '^-',color='#1a9641',lw=2.2,ms=10,markeredgecolor='#1a1a1a',markeredgewidth=0.8,label='MAE')
    ax.set_xlabel('Lead time (weeks before drought peak)',fontsize=10); ax.set_ylabel('Error (% yield anomaly)',fontsize=10)
    ax.set_xticks(LEAD_WEEKS); ax.set_xticklabels(lw_lb,fontsize=9)
    ax.legend(fontsize=9,framealpha=0.9); ax.spines[['top','right']].set_visible(False); lbl(ax,'b')
    ax=axes[2]
    for i,d in enumerate(DISTRICTS):
        p_vals=[]
        for yr in YEARS:
            sub_e=_era5[(_era5['year']==yr)&(_era5['district']==d)].sort_values('date').copy()
            sub_e['doy']=sub_e['date'].dt.dayofyear
            if len(sub_e)==0: p_vals.append([np.nan]*len(LEAD_WEEKS)); continue
            pk=int(np.argmin(np.abs(sub_e['doy'].values-PEAK_DOY)))
            rp=[]
            for nw in LEAD_WEEKS:
                we=max(0,pk-nw*7); ws=max(0,we-28)
                rp.append(float(sub_e['precip_mm'].iloc[ws:we].mean()) if we>ws else np.nan)
            p_vals.append(rp)
        mean_p=np.nanmean(p_vals,axis=0)
        ax.plot(LEAD_WEEKS,mean_p,'o-',color=plt.cm.tab10(i/6),lw=1.5,ms=6,label=d,alpha=0.85)
    ax.set_xlabel('Lead time (weeks)',fontsize=10); ax.set_ylabel('28-day precip. at lead (mm day⁻¹)',fontsize=9)
    ax.set_xticks(LEAD_WEEKS); ax.set_xticklabels(lw_lb,fontsize=9)
    ax.legend(fontsize=7.5,ncol=2,framealpha=0.9); ax.spines[['top','right']].set_visible(False); lbl(ax,'c')
    save(fig,'FIG10_Early_Warning','FIG10_Early_Warning')

# ══════════════════════════════════════════════════════════════════════════════
# FIG11 — SMAP L4 HDF5 Spatial + Temporal
# ══════════════════════════════════════════════════════════════════════════════
def fig11():
    print("\n── FIG11 ──")
    import geopandas as gpd
    smap_files=sorted(glob.glob(os.path.join(DATA_ROOT,'SMAP_L4_SM_gph_*.h5')))
    if not smap_files: print("  ⚠  No SMAP HDF5 files found — skipping FIG11"); return
    gdf_study=gpd.read_file(os.path.join(DATA_ROOT,'study_districts.geojson'))
    gdf_study['NAME_3']=gdf_study['NAME_3'].str.strip()
    def parse_dt(fp): return datetime.strptime(os.path.basename(fp).split('_')[4][:8],'%Y%m%d')
    def extract(fp):
        with h5py.File(fp,'r') as f:
            lat=f['cell_lat'][:]; lon=f['cell_lon'][:]
            sm_s=f['Geophysical_Data/sm_surface'][:]; sm_r=f['Geophysical_Data/sm_rootzone'][:]
            et=f['Geophysical_Data/land_evapotranspiration_flux'][:]; ts=f['Geophysical_Data/surface_temp'][:]
        mask=(lat>=LAT_MIN)&(lat<=LAT_MAX)&(lon>=LON_MIN)&(lon<=LON_MAX)
        return {'lat':lat[mask],'lon':lon[mask],'sm_s':sm_s[mask],'sm_r':sm_r[mask],'et':et[mask],'ts':ts[mask]}
    scenes=sorted([{'date':parse_dt(f),**extract(f)} for f in smap_files],key=lambda x:x['date'])
    sm_means=[np.nanmean(np.where(np.array(s['sm_s'],float)<0,np.nan,s['sm_s'])) for s in scenes]
    sm_rz   =[np.nanmean(np.where(np.array(s['sm_r'],float)<0,np.nan,s['sm_r'])) for s in scenes]
    et_means=[np.nanmean(np.where(np.array(s['et'],float)<-900,np.nan,np.array(s['et'],float)))*86400*1000 for s in scenes]
    t_means =[np.nanmean(np.where(np.array(s['ts'],float)<100,np.nan,np.array(s['ts'],float)))-273.15 for s in scenes]
    dts=[s['date'] for s in scenes]
    onset24=datetime(2024,6,1); peak24=datetime(2024,7,15); end24=datetime(2024,9,30)

    # FIG11A — temporal
    fig,axes=plt.subplots(4,1,figsize=(12,14))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.42,left=0.10,right=0.97,top=0.98,bottom=0.06)
    def sh24(ax):
        ax.axvspan(onset24,end24,alpha=0.13,color='#e6821e',zorder=1)
        ax.axvline(onset24,color='#e6821e',lw=1.3,ls='--',alpha=0.85,zorder=3)
        ax.axvline(peak24, color='#e6821e',lw=2.0,ls='-', alpha=0.95,zorder=4)
        ax.spines[['top','right']].set_visible(False)
    for ai,(data,ylab,col,lab) in enumerate([
        (sm_means,'Surface SM (m³ m⁻³)',   '#d73027','Surface (0–5 cm)'),
        (sm_rz,   'Root-zone SM (m³ m⁻³)', '#5e3c99','Root-zone (0–100 cm)'),
        (et_means,'ET flux (mm day⁻¹)',     '#1a9641','Land ET'),
        (t_means, 'Surface temp. (°C)',     '#b2182b','Surface temperature')]):
        ax=axes[ai]
        ax.plot(dts,data,'o-',color=col,lw=2.0,ms=9,markeredgecolor='#1a1a1a',markeredgewidth=0.7,label=lab)
        ax.fill_between(dts,data,alpha=0.12,color=col)
        sh24(ax); ax.set_ylabel(ylab,fontsize=9.5); ax.legend(fontsize=8,loc='upper right',framealpha=0.9)
        for d2,v in zip(dts,data):
            ax.text(d2,v+0.003,f'{v:.3f}' if ai<2 else f'{v:.1f}',ha='center',fontsize=7.5,color=col,fontweight='bold')
        lbl(ax,chr(97+ai))
    leg_h=[mpatches.Patch(color='#e6821e',alpha=0.25,label='2024 drought period'),
           plt.Line2D([0],[0],color='#e6821e',lw=2.0,ls='-', label='Drought peak (15 Jul 2024)'),
           plt.Line2D([0],[0],color='#e6821e',lw=1.3,ls='--',label='Drought onset')]
    fig.legend(handles=leg_h,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(0.5,0.005),framealpha=0.95,edgecolor='#aaa')
    save(fig,'FIG11_SMAP','FIG11A_SMAP_Temporal_Evolution')

    # FIG11B — spatial maps
    n=len(scenes); ncols=min(n,5); nrows=(n+ncols-1)//ncols
    fig,axes_g=plt.subplots(nrows,ncols,figsize=(4*ncols,3.8*nrows))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.30,wspace=0.18,left=0.04,right=0.93,top=0.97,bottom=0.04)
    axes_flat=axes_g.flatten() if nrows>1 else (list(axes_g) if ncols>1 else [axes_g])
    for ai,sc in enumerate(scenes):
        ax=axes_flat[ai]
        sm=np.where(np.array(sc['sm_s'],float)<0,np.nan,np.array(sc['sm_s'],float))
        ax.scatter(sc['lon'],sc['lat'],c=sm,cmap='YlOrRd_r',vmin=0.05,vmax=0.45,s=6,alpha=0.85,linewidths=0,rasterized=True)
        gdf_study.boundary.plot(ax=ax,color='#1a1a1a',linewidth=1.0)
        ax.set_xlim(LON_MIN-0.1,LON_MAX+0.1); ax.set_ylim(LAT_MIN-0.1,LAT_MAX+0.1)
        col_d='#d6604d' if sc['date']>=onset24 else '#333333'
        ax.set_xlabel(sc['date'].strftime('%d %b %Y'),fontsize=9,color=col_d,fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03,0.96,f"{np.nanmean(sm):.3f}",transform=ax.transAxes,fontsize=7.5,va='top',
                bbox=dict(facecolor='white',alpha=0.75,edgecolor='none',pad=1.5))
    for ai in range(n,len(axes_flat)): axes_flat[ai].set_visible(False)
    cbar_ax=fig.add_axes([0.94,0.10,0.015,0.84])
    sm_cb=plt.cm.ScalarMappable(cmap='YlOrRd_r',norm=plt.Normalize(0.05,0.45)); sm_cb.set_array([])
    cb=fig.colorbar(sm_cb,cax=cbar_ax); cb.set_label('SM surface (m³ m⁻³)',fontsize=9,fontweight='bold'); cb.ax.tick_params(labelsize=7.5)
    save(fig,'FIG11_SMAP','FIG11B_SMAP_SM_Surface_Maps')

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
FIG_MAP = {1:fig01,2:fig02,3:fig03,4:fig04,5:fig05,
           6:fig06,7:fig07,8:fig08,9:fig09,10:fig10,11:fig11}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Nepal drought figures')
    parser.add_argument('--figs',nargs='+',type=int,
                        help='Figure numbers e.g. --figs 1 2 6  (default: all)')
    args = parser.parse_args()
    if not os.path.isdir(DATA_ROOT):
        print(f"\n❌  DATA_ROOT not found: {DATA_ROOT}")
        print("    Edit DATA_ROOT at line 38 of this script.")
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    load_all()
    to_run = args.figs or list(range(1,12))
    print(f"\nGenerating figures: {to_run}")
    for n in to_run:
        if n in FIG_MAP: FIG_MAP[n]()
        else: print(f"  ⚠  {n} not valid (1–11)")
    pngs = sorted(glob.glob(os.path.join(OUT,'**','*.png'),recursive=True))
    print(f"\n{'='*55}")
    print(f"Done. {len(pngs)} PNG + PDF + SVG sets saved to:")
    print(f"  {OUT}")
    print(f"{'='*55}")
