import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import glob
import xarray as xr
import sys,os,subprocess
import numpy as np
from cartopy import config
from matplotlib import cm
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Read in the CERES data.
basedir = '/work/bb1131/b380873/tropic_vis/'
olr_file = basedir + 'ERA5_OLR-20170805-20170809.nc'
olr_data = xr.open_dataset(olr_file)
olr_vals = np.abs(olr_data.ttr.values)
lons = olr_data.longitude
lats = olr_data.latitude
zeit = olr_data.time.values

# Filter for when the simulation occurred.
starttime = datetime(2017,8,7,6,30)
endtime = datetime(2017,8,8,18,0)
ii = np.argwhere(zeit >= np.datetime64(starttime))[0,0]
jj = np.argwhere(zeit >= np.datetime64(endtime))[0,0]
olr_vals_sub = olr_vals[ii:jj]/3600   
# shape = 120, 721, 1440, divide by 3600 s (for hourly data)
zeit_sub = zeit[ii:jj]

# pulling this from the following stackoverflow
# https://stackoverflow.com/questions/30030328/correct-placement-of-colorbar-relative-to-geo-axes-cartopy
def resize_colobar(event):
    plt.draw()

    posn = ax.get_position()
    cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                          0.04, posn.height])


fig, ax = plt.subplots(1,1,figsize=(11,3.7),
                       subplot_kw={'projection': ccrs.PlateCarree()})
cbar_ax = fig.add_axes([0, 0, 0.1, 0.1])
fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)

levs = np.linspace(100,390,15)
def animate(i):
    print(i)
    ax.clear()
    print(np.nanmin(olr_vals_sub[i]),np.nanmean(olr_vals_sub[i]),np.nanmax(olr_vals_sub[i]))
    im = ax.contourf(lons,lats,olr_vals_sub[i],levels=levs,cmap=cm.viridis,\
                  transform=ccrs.PlateCarree()) 

    fig.canvas.mpl_connect('resize_event', resize_colobar)
    plt.colorbar(im,label=r'W m$^{-2}$',cax=cbar_ax)

    ax.set_title('ERA5 TOA OLR: '+ str(zeit_sub[i]))
    gl = ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,
                linewidth=1,color='gray',alpha=0.5,linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax.set_extent([55,170,5,38],crs=ccrs.PlateCarree())
    im.set_clim([100,400])
    ax.coastlines()
    resize_colobar(None)


ananas = ani.FuncAnimation(fig,animate,jj-ii,interval=700,blit=False)
ananas.save('olr_ERA5.mp4')
plt.show()
