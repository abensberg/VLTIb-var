#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:41:05 2023

@author: abensberg, jkobus
"""
import numpy as np
from astroplan import Observer
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import streamlit as st
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib as mpl
from scipy import interpolate
mpl.rcParams.update({'lines.linewidth': 2})
mpl.rcParams['lines.markersize'] = 3

@st.cache_data
def fft(imap, pixelsize, u, v, wave):
    
    fft=np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(imap)))
    freq=np.fft.fftshift(np.fft.fftfreq(imap.shape[0],pixelsize))
    
    uv_lambda = np.transpose([v / wave, u / wave])
    real=interpolate.interpn((freq,freq),np.real(fft),uv_lambda)
    imag=interpolate.interpn((freq,freq),np.imag(fft),uv_lambda)
    
    return real+imag*1j

def get_rise_and_set_HA(declination = -24*u.deg, 
                        obs_lat = -24.63*u.deg, 
                        obs_long = -70.4*u.deg, 
                        obs_height = 2635.43*u.m,
                        min_elevation = 45*u.deg
                        ):
    """
    Calculate the hour angles of astronomical objects at their rise and set times, 
    given the declination of the object, the latitude, longitude, and height of the observer.

    Parameters
    ----------
    declination : float or array-like, Default: -24 degrees
        The declination of the targets in degrees. 
    obs_lat : float or astropy.units.Quantity, Default: -24.63 degrees.
        The latitude of the observer in degrees. 
    obs_long : float or astropy.units.Quantity, Default: -70.4 degrees.
        The longitude of the observer in degrees. 
    obs_height : float or astropy.units.Quantity, Default: 2635.43 meters.
        The height of the observer above sea level in meters. 
    min_elevation : float or astropy.units.Quantity, Default: 45 degrees.
        Minimum altitude at which the target given by its inclination is observavle.

    Returns
    -------
    rise : astropy.units.Quantity
        The hour angles of the targets (declinations) at their rise times.
    sett : astropy.units.Quantity
        The hour angles of the targets (declinations) at their set times.
   """

    
    vlt_location = EarthLocation(lat=obs_lat, lon=obs_long, height=obs_height)
    vlt = Observer(location=vlt_location)
    
    rise = []
    sett = []
    for dec in np.atleast_1d(declination):
        target = SkyCoord(0, dec, unit="deg")
        
        # hour angle of rise 
        rise_time = vlt.target_rise_time(Time('2012-12-13 00:00:00'), target, which="next", horizon = min_elevation)
        set_time = vlt.target_set_time(Time('2012-12-13 00:00:00'), target, which="next", horizon = min_elevation)
        try:
            HA_rise = vlt.target_hour_angle(rise_time, target)
            if HA_rise > 12 * u.hourangle:
                HA_rise -= 24 * u.hourangle
            HA_set = vlt.target_hour_angle(set_time, target)
            if HA_set > 12 * u.hourangle:
                HA_set -= 24 * u.hourangle
            rise.append(HA_rise)
            sett.append(HA_set)
            # print(HA_rise, HA_set)
        except:
            rise.append(u.Quantity(np.nan, u.hourangle))
            sett.append(u.Quantity(np.nan, u.hourangle))
            
    return u.Quantity(rise), u.Quantity(sett)

@st.cache_data
def get_uv(DEC_deg, HA_hour, use_telescopes, phi_deg=-24.6279483, quiet=True):
    """ Returns the (u,v) coordinates for VLTI observations.

        This function calculates the (u,v) coordinates
        and the corresponding projected baselines and position angles
        for given pairs of VLTI telescopes,
        declination of the object to be observed
        and hour angles for the observations.

        Parameters
        ----------
        DEC_deg : float
            Declination of the object to be observed in degree
        HA_hour : array_like of floats,
            1D list or 1D numpy array of hour angles of the
            observations in hours
        use_telescopes : array_like of int
            1D or 2D array_like of telescope indices, e.g. UT-configuration: [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]].
        phi_deg: float, default: -24.6279483
                 Latitude of the observatory.
                 The default value -24.6279483 applies for the VLTI.
        quiet: bool, default: True
               If set to False it will print some info.


        Returns
        -------
        uv_coords: 3D numpy array of floats
                   ndarray of (u,v) coordinates in meter,
                   dimensions: i_HA, i_telescope_pair, u/v
        BL: 2D numpy array of floats
            ndarray of projected baselines in meter,
            dimensions: i_HA, i_telescope_pair
        PA: 2D numpy array of floats
            ndarray of position angles of the telescope pairs in degree,
            dimensions: i_HA, i_telescope_pair
        alt_deg: 1D numpy array of floats
                 Altitude of the object  to be observed in degree,
                 dimension: i_HA

        Examples
        --------
        (u,v) coordinates for one observations at *HA_hour* = 0h of an object at *DEC_deg* = 24° with the UT baseline UT1 - UT2:

        >>> uv, bl, pa, alt = salamander.interferometry.get_uv(24, [0], [0,1])
        >>> uv
        array([[[-24.812      -33.60050621]]])
        >>> bl
        array([[41.7687606]])

        Multiple hour angles and telescope pairs:

        >>> uv, bl, pa, alt = salamander.interferometry.get_uv(24, [-1,0,1], [[0,1],[0,2]])
        >>> uv
        array([[[-18.48346725 -31.28212508]
              [-24.812      -33.60050621]
              [-29.44963596 -36.5061026 ]]
              [[-43.63987179 -51.91032669]
              [-54.84       -57.18371652]
              [-62.30287284 -63.45647076]]])

    """

    list_telescopes = ['0']*34
    ENU_coords      = np.zeros((34,3))

    f_stations = open(__file__.rsplit('/',1)[0] + '/VLTI_stations.dat','r')
    line       = f_stations.readline()
    i = 0
    while line:
        data = str.split(line)
        if data[0] != '#':
            list_telescopes[i] = data[1]
            ENU_coords[i,0]    = np.float64(data[2])    # East
            ENU_coords[i,1]    = np.float64(data[3])    # North
            ENU_coords[i,2]    = 0                      # Up; =0 because telescopes are on same level
            i += 1
        del data, line
        line = f_stations.readline()
    f_stations.close()

    # calculate baseline vector (ENU) of the telescopes !on the ground!
    #
    use_telescopes = np.array(use_telescopes)
    if use_telescopes.ndim == 1:
        use_telescopes = np.array([use_telescopes])
    n_pairs = len(use_telescopes)
    B_ENU = (ENU_coords[use_telescopes[:,0]]-ENU_coords[use_telescopes[:,1]])

    if not quiet:
        used_pairs = np.array(['00-00']*n_pairs)
        for i in range(n_pairs):
            used_pairs[i] = list_telescopes[use_telescopes[i,0]] + '-' + list_telescopes[use_telescopes[i,1]]
        print('')
        print(used_pairs)
        print('')
        print('Baseline East [m]:  ' + str(B_ENU[:,0]))
        print('Baseline North [m]: ' + str(B_ENU[:,1]))

    # conversion from degree to radians
    #
    phi_rad = phi_deg*np.pi/180
    DEC_rad = DEC_deg*np.pi/180
    HA_rad  = np.array(HA_hour)*np.pi/12

    n_baseline = len(HA_rad)

    alt_deg = np.arcsin(np.sin(phi_rad)*np.sin(DEC_rad) + np.cos(phi_rad)*np.cos(DEC_rad)*np.cos(HA_rad))*180/np.pi


    # transformation matrices from ENU to uvw
    # correct for equator; x -> East , y -> along Earth's rotation axis, z -> to zenith of equator (celestial equator)
    #
    matrix_phi = np.array( [[1,0,0],[0,np.cos(phi_rad),np.sin(phi_rad)],[0,-np.sin(phi_rad),np.cos(phi_rad)]] )
    # correct for declination; x -> East , y -> North, z -> to star
    #
    matrix_DEC = np.array( [[1,0,0],[0,np.cos(DEC_rad),-np.sin(DEC_rad)],[0,np.sin(DEC_rad),np.cos(DEC_rad)]] )

    uvw_coords = np.zeros((n_pairs,n_baseline,3))
    for i in range(n_pairs):
        for j in range(n_baseline):
            matrix_HA = np.array( [[np.cos(HA_rad[j]),0,np.sin(HA_rad[j])],[0,1,0],[-np.sin(HA_rad[j]),0,np.cos(HA_rad[j])]] )
            uvw_coords[i,j,:] = np.dot( matrix_DEC, np.dot( matrix_HA, np.dot( matrix_phi, np.array( [B_ENU[i,0],B_ENU[i,1],0]) ) ) )

    if not quiet:
        print('')
        print('u [m]:  ' + str(uvw_coords[:,:,0]))
        print('v [m]:  ' + str(uvw_coords[:,:,1]))
        #print('w:  ' + str(uvw_coords[:,:,2]))

    # baseline (BL) in meter and position angle (PA) in degrees in the sky (=uv-plane); ignore w component
    #
    BL = np.linalg.norm(uvw_coords[:,:,:2],axis=-1)

    # PA is defined from North (v) to East (u) or from South (-v) to West (-u); [0,180[
    #
    PA_deg = np.arctan(uvw_coords[:,:,0]/uvw_coords[:,:,1]) * 180/np.pi
    PA_deg[PA_deg<0] += 180

    if not quiet:
        print('')
        print('BL [m]: ' + str(BL))
        print('PA [deg]: ' + str(PA_deg))

    return uvw_coords[:,:,:2], BL, PA_deg, alt_deg

@st.cache_data
def calc_vis(imap, u_m, v_m, wave_m, pixelsize_rad):
    uv_shape = u_m.shape
    u_m = u_m.flatten()
    v_m = v_m.flatten()
    
    u_m = np.ascontiguousarray(u_m)
    v_m = np.ascontiguousarray(v_m)
    imap = np.ascontiguousarray(imap)
    
    corr_flux = fft(imap, pixelsize_rad, u_m, v_m, wave_m)
    norm = fft(imap, pixelsize_rad, np.zeros((1,)), np.zeros((1,)), wave_m)
    
    # corr_flux = sampleImage(imap, 
    #                         pixelsize_rad, 
    #                         u_m / wave_m, v_m / wave_m,
    #                         PA = 0)
    # norm = sampleImage(imap,pixelsize_rad, np.zeros((1,)), np.zeros((1,)), PA = 0)
    
    corr_flux = corr_flux.reshape(uv_shape)
    
    if norm != 0:
        vis = np.absolute(corr_flux / np.absolute(norm))
    else:
        vis = np.zeros(np.shape(corr_flux))
    
    cps = []
    for cp_idx in range(4): # create baseline triangles for use_telescopes = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
        if cp_idx == 0:
            bl_idxs = [0,3,1]
            signs = [1,1,-1]
        elif cp_idx == 1:
            bl_idxs = [0,4,2]
            signs = [1,1,-1]
        elif cp_idx == 2:
            bl_idxs = [4,5,3]
            signs = [1,-1,-1]
        elif cp_idx == 3:
            bl_idxs = [2,5,1]
            signs = [1,-1,-1]
            
        cp = 0      
        for sign, bl_idx in zip(signs, bl_idxs):
            cp += sign * np.angle(corr_flux[bl_idx], deg = True) 
    
        cps.append(cp)
        
    return vis, np.array(cps)

def name2conf(name):
    conf = {"UT":[[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],
            "small":[[4,8],[4,16],[4,13],[8,16],[8,13],[16,13]],
            "medium":[[31,22],[31,16],[31,27],[22,16],[22,27],[16,27]],
            "large":[[4,21],[4,26],[4,31],[21,26],[21,31],[26,31]],
            "extended":[[4,11],[4,26],[4,30],[11,26],[11,30],[26,30]]}
    return conf[name]

@st.cache_data
def loadimap(model, wave, fname=None, pixsizerad = 1.15e-9, npix = 1200, sig = 20):
    if model=='RT face-on':
        fname = __file__.rsplit('/',1)[0] +'/stdata/'+str(wave)+'_0.npy'
    if model=='RT inclined':
        fname = __file__.rsplit('/',1)[0] +'/stdata/'+str(wave)+'_45.npy'
    
    if 'gaussian' not in model:
        imap = np.load(fname)
    else:
        if model=='gaussian face-on':
            sigmax = sig * pixsizerad
            sigmay = sigmax
            # get coordinates for given pixel resolution
            y,x = np.indices((npix, npix))
            # center = np.floor_divide((npix, npix), 2)
            center = (int(npix/2), int(npix/2))
            x -= center[0]
            y -= center[1]
            x0 = 0
            y0 = 0
            x = x * pixsizerad
            y = y * pixsizerad
            imap = np.exp(- (x-x0)**2/(2*sigmax**2) - (y-y0)**2/(2*sigmay**2)) 
        if model=='gaussian inclined':
            sigmax = sig * pixsizerad
            sigmay = 0.5*sigmax
            # get coordinates for given pixel resolution
            y,x = np.indices((npix, npix))
            # center = np.floor_divide((npix, npix), 2)
            center = (int(npix/2), int(npix/2))
            x -= center[0]
            y -= center[1]
            x0 = 0
            y0 = 0
            x = x * pixsizerad
            y = y * pixsizerad
            imap = np.exp(- (x-x0)**2/(2*sigmax**2) - (y-y0)**2/(2*sigmay**2))
            
    if (imap.shape[0] % 2) != 0:
        imap = imap[:-1,:]
    if (imap.shape[1] % 2) != 0:
        imap = imap[:,:-1]
    
    return imap, pixsizerad

#
# Applet
#

st.write("""
          # VLTI B-VAR
          ### The impact of Baseline VARiations on VLTI observations
          """)

st.markdown("""
            Variability is a vital feature in protoplanetary disks. However, the measured visbility and closure phase of an object can also vary when comparing observations done at different hour angles due to different projected baselines. VLTI B-VAR allows to estimate this effect using four different disk models. It is also possible to upload custom intensity maps from which closure phase and visibilities should be calculated.
            A detailed study of the influence of protoplanetary disk variability on the visiblities and closure phases can be found in [Bensberg,Kobus & Wolf 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A.126B/abstract).
        """)

st.write("""
            Use the slider and buttons to choose your set of parameters for the calculations. In case you want to upload your own custom model intensity map, please make sure that the uploaded file is a numpy file (.npy) containing a 2D numpy array where the dimensions of the image are equal for x- and y-axis.
        """)
         
col1, col2, col3 = st.columns([4,4,3])

with col1:
    config = st.radio(
        "Choose VLTI configuration:",
        ('UT', 'small', 'medium', 'large', 'extended'))

with col2:
    model = st.radio(
        "Choose disk model:",
        ('RT face-on', 'RT inclined', 'gaussian face-on', 'gaussian inclined', 'custom'))
        
if model == 'custom':
    fname = st.file_uploader("Upload custom intensity map", type=['npy'],
                              help='Upload intensity map in numpy array format.', label_visibility='hidden')
        
col4, col5 = st.columns(2,gap='medium')

with col4:
    dec = st.slider('Declination in deg', -69, 20, -24, 1)

with col5:
    wave = st.select_slider(r'Wavelength in μm',[1.6,2.2,3.5,4.8,10],3.5)
        
wave_m = wave * 1e-6

ha_hour_lims = get_rise_and_set_HA(dec*u.deg)
ha_hour = np.linspace(ha_hour_lims[0].value, ha_hour_lims[1].value,100).flatten()

uv_coords, BL, PA_deg, alt_deg = get_uv(dec, ha_hour, name2conf(config))
u_m = uv_coords[:,:,0]
v_m = uv_coords[:,:,1]
if model == 'custom':
    if fname == None:
        imap = np.zeros((100,100))
        pixelsize = 1.15e-9
    else:
        imap, pixelsize = loadimap(model, wave, fname)
else:
    imap, pixelsize = loadimap(model, wave)

vis, cp = calc_vis(imap, u_m, v_m, wave_m, pixelsize)

with col3:
    fig, ax = plt.subplots()
        
    if 'RT' in model:
        vmax = np.unique(imap)[-2]
        norm = clr.LogNorm(vmin=vmax*1e-5, vmax=vmax)
    else:
        vmax = imap.max()    
        norm = clr.Normalize(vmin=0, vmax=vmax)
        
    ax.imshow(imap,norm=norm,interpolation='none',origin='lower', extent=(-1,1,-1,1))
    ax.set_xlim(-0.3,0.3)
    ax.set_ylim(-0.3,0.3)
    ax.axis('off')
    # ax.set_title('Model preview:',fontsize=15)
    st.pyplot(fig)


fig2, axs = plt.subplots(2,2, figsize=(9,7))
for i_bl in range(6):
    line, = axs[0,0].plot(ha_hour, vis[i_bl], alpha = 0.8)
    if i_bl < 4:
        axs[0,1].plot(ha_hour, cp[i_bl], "-", c = "k", alpha = 0.8)
    axs[1,0].plot(ha_hour, BL[i_bl, :], "-", color = line.get_color(), alpha = 0.8)
    axs[1,1].scatter(u_m[i_bl,:], v_m[i_bl,:], color = line.get_color()) # uv coverage
    axs[1,1].scatter(-u_m[i_bl,:], -v_m[i_bl,:], color = line.get_color()) # ''
axs[0,0].set_xlabel("hour angle")
axs[0,0].set_ylabel("visibility")
axs[1,1].axis('equal')
ymin, ymax = axs[0,1].get_ylim()
if abs(ymin) < 1 or abs(ymax) < 1:
    axs[0,1].set_ylim(-1,1)
axs[0,1].set_xlabel("hour angle")
axs[0,1].set_ylabel("closure phase in deg")
axs[1,0].set_xlabel("hour angle")
axs[1,0].set_ylabel("baseline in m")
axs[1,1].set_xlabel("u in m")
axs[1,1].set_ylabel("v in m")

fig2.tight_layout()
st.pyplot(fig2)
