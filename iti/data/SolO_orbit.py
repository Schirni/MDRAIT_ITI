import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from datetime import timedelta, datetime
import astropy.units as u
from astropy.coordinates import Longitude

from sunpy.coordinates import HeliocentricEarthEcliptic, get_body_heliographic_stonyhurst, get_horizons_coord
from sunpy.time import parse_time

obstime = parse_time('2022/03/7')
hee_frame = HeliocentricEarthEcliptic(obstime=obstime)

def get_first_orbit(coord):
    lon = coord.transform_to(hee_frame).spherical.lon
    shifted = Longitude(lon - lon[0])
    ends = np.flatnonzero(np.diff(shifted) < 0)
    if ends.size > 0:
        return coord[:ends[0]]
    return coord

planets = ['Mercury', 'Venus', 'Earth', 'Mars']
times = obstime + np.arange(700) * u.day
planet_coords = {planet: get_first_orbit(get_body_heliographic_stonyhurst(planet, times))
                 for planet in planets}

stereo_a = get_horizons_coord('STEREO-A', obstime)
stereo_b = get_horizons_coord('STEREO-B', obstime)

#missions = ['Parker Solar Probe', 'Solar Orbiter', 'BepiColombo']
missions = ['Solar Orbiter']
mission_labels = {'Solar Orbiter': 'SO'}
#mission_labels = {'Parker Solar Probe': 'PSP', 'Solar Orbiter': 'SO', 'BepiColombo': 'BEPICOLOMBO'}
mission_coords = {mission: get_first_orbit(get_horizons_coord(mission, {'start': obstime,
                                                                        'stop': obstime + 1 * u.yr,
                                                                        'step': '1d'}))
                  for mission in missions}

def coord_to_heexy(coord):
    coord = coord.transform_to(hee_frame)
    coord.representation_type = 'cartesian'
    return coord.y.to_value('AU'), coord.x.to_value('AU')

mpl.rcParams.update({'figure.facecolor': 'black',
                     'axes.edgecolor': 'white',
                     'axes.facecolor': 'black',
                     'axes.labelcolor': 'white',
                     'axes.titlecolor': 'white',
                     'lines.linewidth': 1,
                     'xtick.color': 'white',
                     'xtick.direction': 'in',
                     'xtick.top': True,
                     'ytick.color': 'white',
                     'ytick.direction': 'in',
                     'ytick.right': True})

fig = plt.figure()
ax = fig.add_subplot()

ax.set_xlim(-2.15, 2.15)
ax.set_xlabel('Y (HEE)')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(0.1))

ax.set_ylim(1.8, -1.8)
ax.set_ylabel('X (HEE)')
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

ax.set_title(obstime.strftime('%d-%b-%Y %H:%M UT'))
ax.set_aspect('equal')

ax.plot([0, 0], [0, 2], linestyle='dotted', color='gray')

for planet, coord in planet_coords.items():
    ax.plot(*coord_to_heexy(coord), linestyle='dashed', color='gray')

    if planet == 'Earth':
        color, markersize, offset = 'lime', 10, 0.1
    else:
        color, markersize, offset = 'gray', None, 0.05

    x, y = coord_to_heexy(coord[0])
    ax.plot(x, y, 'o', markersize=markersize, color=color)
    ax.text(x + offset, y, planet, color=color)

for stereo, label, color in [(stereo_a, 'A', 'red'), (stereo_b, 'B', 'blue')]:
    x, y = coord_to_heexy(stereo)
    ax.plot([0, 5*x], [0, 5*y], linestyle='dotted', color='gray')
    ax.plot(x, y, 'o', color=color)
    ax.text(x + 0.1, y, label, color=color, fontsize=18)


ax.plot(0, 0, 'o', markersize=15, color='yellow')
ax.text(0.12, 0, 'Sun', color='yellow')

for mission, coord in mission_coords.items():
    color = 'magenta' if mission == 'Solar Orbiter' else 'orange'

    ax.plot(*coord_to_heexy(coord), linestyle='dashed', color=color)

    x, y = coord_to_heexy(coord[0])
    ax.plot(x, y, 'o', color=color)
    ax.text(x + 0.05, y, mission_labels[mission], color=color)

plt.show()

# This is necessary to reset the Matplotlib settings after plotting for our documentation.
# You don't need this in your own code.
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'axes.titlecolor': 'black'})