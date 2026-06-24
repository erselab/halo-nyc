#!/usr/bin/env python

"""
Create a landmask array file for use in north america ctlpdm inversions.

Requires:
	ctregions.nc - File with 0/1 for global 1 degree cells created from
	CarbonTracker data.
	config file - Configuration file

Output
	region.npy - A numpy file with 0/1 designation for land cells in domain
	region.txt - Same as region.nc but in text format, rows = latitudes, columns = longitudes
	landmask file - numpy file with location of land cells in domain
		This is a 2xnumlandcells array. Dimension 0 has latitude cell numbers with land,
		dimension 1 has longitude cell numbers with land.
"""
from __future__ import print_function

import argparse
import numpy
import netCDF4

import lpdm


parser = argparse.ArgumentParser(description="Create a land mask file from the global regions file made from carbontracker data.")

parser.add_argument('-c', '--configfile', default='config.ini', help="Specify inversion configuration file to use.")
parser.add_argument('regionsfile', help="Global regions file with 0/1 for land")
options = parser.parse_args()

ctl = lpdm.lpdm(options.configfile)

regionsfile = options.regionsfile

# Read in carbontracker regions file. This has 0/1 for each global 1 degree grid cell
ds = netCDF4.Dataset(regionsfile)
regions = numpy.array(ds.variables['regions'][:])
map_na = regions[ctl.south:ctl.north+1, ctl.west:ctl.east+1]

# remove tip of south america
# latitude < 15 deg, longitude < -80 deg
map_na[0:5, 90:120] = 0

# remove greenland
# latitude > 60 deg, longitude < -60 deg
map_na[50:, 110:] = 0

# northwest tip of greenland
# latitude > 70 deg, longitude < -73 deg
map_na[60:, 97:] = 0


numpy.save("region", map_na)
numpy.savetxt("region.txt", map_na, fmt="%.0f")

# find the indices in the map_na array where there is land.
landmaparr = numpy.where(map_na == 1)
#print landmaparr


numpy.save(ctl.landmask_file, landmaparr)
