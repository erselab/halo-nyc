    #!/usr/bin/env python

"""
Script to create the zhsp.txt file.

Takes the input files sprior.npy, obs.txt, bg.txt, calculates
hsprior by convolving sprior with h, then subtracts hsprior and
background values from bg.txt from the observation values in obs.txt
"""

from __future__ import print_function

import sys
import argparse
import numpy

import lpdm

#########################################################################

epilog = """\
obsfile        : text file with observation value, one line per receptor.
spriorfile     : file with sprior data. Either .npy or netcdf format.
backgroundfile : text file with background value, one line per receptor.
"""

parser = argparse.ArgumentParser(description="Calculate zhsp file.", epilog=epilog)
parser.add_argument('-o', '--other', default=None, help="text file with other values to subtract from observations, one line per receptor.")
parser.add_argument('-c', '--config', default="config.ini", help="set configuration file to use.  Default is 'config.ini'")
parser.add_argument('obsfile', help="File with observations for each receptor")
parser.add_argument('sprior', help="File with sprior data")
parser.add_argument('bg', help="File with background values for each receptor")

options = parser.parse_args()

ctl = lpdm.lpdm(options.config)

# Read in observation values.
try:
	obs = numpy.loadtxt(options.obsfile)
except Exception as e:
	sys.exit("Error reading observation file: %s %s" % (obsfile, e))

#print obs
numobs = obs.shape[0]
if obs.ndim > 1: obs = obs[:, 0]


#config["num_obs"] = numobs


# -- hsprior
# try reading file as a numpy .npy file first,
# if that doesn't work, try reading it as netcdf file.
print("making hsprior...")
sprior = ctl.load_file(options.sprior, "sprior")

hsp = ctl.convolve(sprior)

hspriorfile = ctl.workdir + "/hsprior.txt"
print("saving hsprior in ", hspriorfile)
numpy.savetxt(hspriorfile, hsp)


# Get background values previously made using make_bg.py
bg = numpy.loadtxt(options.bg)
if bg.ndim > 1: bg = bg[:, 0]


# subtract background values from observation
obsbg = obs - bg
bgfile = ctl.workdir + "/obs-bg.txt"
print("Saving obs - bg in %s..." % bgfile)
numpy.savetxt(bgfile, obsbg, fmt='%12.3f')


# Subtract hother and hsp from bg
print("making zhsp")
if options.other:
    hsother = numpy.loadtxt(options.other)
    zhsp = obsbg - hsother - hsp
else:
    zhsp = obsbg - hsp


zfile = ctl.workdir + "/zhsp.txt"
print("Saving zhsp in %s..." % zfile)
numpy.savetxt(zfile, zhsp, fmt="%12.6f")
