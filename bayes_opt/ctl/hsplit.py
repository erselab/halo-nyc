#!/usr/bin/env python

"""
hsplit.py - Go through the netcdf footprint files and create the H slices

Do this by stepping through the observations.  For each obs, read in the
netcdf file.  Then step through the H block timesteps, find the footprint values
that match the timestep, and write any nonzero values to a temporary text H slice file.
Each subsequent observation footprint will append to the H text files.
After stepping through all timesteps, reformat the text files to binary format.

---
for each obs:
    read footprint
    for each timestep
        append values to text files, H1, H2, H3, ...

reformat files to binary format
---

This way we read each netcdf file only 1 time.

Requires:
    receptor list - list of footprint file names to use
    footprint files - stilt footprint files
    landmask - landmask file which defines land cells in the domain of interest.  See mklm.py
    config file - Configuration file

Output:
    H strip sparse files.  There is one file for each timestep.  Files are in numpy format.
"""

from __future__ import print_function

import sys
import datetime
import argparse
from collections import defaultdict

import lpdm


parser = argparse.ArgumentParser(description="Create sparse H strip files, one for each time step. ")
parser.add_argument('-c', '--config',
                    default="config.ini",
                    help="set configuration file to use.  Default is 'config.ini'")
parser.add_argument('--debug',
                    action="store_true",
                    default=False,
                    help="Print extra debugging information.")

parser.add_argument('receptorlist', help="File with list of receptor names")

options = parser.parse_args()

ctl = lpdm.lpdm(options.config, debug=options.debug)

t0 = datetime.datetime.now()

# Loop through every receptor file
try:
    f = open(options.receptorlist)
    receptors = f.readlines()
    f.close()
except Exception as e:
    sys.exit("Error reading receptor file %s: %s" % (options.receptorlist, e))

num_obs = len(receptors)

# prepare directories to hold H strips
tmpdir, hdir = ctl.prepare_H()

for obsnum, line in enumerate(receptors):

    filename = line.strip()

    t1 = datetime.datetime.now()

    # read the netcdf file for the obs.
    # grid returned is (ntimesteps x lat x lon) for 1 degree grid
    # Also extract inversion domain from the footprint domain
    # in case the footprint domain is different size than inversion domain
    grid, griddates, lats, lons = ctl.get_footprint(filename, extract_domain=False)
    if grid is None:
        continue  # if can't read footprint file, skip and continue

    # extract land cells.  This also changes shape of grid to 2d numobs x numlandcells
    grid = ctl.land_grid(grid)

    # Loop through each of the grid dates, find ones with dates in desired range.
    # We're going backwards in time, so skip dates after the end date, quit when before start date.
    # If date within range, extract non zero grid indices, write those to file for correct inversion timestep
    # We want to sum up the footprint values for each grid cell for each inversion timestep.
    # Each timestep is (normally) 3 hours, where the footprint timestep is 1 hour.
    # So we need to sum up the 3 footprint hours for the inversion timestep.
    # e.g. if inversion timestep is 3 hours, then sum up hours 0, 1, 2 for timestep 1; 3, 4, 5 for timestep 2 ...
    xx = defaultdict(int)  # sets initialized value to 0
    for gidx, gdate in enumerate(griddates):
        if gdate < ctl.sdate: break
        if gdate >= ctl.edate: continue

        td = gdate - ctl.sdate                     # time since inversion start date,
        nhours = td.days*24 + td.seconds//3600     # in number of hours
        timestep = nhours // ctl.hrsperstep + 1    # inversion timestep number
        nz = grid[gidx].nonzero()[0]               # location of non zero data points
        for cellnum in nz:
            val = grid[gidx, cellnum]
            if val > -3e34:
                # add value to this timestep,cellnum
                key = (timestep, cellnum)
                xx[key] += val

    # now we take the summed up H values and
    # make a dict that is indexed by timestep only,
    # i.e. contains all data for each timestep for this observation
    b = defaultdict(list)
    for key in xx:
        (timestep, cellnum) = key
        b[timestep].append((obsnum, cellnum, xx[key]))


    # append data to temporary text files,
    for timestep in b:
        tmpfile = ctl.get_tmp_h_file(timestep)
        f = open(tmpfile, "a")
        for (nobs, cellnum, val) in b[timestep]:
            f.write("%d %d %15.8e\n" % (nobs, cellnum, val))
        f.close()


    t2 = datetime.datetime.now()
    print("Finished obs num ", obsnum, "of", num_obs, t2-t1, t2-t0)


# now reformat the temporary text files into final binary files

for i in range(1, ctl.ntimesteps + 1):
    ctl.convert_h_file(i, num_obs)


t2 = datetime.datetime.now()
print("Finished. Total elapsed time", t2-t0)
