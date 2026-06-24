#!/usr/bin/env python

"""
# Compute the posterior uncertainty covariance.
"""
from __future__ import print_function

import sys
import argparse
import datetime
import numpy
import netCDF4

import lpdm

####################################################################
def getHQSum(ctl, start_ts, end_ts, num_obs):
    """ Read in the saved hq files over certain time steps and compute sum """

    hqsum = numpy.zeros((num_obs, ctl.nlandcells))

    t0 = datetime.datetime.now()
    for i in range(start_ts, end_ts):
        t1 = datetime.datetime.now()

        # hq slice files are numbered 1 .. ntimesteps
        # hq shape is #obs x #cells
        hqfile = ctl.hq_dir + "/HQ%04d.npy" % (i+1)
        hq = numpy.load(hqfile)
        hqsum = hqsum + hq

        t2a = datetime.datetime.now()
        print("Step %4d" % (i), "of ", start_ts, "to", end_ts-1, "done. Step time ", t2a-t1, "Total elapsed time ", t2a-t0)

    return hqsum


####################################################################
def getQSum(start_ts, end_ts, sigma, D, E):
    """
    # Qsum is (S^T * D * S) * E
    # where
    #   S is time-space varying sigma (#timesteps x # cells)
    #   D is temporal covariance (#timesteps x # timesteps)
    #   S^T is transpose of S (#cells x #timesteps)
    #   E is spatial covariance (#cells x #cells)
    """

    x = numpy.dot(sigma[start_ts:end_ts].T, D[start_ts:end_ts, start_ts:end_ts])

    # multiplication with E is an element by element multiply, NOT matrix multiply
    # In this case E is a numpy array, so the * will do element multiply
    qsum = numpy.dot(x, sigma[start_ts:end_ts]) * E

    return qsum


####################################################################
def mkvnc(filename, name, v, description=None):
    """ Create the netcdf file for given v array """

    ds = netCDF4.Dataset(filename, 'w', format='NETCDF4')

    size = v.shape[0]

    ds.createDimension('size', size)
    grid = ds.createVariable(name, 'double', ('size', 'size'))
    if description is None:
        grid.description = "aggregated posterior uncertainty covariance"
    else:
        grid.description = description
    grid[:] = v

    ds.close()


####################################################################
def dateToTimestep(ctl, date):
    """ Calculate timestep number for a given date """

    (syr, smon, sday) = date.split("-")
    ddate = datetime.datetime(int(syr), int(smon), int(sday))
    td = ddate - ctl.sdate
    ts = td.days * ctl.steps_per_day

    return int(ts)


####################################################################
def make_vshat(start_ts, end_ts, ctl, TC, E, hqhti, sigmafile, num_obs):
    """ Calculate vshat for the timesteps from start_ts to end_ts. """


    # get hqsum for flux
    hqsum = getHQSum(ctl, start_ts, end_ts, num_obs)
    print("hqsum shape is ", hqsum.shape)


    print("compute qsum for flux")
    # get time, space varying sigma from saved file
    sigma = ctl.load_file(sigmafile, "sigma")
    qsum = getQSum(start_ts, end_ts, sigma, TC, E)


    print("compute vshat")
    a = numpy.dot(hqsum.T, hqhti)
    a = numpy.dot(a, hqsum)
    vshat = (qsum - a) / (end_ts - start_ts)**2

    filename = "qsum_%d-%d.nc" % (start_ts, end_ts)
    varname = "qsum"
    mkvnc(filename, varname, qsum, "Sum of Q matrix between time steps %s and %s" % (start_ts, end_ts))

    return vshat


####################################################################
if __name__ == '__main__':

    numpy.set_printoptions(precision=2, edgeitems=10, linewidth=240)

    parser = argparse.ArgumentParser(description="Compute the posterior uncertainty covariance. ")
    parser.add_argument('-c', '--config', default="config.ini", help="set configuration file to use.  Default is 'config.ini'")
    parser.add_argument('sigmafile', help="sigma file used in hsigma.py")

    options = parser.parse_args()

    ctl = lpdm.lpdm(options.config)

    tstart = datetime.datetime.now()

    # Get temporal covariance matrix
    print("Calculating temporal covariance...")
    TC = ctl.makeTemporalCovariance()

    # Get spatial covariance matrix.
    # E is exp(-Xs/ls) where Xs is distance between cells, ls is spatial correlation parameter
    print("Loading spatial covariance...")
    heff = numpy.load(ctl.sp_cov_file)    # distance matrix
    spatial_cl = ctl.spatial_corr_length
    E = numpy.exp(-heff/spatial_cl)

    # get the (hqht +r)^-1 computed from inversion
    print("Reading hqhti.npy...")
    hqhti = numpy.load("hqhti.npy")
    num_obs = hqhti.shape[0]
#    config["num_obs"] = num_obs


    # hard coded dates for aggregated covariance
    covar_startdates = ctl.vshat_dates
    for i in range(len(covar_startdates)-1):
        print(i, covar_startdates[i])

        # convert date to timestep number
        start_ts = dateToTimestep(ctl, covar_startdates[i])
        end_ts = dateToTimestep(ctl, covar_startdates[i+1])
        print(start_ts, end_ts)

        vshat = make_vshat(start_ts, end_ts, ctl, TC, E, hqhti, options.sigmafile, num_obs)

        filename = "vshat_%d-%d" % (start_ts, end_ts)
        numpy.save(filename, vshat)
        mkvnc(filename + ".nc", "vshat", vshat)


    # now redo for entire time span, because we can't be certain that
    # it was covered by config['vshat_dates']
    start_ts = dateToTimestep(ctl, ctl.start_date)
    end_ts = dateToTimestep(ctl, ctl.end_date)
    vshat = make_vshat(start_ts, end_ts, ctl, TC, E, hqhti, options.sigmafile, num_obs)

    numpy.save("vshat", vshat)
    mkvnc("vshat.nc", "vshat", vshat)
#    numpy.save("hqsum_a", hqsum_a)

    tend = datetime.datetime.now()
    print("Total elapsed time is", tend - tstart)
