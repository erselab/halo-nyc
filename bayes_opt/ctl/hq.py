#!/usr/bin/env python

"""
# bayesian inversion
#
# Solve the equation
#  shat = sprior + HQT*inverse(HQHT+R)*(z-hsp)
#
# This script calculates the HQ and HQHT parts of the equation.
#
# This version uses python multiprocessing to split up the jobs.
# Creates a hqht block for each process, than combines them to
# get one final hqht
"""

from __future__ import print_function

import os
import sys
import datetime
import multiprocessing
import argparse
import numpy

from scipy.sparse import csr_matrix

import lpdm

####################################################################
def makeHQBlocks(ctl, start, end, sigmafile, numobs):
    """ Compute transpose(HQH).

    Use the method of Yadav and Michalak, 2012 for expressing Q as
    the kronecker product of D and E, where D is the temporal covariance
    and E is the spatial covariance.  HQ can then be computed by
    splitting up H into smaller blocks and multiplying those by D columns,
    then by E, and summing up all the blocks.

    This routine is called via the python multiprocessing module, so that
    multiple processes can each work on a section of H, defined by start and end.

    The HQ blocks are stored for later use, and the HQHT for each process is stored.
    The final HQHT can be computed by adding up each of the individual HQHT arrays.

    Input:
        config: configuration options
        start:  starting time step number
        end:    ending time step number
    """

    ntimesteps = ctl.ntimesteps


    print("making temporal covariance values...")
    TC = ctl.makeTemporalCovariance()
    print(TC[TC.shape[0]//2,:])
    print("TC shape is ", TC.shape)

    print("getting spatial covariance values...")
    heff = ctl.get_sp_cov()
    spatial_cl = ctl.spatial_corr_length
    if spatial_cl > 0:
        SC = numpy.matrix(numpy.exp(-heff/spatial_cl))
    else:
        SC = numpy.matrix(numpy.eye(heff.shape[0]))

    # initialize hqht to zero
    hqht = numpy.zeros((numobs, numobs))

    sigma = ctl.load_file(sigmafile, "sigma")

    print("starting ", start, "to", end)
    t0 = datetime.datetime.now()
    for i in range(start, end):
        t1 = datetime.datetime.now()

        # we can do the matrix operations in this loop with the sparse matrices
        hq = csr_matrix((numobs, ctl.nlandcells))

        for j in range(ntimesteps):

            # skip if temporal covariance is 0
            if TC[i, j] == 0: continue

            # h slice files are numbered 1 .. ntimesteps
            # hsigma files already have h*sigma applied
            h = ctl.read_sparse_h(j, hdir=ctl.hsig_dir)

            hq = hq + h*TC[i, j]

        hq = hq * SC        # this will create a dense hq
        hq = numpy.array(hq) * sigma[i]        # this will create a dense hq


        # The hq strips are needed later in qht_s, so save them here to file.
        filename = ctl.get_hq_file(i+1)
        numpy.save(filename, hq)

        h = ctl.read_sparse_h(i)
        hqht = hqht + (hq * h.T)

        t2a = datetime.datetime.now()
        print("Step %4d" % i, "of %4d - %4d" % (start, end), "done. Step time ", t2a-t1, "Total elapsed time ", t2a-t0)


    # each process stores it's own hqht, we'll combine them later.
    name = multiprocessing.current_process().name
    filename = ctl.hq_dir + "/HQHT_%s" % (name)
    numpy.save(filename, hqht)



####################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Calculate HQ. ")
    parser.add_argument('-m', '--multiprocs', type=int, default=0, help="Use multiple threads.")
    parser.add_argument('-c', '--config', default="config.ini", help="set configuration file to use.  Default is 'config.ini'")
    parser.add_argument('receptorlist', help="File with list of receptor names")
    parser.add_argument('sigmafile', help="File with sigma values to apply to H")

    options = parser.parse_args()

    try:
        f = open(options.receptorlist)
        files = f.readlines()
        f.close()
    except IOError as e:
        sys.exit("File %s cannot open. %s" % (options.receptorlist, e), file=sys.stderr)

    numobs = len(files)

    tstart = datetime.datetime.now()

    numpy.set_printoptions(precision=6, edgeitems=10, linewidth=150)

    ctl = lpdm.lpdm(options.config)
    ctl.prepare_HQ()

    ntimesteps = ctl.ntimesteps

    if options.multiprocs == 0:
        makeHQBlocks(ctl, 0, ntimesteps, options.sigmafile, numobs)

        filename = ctl.hq_dir + "/HQHT_%s.npy" % ("MainProcess")
        newfilename = ctl.hq_dir + "/HQHT.npy"
        os.rename(filename, newfilename)

    else:
        # start nprocs processes for making the hq pieces.
        nprocs = options.multiprocs
        jobs = []
        for i in range(nprocs):
            start = int(ntimesteps/nprocs)*i
            end = int(ntimesteps/nprocs)*(i+1)
            if i == nprocs-1 and end < ntimesteps: end = ntimesteps
            p = multiprocessing.Process(target=makeHQBlocks, args=(ctl, start, end, options.sigmafile, numobs))
            jobs.append(p)
            p.start()

        # wait for all the jobs to finish before continuing
        for p in jobs:
            p.join()

        print("combining hqht...")
        # initialize hqht to zero
        hqht = numpy.zeros((numobs, numobs))
        for p in jobs:
            name = p.name
            filename = ctl.hq_dir + "/HQHT_%s.npy" % (name)
            a = numpy.load(filename)
            hqht = hqht + a

        filename = ctl.hq_dir + "/HQHT"
        numpy.save(filename, hqht)

    tdone = datetime.datetime.now()
    print("Finished. Elapsed time", tdone-tstart)
