#!/usr/bin/env python

"""
bayesian inversion

 Solve the equation
  shat = sprior + HQT*inverse(HQHT+R)*(z-hsp)

Input files:
    HQHT.npy, HQHT_pbl.npy, HQHT_trans.npy, HQHT_ftrop.npy
    r.txt
    zhsp.txt
    sprior.npy
    All HQ slice files

Output files:
    hqhti.npy
    hqht_zhsp.npy
    qhts.npy
    hshat.txt
    shat_flux.npy, shat_flux.nc

No boundary value optimization

"""

from __future__ import print_function

import argparse
import datetime
import numpy
import pdb
import lpdm


####################################################################
def qht_s(ctl, hqht_zhsp):
    """ compute hqt*[inverse(hqht+r)*(z-h*sp)]

    inverse(hqht+r)*(z-h*sp) is given as the array hqht_zhsp.
    hq is read in from files created previously.
    """

    ntimesteps = ctl.ntimesteps

    qhts = numpy.zeros((ntimesteps, ctl.nlandcells))

    t0 = datetime.datetime.now()
    for i in range(ntimesteps):
        t1 = datetime.datetime.now()

        # hq slice files are numbered 1 .. ntimesteps
        # hq shape is #obs x #cells
        hqfile = ctl.hq_dir + "/HQ%04d.npy" % (i+1)
        hq = numpy.load(hqfile)

        qhts[i] = numpy.dot(hq.T, hqht_zhsp)

        t2a = datetime.datetime.now()
        print("Step %4d" % (i+1), "of ", ntimesteps, "done. Step time ", t2a-t1, "Total elapsed time ", t2a-t0)

    return qhts


####################################################################
if __name__ == '__main__':

    numpy.set_printoptions(precision=6, edgeitems=10, linewidth=150)

    parser = argparse.ArgumentParser(description="Do bayesian inversion. ")
    parser.add_argument('-c', '--config',
                        default="config.ini",
                        help="set configuration file to use.  Default is 'config.ini'")
    parser.add_argument('-r', '--resume',
                        action="store_true",
                        default=False, help="resume calculations at the qhts step")
    parser.add_argument('sprior', help="File with sprior data")
    parser.add_argument('rfile', help="File with r, model-data mismatch")

    options = parser.parse_args()

    ctl = lpdm.lpdm(options.config)

    tstart = datetime.datetime.now()


    # Get hqht calculated previously.
    # shape of hqht is numobs x numobs
    print("Getting hqht...")
    filename = ctl.hq_dir + "/HQHT.npy"
    hqflux = numpy.load(filename)
    numobs = hqflux.shape[0]

    hqht = hqflux

    print("Adding r to hqht...")
    r = numpy.loadtxt(options.rfile)
    for i in range(numobs):
        hqht[i, i] = hqht[i, i] + r[i]


    # calculate inverse of hqht
    # hqhti is needed later in aposterior variance calculations
    if options.resume:
        print("Loading hqhti...")
        hqhti = numpy.load(ctl.workdir+"/hqhti.npy")
    else:
        print("Calculate inverse of hqht...")
        hqhti = numpy.linalg.inv(hqht)
        numpy.save("hqhti", hqhti)


    print("Getting zhsp...")
    zfile = ctl.workdir + "/zhsp.txt"
    zhsp = numpy.loadtxt(zfile)

    if options.resume:
        print("Loading hqht_zhs...")
        hqht_zhsp = numpy.load(ctl.workdir+"/hqht_zhsp.npy")
    else:
        print("Calculating hqhti*zhsp...")
        hqht_zhsp = numpy.dot(hqhti, zhsp)
        numpy.save(ctl.workdir+"/hqht_zhsp", hqht_zhsp)

    # compute qht*[inverse(hqht+r)*(z-h*sp)]
    print("Calculating qhts...")
    qhts = qht_s(ctl, hqht_zhsp)
    print("qhts shape is ", qhts.shape)
    numpy.save(ctl.workdir+"/qhts", qhts)

    print("Getting sprior...")
    sp = ctl.load_file(options.sprior, "sprior")
    print("sp shape is ", sp.shape)


    print("Calculate shat...")
    shat = sp + qhts

    print("Saving shat in netcdf format...")
    numpy.save("shat", shat)
    ctl.make_ncdf(ctl.workdir+"/shat.nc", "shat", shat)


    print("Making hshat...")
    hshat = ctl.convolve(shat)
    numpy.savetxt(ctl.workdir+"/hshat.txt", hshat)


    tdone = datetime.datetime.now()
    print("Finished. Elapsed time", tdone-tstart)
