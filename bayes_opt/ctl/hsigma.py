#!/usr/bin/env python

"""
# apply sigma values to h strips

# take sigma values for each grid cell previously calculated with make_sigma.py,
# multiply that with each h strip to produce sigma*h, and store that result.
"""

from __future__ import print_function

import datetime
import argparse
import numpy
from scipy.sparse import csr_matrix

import lpdm

parser = argparse.ArgumentParser(description="Multiply sigma with H strip files to get Hsigma. ")
parser.add_argument('-c', '--config',
                    default="config.ini",
                    help="set configuration file to use.  Default is 'config.ini'")
parser.add_argument('sigmafile', help="File with sigma values to apply to H")

options = parser.parse_args()

ctl = lpdm.lpdm(options.config)
ntimesteps = ctl.ntimesteps

sigma = ctl.load_file(options.sigmafile, "sigma")

ctl.prepare_Hsigma()

t1 = datetime.datetime.now()

for j in range(ntimesteps):

    # h slice files are numbered 1 .. ntimesteps
    h = ctl.read_sparse_h(j)

    b = numpy.multiply(h.todense(), sigma[j])
    b = csr_matrix(b)    # convert back to sparse matrix

    hsigfile = ctl.hsig_dir + "/H%04d.npz" % (j+1)
    ctl.write_sparse(hsigfile, b)

    t2 = datetime.datetime.now()
    print(j, "of", ntimesteps, hsigfile, t2 - t1)
