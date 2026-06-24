#!/usr/bin/env python

"""
calculate distances between each cell in the domain.
"""

from __future__ import print_function


import datetime
import argparse
from math import atan, tan, sin, cos, pi, sqrt, atan2, radians
import numpy

import lpdm



ELLIPSOIDS = {
    # model           major (km)   minor (km)     flattening
    'WGS-84':        (6378.137,    6356.7523142,  1 / 298.257223563),
    'GRS-80':        (6378.137,    6356.7523141,  1 / 298.257222101),
    'Airy (1830)':   (6377.563396, 6356.256909,   1 / 299.3249646),
    'Intl 1924':     (6378.388,    6356.911946,   1 / 297.0),
    'Clarke (1880)': (6378.249145, 6356.51486955, 1 / 293.465),
    'GRS-67':        (6378.1600,   6356.774719,   1 / 298.25)
}

####################################################################
def makeSpatialDistance(lats, lons):
    """ Make a 2d array containing distances from every grid cell to
    every other grid cell
    """

    m = len(lats)
    dist = numpy.zeros((m, m))

    # fills only half of the dist matrix.  But because it's symmetric
    # fill the other half with its transpose.
    # diagonal is zero, because it's the cell distance to itself
    for i in range(m):
        print(i, "of", m)
        for j in range(i+1, m):
            d = vincenty_distance((lats[i], lons[i]), (lats[j], lons[j]))
            dist[i, j] = d

    # fill up the other half
    dist = dist + dist.T

    return dist


###################################################################
def vincenty_distance(a, b):
    """ Calculate distance between locations a and b.
    a and b are tuples of (lat, lon) in decimal degrees.
    modified slightly from geocode distance.py
    Return the distance in kilometers between two points a and b.
    a and b are tuples of (lat, lon)
    Original code from https://github.com/geopy/geopy
    """

    ELLIPSOID = "WGS-84"

    a_latitude, a_longitude = a
    b_latitude, b_longitude = b
    lat1, lng1 = radians(a_latitude), radians(a_longitude)
    lat2, lng2 = radians(b_latitude), radians(b_longitude)

    major, minor, f = ELLIPSOIDS[ELLIPSOID]

    delta_lng = lng2 - lng1

    reduced_lat1 = atan((1 - f) * tan(lat1))
    reduced_lat2 = atan((1 - f) * tan(lat2))

    sin_reduced1, cos_reduced1 = sin(reduced_lat1), cos(reduced_lat1)
    sin_reduced2, cos_reduced2 = sin(reduced_lat2), cos(reduced_lat2)

    lambda_lng = delta_lng
    lambda_prime = 2 * pi

    iter_limit = 20

    while abs(lambda_lng - lambda_prime) > 10e-12 and iter_limit > 0:
        sin_lambda_lng, cos_lambda_lng = sin(lambda_lng), cos(lambda_lng)

        sin_sigma = sqrt(
            (cos_reduced2 * sin_lambda_lng) ** 2 +
            (cos_reduced1 * sin_reduced2 -
             sin_reduced1 * cos_reduced2 * cos_lambda_lng) ** 2
        )

        if sin_sigma == 0:
            return 0 # Coincident points

        cos_sigma = (
            sin_reduced1 * sin_reduced2 +
            cos_reduced1 * cos_reduced2 * cos_lambda_lng
        )

        sigma = atan2(sin_sigma, cos_sigma)

        sin_alpha = (cos_reduced1 * cos_reduced2 * sin_lambda_lng / sin_sigma)
        cos_sq_alpha = 1 - sin_alpha ** 2

        if cos_sq_alpha != 0:
            cos2_sigma_m = cos_sigma - 2 * (sin_reduced1 * sin_reduced2 / cos_sq_alpha)
        else:
            cos2_sigma_m = 0.0 # Equatorial line

        C = f / 16. * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

        lambda_prime = lambda_lng
        lambda_lng = (
            delta_lng + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (
                cos2_sigma_m + C * cos_sigma * (
                -1 + 2 * cos2_sigma_m ** 2
                )
                )
            )
        )
        iter_limit -= 1

    if iter_limit == 0:
        raise ValueError("Vincenty formula failed to converge!")

    u_sq = cos_sq_alpha * (major ** 2 - minor ** 2) / minor ** 2

    A = 1 + u_sq / 16384. * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))

    B = u_sq / 1024. * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    delta_sigma = (
        B * sin_sigma * (
            cos2_sigma_m + B / 4. * (
                cos_sigma * (
                    -1 + 2 * cos2_sigma_m ** 2
                ) - B / 6. * cos2_sigma_m * (
                    -3 + 4 * sin_sigma ** 2
                ) * (
                    -3 + 4 * cos2_sigma_m ** 2
                )
            )
        )
    )

    s = minor * A * (sigma - delta_sigma)
    return s


##############################################################################

if __name__ == '__main__':

    epilog = "Output is a numpy save file with array (nlandcells x nlandcells) of distances between land cells in kilometers."""
    description = "Calculate distances between land cells for an inversion. "
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-c', '--config', default="config.ini", help="Set configuration file to use.  Default is 'config.ini'")

    options = parser.parse_args()

    print("Making spatial distances...")
    t0 = datetime.datetime.now()

    ctl = lpdm.lpdm(options.config)

    heff = makeSpatialDistance(ctl.lats, ctl.lons)
    numpy.save(ctl.sp_cov_file, heff)
    t1 = datetime.datetime.now()
    print("Finished making spatial distance.  Elapsed time ", t1-t0)
