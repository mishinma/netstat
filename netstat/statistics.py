""" Module for calculating graph statistics """

import numpy as np


def get_distance_of_index(index, dist_distr):
    i = 0
    upper_bound = dist_distr[i]
    while index > upper_bound:
        i += 1
        upper_bound += dist_distr[i]
    # Index i represents distance i+1
    return i+1


def compute_median(dist_distr):
    median_index = np.median(np.arange(np.sum(dist_distr)))
    is_integer = median_index == int(median_index)
    dist = get_distance_of_index(int(median_index), dist_distr)
    if is_integer:
        return dist
    else:
        return (2*dist+1)/2.0


def compute_eff_diameter(dist_distr):
    eff_diameter_index = np.percentile(np.arange(np.sum(dist_distr)), 90, interpolation='linear')
    dist = get_distance_of_index(int(eff_diameter_index), dist_distr)
    return dist


def compute_statistics(dist_distr):
    mean = np.sum([(i+1)*dist_distr[i] for i in xrange(dist_distr.size)])/float(np.sum(dist_distr))
    median = compute_median(dist_distr)
    diameter = dist_distr.size
    eff_diameter = compute_eff_diameter(dist_distr)
    print 'Mean: %d' % mean
    print 'Median: %d' % median
    print 'Diameter: %d' % diameter
    print 'Effective Diameter: %d' % eff_diameter
