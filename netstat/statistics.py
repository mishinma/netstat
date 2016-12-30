""" Module for calculating graph statistics """

import numpy as np


def normalize(distr):
    return distr*1.0/np.sum(distr)


def mean(distr):
    distr_zip = np.vstack((np.arange(distr.size), distr)).T  # Zip indices and values
    mn = np.sum(np.prod(distr_zip, axis=1))
    return mn


def quantile(distr, phi):
    distr_cmf = np.cumsum(distr)
    distr_ge_phi = np.logical_or(np.isclose(distr_cmf, phi), distr_cmf > phi)
    return np.argmax(distr_ge_phi)


def median(distr):
    return quantile(distr, 0.5)


def effective(distr):
    return quantile(distr, 0.9)


def mode(distr):
    return distr.size - 1


def compute_statistics(distr):
    # Trim zeros and normalize
    distr = np.trim_zeros(distr, trim='b')
    distr = normalize(distr)
    return (
        mean(distr),
        median(distr),
        mode(distr),
        effective(distr)
    )
