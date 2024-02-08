#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Function to calculate the wavefront error as a function of wavelength
"""


def phase_to_wfe(wl_point,phase_difference):
    """
    Calculating wavefront error for the phase at a chosen wavelength.

    Parameters
    ----------
    wl_point : TYPE
        Chosen wavelength point.
    phase_difference : TYPE
        DESCRIPTION.

    Returns
    -------
    wavefront_error : TYPE
        Phase difference at wl_point in degrees.

    """
    wavefront_error = (phase_difference/360) * wl_point
    return wavefront_error