"""This module contains the definitions for several spatial-based parameters. 
These include:

    S_ds,
    S_td,
    S_tdi,
    S_rw,
    S_rw,
    S_hw,
    S_fd,
    S_cl,
    S_tr

With the exception of S_ds, all spatial parameters are calculated with respect 
to the Fourier spectrum of a surface, which can be easily obtained applying a
Fast Fourier Transform (FFT) algorithm to the surface. For best results, the zero-frequency
component of the spectrum should be shifted to the center as well. `scipy.fft` is
a good library for doing all of this. Calculating spatial parameters requires that 
the surface be quadratic (i.e. square).

`M` appears as an argument to the definitions of many spatial parameters. Certain
spatial parameters (S_td and S_rw, for instance) rely on drawing semicircles or 
radial lines across a surface, and `M` specifies how many such geometries are drawn. 
High `M` makes the calculation of a spatial parameter more granular (but takes more 
time) while low `M` makes it rougher (but is faster). Generally, `M` should not 
exceed the size of either of the dimensions of a surface, and for a NxN surface, 
`M = N` is a good default. However, `M` should be set constant when dealing with 
surfaces of varying sizes to preserve area independence. So, given an NxN surface 
and a (N - 1)x(N - 1) surface, if you want to be able to accurately compare them 
in terms of their spatial parameters, you should set `M = N` when calculating 
either surface for best results. 

See http://www.imagemet.com/WebHelp6/Default.htm#RoughnessParameters/Roughness_Parameters.htm
for an in-depth description of each parameter.
"""

import numpy as np
from scipy.stats import linregress

from dr.utils import (find_summits,
                      find_local_maxs,
                      construct_interpolator,
                      angular_amplitude_sum,
                      radial_amplitude_sum,
                      get_acf,
                      query_acf)

from dr.parameters.amplitude import S_z


def S_ds(channel: np.array,
         dx: float,
         dy: float, *,
         strict: bool = False,
         summits: np.array = None) -> float:
    """Calculates the density of summits.

    What constitutes a summit varies across sources. Under one definition, a
    summit merely constitutes a local maximum (i.e. a point that is higher than
    all 8 of its neighbors). A more stringent definition refers to such local
    maximums as 'peaks' and instead requires summits, in addition to being peaks,
    to be separated by at least 1% of the minimum X or Y dimension and be at
    least 5% of S_z above the mean height.

    Args:
        channel: The MxN channel, with the best
        fitting first order plane subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        strict: Whether or not to apply the stricter definition of a summit.
        Defaults to `False`. If `summits is None` and `strict` is `False`, the
        local maximums are used. If If `summits is None` and `strict` is `True`,
        summits are calculated using the more stringent definition.
        summits: An array of indices indicating locations of summits. Defaults
        to None. If `summits is not None`, overrides `strict`.

    Returns:
        The density of summits

    Notes:
        Is area independent.
    """
    M, N = channel.shape

    if summits is None:
        if strict:
            # Use stricter definition
            s_z = S_z(channel)
            summits = find_summits(channel, s_z)
        else:
            # Use loose definition
            summits = find_local_maxs(channel, return_indices=True)

    return len(summits) / (M * N * dx * dy)


def S_td(spectrum: np.array,
         M: int = None, *,
         amplitudes: np.array = None,
         interpolator=None) -> float:
    """Calculates the texture direction.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at
        the center.
        M: The number of equiangularly separated lines to calculate the spectrum
        for. Defaults to `None`. If `None`, `spectrum.shape[0]` is used.
        amplitudes: The sums of the amplitudes along `M` angles from `0` to
        `M / 2`. Defaults to `None`. If `None`, the amplitudes are calculated.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None,
        bilinear interpolation is used. Recommended if calling repeatedly.

    Returns:
        The angle of the dominating texture.

    Notes:
        Is area independent.
    """
    dim, _ = spectrum.shape

    if M is None:
        M = dim

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    if amplitudes is None:
        amplitudes = [angular_amplitude_sum(spectrum=spectrum,
                                            angle=i * np.pi / M,
                                            interpolator=interpolator)
                      for i in np.arange(0, M)]

    # Get the angle along which the amplitude sum is the greatest
    angle_max = np.argmax(amplitudes) * np.pi / M

    # NOTE: depending on which source you consult, the texture direction is either
    # angle_max as is, or angle_max - (pi / 2) (i.e. an angle perpendicular to it)
    # So long we're consistent with our choice, it shouldn't affect the richness
    # of this feature

    return np.degrees(angle_max)


def S_tdi(spectrum: np.array,
          M: int = None, *,
          amplitudes: np.array = None,
          interpolator=None) -> float:
    """Calculates the texture direction index.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at
        the center.
        M: The number of equiangularly separated lines to calculate the
        spectrum for. Defaults to `None`. If `None`, `spectrum.shape[0]` is used.
        amplitudes: The sums of the amplitudes along `M` angles from `0` to
        `M / 2`. Defaults to `None`. If `None`, the amplitudes are calculated.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None,
        bilinear interpolation is used. Recommended if calling repeatedly.

    Returns:
        The texture direction index

    Notes:
        Is area independent.
    """
    dim, _ = spectrum.shape

    if M is None:
        M = dim

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    if amplitudes is None:
        amplitudes = [angular_amplitude_sum(spectrum=spectrum,
                                            angle=i * np.pi / M,
                                            interpolator=interpolator)
                      for i in np.arange(0, M)]

    A_max = np.max(amplitudes)
    return np.sum(amplitudes) / (M * A_max)


def S_rw(spectrum: np.array,
         dx: float,
         M: int = None, *,
         amplitudes: np.array = None,
         interpolator=None) -> float:
    """Calculates the dominating radial wavelength

    NOTE: not found in Nanoscope's S parameters.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at
        the center.
        dx: The pixel separation distance along the x-dimension.
        M: 2x number of equidistantly separated semicircles to calculate the
        spectrum for (i.e. 512 yields 256 circles) Defaults to `None`. If
        `None`, `spectrum.shape[0]` is used.
        amplitudes: The sums of the amplitudes along `M / 2` semicircles from `0`
        to  `M`. Defaults to `None`. If `None`, the amplitudes are calculated.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None,
        bilinear interpolation is used. Recommended if calling repeatedly.


    Returns:
        The radial wavelength

    Notes:
        Unclear if area independent.
    """

    dim, _ = spectrum.shape

    if M is None:
        M = dim

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    if amplitudes is None:
        amplitudes = [radial_amplitude_sum(spectrum=spectrum,
                                           radius=i,
                                           interpolator=interpolator)
                      for i in np.linspace(1, dim // 2, M // 2)]

    # Get radius for which `radial_amplitude_sum` is the greatest is the great
    r_max = np.linspace(1, dim // 2, M // 2)[np.argmax(amplitudes)]
    return dx * (M - 1) / r_max


def S_rwi(spectrum: np.array,
          M: int = None, *,
          amplitudes: np.array = None,
          interpolator=None) -> float:
    """Calculates the radial wave index

    NOTE: not found in Nanoscope's S parameters.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at
        the center.
        M: 2x number of equidistantly separated semicircles to calculate the
        spectrum for (i.e. 512 yields 256 circles) Defaults to `None`.
        If `None`, `spectrum.shape[0]` is used.
        amplitudes: The sums of the amplitudes along `M / 2` semicircles from `0`
        to  `M`. Defaults to `None`. If `None`, the amplitudes are calculated.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None,
        bilinear interpolation is used. Recommended if calling repeatedly.


    Returns:
        The radial wave index

    Notes:
        Unclear if area independent.
    """
    dim, _ = spectrum.shape

    if M is None:
        M = dim

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    if amplitudes is None:
        amplitudes = [radial_amplitude_sum(spectrum=spectrum,
                                           radius=i,
                                           interpolator=interpolator)
                      for i in np.linspace(1, dim // 2, M // 2)]

    amplitude_max = np.max(amplitudes)
    return (2 / (M * amplitude_max)) * np.sum(amplitudes)


def S_hw(spectrum: np.array,
         dx: float,
         M: int = None, *,
         amplitudes: np.array = None,
         interpolator=None) -> float:
    """Calculates the mean half wavelength

    NOTE: not found in Nanoscope's S parameters.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at 
        the center.
        dx: The pixel separation distance along the x-dimension.
        M: 2x number of equidistantly separated semicircles to calculate the 
        spectrum for (i.e. 512 yields 256 circles) Defaults to `None`.
        If `None`, `spectrum.shape[0]` is used.
        amplitudes: The sums of the amplitudes along `M / 2` semicircles from `0` 
        to  `M`. Defaults to `None`. If `None`, the amplitudes are calculated.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None, 
        bilinear interpolation is used. Recommended if calling repeatedly.


    Returns:
        The mean half wavelength

    Notes:
        Unclear if area independent.
    """
    dim, _ = spectrum.shape

    if M is None:
        M = dim

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    if amplitudes is None:
        amplitudes = [radial_amplitude_sum(spectrum=spectrum,
                                           radius=i,
                                           interpolator=interpolator)
                      for i in np.linspace(1, dim // 2, M // 2)]

    # Get for integrated amplitude sum for every semicircle
    integrated_amplitude_sums = [np.sum(amplitudes[:r])
                                 for r in range(1, M // 2)]

    # Apply definition of r_05; see definition for S_hw
    r_05 = np.argmin(
        np.abs((integrated_amplitude_sums / integrated_amplitude_sums[-1]) - 0.5))

    # Radii may have a fractional component, so the index provided by argmin may
    # not truly indicate r_05 yet
    r_05 = np.linspace(1, dim // 2, M // 2)[r_05]
    return dx * (dim - 1) / r_05


def S_fd(spectrum: np.array,
         xf: np.array,
         M: int = None, *,
         amplitude_profiles: np.array = None,
         epsilon: float = 0.001,
         interpolator=None) -> float:
    """Calculates the fractal dimension.

    NOTE: not found in Nanoscope's S parameters.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at 
        the center.
        xf: Frequencies along x-dimension
        M: The number of equiangularly separated lines to calculate the
        spectrum for. Defaults to `None`. If `None`, `spectrum.shape[0]` is used.
        If `None`, `spectrum.shape[0]` is used.
        amplitude_profiles: The profiles of the amplitudes along `M / 2` semicircles from 
        `0` to  `M`. Defaults to `None`. If `None`, the amplitude profiles are calculated.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None, 
        bilinear interpolation is used. Recommended if calling repeatedly.


    Returns:
        The fractal dimension

    Notes:
        Is area independent.
    """
    dim, _ = spectrum.shape

    if M is None:
        M = dim

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    if amplitude_profiles is None:
        amplitude_profiles = [angular_amplitude_sum(spectrum=spectrum,
                                                    angle=i * np.pi / M,
                                                    interpolator=interpolator,
                                                    return_profile=True)
                              for i in np.arange(0, M)]
    # Get the fractal dimension along every angle
    fds = []
    for freqs, profile in amplitude_profiles:
        # Add small error term to prevent log(0) errors
        xf_log = np.log([np.abs(xf[coord]) + epsilon for coord in freqs])
        profile_log = np.log(profile + epsilon)

        # Calculate fractal dimension along this angle
        slope = linregress(xf_log, profile_log).slope
        fds.append((slope + 6) / 2)

    # Use nanmean in case epsilon trick did not work
    return np.nanmean(fds)


def S_cl(spectrum: np.array,
         correlation: float,
         dx: float,
         dy: float,
         tol: float = 0.001, *,
         acf: np.array = None,
         zero_lag_idx: (int, int) = None) -> float:
    """Calculate correlation length for the fastest decay to a specified correlation
    on the ACF of a channel.

    NOTE: The results obtained by this and Nanoscope differ, but not by a huge
    amount. The two implementations could be following the same procedure, but
    because of the imprecision behind using a discrete autocorrelation function
    one must specify a certain tolerance for the correlation (i.e. 20% +- 0.1 %).
    Because we do not know what tolerance Nanoscope uses, it's difficult to
    exactly replicate its results

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component
        is at the center.
        correlation: The correlation to which the decay is measured.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        tol: Tolerance for the correlation. Defaults to 0.01.
        acf: The autocorrelation function for `spectrum`.
        zero_lag_idx: The location of the zero-lag component for `autocorrelation`.

    Returns:
        The correlation length

    Notes:
        Is area independent, with some modifications.
    """
    M, _ = spectrum.shape

    # Get autocorrelation function (zero lag index is typically in the center)
    if acf is None or zero_lag_idx is None:
        acf, zero_lag_idx = get_acf(spectrum)
    x0, y0 = zero_lag_idx

    # Get indices where the acf == correlation +- tol
    xidx, yidx = query_acf(acf, correlation=correlation, tol=tol)

    # Result is meaningless if no indices are found
    if len(xidx) == 0:
        return np.nan

    # Calculate lengths
    radii = [np.sqrt(((x - x0) * dx) ** 2 + ((y - y0) * dy) ** 2)
             for x, y in zip(xidx, yidx)]

    # We normalize the correlation length w.r.t. distance to the edge
    return np.sqrt(((0 - x0) * dx) ** 2 + ((0 - y0) * dy) ** 2) / M


def S_tr(spectrum: np.array,
         correlation: float,
         dx: float,
         dy: float,
         tol: float = 0.001, *,
         acf: np.array = None,
         zero_lag_idx: (int, int) = None) -> float:
    """Calculate texture aspect ratio for a specified correlation.

    Texture aspect ratio is defined as the ratio of the correlation length
    (i.e. distance of fastest decay) and of the distance of slowest decay.

    NOTE: The results obtained by this and Nanoscope differ, but not by a huge
    amount. The two implementations could be following the same procedure, but
    because of the imprecision behind using a discrete autocorrelation function
    one must specify a certain tolerance for the correlation (i.e. 20% +- 0.1 %).
    Because we do not know what tolerance Nanoscope uses, it's difficult to
    exactly replicate its results

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component
        is at the center.
        correlation: The correlation to which the decay is measured.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        tol: Tolerance for the correlation. Defaults to 0.01.
        acf: The autocorrelation function for `spectrum`.
        zero_lag_idx: The location of the zero-lag component for `autocorrelation`.

    Returns:
        The texture aspect ratio

    Notes:
        Is area independent.
    """
    M, _ = spectrum.shape

    # Get autocorrelation function (zero lag index is typically in the center)
    if acf is None or zero_lag_idx is None:
        acf, zero_lag_idx = get_acf(spectrum)
    x0, y0 = zero_lag_idx

    # Get indices where the acf == correlation +- tol
    xidx, yidx = query_acf(acf, correlation=correlation, tol=tol)

    # Result is meaningless if no indices are found
    if len(xidx) == 0:
        return np.nan

    # Calculate lengths and sort
    radii = sorted([np.sqrt(((x - x0) * dx) ** 2 + ((y - y0) * dy) ** 2)
                    for x, y in zip(xidx, yidx)])

    # Multiplying by M creates area independence
    return radii[0] * M / radii[-1]
