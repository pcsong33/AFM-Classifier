"""This module contains a number of helper functions for use in calculating surface
parameters. 

While some of the utilities are specific to certain types of parameters,
(for instance, get_fourier_spectrum, angular_amplitude_sum, etc are used for 
spatial parameters, whereas bearing_height and trap_rule are used for functional 
parameters), others see use in every parameter."""

import itertools
from functools import partial, reduce

import numpy as np
from scipy.linalg import lstsq
from scipy.fft import fftfreq, fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import LinearNDInterpolator
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from dr.load import load_table

NOT_QUADRATIC_EXN = Exception('fit_and_subtract: image is not quadratic')


def fit_and_subtract(channel: np.array) -> np.array:
    """Finds the best first-order fit of the data using least-squares and subtracts it
      from the data

    Args:
        channel: An MxM or (M^2x1) float array.

    Returns:
        The transformed Mxm channel
    """
    # Generate points
    shape = channel.shape
    if len(shape) < 2:
        M = int(np.sqrt(shape))
        if M - int(M) != 0:
            raise NOT_QUADRATIC_EXN
    else:
        M, N = shape
        if M != N:
            raise NOT_QUADRATIC_EXN

    dim = np.arange(0, M, 1)
    points = np.array(list(itertools.product(dim, dim)))

    # Add bias (intercept) terms
    A = np.c_[points, np.ones(points.shape[0])]
    target = channel.flatten()

    # Solve for coefficients of the plane and construct the fit
    C, _, _, _ = lstsq(A, target)
    fit = C[0] * points[:, 0] + C[1] * points[:, 1] + C[2]
    return (target - fit).reshape(M, M)


def get_deltas(channel: np.array, len_x: float, len_y: float) -> (float, float):
    """Calculate pixel separation distances for each dimension

    Args:
        channel:  An MxN float array.
        len_x (float): Length of channel (in micrometers, millimeters, etc.)
        along the x-dimension
        len_y (float): Length of channel (in micrometers, millimeters, etc.)
        along the y-dimension

    Returns:
        (float, float): The two pixel separation distances
    """
    M, N = channel.shape
    return M / len_x, N / len_y


def neighbors(a, r_idx, c_idx):
    """Returns the neighbors of the point at a[r_idx, c_idx]"""
    x, y = a.shape

    def check_neighbors(i, j): return all(
        (i >= 0, i < x, j >= 0, j < y, i != r_idx or j != c_idx))
    neighbors = [[a[i][j] for j in range(c_idx - 1, c_idx + 2) if check_neighbors(i, j)]
                 for i in range(r_idx - 1, r_idx + 2)]
    return np.array(sum(neighbors, []))


def is_local_extrema(a, i, j, compare):
    """Determines if a[i, j] is a local extrema"""
    nbs = neighbors(a, i, j)
    if len(nbs) != 8:
        return False

    candidate = a[i, j]
    return all((compare(candidate, nb) for nb in nbs))


def find_local_extrema(a: np.array, compare, return_indices: bool = False) -> np.array:
    """Finds local extrema of a 2D array. Local extrema are found with respect
    to a point's 8 neighbors.

    Args:
        a: The 2D array.
        compare: Comparison function to use.
        return_indices: Whether or not to return indices instead
        of the values of the extrema themselves. Defaults to False.

    Returns:
        The local extrema (or their indices)
    """
    x, y = a.shape
    if return_indices:
        return [(i, j) for j in range(y) for i in range(x) if is_local_extrema(a, i, j, compare)]
    return np.array([a[i, j] for j in range(y) for i in range(x) if is_local_extrema(a, i, j, compare)])


find_local_maxs = partial(find_local_extrema, compare=np.greater)
find_local_mins = partial(find_local_extrema, compare=np.less)


def find_summits(a: np.array, S_z: float) -> list:
    """Finds summits of a channel. Summits are constrained to be separated by at
    least 1% of the minimum X or Y dimension and must also be at least 5% of S_z
    above the mean height.


    Args:
        a: The 2D channel.
        S_z: The maximum peak height.

    Returns:
        The indices (as tuples) of the summits of the channel.
    """
    # All summits must firstly be local_maxs (i.e. peaks)
    peaks_idxs = find_local_maxs(a, return_indices=True)
    peaks_heights = [a[idx] for idx in peaks_idxs]
    peaks = zip(peaks_idxs, peaks_heights)

    # The minimum height threshold for a peak to be considered a summit
    threshold = (0.05 * S_z) + np.mean(a)
    last_idx = np.inf

    # Determines minimum distance by which peaks must be separated to be considered summits
    MIN_DIMENSION = np.argmin(a.shape)
    MIN_DIFF = 0.01 * a.shape[np.argmin(a.shape)]

    summits = []
    for peak_idx, peak_height in peaks:
        # Consider a peak to be a summit iff its height is above the threshold and
        # it is sufficiently separated from the last peak along the minimum dimension
        if np.abs(peak_idx[MIN_DIMENSION] - last_idx) >= MIN_DIFF and peak_height >= threshold:
            summits.append(peak_idx)
            last_idx = peak_idx[MIN_DIMENSION]

    return summits


def bearing_area(x):
    """Bearing area curve (Abott Curve) is the inverse CDF of the surface profile's height.
     Returns array of x-values(bin_edges) and y-values(percents)"""
    num_bins = 500
    counts, bin_edges = np.histogram(x, bins=num_bins)
    cdf = np.cumsum(counts)
    percents = 1 - cdf/cdf[-1]
    return bin_edges[::-1], percents[::-1]


def func_shift(x):
    """Shifts function vertically so all y-values are positive"""
    bin_edges, percents = bearing_area(x)
    if min(bin_edges) < 0:
        bin_edges += abs(min(bin_edges))
    return bin_edges, percents


def bearing_height(x, xval):
    """Determines the value of the bearing area curve for a specific value along the x-axis"""
    bin_edges, percents = bearing_area(x)
    if xval == 1:
        return bin_edges[-1]
    result = np.where(percents >= xval)
    return bin_edges[result[0][0]]


def trap_rule(x, a, over=True):
    """Calculates area under curve, using trapezoid rule. If over = True, integration starts at a.
    If over = False, integration ends at a"""
    bin_edges, percents = func_shift(x)
    if over:
        result = np.where(percents >= a)[0]
    else:
        result = np.where(percents <= a)[0]
    y_values = bin_edges[result]
    x_values = percents[result]
    area = np.trapz(y_values, x=x_values)
    # breakpoint()
    return area


def area_above(x, a, b):
    """Calculates area above curve, below horizontal line, used in S_ci and S_vi calculations"""
    bin_edges, percents = func_shift(x)
    result = np.where(percents >= a)
    area_between = ((bin_edges[result[0][0]] * (b-a)
                     ) - trap_rule(x, a, over=True))
    return area_between


def point_slope(x, xvalue, yvalue, slope):
    """Returns y-value of points at the vertical axes, the intersection at 0% and 100%"""
    return slope * (x - xvalue) + yvalue


def find_decline(x):
    """Calculates flattest 40% slope of bearing area curve. Returns vertical axis intersection heights 
    using the flattest 40% slope. """
    bin_edges, percents = func_shift(x)
    sixty_index = np.where(percents >= 0.6)[0][0]
    slope_list = []
    point_list = []
    i_index = 0
    for i in percents[:sixty_index]:
        forty_mask = np.where(percents >= 0.4 + i)
        if len(forty_mask[0]) == 0:
            forty_index = -1
        else:
            forty_index = forty_mask[0][0]
        slope = (bin_edges[forty_index] - bin_edges[i_index]) / 0.4
        i_index += 1
        slope_list.append(slope)
        point_list.append(i)
    flat_slope = max(slope_list)
    point_index = slope_list.index(flat_slope)
    xval = point_list[point_index]
    yval = bin_edges[point_index]
    zero_axis = point_slope(0, xval, yval, flat_slope)
    hundred_axis = point_slope(1, xval, yval, flat_slope)
    return zero_axis, hundred_axis


def get_fourier_spectrum(channel: np.array) -> (np.array, np.array, np.array):
    """Applies 2D DFFT to a channel and shifts it such that the DC-component
    (zero-frequency component) is centered

    Args:
        channel: The MxM channel to be transformed.

    Returns:
        The transformed and shifted spectrum of the channel.
        Frequencies along x-dimension
        Frequencies along y-dimension (equal to those along x-dimension)
    """

    M, _ = channel.shape

    xf = fftshift(fftfreq(M))
    yf = fftshift(fftfreq(M))
    return fftshift(fft2(channel)), xf, yf


def construct_interpolator(values: np.array):
    """Convenience function for creating interpolators"""
    M, _ = values.shape
    dim = np.arange(0, M, 1)
    points = np.array(list(itertools.product(dim, dim)))
    return LinearNDInterpolator(points, values.reshape(M ** 2))


def angular_amplitude_sum(spectrum: np.array,
                          angle: float,
                          interpolator=None,
                          return_profile=False) -> float:
    """Calculates the amplitude sum along a specified radial line.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at 
        the center.
        angle: The angle of the radial line, in radians.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None, 
        bilinear interpolation is used. Recommended if calling repeatedly.
        return_profile: Whether to return the amplitudes themselves
        instead of their sum. Defaults to False.

    Returns:
        The amplitude sum
    """
    M, _ = spectrum.shape

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    # Both collections are only relevant when return_profiles is True
    freqs = []
    profile = []

    for i in range(1, M // 2):
        p = min((M // 2) + i * np.cos(angle), M - 1)
        q = min((M // 2) + i * np.sin(angle), M - 1)

        # Check if interpolation is necessary by checking whether p and q are integers
        p_i = int(p)
        q_i = int(q)

        if (p - p_i != 0 or q - q_i != 0):
            amplitude = interpolator(p, q)
        else:
            amplitude = spectrum[p_i, q_i]

        freqs.append(p_i)
        profile.append(np.abs(amplitude))

    if return_profile:
        return freqs, np.array(profile)
    else:
        return np.sum(profile)


def radial_amplitude_sum(spectrum: np.array,
                         radius: float,
                         M: int = None,
                         interpolator=None) -> float:
    """Calculates the amplitude sum along a specified semicircle.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component
        is at the center.
        radius: The radius of the semicircle.
        interpolator: An interpolation function called to
        provide interpolated values for the spectrum where necessary. If None, 
        bilinear interpolation is used. Recommended if calling repeatedly.

    Returns:
        The amplitude sum
    """
    M, _ = spectrum.shape
    S = 0

    if interpolator is None:
        interpolator = construct_interpolator(spectrum)

    for i in range(1, M):
        # Restrict p and q to be less than or equal to M - 1
        p = min((M // 2) + radius * np.cos(i * np.pi / M), M - 1)
        q = min((M // 2) + radius * np.sin(i * np.pi / M), M - 1)

        # Check if interpolation is necessary by checking whether p and q are integers
        p_i = int(p)
        q_i = int(q)

        if (p - p_i != 0 or q - q_i != 0):
            amplitude = interpolator(p, q)
        else:
            amplitude = spectrum[p_i, q_i]

        S += np.abs(amplitude)
    return S


def get_acf(spectrum: np.array) -> (np.array, (int, int)):
    """Given a spectrum, returns the autocorrelation function.

    Args:
        spectrum: The MxM Fourier spectrum, shifted so that DC-component is at 
        the center.

    Returns:
        The autocorrelation function, scaled between -1 and 1.
        The indices corresponding to where lag is zero.
    """
    M, _ = spectrum.shape

    # Obtain autovariance, by Weiner-Khinchin theorem
    power_spectrum = np.abs(spectrum) ** 2
    autocovariance = np.real(ifft2(ifftshift(power_spectrum))) / M ** 2
    variance = autocovariance.flat[0]

    # Normalize autocovariance to obtain autocorrelation
    # We also shift it s.t. the zero-lag component is in the center
    autocorrelation = fftshift(autocovariance / variance)

    zero_lag_indices = np.unravel_index(autocorrelation.argmax(),
                                        autocorrelation.shape)

    return autocorrelation, zero_lag_indices


def query_acf(acf: np.array,
              correlation: float,
              tol: float,
              max_iter: int = 5) -> (np.array, np.array):
    """Returns indices of autocorrelation function where the correlation is within
    correlation +- tol. If none are found, the tolerance is increased until at 
    least one candidate is found or max_iter is exceeded.

    Args:
        acf: The autocorrelation function.
        correlation: The correlation for which to find corresponding  indices.
        tol: Self-explanatory.
        max_iter: The maximum number of iterations for increasing `tol`.

    Returns:
        The indices of the ACF where it is equal to correlation +- tol. May be empty.
    """
    # Get indices where the correlation is close `correlation +- tol`
    # If none are found, increase tol until at least 1 is found or max_iter is exceeded
    correlation_mask = np.isclose(acf, correlation, tol)

    i = 0
    while correlation_mask.sum() == 0 and i <= max_iter:
        tol *= 10
        correlation_mask = np.isclose(acf, correlation, tol)
        i += 1

    return np.where(correlation_mask)
