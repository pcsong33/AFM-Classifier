"""This module contains the definitions for several hybrid parameters. These 
include:

    S_sc,
    S_2a,
    S_3a,
    S_dr,
    S_dq,
    S_dq6

Hybrid parameters measure gradients over a surface, particularly
ones over local features like local minimums and maximums of the surface (where
a point has a greater value than all 8 of its neighbors).

Calculating these gradients relies on knowing the surface's pixel separation distances
in each dimension (dx and dy), which are calculated as `length of dimension / # of 
pixels in dimension`. These serves as differential distances used to numerically 
differentiate the surfaces.

See http://www.imagemet.com/WebHelp6/Default.htm#RoughnessParameters/Roughness_Parameters.htm
for an in-depth description of each parameter.
"""

import numpy as np

from dr.utils import find_summits, find_local_maxs
from dr.parameters.amplitude import S_z


def S_sc(channel: np.array,
         dx: float,
         dy: float, *,
         strict: bool = False,
         summits: np.array = None) -> float:
    """Calculates the mean summit curvature.

    What constitutes a summit varies across sources. Under one definition, a
    summit merely constitutes a local maximum (i.e. a point that is higher than
    all 8 of its neighbors). A more stringent definition refers to such local
    maximums as 'peaks' and instead requires summits, in addition to being peaks,
    to be separated by at least 1% of the minimum X or Y dimension and be at
    least 5% of S_z above the mean height.

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        strict: Whether or not to apply the stricter definition of a summit.
        Defaults to `False`. If `summits is None` and `strict` is `False`, the
        local maximums are used. If If `summits is None` and `strict` is `True`,
        summits are calculated using the more stringent definition.
        summits: An array of indices indicating locations of summits. Defaults 
        to None. If `summits is not None`, overrides `strict`.

    Returns:
        The mean summit curvature.

    Notes:
        Is area independent.
    """
    # Calculate relevant second-order gradients
    channel_dy, channel_dx = np.gradient(channel, dx, dy)
    channel_dyy, _ = np.gradient(channel_dy, dx, dy)
    _, channel_dxx = np.gradient(channel_dx, dx, dy)

    if summits is None:
        if strict:
            # Use stricter definition
            s_z = S_z(channel)
            summits = find_summits(channel, s_z)
        else:
            # Use loose definition
            summits = find_local_maxs(channel, return_indices=True)

    N = len(summits)
    return (np.sum([(channel_dxx[idx] + channel_dyy[idx])
                    for idx in summits])) / (-2 * N)


def S_2a(channel: np.array, dx: float, dy: float) -> float:
    """Calculates the projected surface area

    Args:
        channel: The MxN channel.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.

    Returns:
        The projected surface area

    Notes:
        Is NOT area independent.
    """
    M, N = channel.shape
    return M * N * dx * dy


def S_3a(channel: np.array, dx: float, dy: float) -> float:
    """Calculates the surface area

    NOTE: does not match Nanoscope's calculation

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.

    Returns:
        The surface area

    Notes:
        Is NOT area independent.
    """
    M, N = channel.shape
    S = 0
    for i in range(M - 1):
        for j in range(N - 1):
            # Could be prettier, but this implements the literal definition
            S += (
                (np.sqrt(dy ** 2 + (channel[i][j] - channel[i][j + 1]) ** 2) + np.sqrt(
                    dy ** 2 + (channel[i + 1][j] - channel[i + 1][j + 1]) ** 2))
                * (np.sqrt(dx ** 2 + (channel[i][j] - channel[i + 1][j]) ** 2) + np.sqrt(dx ** 2 + (channel[i][j + 1] - channel[i + 1][j + 1]) ** 2))) / 4
    return S


def S_dr(channel: np.array,
         dx: float,
         dy: float, *,
         s_2a: float = None,
         s_3a: float = None) -> float:
    """Calculates the surface area ratio

    NOTE: does not match Nanoscope's calculation, as a direct consequence of
    S_3a not matching

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        s_2a: The projected surface area. Defaults to `None`. If `None`, `s_2a`
        is set to `S_2a(channel)`.
        s_3a: The surface area. Defaults to `None`. If `None`, `s_3a` is set to 
        `S_3a(channel)`.

    Returns:   
        The surface area ratio

    Notes:
        Is area independent.
    """
    s_2a = S_2a(channel, dx=dx, dy=dy) if s_2a is None else s_2a
    s_3a = S_3a(channel, dx=dx, dy=dy) if s_3a is None else s_3a
    return 100 * (s_3a - s_2a) / s_2a


def S_dq(channel: np.array, dx: float, dy: float) -> float:
    """Calculates the root mean square gradient

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.

    Returns:
        The root mean square gradient

    Notes:
        Is area independent.
    """
    M, N = channel.shape
    channel_dy, channel_dx = np.gradient(channel, dx, dy)
    S = np.sqrt((np.sum(channel_dy ** 2) + np.sum(channel_dx ** 2)) / (M * N))

    # While the definition doesn't require converting the summation into degrees
    # like this, NanoScope Analysis does, so we follow its convention
    return np.degrees(np.arctan(S))


def S_dq6(channel: np.array, dx: float, dy: float) -> float:
    """Calculates the area root mean square slope

    NOTE: not found in Nanoscope's parameters

    Args:
        channel : The MxN channel, with the best fitting first order plane 
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.

    Returns:
        float: The area root mean square slope

    Notes:
        Is area independent.
    """
    M, N = channel.shape

    def calculate_summand(x_idx, y_idx):
        # Could be prettier, but this implements the literal definition
        left_term = ((-channel[x_idx - 3, y_idx]
                      + 9 * channel[x_idx - 2, y_idx]
                      - 45 * channel[x_idx - 1, y_idx]
                      + 45 * channel[x_idx + 1, y_idx]
                      - 9 * channel[x_idx + 2, y_idx]
                      + channel[x_idx + 3, y_idx]) * (1 / (60 * dx))) ** 2
        right_term = ((-channel[x_idx, y_idx - 3]
                       + 9 * channel[x_idx, y_idx - 2]
                       - 45 * channel[x_idx, y_idx - 1]
                       + 45 * channel[x_idx, y_idx + 1]
                       - 9 * channel[x_idx, y_idx + 2]
                       + channel[x_idx, y_idx + 3]) * (1 / (60 * dy))) ** 2
        return np.sqrt(left_term + right_term)

    S = np.mean(
        [calculate_summand(x_idx, y_idx)
         for x_idx in range(3, M - 3)
         for y_idx in range(3, N - 3)])
    return np.degrees(np.arctan(S))
