"""This module contains the definitions for several amplitude-based parameters. 
These include:

    S_a,
    S_q,
    S_sk,
    S_ku,
    S_z, S_t, S_y (all equivalent)
    S_10z,
    S_v,
    S_p,
    S_mean,

These parameters measure properties relating to the distribution of the values 
over surface, such as the average value, the variance of said values, the 
maximum/minimum value, etc.

See http://www.imagemet.com/WebHelp6/Default.htm#RoughnessParameters/Roughness_Parameters.htm
for an in-depth description of each parameter.
"""

import numpy as np

from dr.utils import find_local_maxs, find_local_mins


def S_a(channel: np.array) -> float:
    """Calculates the roughness average

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.

    Returns:
        The roughness average

    Notes:
        Is area independent.
    """
    return np.mean(np.abs(channel))


def S_q(channel: np.array) -> float:
    """Calculates the root mean square parameter, S_q.

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.

    Returns:
        The standard deviation of `channel`.

    Notes:
        Is area independent.
    """
    return channel.std()


def S_sk(channel: np.array, *, std: float = None) -> float:
    """Calculates the surface skewedness.

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.
        std: The standard deviation of `channel. Defaults to None. If None, std 
        is set to `S_q(channel)`.

    Returns:
        The surface skewedness

    Notes:
        Is area independent.
    """
    std = S_q(channel) if std is None else std

    return np.mean(channel ** 3) / std ** 3


def S_ku(channel: np.array, *, std: float = None) -> float:
    """Calculates the surface kurtosis.

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.
        std: The standard deviation of `channel. Defaults to None. If None, std 
        is set to `S_q(channel)`.

    Returns:
        The surface kurtosis

    Notes:
        Is area independent.
    """
    std = S_q(channel) if std is None else std

    return np.mean(channel ** 4) / std ** 4


def S_z(channel: np.array) -> float:
    """Calculates the peak-peak height.

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.

    Returns:
        The peak-peak height

    Notes:
        Is area independent.
    """
    return np.max(channel) - np.min(channel)


# Aliases for S_z
S_y = S_t = S_z


def S_10z(channel: np.array, *,
          local_maxs: np.array = None,
          local_mins: np.array = None) -> float:
    """Calculates the ten point height.

    NOTE: not found in Nanoscope's S parameters

    Args:
        channel: The MxN channel, with the best
        fitting first order plane subtracted.
        local_maxs: The local maximums of `channel`. Defaults to None. If None,
        `local_maxs` is set to `dr.utils.find_local_maxs(channel).` See 
        `dr.utils.find_local_maxs` for more details. 
        local_mins: The local minimums of `channel`. Defaults to None. If None,
        `local_mins` is set to `dr.utils.find_local_mins(channel).` See 
        `dr.utils.find_local_maxs` for more details. 

    Returns:
        The ten point height

    Notes:
        Is area independent.
    """
    # Get 5 highest/lowest local maximums/minimums
    if local_maxs is None:
        local_maxs = find_local_maxs(channel)

    if local_mins is None:
        local_mins = find_local_mins(channel)

    highest = np.sort(local_maxs)[-5:]
    lowest = np.sort(local_mins)[:5]

    # Parameter is undefined if there aren't at least 5 of either
    if len(highest) != 5 or len(lowest) != 5:
        return np.nan

    return np.sum(np.abs(highest)) + np.sum(np.abs(lowest)) / 5


def S_v(channel: np.array) -> float:
    """Calculates the maximum valley depth.

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.

    Returns:
        The maximum valley depth

    Notes:
        Is area independent.
    """
    return np.min(channel)


def S_p(channel: np.array) -> float:
    """Calculates the maximum peak height

    Args:
        channel: The MxN channel, with the best fitting first order plane 
        subtracted.

    Returns:
        The maximum peak height

    Notes:
        Is area independent.
    """
    return np.max(channel)


def S_mean(channel: np.array) -> float:
    """Calculates the mean height of all pixels

    Args:
        channel: The MxN channel. This should be the original channel, not the 
        one with the first-order fitted subtracted. By definition, the mean of 
        the `fit_and_subtract`'ed channel is trivially 0.

    Returns:
        The mean height

    Notes:
        Is area independent.
    """
    return np.mean(channel)
