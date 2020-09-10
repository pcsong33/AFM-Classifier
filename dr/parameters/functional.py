import numpy as np

from dr.utils import bearing_height, area_above, func_shift, find_decline, trap_rule
from dr.parameters.amplitude import S_q


def S_bi(channel: np.array, *, std: float = None) -> float:
    """Calculates the surface bearing index parameter, S_bi.

    Args:
        channel: The MxN channel, with the best fitting first order plane
        subtracted.
        std: The standard deviation of `channel. Defaults to None. If None, std
        is set to `S_q(channel)`.
    Returns:
        Returns the root mean square value divided by the height of the bearing area curve at ratio 5%.
    Notes:
        Is area independent.
    """
    std = S_q(channel) if std is None else std
    return std / bearing_height(channel, .05)


def S_ci(channel: np.array, dx: float, dy: float, *, std: float = None) -> float:
    """Calculates the core fluid retention index, S_ci. The values v_five and v_eight describe
        the void area above the bearing area ratio curve and under the horizontal line
        drawn at bearing area ratios of 5% and 8% respectively. The numerator
        (v_five - v_eight) describes the air in the core zone.

    Args:
        channel: The MxN channel, with the best fitting first order plane
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        std: The standard deviation of `channel. Defaults to None. If None, std
        is set to `S_q(channel)`.
    Returns:
        Returns the core fluid retention index.
    Notes:
        Is area independent.
    """
    M, N = channel.shape
    std = S_q(channel) if std is None else std
    v_eight = area_above(channel, 0.8, 1)
    v_five = area_above(channel, 0.05, 1)
    return (v_five - v_eight) / std


def S_vi(channel: np.array, dx: float, dy: float, *, std: float = None) -> float:
    """Calculates the valley fluid retention index, S_vi. Similar to in the calculation of S_ci,
        the value v_eight describes the void area above the bearing area ratio curve and
        under the horizontal line drawn at bearing area ratios of 8%. The numerator (v_eight)
        describes the air in the valley zone.

    Args:
        channel: The MxN channel, with the best fitting first order plane
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
        std: The standard deviation of `channel. Defaults to None. If None, std
        is set to `S_q(channel)`.
    Returns:
        Returns the valley fluid retention index.
    Notes:
        Is area independent.
    """
    M, N = channel.shape
    std = S_q(channel) if std is None else std
    v_eight = area_above(channel, 0.8, 1)
    return v_eight / std


def S_pk(channel: np.array, dx: float, dy: float) -> float:
    """Calculates the reduced summit height, S_pk. Calculates the height of the triangle
        created from drawing a straight line drawn from the intersection point between
        the bearing area ratio curve at 0% and the upper horizontal line of the least
        mean squares line.

    Args:
        channel: The MxN channel, with the best fitting first order plane
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
    Returns:
        Returns the reduced summit height.
    Notes:
        Is area independent.
    """
    bin_edges, percents = func_shift(channel)
    left_height, _ = find_decline(channel)
    length_index = np.where(bin_edges <= left_height)[0][0]
    length = percents[length_index]
    curve_area = trap_rule(channel, length, over=False)
    area_between = curve_area - (left_height * length)
    height = 2 * area_between / length
    return height


def S_vk(channel: np.array) -> float:
    """Calculates the reduced summit height, S_pk. This parameter calculates
        the height of the triangle created from drawing a straight line drawn
        from the intersection point between the bearing area ratio curve at
        100% and the lower horizontal line of the least mean squares line.

    Args:
        channel: The MxN channel, with the best fitting first order plane
        subtracted.
        dx: The pixel separation distance along the x-dimension.
        dy: The pixel separation distance along the y-dimension.
    Returns:
        Returns the reduced summit height.
    Notes:
        Is area independent.
    """
    bin_edges, percents = func_shift(channel)
    _, right_height = find_decline(channel)
    length_index = np.where(bin_edges <= right_height)[0][0]
    length = percents[length_index]
    curve_area = trap_rule(channel, length, over=True)
    area_between = (right_height * (1 - length)) - curve_area
    height = 2 * area_between / (1 - length)
    return height


def S_k(channel: np.array) -> float:
    """Calculates the core roughness depth, s_k. This parameter calculates
    the height of the triangle created from drawing a straight line drawn
    from the intersection point between the bearing area ratio curve at
    100% and the lower horizontal line of the least mean squares line.

    Args:
        channel: The MxN channel, with the best fitting first order plane
        subtracted.
    Returns:
        Returns the reduced valley depth.
    Notes:
        Is area independent.
    """
    zero, hundred = find_decline(channel)
    return zero - hundred


def S_dc(channel: np.array, l: int, h: int) -> np.array:
    """Set of parameters describing height differences between certain bearing area ratios;
    l and h denotes the lower and upper bearing area ratios of the interval. Sdcl is the height
    value at bearing area ratio at l % and Sdch is the height at h % """

    return bearing_height(channel, l / 100) - bearing_height(channel, h / 100)
