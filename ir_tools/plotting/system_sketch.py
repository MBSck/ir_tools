from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


def create_grid(
    dim: int,
    pixel_size: u.au,
    inclination: Optional[u.one] = None,
    position_angle: Optional[u.deg] = None,
) -> np.ndarray:
    """Creates the sketch's grid.

    Parameters
    ----------
    dim : float

    Returns
    -------
    image_grid : astropy.units.au
        A two dimensional linear or logarithmic grid.
    """
    xx = np.linspace(-0.5, 0.5, dim, endpoint=False) * pixel_size * dim
    xx, yy = np.meshgrid(xx, xx)
    if position_angle is not None:
        xx_rot = xx * np.cos(position_angle) - yy * np.sin(position_angle)
        yy_rot = xx * np.sin(position_angle) + yy * np.cos(position_angle)
    else:
        xx_rot, yy_rot = xx, yy

    if inclination is not None:
        xx_rot *= np.cos(inclination)

    return np.hypot(xx_rot, yy_rot)


def compute_eccentric_anomaly(eccentric_anomaly, mean_anomaly, eccentricity):
    """Computes the eccentric anomaly."""
    return eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly


def compute_true_anomaly(eccentric_anomaly, eccentricity):
    return 2 * np.arctan(
        np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(eccentric_anomaly / 2)
    )


# TODO: Fully understand this
def solve_eccentric_anomaly(mean_anomaly, eccentricity):
    """Solves the eccentric anomaly numerically

    NOTES
    -----
    A method defined similar to the binary search
    from J. Meeus 1991 Chapter 29 Third Method.
    """
    mean_anomaly %= 2 * np.pi
    if mean_anomaly > np.pi:
        mean_anomaly = 2 * np.pi - mean_anomaly
        sign = -1
    else:
        sign = 1

    e0, d0 = np.pi / 2, np.pi / 4
    for _ in range(33):
        m1 = e0 - eccentricity * np.sin(e0)
        e0 = e0 + d0 * np.sign(mean_anomaly - m1)
        d0 /= 2

    return e0 * sign


# TODO: Code this in julia -> Should look very much nicer
def get_orbit(
    period: u.yr,
    eccentricity: u.one,
    semi_major_axis: u.au,
    inclination: u.deg,
    periapsis_time: u.yr,
    argument_periapsis: u.deg,
    long_ascending_node: u.deg,
):
    rhos, thetas = [], []
    for epoch in np.linspace(periapsis_time, periapsis_time + period, 1024):
        mean_anomaly = 2 * np.pi * (epoch - periapsis_time) / period
        eccentric_anomaly = solve_eccentric_anomaly(
            mean_anomaly.value, eccentricity.value
        )
        true_anomaly = compute_true_anomaly(eccentric_anomaly, eccentricity)

        # NOTE: Up to here soley Kepler, see Chapter 55 for these calculations
        theta = np.arctan2(
            np.cos(true_anomaly + argument_periapsis.to(u.rad)),
            np.sin(true_anomaly + argument_periapsis.to(u.rad)) * np.cos(inclination),
        )
        theta = (theta + long_ascending_node.to(u.rad)).to(u.deg)

        radius = semi_major_axis * (1 - eccentricity * np.cos(eccentric_anomaly))
        rho = (
            radius
            * np.sin(true_anomaly + argument_periapsis.to(u.rad))
            / np.cos(theta - long_ascending_node)
        )

        rhos.append(rho)
        thetas.append(theta)

    return u.Quantity(rhos), u.Quantity(thetas)


def partition_list(lst):
    if len(lst) % 2 != 0:
        lst.append(None)
    return [(lst[i], lst[i + 1]) for i in range(0, len(lst), 2)]


def disc_image(
    radii: np.ndarray,
    dim: int,
    pixel_size: u.au,
    inclination: Optional[u.one] = None,
    position_angle: Optional[u.deg] = None,
) -> np.ndarray:
    radial_profiles = []
    for lower, upper in partition_list(radii.value.tolist()) * radii.unit:
        grid = create_grid(dim, pixel_size, inclination, position_angle)

        if upper is None:
            radial_profile = (grid > lower).astype(int)
        else:
            radial_profile = ((grid > lower) & (grid < upper)).astype(int)

        radial_profiles.append(radial_profile)

    return np.sum(radial_profiles, axis=0)


def make_system_sketch(
    dim: int,
    pixel_size: u.au,
    radii: u.au,
) -> None:
    extent = [sign * 0.5 * dim * pixel_size.value for sign in [1, -1, -1, 1]]
    rho, theta = get_orbit(
        23.5 * u.yr,
        0.47 * u.one,
        10.8 * u.au,
        149.47 * u.deg,
        2020.42 * u.yr,
        186.45 * u.deg,
        161.51 * u.deg,
    )
    apihel_ind = np.abs(rho).argmax()
    x_orbit, y_orbit = (rho * np.cos(theta)).value, (rho * np.sin(theta)).value

    fig, ax = plt.subplots(figsize=(15, 11))
    ax.imshow(disc_image(radii, dim, pixel_size), cmap="Greys", extent=extent)
    ax.scatter(0, 0, color="orange", marker="*", s=100, label="HD142527A")
    ax.scatter(
        x_orbit[apihel_ind],
        -y_orbit[apihel_ind],
        color="blue",
        marker="*",
        s=100,
        label="HD142527B",
    )
    ax.plot(x_orbit, -y_orbit, color="blue")
    ax.set_xlabel(r"$\Delta\alpha$ (AU)")
    ax.set_ylabel(r"$\Delta\delta$ (AU)")

    disc_label = "Inner disc"
    custom_label = Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="black",
        markersize=10,
        label=disc_label,
    )
    handles, labels = plt.gca().get_legend_handles_labels()

    handles.append(custom_label)
    labels.append(disc_label)
    ax.legend(loc="upper left", handles=handles, labels=labels)

    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper right")
    extent = [sign * 0.5 * dim * 11.5 * pixel_size.value for sign in [1, -1, -1, 1]]
    outer_disc_image = Image.open("sphere_disc.png")
    ax_inset.imshow(outer_disc_image, cmap="Greys", extent=extent)
    ax_inset.scatter(0, 0, color="orange", marker="*", s=5)
    ax_inset.plot(x_orbit, -y_orbit, color="blue")
    ax_inset.scatter(
        x_orbit[apihel_ind], -y_orbit[apihel_ind], color="blue", marker="*", s=5
    )
    ax_inset.set_xlabel(r"$\Delta\alpha$ (AU)", fontsize=8)
    ax_inset.set_ylabel(r"$\Delta\delta$ (AU)", fontsize=8)
    ax_inset.axhline(0, color="lightgray", linewidth=0.5, ls="--")
    ax_inset.axvline(0, color="lightgray", linewidth=0.5, ls="--")

    plt.tight_layout()
    plt.savefig("system_sketch.pdf")
    # plt.show()


if __name__ == "__main__":
    make_system_sketch(4096, 0.01 * u.au, [1, 2, 4, 6] * u.au)
