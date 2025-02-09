from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np

from .component import FourierComponent
from .options import OPTIONS
from .utils import (
    compute_effective_baselines,
)


def plot_interferometric_observables(
    wavelength_range: u.um,
    components: List[FourierComponent],
    component_labels: List[str],
    save_dir: Path | None = None,
) -> None:
    """Plots the observables of the model.

    Parameters
    ----------
    wavelength_range : astropy.units.m
    sed_scaling : str, optional
        The scaling of the SED. "nu" for the flux to be
        in Jy times Hz. If "lambda" the flux is in Jy times m.
        If "none" the flux is in Jy.
        The default is "nu".
    """
    save_dir = Path.cwd() if save_dir is None else save_dir
    wavelength = np.linspace(wavelength_range[0], wavelength_range[1], OPTIONS.plot.dim)
    _, vis, t3, vis_comps = compute_observables(
        components, wavelength=wavelength, rcomponents=True
    )

    vis_data = OPTIONS.data.vis if "vis" in OPTIONS.fit.data else OPTIONS.data.vis2

    effective_baselines, baseline_angles = compute_effective_baselines(
        vis_data.ucoord,
        vis_data.vcoord,
        components[1].cinc(),
        components[1].pa(),
        return_zero=False,
    )
    baseline_angles = baseline_angles.to(u.deg)

    num_plots = len(effective_baselines)
    cols = int(str(num_plots)[: int(np.floor(np.log10(num_plots)))])
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(30, 30),
        facecolor=OPTIONS.plot.color.background,
        sharex=True,
        constrained_layout=True,
    )
    axes = axes.flatten()
    if "vis" in OPTIONS.fit.data:
        if OPTIONS.model.output != "normed":
            y_label = "Correlated fluxes (Jy)"
            ylims = [None, None]
        else:
            y_label = "Visibilities (Normalized)"
            ylims = [0, 1]
    else:
        y_label = "Visibilities Squared (Normalized)"
        ylims = [0, 1]

    for index, (baseline, baseline_angle) in enumerate(
        zip(effective_baselines, baseline_angles)
    ):
        ax = axes[index]
        set_axes_color(ax, OPTIONS.plot.color.background)
        ax.plot(
            wavelength,
            vis[:, index],
            label=rf"B={baseline.value:.2f} m, $\phi$={baseline_angle.value:.2f}$^\circ$",
        )

        for comp_index, vis_comp in enumerate(vis_comps):
            ax.plot(wavelength, vis_comp[:, index], label=component_labels[comp_index])

        ax.set_ylim(ylims)
        ax.legend()

    fig.subplots_adjust(left=0.2, bottom=0.2)
    fig.text(0.5, 0.04, r"$\lambda$ ($\mathrm{\mu}$m)", ha="center", fontsize=16)
    fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)
    plt.savefig(save_dir / "vis_vs_baseline.png", format="png")
    plt.close()

    if "t3" in OPTIONS.fit.data:
        effective_baselines, baseline_angles = compute_effective_baselines(
            OPTIONS.data.t3.u123coord,
            OPTIONS.data.t3.v123coord,
            components[1].cinc(),
            components[1].pa(),
            longest=True,
            return_zero=False,
        )
        baseline_angles = baseline_angles.to(u.deg)

        num_plots = len(effective_baselines)
        cols = int(str(num_plots)[: int(np.floor(np.log10(num_plots)))])
        rows = int(np.ceil(num_plots / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(30, 30),
            facecolor=OPTIONS.plot.color.background,
            sharex=True,
            constrained_layout=True,
        )
        axes = axes.flatten()
        for index, (baseline, baseline_angle) in enumerate(
            zip(effective_baselines, baseline_angles)
        ):
            ax = axes[index]
            set_axes_color(ax, OPTIONS.plot.color.background)
            ax.plot(wavelength, t3[:, index], label=f"B={baseline.value:.2f} m")
            ax.legend()

        fig.subplots_adjust(left=0.2, bottom=0.2)
        fig.text(0.5, 0.04, r"$\lambda$ ($\mathrm{\mu}$m)", ha="center", fontsize=16)
        fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)
        plt.savefig(save_dir / "t3_vs_baseline.png", format="png")
        plt.close()
