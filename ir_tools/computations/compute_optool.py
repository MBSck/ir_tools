from functools import reduce
from pathlib import Path
from typing import List

import optool
import toml

OPTOOL_EXEC = Path().home() / ".local" / "bin" / "optool"
LNK_DIR = Path().home() / "Documents" / "Code" / "optool" / "lnk_data" / "ad"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "opacities"
with open(DATA_DIR / "lnk" / "lnks.toml", "r") as toml_file:
    CONFIG = toml.load(toml_file)


def compute_stoichiometries(
    variant: str,
    grain_size: float,
    wl_range: List[float] = [0.1, 100],
    wl_dim: int = 1000,
) -> None:
    """Does the optool optool calculation."""
    first, second, additional = (
        CONFIG[variant]["first"],
        CONFIG[variant]["second"],
        CONFIG[variant]["additional"],
    )
    command = [
        str(OPTOOL_EXEC),
        f"-l {wl_range[0]} {wl_range[1]} {wl_dim}",
        f"-a {grain_size}",
        *additional,
    ]

    if first and second:
        paths = [LNK_DIR / f"{first}-{coord}-{second}.lnk" for coord in ["x", "y", "z"]]
    else:
        paths = [first]

    particle = reduce(
        lambda x, y: x + y,
        [optool.particle(" ".join([*command, f"-c {path}"])) for path in paths],
    )

    # pz2 = optool.particle("~/bin/optool "+dhs_opt+" "+wl_opt+" -a "+gs+" /Users/jvarga/Dokumentumok/MATISSE/pro/optool/lnk_data/ad/oliSC_z_551K.lnk")

    # plt.plot(
    #     pmix2.lam, pmix2.kabs[0, :], label="forsterite DHS Zeidler monocrystalline"
    # )


if __name__ == "__main__":
    # do_optool_calculation("forsterite", 0.1)
    compute_stoichiometries("pyroxene", 0.1)

    # plt.plot(wl_cont, cont_opacity, label="QVAL")
    # plt.plot(pz.lam, pz.kabs.flatten(), label="Zubko")
    # plt.plot(pp.lam, pp.kabs.flatten(), label="Preibisch")
    # plt.plot()
    #
    # plt.xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    # plt.ylabel(r"$\kappa_{\mathrm{abs}}$")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.legend()
    # plt.title("amorphous carbon comparison".title())
    # plt.savefig("amorphous_carbon_comparison.pdf", format="pdf")
