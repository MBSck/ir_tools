from pathlib import Path

import numpy as np
import optool
import matplotlib.pyplot as plt
from ppdmod.utils import load_data, qval_to_opacity


if __name__ == "__main__":
    wl = np.load("grf_wl_grid.npy")
    # with open("grf_wl_grid.txt", "w") as f:
    #     f.write(f"{wl.size}\n")
    #     for w in wl:
    #         f.write(f"{w}\n")

    cont_opacity_file = Path("/Users/scheuck/Data/opacities/qval") / "Q_amorph_c_rv0.1.dat"
    wl_cont, cont_opacity = load_data(cont_opacity_file, load_func=qval_to_opacity)

    lnk_path = Path("/Users/scheuck/Code/optool/lnk_data")
    optool_path = Path("/Users/scheuck/Code/optool/optool")
    pz = optool.particle(f"{optool_path} c-z -l grf_wl_grid.txt -a 0.1")
    pp = optool.particle(f"{optool_path} c-p -l grf_wl_grid.txt -a 0.1")

    np.save("preibisch_amorph_c_rv0.1.npy", np.array([pz.lam, pz.kabs.flatten()]))

    plt.plot(wl_cont, cont_opacity, label="QVAL")
    plt.plot(pz.lam, pz.kabs.flatten(), label="Zubko")
    plt.plot(pp.lam, pp.kabs.flatten(), label="Preibisch")
    plt.plot()

    plt.xlabel(r"Wavelength ($\mathrm{\mu}$m)")
    plt.ylabel(r"$\kappa_{\mathrm{abs}}$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("amorphous carbon comparison".title())
    plt.savefig("amorphous_carbon_comparison.pdf", format="pdf")
