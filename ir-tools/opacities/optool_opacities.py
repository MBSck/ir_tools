from pathlib import Path

import numpy as np
import optool
import matplotlib.pyplot as plt
from ppdmod.utils import load_data, qval_to_opacity

STR_TO_LNK = {

    "oliSC_x_551K"
}

# TODO: Make lnk data dir
def do_optool_calculation():
    gsize = 0.1 #um
    gs = '%f'%gsize
    dhs_opt = ' -dhs 0.8'
    wl_grid = '-l 0.05 50 10000'

    px2 = optool.particle("~/bin/optool "+dhs_opt+" "+wl_opt+" -a "+gs+)
    py2 = optool.particle("~/bin/optool "+dhs_opt+" "+wl_opt+" -a "+gs+" /Users/jvarga/Dokumentumok/MATISSE/pro/optool/lnk_data/ad/oliSC_y_551K.lnk")
    pz2 = optool.particle("~/bin/optool "+dhs_opt+" "+wl_opt+" -a "+gs+" /Users/jvarga/Dokumentumok/MATISSE/pro/optool/lnk_data/ad/oliSC_z_551K.lnk")
    pmix2 = (px2+py2+pz2)/3.

    plt.plot(pmix2.lam,pmix2.kabs[0,:],label='forsterite DHS Zeidler monocrystalline')



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
