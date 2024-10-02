from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



if __name__ == "__main__":
    opacities = Path("/Users/scheuck/Data/opacities/grf")
    plot_dir = opacities / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    for file in opacities.glob("*.Kappa"):
        wl, op = np.loadtxt(file, unpack=True, usecols=(0, 2))
        plt.plot(wl, op)
        plt.yscale("log")
        plt.xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
        plt.ylabel(r"$\kappa$ (cm$^2$ g$^{-1}$")
        plt.xlim([-5, 50])
        plt.title(file.stem)
        plt.savefig(plot_dir / f"{file.stem}.pdf", format="pdf")
        plt.close()
