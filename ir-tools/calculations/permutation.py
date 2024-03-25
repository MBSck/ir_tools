from pathlib import Path

import numpy as np
from astropy.io import fits

def calc_perm(coord, c1oord, c2oord):
    perm1, perm2 = [], []
    for c1, c2 in zip(c1oord, c2oord):
        index1, index2 = np.where(coord == c1), np.where(coord == c2)
        perm1.append(index1)
        perm2.append(index2)
    return np.hstack(perm1), np.hstack(perm2)


def get_permutation(fits_file: Path):
    with fits.open(fits_file) as hdul:
        ucoord, vcoord = hdul["oi_vis"].data["UCOORD"], hdul["oi_vis"].data["VCOORD"]
        u1coord, u2coord = hdul["oi_t3"].data["U1COORD"], hdul["oi_t3"].data["U2COORD"]
        v1coord, v2coord = hdul["oi_t3"].data["v1COORD"], hdul["oi_t3"].data["v2COORD"]

    u1perm, u2perm = calc_perm(np.sort(ucoord), u1coord, u2coord)
    v1perm, v2perm = calc_perm(np.sort(vcoord), v1coord, v2coord)
    breakpoint()

if __name__ == "__main__":
    fits_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/nband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits")
    get_permutation(fits_file)
