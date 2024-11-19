from pathlib import Path
from typing import List

from astropy.io import fits
from tqdm import tqdm


def show_tpls(directory: Path):
    tpls = []
    for f in list(Path(directory).glob('*fits')):
        with fits.open(f, "readonly") as hdul:
            tpl = hdul[0].header["hierarch eso tpl start"]
            if tpl not in tpls:
                tpls.append(tpl)
    return tpls


def remove_tpls(directory: Path, tpl_to_keep: List[str]):
    files = list(Path(directory).glob('*fits'))
    for f in tqdm(files):
        with fits.open(f, "readonly") as hdul:
            header = hdul[0].header
        tpl_start = header["hierarch eso tpl start"]
        if tpl_start not in tpl_to_keep:
            f.unlink()


if __name__ == "__main__":
    print(show_tpls("2022-04-23-other-calib"))
    remove_tpls("2022-04-23-other-calib",
                ["2022-04-23T03:05:25", "2022-04-21T06:47:05"])
    print(show_tpls("2022-04-23-other-calib"))

