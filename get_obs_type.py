from pathlib import Path

from matadrs.utils.readout import ReadoutFits


def print_types(folder: Path) -> None:
    print(folder.name)
    for f in folder.glob("*fits"):
        print(f.name)
        readout = ReadoutFits(f)
        try:
            print(readout.primary_header["ESO PRO CATG"])
        except KeyError:
            try:
                print(readout.primary_header["ESO DPR CATG"])
                print(readout.primary_header["ESO DPR TYPE"])
                print(readout.primary_header["ESO DPR TECH"])
            except KeyError:
                pass
        print(f"{'':-^50}")


if __name__ == "__main__":
    folder = Path("/data/beegfs/astro-storage/groups/matisse/scheuck/data/matisse/GTO/hd142666/raw/test")
    print_types(folder / "calib_files")
