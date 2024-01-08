from pathlib import Path

from matadrs.utils.readout import ReadoutFits


if __name__ == "__main__":
    path = Path("/data/beegfs/astro-storage/groups/matisse/scheuck/data/matisse/GTO/hd142527/raw/large")
    for folder in list(path.glob("20*")):
        print(folder.name)
        for f in (folder / "calib_files").glob("*fits"):
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
        break
