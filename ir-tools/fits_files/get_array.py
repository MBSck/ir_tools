from pathlib import Path

from matadrs.utils.readout import ReadoutFits


if __name__ == "__main__":
    path = Path("/data/beegfs/astro-storage/groups/matisse/scheuck/data/matisse/GTO/hd142527")
    for folder in path.glob("20*"):
        for f in folder.glob("*fits"):
            readout = ReadoutFits(f)
            try:
                if "ESO ISS BASELINE" in readout.primary_header:
                    array = readout.primary_header["ESO ISS BASELINE"]
                else:
                    array = readout.primary_header["ESO OBS BASELINE"]
                break
            except:
                pass
        print(folder.name)
        print(array)
        print(f"{'':-^50}")
