from pathlib import Path

from matadrs.utils.tools import add_instrument

if __name__ == "__main__":
    folder = Path("/Users/scheuck/Code/modelling/ppdmod/tests/data/aspro")
    for fits in folder.glob("*fits"):
        add_instrument(fits, "matisse", add_error=True)
