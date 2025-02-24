from itertools import permutations
from typing import Dict


def add_array_config(
    dictionary: Dict[str, str], key: str, value: str
) -> Dict[str, str]:
    """Adds all permutations of the configuration to the dictionary."""
    perms = map(lambda x: "-".join(x), permutations(key.split("-")))
    perms = ["-".join(perm.split("-")) for perm in perms]
    return {**dictionary, **{perm: value for perm in perms}}


SOURCE_INFO_KEYS = [
    "source",
    "tpl_start",
    "year",
    "date",
    "sci_seeing_start",
    "sci_seeing_end",
    "sci_tau0_start",
    "sci_tau0_end",
    "cal_seeing",
    "cal_tau0",
    "array",
    "band",
    "chopped",
    "pipe_version",
    "cal_name",
    "cal_ra",
    "cal_dec",
    "cal_diam",
    "cal_diam_err",
    "file_name",
]

FIELDS_TO_KEY = {
    "Spectral Type": "SP_TYPE",
    "RA": "RA",
    "DEC": "DEC",
    "source": "source",
}

LATTICE_STRUCTURE = {
    "pyroxene": "amorphous",
    "forsterite": "crystalline",
    "enstatite": "crystalline",
    "silica": "crystalline",
    "carbon": "amorphous",
}

CHEMICAL_FORMULAS = {
    "pyroxene": r"\ce{Mg_{x}Fe_{1-x}SiO3}",
    "forsterite": r"\ce{Mg2SiO4}",
    "enstatite": r"\ce{MgSiO3}",
    "silica": r"\ce{SiO2}",
    "carbon": r"\ce{C}",
}

ARRAY_CONFIGS = {}
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B2-D0-C1", "smallAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-B2-C1-D0", "smallAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "K0-G2-D0-J3", "mediumAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-H0-G1-I1", "mediumAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-G2-J3-K0", "mediumAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-D0-J3", "mediumAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G2-J2-J3", "largeAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-J3", "largeAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-K0", "largeAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-I1", "largeAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-J3", "largeAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-J2-J6", "extendedAT")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "U1-U2-U3-U4", "UTs")
