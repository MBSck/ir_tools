from itertools import permutations
from typing import Dict


def add_array_config(
    dictionary: Dict[str, str], key: str, value: str
) -> Dict[str, str]:
    """Adds all permutations of the configuration to the dictionary."""
    perms = map(lambda x: "-".join(x), permutations(key.split("-")))
    perms = ["".join(perm.split("-")) for perm in perms]
    return {**dictionary, **{perm: value for perm in perms}}


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
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B2-D0-C1", "AT small")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-B2-C1-D0", "AT small")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "K0-G2-D0-J3", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-H0-G1-I1", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-G2-J3-K0", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-D0-J3", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G2-J2-J3", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-J3", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-K0", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-I1", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-J3", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-J2-J6", "AT extended")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "U1-U2-U3-U4", "UTs")
