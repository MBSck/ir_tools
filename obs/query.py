from pprint import pprint

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astroquery.ipac.irsa import Irsa
from astroquery.vizier import Vizier

def wise_to_flux(flux: float, lband: bool):
    if lband:
        return np.round(309.54 * 10.0 ** (-flux / 2.5), 2)
    return np.round(31.674 * 10.0 ** (-flux / 2.5), 2)


def query_and_filter_catalogs(object_name: str, radius=1*u.arcsec):
    """Query Vizier for the specified object and filter the catalogs to
    only include those with WISE data and multiple epochs."""
    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.query_object(object_name, radius=radius)
    multi_epoch_catalogs = []
    
    for catalog in catalogs:
        has_wise_columns = any(col for col in catalog.colnames if any(f"W{index}" in col for index in range(1, 5)))
        
        # has_multiple_epochs(catalog):
        if has_wise_columns:
            multi_epoch_catalogs.append(catalog)
    
    return multi_epoch_catalogs


def plot_multi_epoch(object_name: str) -> None:
    """Plot the multi-epoch WISE data for the specified object."""
    table = Irsa.query_region(object_name, catalog="allwise_p3as_mep",
                              spatial="Cone", radius=1 * u.arcsec)
    w1, w3 = table["w1mpro_ep"], table["w3mpro_ep"]
    w1_sigma, w3_sigma = table["w1sigmpro_ep"], table["w3sigmpro_ep"]
    w1_err, w3_err = w1_sigma/w1, w3_sigma/w3
    w1, w3 = wise_to_flux(w1, True), wise_to_flux(w3, False)
    w1_err, w3_err = w1*w1_err, w3*w3_err

    labels = Time(table["mjd"].data, format="mjd")
    ind = np.argsort(labels)
    labels = labels[ind]
    w1, w3 = w1[ind], w3[ind]
    w1_err, w3_err = w1_err[ind], w3_err[ind]
    isot = [label.isot[:16] for label in labels]
    errorbar_kwargs = {"fmt": "o", "ecolor": "gray", "capsize": 5}

    _, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7))
    ax.errorbar(isot, w1, w1_err, **errorbar_kwargs)
    bx.errorbar(isot, w3, w3_err, c="orange", **errorbar_kwargs)

    ax.set_ylabel("Flux (Jy)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylim([0, None])
    ax.set_title(f"{object_name} W1")
    ax.tick_params(axis="x", labelsize=6, rotation=45)

    bx.set_ylabel("Flux (Jy)")
    bx.set_xlabel("Time (UTC)")
    bx.set_title(f"{object_name} W3")
    bx.set_ylim([0, None])
    bx.tick_params(axis="x", labelsize=6, rotation=45)

    plt.savefig("time_variability.pdf", format="pdf")


if __name__ == "__main__":
    # dic = query("HP Cha")
    # pprint(dic)

    # lflux = wise_to_flux(dic["W1mag"], True)
    # nflux = wise_to_flux(dic["W3mag"], False)
    # print(lflux, nflux)

    # catalog_list = Vizier.find_catalogs("WISE")
    # breakpoint()
    # pprint({k:v.description for k,v in catalog_list.items()})
    
    # catalogs = Irsa.list_catalogs()
    plot_multi_epoch("hd142666")
