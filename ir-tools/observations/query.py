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


def query_and_filter_catalogs(object_name: str, radius=1 * u.arcsec):
    """Query Vizier for the specified object and filter the catalogs to
    only include those with WISE data and multiple epochs."""
    Vizier.ROW_LIMIT = -1
    catalogs = Vizier.query_object(object_name, radius=radius)
    multi_epoch_catalogs = []

    for catalog in catalogs:
        has_wise_columns = any(
            col
            for col in catalog.colnames
            if any(f"W{index}" in col for index in range(1, 5))
        )

        # has_multiple_epochs(catalog):
        if has_wise_columns:
            multi_epoch_catalogs.append(catalog)

    return multi_epoch_catalogs


# TODO: Check AKARI as well for the targets
def query_catalog(object_name: str, catalog: str, index: int):
    """Query the specified catalog for the specified object
    from a multi epoch catalog."""
    if catalog == "wise":
        wise = Irsa.query_region(
            object_name, catalog="allwise_p3as_mep", spatial="Cone", radius=1 * u.arcsec
        )

        w, w_err = wise[f"w{index}mpro_ep"], wise[f"w{index}sigmpro_ep"]
        # w_err_ratio = w_err / w

        labels = Time(wise["mjd"].data, format="mjd")
        ind = np.argsort(labels)
        labels = labels[ind]
        w, w_err = w[ind], w_err[ind]
        isot = [label.isot for label in labels]
        data = {
            k: {"time": [], "value": [], "error": []}
            for k in np.unique([iso.split("T")[0] for iso in isot])
        }

        for w_entry, w_err_entry, iso in zip(w, w_err, isot):
            key, time = iso.split("T")
            data[key]["time"].append(time)
            data[key]["value"].append(w_entry)
            data[key]["error"].append(w_err_entry)
    return {k: {ki: np.array(vi) for ki, vi in v.items()} for k, v in data.items()}


def plot_multi_epoch(object_name: str) -> None:
    """Plot the multi-epoch WISE data for the specified object."""
    # akari_data = query_catalog(object_name, "akari")

    _, axarr = plt.subplots(1, 2, figsize=(15, 7))
    for index, ax in zip([1, 3], axarr.flatten()):
        wise_data = query_catalog(object_name, "wise", index)
        errorbar_kwargs = {"fmt": "o", "ecolor": "gray", "capsize": 5}
        for key, value in wise_data.items():
            ax.errorbar(
                value["time"],
                value["value"],
                value["error"],
                label=key,
                **errorbar_kwargs,
            )
        ax.set_ylabel("Luminosity (mag)")
        ax.set_xlabel("Time (UTC)")
        # ax.set_ylim([0, None])
        ax.set_title(f"{object_name.upper()} WISE{index}")
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.legend()

    plt.savefig(f"{object_name}_time_variability.pdf", format="pdf")


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
    plot_multi_epoch("hd142527")
