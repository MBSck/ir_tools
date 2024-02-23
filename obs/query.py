from pprint import pprint

from p2obp.backend.query import query


def wise_to_flux(flux: float, lband: bool):
    if lband:
        return round(309.54 * 10.0**(-flux/2.5), 2)
    return round(31.674 * 10.0**(-flux/2.5), 2)


if __name__ == "__main__":
    dic = query("HP Cha")
    pprint(dic)

    lflux = wise_to_flux(dic["W1mag"], True)
    nflux = wise_to_flux(dic["W3mag"], False)
    print(lflux, nflux)
