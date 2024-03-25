import astropy.units as u
from astropy.coordinates import SkyCoord


if __name__ == "__main__":
    north = ("16h11m31.3598354904s", "-18d38m25.778047848s")
    south = ("16h11m31.3075463616s", "-18d38m26.889237456s")
    coord_north = SkyCoord(ra=north[0], dec=north[1], frame="icrs")
    coord_south = SkyCoord(ra=south[0], dec=south[1], frame="icrs")
    ra = (coord_north.ra-coord_south.ra).to(u.arcsec)
    dec = (coord_north.dec-coord_south.dec).to(u.arcsec)
    breakpoint()
