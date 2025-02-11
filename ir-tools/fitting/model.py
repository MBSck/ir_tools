import astropy.units as u
import numpy as np

from oimodeler.oimComponent import oimComponentRadialProfile
from oimodeler.oimOptions import oimOptions
from oimodeler.oimParam import oimParam
from oimodeler.oimUtils import blackbody, convert_distance_to_angle


class oimTempGrad(oimComponentRadialProfile):
    """A ring defined by a radial temperature profile in r^q and a radial dust
    surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [au].
    rout : float
        Outer radius of the disk [au].
    Tin : float
        Inner radius temperature [K].
    Mdust : float
        Mass of the dusty disk [M_sun].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float or oimInterp
        Dust mass absorption coefficient [cm2.g-1].
    dist : float
        Distance of the star [pc].

    Attributes
    ----------
    params : dict with keys of str and values of oimParam
        Dictionary of parameters.
    _r : array_like
    _wl : array_like
        Wavelengths [micron].
    _t : array_like
        Times [second].

    Methods
    -------
    _radialProfileFunction(r, wl, t)
        Calculates a radial temperature gradient profile via a dust-surface
        density- and temperature profile.
    """

    name = "Temperature Gradient"
    shortname = "TempGrad"
    elliptic = True
    temperature = "const"

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.params["rin"] = oimParam(
            name="rin", value=0, unit=u.au, description="Inner radius of the disk"
        )
        self.params["rout"] = oimParam(
            name="rout", value=0, unit=u.au, description="Outer radius of the disk"
        )
        self.params["p"] = oimParam(
            name="p",
            value=-1,
            unit=u.one,
            description="Power-law exponent for the dust surface density profile",
        )
        self.params["sigma0"] = oimParam(
            name="sigma0",
            value=1e-3,
            unit=u.g / u.cm**2,
            description="The surface density at a reference radius",
        )
        self.params["q"] = oimParam(
            name="q",
            value=-1,
            unit=u.one,
            free=False,
            description="Power-law exponent for the temperature profile",
        )
        self.params["temp0"] = oimParam(
            name="temp0",
            value=1500,
            unit=u.K,
            free=False,
            description="The temperature at a reference radius",
        )
        self.params["kappa_abs"] = oimParam(
            name="kappa_abs",
            value=0,
            unit=u.cm**2 / u.g,
            free=False,
            description="Dust mass absorption coefficient",
        )
        self.params["kappa_cont"] = oimParam(
            name="kappa_abs",
            value=0,
            unit=u.cm**2 / u.g,
            free=False,
            description="Dust mass absorption coefficient",
        )
        self.params["cont_weight"] = oimParam(
            name="cont_weight",
            value=1,
            unit=u.one,
            free=False,
            description="The continuum weight",
        )
        self.params["dist"] = oimParam(
            name="dist",
            value=0,
            unit=u.pc,
            free=False,
            description="Distance of the star",
        )
        self.params["eff_temp"] = oimParam(
            name="eff_temp",
            value=0,
            unit=u.K,
            free=False,
            description="Effective temperature of the star",
        )
        self.params["eff_radius"] = oimParam(
            name="eff_radius",
            value=0,
            unit=u.Rsun,
            free=False,
            description="Effective radius of the star",
        )
        self.params["r0"] = oimParam(
            name="r0",
            value=1,
            unit=u.au,
            free=False,
            description="The reference radius",
        )

        self._eval(**kwargs)
        self.params["f"].free = False
        if self.temperature == "power":
            self.params["q"].free = True
            self.params["temp0"].free = True

    @property
    def _r(self):
        """Gets the radial profile (au)."""
        rin = convert_distance_to_angle(
            self.params["rin"].value, self.params["dist"].value
        )
        rout = convert_distance_to_angle(
            self.params["rout"].value, self.params["dist"].value
        )
        if oimOptions.model.grid.type == "linear":
            return np.linspace(rin, rout, self.params["dim"].value)

        return np.logspace(
            0.0 if rin == 0 else np.log10(rin), np.log10(rout), self.params["dim"].value
        )

    @_r.setter
    def _r(self, value):
        return

    def _radialProfileFunction(
        self, r: np.ndarray, wl: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Calculates a radial temperature gradient profile via a dust-surface
        density- and temperature profile.

        Parameters
        ----------
        r : numpy.ndarray
            Radial grid [mas].
        wl : numpy.ndarray
            Wavelengths [micron].
        t : numpy.ndarray
            Times [second].

        Results
        -------
        radial_profile : numpy.ndarray
        """
        # HACK: Sets the multi wavelength coordinates properly.
        # Does not account for time, improves computation time.
        wl = np.unique(wl)
        kappa_abs = self.params["kappa_abs"](wl, t)
        if len(r.shape) == 3:
            r = r[0, 0][np.newaxis, np.newaxis, :]
            wl, kappa_abs = map(lambda x: x[np.newaxis, :, np.newaxis], [wl, kappa_abs])
        else:
            wl, kappa_abs = map(lambda x: x[:, np.newaxis], [wl, kappa_abs])
            r = r[np.newaxis, :]

        r0 = convert_distance_to_angle(
            self.params["r0"].value, self.params["dist"].value
        )
        if self.temperature == "const":
            eff_radius = self.params["eff_radius"](wl, t)
            eff_temp = self.params["eff_temp"](wl, t)
            eff_radius = convert_distance_to_angle(
                eff_radius * self.params["eff_radius"].unit.to(u.au),
                self.params["dist"].value,
            )
            temp = np.sqrt(eff_radius / (2 * r)) * eff_temp
        else:
            q, temp0 = self.params["q"](wl, t), self.params["temp0"](wl, t)
            temp = temp0 * (r / r0) ** q

        sigma = self.params["sigma0"](wl, t) * (r / r0) ** self.params["p"](wl, t)
        epsilon = 1 - np.exp(-sigma * kappa_abs)
        spectral_density = blackbody(temp, wl) * epsilon

        if len(r.shape) == 3:
            rin = convert_distance_to_angle(
                self.params["rin"].value, self.params["dist"].value
            )
            rout = convert_distance_to_angle(
                self.params["rout"].value, self.params["dist"].value
            )
            radial_profile = ((r > rin) & (r < rout)).astype(int)
            image = np.nan_to_num(radial_profile * spectral_density, nan=0)
            return image

        return spectral_density


class oimAsymTempGrad(oimTempGrad):
    """A ring defined by a radial temperature profile in r^q and a radial dust
    surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [au].
    rout : float
        Outer radius of the disk [au].
    Tin : float
        Inner radius temperature [K].
    Mdust : float
        Mass of the dusty disk [M_sun].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float or oimInterp
        Dust mass absorption coefficient [cm2.g-1].
    dist : float
        Distance of the star [pc].

    Attributes
    ----------
    params : dict with keys of str and values of oimParam
        Dictionary of parameters.
    _r : array_like
    _wl : array_like
        Wavelengths [micron].
    _t : array_like
        Times [second].

    Methods
    -------
    _radialProfileFunction(r, wl, t)
        Calculates a radial temperature gradient profile via a dust-surface
        density- and temperature profile.
    """

    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    elliptic = True
    asymmetric = True
