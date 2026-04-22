"""
sparcs.components.solar.bifacial.temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``temperature`` module contains functions for modeling temperature of
PV modules and cells.

"""


# noinspection SpellCheckingInspection
def pvsyst_cell(
    poa_front,
    poa_back,
    temp_air,
    wind_speed=1.0,
    u_c=29.0,
    u_v=0.0,
    module_bifaciality=0,
    module_efficiency=0.1,
    alpha_absorption=0.9,
):
    r"""
    Calculate cell temperature using an empirical heat loss factor model
    as implemented in PVsyst, adjusted by a bifaciality factor.

    Parameters
    ----------
    poa_front: numeric
        Total incident irradiance on the front surface of the PV modules (W/m2).

    poa_back: numeric
        Total incident irradiance on the back surface of the PV modules (W/m2).

    temp_air : numeric
        Ambient dry bulb temperature [C].

    wind_speed : numeric, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value 1.0 m/s is the wind
        speed at module height used to determine NOCT. [m/s]

    u_c : float, default 29.0
        Combined heat loss factor coefficient. The default value is
        representative of freestanding modules with the rear surfaces exposed
        to open air (e.g., rack mounted). Parameter :math:`U_{c}` in
        :eq:`pvsyst`.
        :math:`\left[\frac{\text{W}/{\text{m}^2}}{\text{C}}\right]`

    u_v : float, default 0.0
        Combined heat loss factor influenced by wind. Parameter :math:`U_{v}`
        in :eq:`pvsyst`.
        :math:`\left[ \frac{\text{W}/\text{m}^2}{\text{C}\ \left( \text{m/s} \right)} \right]`

    module_bifaciality : numeric, default 0
        Module bifaciality factor as a fraction.

    module_efficiency : numeric, default 0.1
        Module external efficiency as a fraction. Parameter :math:`\eta_{m}`
        in :eq:`pvsyst`. Calculate as
        :math:`\eta_{m} = DC\ power / (POA\ irradiance \times module\ area)`.

    alpha_absorption : numeric, default 0.9
        Absorption coefficient. Parameter :math:`\alpha` in :eq:`pvsyst`.

    Returns
    -------
    numeric, values in degrees Celsius


    References
    ----------
    .. [1] "PVsyst 6 Help", Files.pvsyst.com, 2018. [Online]. Available:
       http://files.pvsyst.com/help/index.html. [Accessed: 10- Dec- 2018].

    .. [2] Faiman, D. (2008). "Assessing the outdoor operating temperature of
       photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.

    """  # noqa: E501

    total_loss_factor = u_c + u_v * wind_speed

    heat_input = poa_front * alpha_absorption * (1 - module_efficiency)
    if module_bifaciality > 0:
        heat_input += poa_back * alpha_absorption * (1 - module_efficiency * module_bifaciality)

    temp_difference = heat_input / total_loss_factor
    return temp_air + temp_difference
