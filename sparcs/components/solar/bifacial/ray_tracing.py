# -*- coding: utf-8 -*-
"""
sparcs.components.solar.bifacial.ray_tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""

import logging
import os
from contextlib import redirect_stdout

from bifacial_radiance import AnalysisObj, RadianceObj

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# noinspection PyPep8Naming, SpellCheckingInspection
def get_total_irradiance(
    solar_azimuth,
    solar_elevation,
    surface_azimuth,
    surface_tilt,
    surface_length,
    surface_width,
    module_height,
    gcr,
    albedo,
    timestamp,
    dni,
    dhi,
    module_bifaciality=1,
    modules_stacked=1,
    module_stack_gap=0,
    module_row_gap=0,
    modules=20,
    module_index=None,
    module_mesh=9,
    rows=7,
    row_index=None,
    path=os.path.join("data", "radiance"),
    name="array",
):
    # Redirect print output to logging
    # noinspection PyTypeChecker
    with redirect_stdout(WriteLogger(logger.debug)):
        timestr = f"{name}_{timestamp.strftime('%Y-%m-%d_%H%M')}"
        radiance = Radiance(name, path)

        # Create a custom PV module type.
        radiance.makeModule(
            name=name,
            x=surface_width,
            y=surface_length,
            xgap=module_row_gap,
            ygap=module_stack_gap,
            bifi=module_bifaciality,
            numpanels=modules_stacked,
            tubeParams={"visible": False},
            rewriteModulefile=False,
        )

        # TODO: Handle ground.rad file input, if albedo is not numeric
        # Input albedo number or material name like 'concrete'. To see options, run this without any input.
        radiance.setGround(albedo)

        # Skip if not daylight hours
        sky = radiance.makeSky(timestamp, dhi, dni, solar_elevation, solar_azimuth)
        if not sky:
            return np.zeros(module_mesh), np.zeros(module_mesh)

        analysis = radiance.loadResults(timestamp)
        if analysis is None:
            scene = radiance.makeScene(
                sceneDict={
                    "hub_height": module_height,
                    "azimuth": surface_azimuth,
                    "tilt": surface_tilt,
                    "gcr": gcr,
                    "nMods": modules,
                    "nRows": rows,
                },
                radname=timestr,
            )

            # makeOct combines all the ground, sky and object files into a .oct file.
            octfile = radiance.makeOct(radiance.getfilelist(), timestr)

            # Return an analysis object including the scan dimensions for back irradiance
            analysis = Analysis(octfile, timestr)

            analysis.makeAnalysis(
                scene,
                octfile,
                modWanted=module_index,
                rowWanted=row_index,
                sensorsy=module_mesh,
            )

    return analysis.Wm2Front, analysis.Wm2Back


# noinspection PyPep8Naming, SpellCheckingInspection, PyProtectedMember
class Radiance(RadianceObj):
    def loadResults(self, timeindex):
        time_str = timeindex.strftime("%Y-%m-%d_%H%M")
        results_file = os.path.join("results", f"irr_{self.name}_{time_str}.csv")
        if os.path.isfile(results_file):
            return pd.read_csv(results_file, sep=",")
        return None

    def makeSky(self, timeindex, dni, dhi, sunalt, sunaz):
        """
        Sets and returns sky information using gendaylit.
        Uses user-provided data for sun position and irradiance.

        .. warning::
            This generates the sky at the sun altitude&azimuth provided, make
            sure it is the right position relative to how the weather data got
            created and read (i.e. label right, left or center).


        Parameters
        ------------
        timeindex : pd.Timestamp
            pd.Timestamp of the MetObj (daylight hours only)
        dni: int or float
           Direct Normal Irradiance (DNI) value, in W/m^2
        dhi : int or float
           Diffuse Horizontal Irradiance (DHI) value, in W/m^2
        sunalt : int or float
           Sun altitude (degrees)
        sunaz : int or float
           Sun azimuth (degrees)

        Returns
        -------
        skyname : string
           Filename of sky in /skies/ directory
        """
        print("Sky generated with Gendaylit 2 MANUAL, with DNI: %0.1f, DHI: %0.1f" % (dni, dhi))

        sky_path = "skies"

        if sunalt <= 0 or dhi <= 0:
            self.skyfiles = [None]
            return None

        # Assign Albedos
        try:
            if self.ground.ReflAvg.shape[0] == 1:  # just 1 entry
                groundindex = 0
            else:
                print("Ambiguous albedo entry, Set albedo to single value " "in setGround()")
                return None
        except Exception:
            print("usage: make sure to run setGround() before gendaylit()")
            return None

        # Radiance expects azimuth South = 0, PVlib gives South = 180. Must substract 180 to match.
        sunaz -= 180

        # Note: -W and -O1 are used to create full spectrum analysis in units of Wm-2
        # " -L %s %s -g %s \n" %(dni/.0079, dhi/.0079, self.ground.ReflAvg) +
        skyStr = (
            "# start of sky definition for daylighting studies\n"
            + "# Manual inputs of DNI, DHI, SunAlt and SunAZ into Gendaylit used \n"
            + "!gendaylit -ang %s %s" % (sunalt, sunaz)
        ) + (
            " -W %s %s -g %s -O 1 \n" % (dni, dhi, self.ground.ReflAvg[groundindex])
            + "skyfunc glow sky_mat\n0\n0\n4 1 1 1 0\n"
            + "\nsky_mat source sky\n0\n0\n4 0 0 1 180\n"
            + self.ground._makeGroundString(index=groundindex, cumulativesky=False)
        )

        time_str = timeindex.strftime("%Y-%m-%d_%H%M")
        sky_name = os.path.join(sky_path, f"sky2_{self.name}_{time_str}.rad")

        sky_file = open(sky_name, "w")
        sky_file.write(skyStr)
        sky_file.close()

        self.skyfiles = [sky_name]

        return sky_name


# noinspection PyPep8Naming, SpellCheckingInspection
class Analysis(AnalysisObj):
    """
    Analysis class for performing raytrace to obtain irradiance measurements
    at the array, as well plotting and reporting results.
    """

    def makeAnalysis(self, scene, octfile, modWanted=None, rowWanted=None, sensorsy=9, sensorsx=1, **kwargs):
        frontscan, backscan = self.moduleAnalysis(
            scene,
            modWanted=modWanted,
            rowWanted=rowWanted,
            sensorsy=sensorsy,
            sensorsx=sensorsx,
            **kwargs,
        )

        # Compare the back vs front irradiance
        self.analysis(octfile, self.name, frontscan, backscan)


class WriteLogger:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        for line in message.rstrip().splitlines():
            self.level(line.rstrip())

    def flush(self):
        pass
