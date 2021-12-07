# Core.py contains core functionalities and imports for entire toolset

# Supported Image Formats
__meta_supported_formats__ = ["PNG", "JPEG", "JPEG 2000"]

def getmeta_supp_formats():
    return __meta_supported_formats__

# Supported Color Modes
__meta_supported_modes__ = ["RGB", "RGBA"]

def getmeta_supp_modes():
    return __meta_supported_modes__

# Core Imports

# types and classes
from typing import *
from types import *

# files
import pathlib as pl
import os
from io import open

# manipulation and tabulation
import pandas as pd
import numpy as np
from math import sqrt, acos

# images
from PIL import Image

# time
from time import time_ns, process_time_ns, sleep

# concurrency
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# prebuilt utilities
import utilities

def ColorSpaceAngle(istrain: np.ndarray, cstrain: np.ndarray):
    "Used to calculate the 3D Angle between instance (istrain) and class (cstrain) colors in their color space"
    return acos(istrain * cstrain, (sqrt(sum(np.power(istrain, 2))) * sqrt(sum(np.power(cstrain, 2)))))

def DE_Euclid(istrain: np.ndarray, cstrain: np.ndarray):
    "Common Euclidean color difference calculation, similar to CIE76"
    return sqrt(sum(np.power(istrain - cstrain, 2)))

def DE_HP_sRGB(istrain: np.ndarray, cstrain: np.ndarray):
    "Human Perception sRGB color difference calculation"
    rB = (istrain[0] + cstrain[0]) / 2
    dR, dG, dB = np.power((istrain - cstrain), 2)
    rbuf = (2 + (rB / 256)) * dR
    gbuf = 4 * dG
    bbuf = (2 + ((255 - rB) / 256)) * dB
    return sqrt(rbuf + gbuf + bbuf)

def DE_CIE76(istrain: np.ndarray, cstrain: np.ndarray):
    "CIE 1976 Formula; used on CIELAB coordinates"
    return sqrt(sum(np.power(istrain - cstrain, 2)))

def DE_CIE94(istrain: np.ndarray, cstrain: np.ndarray):
    "CIE 1994 Formula, extended to perceptual non-uniformities; used on CIELAB coordinates"
    pass

def DE_CIEDE2000(istrain: np.ndarray, cstrain: np.ndarray):
    "CIE 2000 Formula, successor to CIE94; used on CIELAB coordinates"
    pass
