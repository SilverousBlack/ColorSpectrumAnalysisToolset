import numpy as np
from math import pow, sqrt

def ColorDataFromHex(buf: str):
    code = buf.lstrip("#")
    vr = int(code[0:2], 16)
    vg = int(code[2:4], 16)
    vb = int(code[4:6], 16)
    return (vr, vg, vb)

def GetHexString(data: np.ndarray):
    rb = str(hex(data[0]).lstrip("0x").rstrip("L")) or "00"
    rbuf = rb if len(rb) == 2 else "0" + rb
    gb = str(hex(data[1]).lstrip("0x").rstrip("L")) or "00"
    gbuf = gb if len(gb) == 2 else "0" + gb
    bb = str(hex(data[2]).lstrip("0x").rstrip("L")) or "00"
    bbuf = bb if len(bb) == 2 else "0" + bb
    return str("#" + rbuf + gbuf + bbuf).upper()

def WhiteData(data: np.ndarray):
    return [data.min(), (data.min() / 255) * 100]

def BlackData(data: np.ndarray):
    return [(255 - data.max()), ((255 - data.max()) / 255) * 100]

def EuclidColorDifference(istrain: np.array, cstrain: np.array):
    return sqrt(np.sum(np.power(istrain - cstrain, 2)))

def CIEDE2000():
    pass

def PixelSummary(locus: str, data: np.ndarray):
    internal = {"locus": locus}
    internal["red"] = data[0]
    internal["green"] = data[1]
    internal["blue"] = data[2]
    internal["transparency"] = 0 if len(data) == 3 else data[3]
    internal["hex"] = GetHexString(data)
    internal["white"], internal["whitepercent"] = WhiteData(data)
    internal["black"], internal["blackpercent"] = BlackData(data)
    return internal
