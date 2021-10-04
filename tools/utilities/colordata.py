import numpy as np

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

def PixelSummary(data: np.ndarray):
    internal = {}
    internal["red"] = data[0]
    internal["green"] = data[1]
    internal["blue"] = data[2]
    internal["transparency"] = 0 if len(data) == 3 else data[3]
    internal["hex"] = GetHexString(data)
    internal["white"], internal["whitepercent"] = WhiteData(data)
    internal["black"], internal["blackpercent"] = BlackData(data)
    return internal
