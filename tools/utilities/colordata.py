import numpy as np

def GetHexString(data: np.ndarray):
    return "0x" + str(hex(data[0].lstrip("0x").rstrip("L"))) + str(hex(data[1].lstrip("0x").rstrip("L"))) + str(hex(data[2].lstrip("0x").rstrip("L")))

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
