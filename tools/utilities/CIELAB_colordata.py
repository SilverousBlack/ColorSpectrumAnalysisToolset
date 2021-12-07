from numpy import ndarray, array
from math import sqrt, atan2

def CIELAB_PixelSUmmary(data: ndarray):
    internal = {"L*": data[0], 
                "a*": data[1],
                "b*": data[2]}
    delta = 6/29
    LL = (data[0] + 16) / 116
    def ffunc(value):
        return pow(value , 3) if value > delta else (3 * pow(delta, 2)) * (value - (4/29)) 
    XX = (95.0489 * ffunc(LL + (data[1] / 500)))
    YY = (100 * ffunc(LL))
    ZZ = (108.8840 * ffunc(LL - (data[2] / 200)))
    internal["tristimulus-X"] = XX
    internal["tristimulus-Y"] = YY
    internal["tristimulus-Z"] = ZZ
    uprime = (4 * XX) / (XX + (15 * YY) + (3 * ZZ))
    vprime = (9 * YY) / (XX + (15 * YY) + (3 * ZZ))
    internal["u*"] = (13 * data[0]) * (uprime - 0.197829)
    internal["v*"] = (13 * data[0]) * (vprime - 0.208148)
    return internal

res = CIELAB_PixelSUmmary(array([50, 128, 128]))
print(res)
