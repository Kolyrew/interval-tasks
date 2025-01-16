import intvalpy as ip
import numpy as np
def scalar_to_interval(x, rad):
    return ip.Interval(x - rad, x + rad)


scalar_to_interval_vec = np.vectorize(scalar_to_interval)