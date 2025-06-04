import math
import numpy as np
import pickle
import pandas as pd
from scipy.special import kl_div


def read_with_pd(path, delimiter="\t", header=None):
    data_pd = pd.read_csv(path, delimiter=delimiter, header=header)
    return data_pd[0].tolist()


def save_pkl(name, obj):
    """save obj with pickle"""
    name = name.replace(".pkl", "")
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name):
    """load a pickle object"""
    name = name.replace(".pkl", "")
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def write_in_file(path_to_file, data):
    with open(path_to_file, "w+") as f:
        for item in data:
            f.write("%s\n" % item)


def scale_range(x, a, b):
    """scale an input between a and b, b>a"""
    assert b > a

    min_x = np.min(x)
    max_x = np.max(x)

    x_scaled = (b - a) * np.divide(x - min_x, max_x - min_x) + a

    return x_scaled, min_x, max_x


def unscale_range(x_scaled, a, b, min_x, max_x):
    """unscale a scaled input"""
    assert b > a
    assert max_x > min_x

    x = np.divide(x_scaled - a, b - a) * (max_x - min_x) + min_x

    new_min = np.min(x)
    new_max = np.max(x)

    return x, new_min, new_max


def get_distributions(arr):

    if len(arr) != 2:
        raise ValueError("Please enter only two arrays")

    temp_arr = np.hstack((arr[0], arr[1]))
    l_bound, u_bound = np.min(temp_arr), np.max(temp_arr)

    bins = np.arange(math.floor(l_bound), math.ceil(u_bound))

    p, _ = np.histogram(arr[0], bins=bins, density=True)
    q, _ = np.histogram(arr[1], bins=bins, density=True)

    return p, q


def KL_div(p, q):
    return sum(np.nan_to_num(kl_div(p, q), posinf=0))


def JS_div(p, q):
    M = 0.5 * (p + q)
    return 0.5 * (KL_div(p, M) + KL_div(q, M))
