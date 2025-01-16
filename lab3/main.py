import os

import numpy as np
import intvalpy as ip
import matplotlib.pyplot as plt
from functools import cmp_to_key

from itools import scalar_to_interval_vec
from utils import read_bin_file_with_numpy

ip.precision.extendedPrecisionQ = False

def mode(X):
    if X is None:
        return None


    boundaries = []
    for el in X:
        boundaries.append((el.a, 'start'))
        boundaries.append((el.b, 'end'))


    boundaries.sort(key=lambda x: (x[0], x[1] == 'end'))

    max_count = 0
    current_count = 0
    mode_intervals = []
    start_point = None
    is_active = False


    for point, boundary_type in boundaries:
        if boundary_type == 'start':
            current_count += 1
            if current_count == max_count and not is_active:
                start_point = point
                is_active = True
            elif current_count > max_count:
                max_count = current_count
                start_point = point
                mode_intervals.clear()
                is_active = True
        else:
            if is_active and current_count == max_count:
                mode_intervals.append(ip.Interval(start_point, point))
                is_active = False
            current_count -= 1

    return mode_intervals



def med_K(X):
    c_inf = [ip.inf(el) for el in X]
    c_sup = [ip.sup(el) for el in X]

    return ip.Interval(np.median(c_inf), np.median(c_sup))


def med_P(X):
    x = sorted(X, key=cmp_to_key(lambda x, y: (x.a + x.b) / 2 - (y.a + y.b) / 2))

    index_med = len(x) // 2

    if len(x) % 2 == 0:
        return (x[index_med - 1] + x[index_med]) / 2

    return x[index_med]


def coefficient_Jakkard(X_data, Y_data=None):
    if Y_data is None:
        x_inf = [ip.inf(x) for x in X_data]
        x_sup = [ip.sup(x) for x in X_data]
        return (min(x_sup) - max(x_inf)) / (max(x_sup) - min(x_inf))

    if isinstance(X_data, ip.ClassicalArithmetic) and isinstance(Y_data, ip.ClassicalArithmetic):
        return (min(ip.sup(X_data), ip.sup(Y_data)) - max(ip.inf(X_data), ip.inf(Y_data))) / \
            (max(ip.sup(X_data), ip.sup(Y_data)) - min(ip.inf(X_data), ip.inf(Y_data)))

    jakkard_v = []
    for x, y in zip(X_data, Y_data):
        coeff = (min(ip.sup(x), ip.sup(y)) - max(ip.inf(x), ip.inf(y))) / (
                max(ip.sup(x), ip.sup(y)) - min(ip.inf(x), ip.inf(y)))
        jakkard_v.append(coeff)

    return jakkard_v



def func_a(a, X, Y):
    return np.mean(coefficient_Jakkard(X + a, Y))


def func_t(t, X, Y):
    return np.mean(coefficient_Jakkard(X * t, Y))


def func_mode_a(a, X, Y):
    return np.mean(coefficient_Jakkard(mode(X + a), Y))


def func_mode_t(t, X, Y):
    return np.mean(coefficient_Jakkard(mode(X * t), Y))


def func_med_p_a(a, X, Y):
    return np.mean(coefficient_Jakkard(med_P(X + a), med_P(Y)))


def func_med_p_t(t, X, Y):
    return np.mean(coefficient_Jakkard(med_P(X * t), med_P(Y)))


def func_med_k_a(a, X, Y):
    return np.mean(coefficient_Jakkard(med_K(X + a), med_K(Y)))


def func_med_k_t(t, X, Y):
    return np.mean(coefficient_Jakkard(med_K(X * t), med_K(Y)))


def draw_func(f, a, b, parametr: str, func="", amount=50):
    X_linsp = np.linspace(a, b, amount)
    y = []

    for i, x in enumerate(X_linsp):
        y.append(f(x))

    y_max = np.max(y)
    ind_max = y.index(y_max)
    x_max = X_linsp[ind_max]
    y_max = np.max(y)
    plt.plot(X_linsp, y, color='g')
    plt.xlabel(f"{parametr}")
    plt.ylabel(f"Ji({parametr}, {func}(X), {func}(Y))")
    plt.axvline(x=x_max, linestyle='dotted', color='r', label=f"x = {np.round(x_max, 4)}")
    plt.grid()
    plt.legend()
    plt.title("Jaccard Index")
    plt.savefig(os.path.join(PATH_IMAGES, "img", f"Jaccadrd-{parametr}-{func}"))
    plt.show()
    print(f"F_{parametr}_{func} = {y_max}, {parametr}_{func} = [{x_max-(0.008, 0.003)[parametr!='a']}, {x_max+(0.008, 0.003)[parametr!='a']}]")


def get_transformed_values():
    x_data = read_bin_file_with_numpy('-0.205_lvl_side_a_fast_data.bin')
    y_data = read_bin_file_with_numpy('0.225_lvl_side_a_fast_data.bin')
    x_voltage = x_data / 16384.0 - 0.5
    y_voltage = y_data / 16384.0 - 0.5
    return x_voltage, y_voltage


PATH_IMAGES = r"/Users/evgeniiromancuk/Documents/Politech/Interval_analysis/Lab3"

def generate_search_data():
    x_voltage, y_voltage = get_transformed_values()
    y_test = np.transpose(y_voltage, axes=(2, 1, 0))
    x_test = np.transpose(x_voltage, axes=(2, 1, 0))


    X, Y = np.mean(x_test, axis=2), np.mean(y_test, axis=2)

    def sigma_filter(data, axis=0, n_sigma=3):
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std
        mask = (data >= lower_bound) & (data <= upper_bound)
        return data * mask

    X_filtered = sigma_filter(X, axis=0, n_sigma=3)
    Y_filtered = sigma_filter(Y, axis=0, n_sigma=3)

    X_filtered, Y_filtered = X_filtered.flatten(), Y_filtered.flatten()

    R = 0.000062
    bound_a_l, bound_a_r = np.min(Y) - np.max(X), np.max(Y) - np.min(X)
    bound_t_l, bound_t_r = np.min(Y) / np.max(X), np.max(Y) / np.min(X)
    X, Y = scalar_to_interval_vec(X_filtered, R), scalar_to_interval_vec(Y_filtered, R)

    draw_func(lambda a: func_a(a, X, Y), bound_a_l, bound_a_r, "a", amount=30)
    draw_func(lambda t: func_t(t, X, Y), 1.2 * bound_t_l, bound_t_r, "t")
    # ----
    draw_func(lambda a: func_med_k_a(a, X, Y), bound_a_l, bound_a_r, "a", "med_k")
    draw_func(lambda t: func_med_k_t(t, X, Y), bound_t_l, bound_t_r, "t", "med_k")
    # ----
    draw_func(lambda a: func_med_p_a(a, X, Y), bound_a_l, bound_a_r, "a", "med_p")
    draw_func(lambda t: func_med_p_t(t, X, Y), bound_t_l, bound_t_r, "t", "med_p")
    # ----
    m_y = mode(Y)
    draw_func(lambda a: func_mode_a(a, X, m_y), bound_a_l, bound_a_r, "a", "mode", amount=50)
    draw_func(lambda t: func_mode_t(t, X, m_y), bound_t_l, bound_t_r, "t", "mode", amount=50)



generate_search_data()