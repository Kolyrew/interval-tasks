import math
import numpy as np
from interval import Interval
from decimal import Decimal, getcontext
import matplotlib.pyplot as plt

getcontext().prec = 50

def det(mid_matrix, alpha):
  i1 = Interval(Decimal(mid_matrix[0][0]) - alpha, Decimal(mid_matrix[0][0]) + alpha)
  i2 = Interval(Decimal(mid_matrix[0][1]) - alpha, Decimal(mid_matrix[0][1]) + alpha)
  i3 = Interval(Decimal(mid_matrix[1][0]) - alpha, Decimal(mid_matrix[1][0]) + alpha)
  i4 = Interval(Decimal(mid_matrix[1][1]) - alpha, Decimal(mid_matrix[1][1]) + alpha)
  det = i1 * i4 - i2 * i3
  return det

def find_upper_bound(mid_matrix):
  alpha = Decimal(1)

  while 0 not in det(mid_matrix, alpha):
    alpha = Decimal(math.exp(k))

  return alpha

def find_alpha(mid_matrix, eps=10e-18):
  a = Decimal(0)
  b = find_upper_bound(mid_matrix)
  k = 0

  alphas = []
  dets = []

  while b - a > eps:
    mid = (a + b) / 2

    alphas.append(mid)
    dets.append(det(mid_matrix, mid))

    if 0 in det(mid_matrix, mid):
      b = mid
    else:
      a = mid

    k += 1

  alpha = (a + b) / 2

  alphas.append(alpha)
  dets.append(det(mid_matrix, alpha))

  return alphas, dets

if __name__ == '__main__':
  mid_matrix = np.array([
    [Decimal('1.05'), Decimal('0.95')],
    [Decimal('1.0'), Decimal('1.0')],
  ])

  alphas, dets = find_alpha(mid_matrix, 0.00001)

  print(len(alphas), alphas[-1])

  theoretical_alpha = Decimal('0.025')

  ks = np.arange(len(alphas))
  alphas = np.array(alphas)

  deltas = np.abs(alphas - theoretical_alpha) / theoretical_alpha
  ys = np.exp(-ks)

  plt.plot(ks, alphas)
  plt.xlabel('k')
  plt.ylabel('α')
  plt.show()

  plt.semilogy(ks, deltas, label='Относительная погрешность')
  plt.semilogy(ks, ys, label='exp(-k)')
  plt.xticks(ks)
  plt.xlabel('k')
  plt.ylabel('δ')
  plt.legend()
  plt.show()