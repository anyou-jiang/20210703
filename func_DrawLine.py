import numpy as np
from math import pi
from math import sin
from math import cos
import matplotlib.pyplot as plt
import scipy.io

def func_DrawLine(Img, X0, Y0, X1, Y1, nG):
    W = Img.shape[0]
    H = Img.shape[1]

    if X0 >= 0 and X0 < W and Y0 >= 0 and Y0 < H:
        Img[X0, Y0] = nG
    if X1 >= 0 and X1 < W and Y1 >= 0 and Y1 < H:
        Img[X1, Y1] = nG

    if abs(X1 - X0) <= abs(Y1 - Y0): # 'tall' shape
        if Y1 < Y0:
            X1, X0 = X0, X1
            Y1, Y0 = Y0, Y1
        if X1 >= X0 and Y1 >= Y0: # /  Y1 >= Y0 always true
            dy = Y1 - Y0
            dx = X1 - X0 # positives
            p = 2 * dx
            n = 2 * dy - 2 * dx
            tn = dy
            while Y0 < Y1:
                if tn >= 0:
                    tn = tn - p
                else:
                    tn = tn + n
                    X0 = X0 + 1
                Y0 = Y0 + 1
                if X0 >= 0 and X0 < W and Y0 >= 0 and Y0 < H:
                    Img[X0, Y0] = nG
        else: # \
            dy = Y1 - Y0
            dx = X1 - X0 # negative
            p = -2 * dx
            n = 2 * dy + 2 * dx
            tn = dy
            while Y0 <= Y1:
                if tn >= 0:
                    tn = tn - p
                else:
                    tn = tn + n
                    X0 = X0 - 1
                Y0 = Y0 + 1
                if X0 >= 0 and X0 < W and Y0 >= 0 and Y0 < H:
                    Img[X0, Y0] = nG
    else: # 'flat' shape
        if X1 < X0:
            X1, X0 = X0, X1
            Y1, Y0 = Y0, Y1
        if X1 >= X0 and Y1 >= Y0: # /
            dy = Y1 - Y0
            dx = X1 - X0
            p = 2 * dy
            n = 2 * dx - 2 * dy
            tn = dx
            while X0 < X1:
                if tn >= 0:
                    tn = tn - p
                else:
                    tn = tn + n
                    Y0 = Y0 + 1
                X0 = X0 + 1
                if X0 >= 0 and X0 < W and Y0 >= 0 and Y0 < H:
                    Img[X0, Y0] = nG
        else: # \
            dy = Y1 - Y0
            dx = X1 - X0
            p = -2 * dy
            n = 2 * dy + 2 * dx
            tn = dx
            while X0 < X1:
                if tn >= 0:
                    tn = tn - p
                else:
                    tn = tn + n
                    Y0 = Y0 - 1
                X0 = X0 + 1
                if X0 >= 0 and X0 < W and Y0 >= 0 and Y0 < H:
                    Img[X0, Y0] = nG

    return Img