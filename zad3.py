#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division             # Division in Python 2.7
import matplotlib
matplotlib.use('Agg')                       # So that we can render files without GUI
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from matplotlib import colors
def read_dem(file):
    f = open(file, "r")
    w,h,od = [int(i) for i in f.readline().split(' ')]
    dem = list()
    for i in range(0, h):
        t = f.readline().split(' ')
        dem.append([float(i) for i in t[:len(t)-1]])
    return [dem,w,h]

def cal_cos(s, v):
    return (s[0]*v[0] + s[1]*v[1] + s[2]*v[2])/((s[0]**2+s[1]**2+s[2]**2)**(1/2) *(v[0]**2+v[1]**2+v[2]**2)**(1/2))

def plot_dem(dem, w, h):
    param = {'xtick.direction': 'in', 'ytick.direction': 'in',
             'text.usetex': True}
    plt.rcParams.update(param)
    f, plots = plt.subplots(nrows=1, ncols=1)

    img = np.zeros((w,h,3))
    maks = 0
    for i in dem:
        t = max(i)
        if t > maks:
            maks = t
    sun = [250,250, 500]
    for i in range(h):
        for j in range(w):
            s = cal_cos(sun, [i,j,dem[i][j]])
            img[i,j] = gradient_hsv_map(dem[i][j]/maks, 100)
    #im = plots.imshow(dem,cmap="terrain")
    im = plots.imshow(img)
    f.savefig('map.pdf')




def plot_color_gradients(gradients, names):
    param = {'xtick.direction': 'in', 'ytick.direction': 'in',
             'text.usetex': True}
    plt.rcParams.update(param)
    # For pretty latex fonts (commented out, because it does not work on some machines)
    #rc('text', usetex=True)
    #rc('font', family='serif', serif=['Times'], size=10)
    rc('legend', fontsize=10)

    column_width_pt = 400         # Show in latex using \the\linewidth
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=1.00, bottom=0.05, left=0.25, right=0.95)


    for ax, gradient, name in zip(axes, gradients, names):
        # Create image with two lines and draw gradient on it
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.25
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=10)

    fig.savefig('my-gradients.pdf')

def hsv2rgb(h, s, v):
    if s == 0:
        R = G = B = v
    else:
        Hi = int(h/60)
        f = h/60 - Hi
        p = v * (100 - s)/100
        q = v * (100 - f * s)/100
        t = v * (100 - s * (1 - f))/100
        if Hi == 0:
            R = v
            G = t
            B = p
        elif Hi == 1:
            R = q
            G = v
            B = p
        elif Hi == 2:
            R = p
            G = v
            B = t
        elif Hi == 3:
            R = p
            G = q
            B = v
        elif Hi == 4:
            R = t
            G = p
            B = v
        else:
            R = v
            G = p
            B = q
    return (R, G, B)

def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    if v <= 0.5:
        return (0, 1-v*2, v*2)
    else:
        return ((v-0.5)*2, 0, 1-(v-0.5)*2)


def gradient_rgb_gbr_full(v):
    if v <= 0.25:
        return (0, 1, v * 4)
    elif v <= 0.5:
        return (0, 1 - (v-0.25)*4, 1)
    elif v <= 0.75:
        return ((v-0.5)*4, 0, 1)
    else:
        return(1, 0, 1 - (v - 0.75) * 4)

def gradient_rgb_wb_custom(v):
    #(1,1,1) -> (1,0,1) -> (0,0,1) -> (0,1,1) ->
    #(0,1,0) -> (1,1,0) -> (1,0,0) -> (0,0,0)
    if v <= 1 / 7:
        return (1,1-v*7,1)
    elif v <= 2 / 7:
        return (1-(7*v-1),0,1)
    elif v <= 3 / 7:
        return (0,(7*v-2),1)
    elif v <= 4 / 7:
        return(0,1,1-(7*v-3))
    elif v <= 5 / 7:
        return ((v*7-4),1,0)
    elif v <= 6 / 7:
        return (1,1-(7*v-5),0)
    else:
        return (1-(7*v-6),0,0)


def gradient_hsv_bw(v):
    return hsv2rgb(0, 0, v)


def gradient_hsv_gbr(v):
    h = (120 +240 * v) % 360
    return hsv2rgb(h, 100, 1)

def gradient_hsv_unknown(v):
    h = 120 - 120 * v
    return hsv2rgb(h, 50, 1)

def gradient_hsv_map(v, s = 50):
    h = 120 - 120 * v
    return hsv2rgb(h, s, 1)

def gradient_hsv_custom(v):
    h = 360 * v
    s = 100 - 100 * v
    return hsv2rgb(h, s, 1)


if __name__ == '__main__':
    def toname(g):
        return g.__name__.replace('gradient_', '').replace('_', '-').upper()

    gradients = (gradient_rgb_bw, gradient_rgb_gbr, gradient_rgb_gbr_full, gradient_rgb_wb_custom,
                 gradient_hsv_bw, gradient_hsv_gbr, gradient_hsv_unknown, gradient_hsv_custom)

    plot_color_gradients(gradients, [toname(g) for g in gradients])

    map = read_dem("big.dem")
    plot_dem(map[0],map[1],map[2])
