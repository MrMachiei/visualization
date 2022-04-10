#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

def read_data(source):
    return pd.read_csv(source, index_col = ["generation", "effort"])

def data_to_mean(data):
    return list(data.mean(axis=1))

def ploter(plt, box):
    ox = list(range(2500, 500001, 2500))
    evolrs = read_data("1evolrs.csv")
    evolrs_mean = data_to_mean(evolrs)
    evol_L, = plt.plot(ox, evolrs_mean, 'bo-', label='1-Evol-RS', markevery=25, markeredgecolor='black')

    coev1rs = read_data("1coevrs.csv")
    coev1rs_mean = data_to_mean(coev1rs)
    coev1_R_L, = plt.plot(ox, coev1rs_mean, 'gv-', label='1-Coev-RS', markevery=25, markeredgecolor='black')

    coev2rs = read_data("2coevrs.csv")
    coev2rs_mean = data_to_mean(coev2rs)
    coev2_R_L, = plt.plot(ox, coev2rs_mean, 'rD-', label='2-Coev-RS', markevery=25, markeredgecolor='black')

    coev1 = read_data("1coev.csv")
    coev1_mean = data_to_mean(coev1)
    coev1_L, = plt.plot(ox, coev1_mean, 'ks-', label='1-Coev', markevery=25, markeredgecolor='black')

    coev2 = read_data("2coev.csv")
    coev2_mean = data_to_mean(coev2)
    coev2_L, = plt.plot(ox, coev2_mean,'md-', label='2-Coev', markevery=25, markeredgecolor='black')

    evolrs2 = evolrs.iloc[-1].tolist()[2::]
    coev1rs2 = coev1rs.iloc[-1].tolist()[2::]
    coev2rs2 = coev2rs.iloc[-1].tolist()[2::]
    coev12 = coev1.iloc[-1].tolist()[2::]
    coev22 = coev2.iloc[-1].tolist()[2::]

    plt.legend(loc='lower right', numpoints=2)

    box.boxplot([evolrs2, coev1rs2, coev2rs2, coev12, coev22], notch=True,
                sym='b+', showmeans=True,
                boxprops={'color':'blue'}, whiskerprops={'linestyle': '--', 'color' : 'blue'},
                meanprops={'marker' : 'o', 'markerfacecolor' : 'blue', 'markeredgecolor' : 'black', 'markersize' : 4},
                medianprops={'color' : 'red'})
def main():
    param = {'xtick.direction': 'in', 'ytick.direction': 'in',
           'text.usetex': True}
    plt.rcParams.update(param)

    f, plots = plt.subplots(nrows = 1, ncols=2)

    line = plots[0]
    box = plots[1]

    line.set_ylabel('Odsetek wygranych gier [\%]')
    line.set_xlabel('Rozegranych gier(x1000)')
    line.set_xlim(left=0, right=500000)
    line.set_ylim([0.6, 1])
    format = lambda x, pos: '{:.0f}'.format(x * 100, pos)
    line.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format))
    format = lambda x, pos: '{:.0f}'.format(x / 1000, pos)
    line.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(format))
    line.grid(True, linestyle='--')
    gora = line.axes.twiny()
    gora.set_xlim(left=0, right=200)
    gora.set_xlabel('Pokolenie')
    gora.set_xticks(list(range(0, 201, 40)))

    box.set_xticklabels(['1-Evol-RS', '1-Coev-RS', '2-Coev-RS', '1-Coev', '2-Coev'], rotation=22.5)
    box.set_ylim([0.6, 1])
    format = lambda x, pos: '{:.0f}'.format(x * 100, pos)
    box.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format))
    box.grid(True, linestyle='--')
    box.yaxis.tick_right()
    ploter(line, box)
    plt.show()
    f.savefig('myplot.pdf')
    plt.close()

if __name__ == '__main__':
    main()