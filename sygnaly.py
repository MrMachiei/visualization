from pylab import *
from numpy import *
import math
from random import uniform
import scipy
from ipywidgets import *

def draw_plots2(A, F, w, LP, f):
    #--- Definiujemy sygnal wejsciowy
    T = 1/F      # Okres sygnalu [s]

    #--- Probkujemy sygnal
    TW = 1/w     # Okres probkowania [s] (co ile sekund pobieramy próbkę)

    t = np.arange(0, LP*T, TW) # Momenty, w których pobieramy próbki (oś OX)

    n = len(t)                 # Liczba próbek
    signal = f(t)

    #--- Rysujemy sygnał (niebieskie kółka)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    ax = fig.add_subplot(121)
    ax.plot(t, signal, 'o')

    #--- Rysujemy sygnał przed spróbkowaniem (dla wizualizacji)
    base_t = np.arange(0, LP * T, 1/200)
    base_signal = f(base_t)

    ax.plot(base_t, base_signal, linestyle='-', color='red')
    ax.set_ylim([min(base_signal), max(base_signal)])
    xlabel("Czas[s]")
    ylabel("Amplituda[m]")

    #--- Wykonujemy FFT
    signal1 = fft.fft(signal)
    signal1 = abs(signal1) # moduł
    # w*LP/2*A -> singal1/w/LP*2
    signal1 = signal1 / (w * LP) * 2 * F

    #--- Rysujemy FFT
    ax = fig.add_subplot(122)
    ymax = max(signal1)
    ax.set_ylim([0.0, max(1.1*ymax, 0.3)])
    xlabel("Czestotliwosc [Hz]")
    ylabel("Amplituda [m]")

    freqs = range(n)
    xt = np.linspace(0, w, n, endpoint=False)
    stem(xt, signal1, '-*', use_line_collection=True)
    print(fft.fft(signal))

    show()


def draw_plots(A, F, w, LP):
    #--- Definiujemy sygnal wejsciowy
    T = 1/F      # Okres sygnalu [s]

    #--- Probkujemy sygnal
    TW = 1/w     # Okres probkowania [s] (co ile sekund pobieramy próbkę)

    t = np.arange(0, LP*T, TW) # Momenty, w których pobieramy próbki (oś OX)
    f = lambda t: A * sin(2*pi*t*F);
    n = len(t)                 # Liczba próbek
    signal = f(t)

    #--- Rysujemy sygnał (niebieskie kółka)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    ax = fig.add_subplot(121)
    ax.plot(t, signal, 'o')

    #--- Rysujemy sygnał przed spróbkowaniem (dla wizualizacji)
    base_t = np.arange(0, LP * T, 1/200)
    base_signal = f(base_t)

    ax.plot(base_t, base_signal, linestyle='-', color='red')
    ax.set_ylim([min(base_signal), max(base_signal)])
    xlabel("Czas[s]")
    ylabel("Amplituda[m]")

    #--- Wykonujemy FFT
    signal1 = fft.fft(signal)
    signal1 = abs(signal1) # moduł
    # w*LP/2*A -> singal1/w/LP*2
    signal1 = signal1 / (w * LP) * 2 * F

    #--- Rysujemy FFT
    ax = fig.add_subplot(122)
    ymax = max(signal1)
    ax.set_ylim([0.0, max(1.1*ymax, 0.3)])
    xlabel("Czestotliwosc [Hz]")
    ylabel("Amplituda [m]")

    freqs = range(n)
    xt = np.linspace(0, w, n, endpoint=False)
    stem(xt, signal1, '-*', use_line_collection=True)

    show()

def szum(t):
    f = lambda t: sin(22*pi*t*1)
    res = f(t)
    for _ in range(100):
        am = uniform(0.1, 0.3)
        f = uniform(2.0, 4.0)
        res += am*sin(22*pi*t*f + pi)
        print(am, f)
    return res



#draw_plots(A=1.0, F=20, w=20, LP=20)
#draw_plots(A=1.0, F=21, w=20, LP=20)

#draw_plots(F=1, A=1, LP=1, w=40)
#draw_plots(F=1, A=2, LP=1, w=40)
#draw_plots(F=1, A=3, LP=1, w=40)


#draw_plots(A=1, F=1, w=50, LP=1)
#draw_plots(A=1, F=1, w=100, LP=1)

#draw_plots(F=10, w=100, A=1, LP=10)
#draw_plots(F=20, w=100, A=1, LP=10)
#draw_plots(F=30, w=100, A=1, LP=10)
#draw_plots(F=40, w=100, A=1, LP=10)


#draw_plots(F=50, w=100, A=1, LP=10)

draw_plots2(F=1, w=20, A=1, LP=10, f= lambda t : ( sin(2*pi*t*1) + sin(4*pi*t*1)))
#draw_plots2(F=1, w=20, A=1, LP=10, f=szum)
#draw_plots2(F=1, w=20, A=1, LP=10, f=lambda t : ( 2*sin(2*pi*t*1) + 0.5))

#draw_plots2(F=1, w=20, A=1, LP=10, f=lambda t : ( sin(2*pi*t*1)))
#draw_plots2(F=1, w=20, A=1, LP=10, f=lambda t : ( sin(2*pi*t*1 + pi/4)))


x = random.random(10)
print(x)
f1 = fft.fft(x)
f2 = fft.ifft(f1)
print(f1)
print(f2)
print((x-f2) <= 0.00001 )