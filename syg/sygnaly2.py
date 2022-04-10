from pylab import *
from numpy import *
import scipy
import scipy.fftpack
import scipy.io.wavfile
import itertools

def zad2(A, F, w, LP, f):
    #--- Definiujemy sygnal wejsciowy
    T = 1/F      # Okres sygnalu [s]

    #--- Probkujemy sygnal
    TW = 1/w     # Okres probkowania [s] (co ile sekund pobieramy próbkę)

    t = np.arange(0, LP*T, TW) # Momenty, w których pobieramy próbki (oś OX)

    n = len(t)                 # Liczba próbek
    signal = f(t)

    #--- Rysujemy sygnał (niebieskie kółka)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    ax = fig.add_subplot(131)
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
    ax = fig.add_subplot(132)
    ymax = max(signal1)
    ax.set_ylim([0.0, max(1.1*ymax, 0.3)])
    xlabel("Czestotliwosc [Hz]")
    ylabel("Amplituda [m]")

    freqs = range(n)
    xt = np.linspace(0, w, n, endpoint=False)
    stem(xt, signal1, '-*', use_line_collection=True)

    signal1[20] = 0
    signal1[180] = 0
    subplot(133)
    plt.xlabel('')
    plt.ylabel('')
    dane = fft.ifft(signal1)
    plot(dane)

    show()

def zad3(A, F, w, LP, f):
    #--- Definiujemy sygnal wejsciowy
    T = 1/F      # Okres sygnalu [s]

    #--- Probkujemy sygnal
    TW = 1/w     # Okres probkowania [s] (co ile sekund pobieramy próbkę)

    t = np.arange(0, LP*T, TW) # Momenty, w których pobieramy próbki (oś OX)

    n = len(t)                 # Liczba próbek
    signal = f(t)

    #--- Rysujemy sygnał (niebieskie kółka)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    ax = fig.add_subplot(131)
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
    ax = fig.add_subplot(132)
    ymax = max(signal1)
    ax.set_ylim([0.0, max(1.1*ymax, 0.3)])
    xlabel("Czestotliwosc [Hz]")
    ylabel("Amplituda [m]")

    freqs = range(n)
    xt = np.linspace(0, w, n, endpoint=False)
    stem(xt, signal1, '-*', use_line_collection=True)

    Y = scipy.fftpack.fftshift(scipy.fftpack.fft(signal1))
    z = np.angle(Y, deg=True)

    subplot(133)
    plt.xlabel('')
    plt.ylabel('')
    title("Faza")
    plot(freqs, z)

    show()

def zad1():
    plamy = np.genfromtxt("spots.txt")

    plt.figure(figsize=(10,10))

    subplot(121)
    plt.xlabel('n')
    plt.ylabel('aktywność')

    plot(plamy)


    subplot(122)
    plt.xlabel('częstotliwość')
    plt.ylabel('spektrum')

    plamy_fft = abs(fft.fft(plamy))
    freq = np.linspace(0, 12, plamy_fft.size)
    stem(freq, plamy_fft, 'r')

    plot(plamy_fft)
    #show()
    plamy_fft[0] = 0
    maxf = freq[argmax(plamy_fft)]
    print("Częstotliwość cyklu aktywności słoneczej", maxf)

def draw(w, original_signal, log_scale = False):
    signal = [s[0] for s in original_signal]

    signal1 = abs(fft.fft(signal))

    signal1 = signal1[::20]
    freqs = np.linspace(0, w, signal1.size)
    stem(freqs, signal1, '-*')
    if log_scale:
        plt.yscale('log')

    show()
    return signal1

def dom_freq(w, original_signal):
    freqs = rfftfreq(original_signal.size, 1/w)

    signal = abs(rfft([s[0] for s in original_signal]))/(original_signal.size/2)

    sig = signal.copy()
    s = 11
    max_f = 0
    temp = 0
    czest = []
    while (s > 10):
        tempmax = sig[max_f:].argmax()
        temp += tempmax
        s = signal[temp]
        if s > 10:
            czest.append(freqs[temp])
            max_f = tempmax + 10
    return czest


def zad4():
    w, original_signal = scipy.io.wavfile.read('err.wav')

    draw(w, original_signal)
    draw(w, original_signal, True)

    print("Dominujące częstotliwości:", end=" ")
    print(dom_freq(w, original_signal))

def zad5():
    a = [1, 2, 3]
    b = [1, 2]
    print(123 * 12)

    # Należy odpowiednio dobrać “padding”. W przeciwnym wypadku na końcu wyniku pojawią się dodatkowe zera.
    A = fft.fft(a, 4)
    B = fft.fft(b, 4)
    C = A * B
    c = abs(ifft(C))
    print(c)
zad1()
zad2(F=1, w=20, A=1, LP=10, f=lambda t: ( sin(2*pi*t*1) + sin(4*pi*t*1)))
zad3(F=1, w=20, A=1, LP=5, f=lambda t: ( sin(2*pi*t*1) + sin(4*pi*t*1)))
zad3(F=1, w=20, A=1, LP=5, f=lambda t: ( sin(2*pi*t*1) + cos(4*pi*t*1)))

zad4()