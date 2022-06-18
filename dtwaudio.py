#%%
import os
from winreg import QueryInfoKey
from wsgiref import headers
#import librosa
import matplotlib.pylab as plt
import numpy as np
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dtw import *
import pandas as pd
#%%
def obliczanie_sciezki(macierz_sledzenia):
    N, M = macierz_sledzenia.shape        #uzyskanie wielkości macierzy czyli uzyskanie wielkości przebiegów x i y
    n = N - 1                                       #uzyskanie wspołrzędjej y końca macierzy
    m = M - 1                                       #uzyskanie wspołrzędnej x końca macierzy
    sciezka = [(n, m)]                              #wspołrzędne początkowe ścieżki
    while n > 0 or m > 0:
        wybor_sciezki = macierz_sledzenia[n, m]     #wybor ruchu
        if wybor_sciezki == 0:                      #ruch po skosie
            n = n - 1
            m = m - 1
        elif wybor_sciezki == 1:                    #ruch w pionie
            n = n - 1
        elif wybor_sciezki == 2:                    #ruch w poziomie
            m = m - 1
        sciezka.append((n, m))                      #dodanie kolejnej wspołrzędnej ścieżki

    sciezka = sciezka[::-1]                         #usuniecie ostatnich współrzędnych ścieżki
    return sciezka                                  #zwrócenie optymalnej ścieżki
#%%
def obliczanie_macierzy_kosztow(macierz_odl):
    N, M = macierz_odl.shape        #uzyskanie wielkości macierzy czyli uzyskanie wielkości przebiegów x i y

    macierz_kosz = np.zeros((N + 1, M + 1))         #stworzenie macierzy (N+1)x(M+1) i wypełnienie jej zerami
    for i in range(1, N + 1):
        macierz_kosz[i, 0] = np.inf                 #wypełnienie dodatkowego rzędu wartościami nieskończonymi
    for i in range(1, M + 1):
        macierz_kosz[0, i] = np.inf                 #wypełnienie dodatkowej kolumny wartościami nieskończonymi

    macierz_sledzenia = np.zeros((N, M))            #stworzenie macierzy Nx(M+1) i wypełnienie jej zerami
    for i in range(1, N+1):
        for j in range(1, M+1):
            D = [macierz_kosz[i-1, j-1],            #stworzenie listy zawierającej możliwe pola ruchu
                 macierz_kosz[i-1, j],
                 macierz_kosz[i, j-1]]
            D_id = np.argmin(D)                     #uzyskanie numeru indeksu o najmniejszej wartości RMS
            macierz_kosz[i, j] = macierz_odl[i-1, j-1] + D[D_id]        #nadpisanie elementu macierzy
            macierz_sledzenia[i-1, j-1] = D_id                          #nadpisanie macierzy śledzenia
    macierz_kosz = macierz_kosz[1:, 1:]             #usunięcie 1 rzędu i kolumny (wartości nieskończone)
    sciezka = obliczanie_sciezki(macierz_sledzenia) #uzyskanie optymalnej sciezki
    return (sciezka, macierz_kosz)                  #zwrócenie ścieżki oraz macierzy kosztów
#%%
directory = 'Audio'
y=[]
f_s=[]
for root, dirs, files in os.walk(directory):
    for filename in files:
        name = os.path.join(root, filename)
        print(name)
        z, f_ss = librosa.load(name)
        y.append(z)
        f_s.append(f_ss)
        plt.figure(figsize=(10, 7))                                             #stworzenie okna na wykres
        plt.plot(z)
        plt.title(name)         
#%%
i1=pd.read_csv("i1.csv",header = None)
i2=pd.read_csv("i2.csv",header = None)
j1=pd.read_csv("j1.csv", sep = ';', header=None)
j2=pd.read_csv("j2.csv",sep = ';', header=None)
#%%
alignment = dtw(i1[0], j1[0], keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")
#%%
## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(i1[], i2, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()
#%%
for r in al:
## Display the warping curve, i.e. the alignment curve
    r.plot(type="threeway")
    dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()
#%%
N = yy[1].shape[0]                                  #uzyskanie wielkości macierzy (ilości punktów) przebiegu x
M = y[0].shape[0]                                  #uzyskanie wielkości macierzy (ilości punktów) przebiegu y
macierz_odl = np.zeros((N, M))                  #stworzenie macierzy NxM i wypełnienie jej zerami
#for i in range(N):
 #   for j in range(M):
  #      macierz_odl[i, j] = abs(x[i] - y[j])    #wyznaczenie wartości poszczególnych elementów macierzy odległości

sciezka, macierz_kosz = obliczanie_macierzy_kosztow(macierz_odl)        #uzyskanie wspórzędnych ścieżki oraz macierzy kosztów
sciezka_x, sciezka_y = zip(*sciezka)            #uzyskanie składowych x i y współrzędnych
odleglosc = (abs(min(y)) + abs(max(yy)))/2 + 0.5     #wyliczenie odległości między dwoma wykresami

print("Alignment cost: {:.4f}".format(macierz_kosz[N - 1, M - 1]))
print("Normalized alignment cost: {:.4f}".format(macierz_kosz[N - 1, M - 1]/(N + M)))

#%%
for a in z:
    fig, wykres = plt.subplots(1, 2, figsize=(25,10))  
    skala_1 = wykres[0].imshow(macierz_odl, cmap=plt.cm.binary, interpolation="nearest", origin="lower")    #wyrysowanie macierzy odległości
    wykres[0].set_title("Macierz odległości", size = 18)        #nadanie nazwy wykresowi
    wykres[0].plot(sciezka_y, sciezka_x)                        #narysowanie ścieżki na macierzy odległości
    fig.colorbar(skala_1, ax=wykres[0])                         #wyrysowanie ramki z podziałem stopniowym

    divider = make_axes_locatable(wykres[1])                    #stworzenie wykresów lokalnych
    przebieg_x = divider.append_axes("left", 1, pad=0.5, sharey=wykres[1])      #utworzenie miejsca na wykres przebiegu x
    przebieg_x.plot(x, np.arange(x.shape[0]))                   #utowrzenie przebiegu x
    przebieg_x.xaxis.set_tick_params(labelbottom=False)         #usunięcie wartości osi x
    przebieg_x.yaxis.set_tick_params(labelleft=False)           #usuniecie wartości osi y

    przebieg_y = divider.append_axes("bottom", 1, pad=0.5, sharex=wykres[1])    #utworzenie miejsca na wykres przebiegu x
    przebieg_y.plot(np.arange(y.shape[0]), y)                   #utowrzenie przebiegu y
    przebieg_y.xaxis.set_tick_params(labelbottom=False)         #usunięcie wartości osi x
    przebieg_y.yaxis.set_tick_params(labelleft=False)           #usunięcie wartości osi y

    skala_2 = wykres[1].imshow(macierz_kosz, cmap=plt.cm.binary, interpolation="nearest", origin="lower")   #wyrysowanie macierzy kosztów
    wykres[1].set_title("Macierz kosztów", size = 18)           #nadanie nazwy wykresowi
    wykres[1].plot(sciezka_y, sciezka_x)                        #narysowanie ścieżki na macierzy kosztów

    fig.colorbar(skala_2, ax=wykres[1])                         #wyrysowanie ramki z podziałem stopniowym
# %%
