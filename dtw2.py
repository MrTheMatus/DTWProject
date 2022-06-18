#%%
from dtw import *
help(DTW)
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
query, fs1 = librosa.load("Audio\\12.05 - (11.02) - telefon - ulica.m4a")
template, fs2 = librosa.load("Audio\IMS-AF-Nagranie6-komputer2.mp3")
#%%
query2 = np.array_split(query,8)
template2 =np.array_split(template,8)
#%%
for i in range(8):
    idx = np.linspace(0, len(query2[i])/22500.0)
## Find the best match with the canonical recursion formula
    alignment = dtw(query2[i], template2[i], keep_internals=True)

## Display the warping curve, i.e. the alignment curve
    alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    dtw(query2[i], template2[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
    print(rabinerJuangStepPattern(6,"c"))
    rabinerJuangStepPattern(6,"c").plot()
# %%
D = librosa.stft(template)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure()
librosa.display.specshow(S_db)
plt.colorbar()
#%%
D = librosa.stft(query)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure()
librosa.display.specshow(S_db)
plt.colorbar()
#%%
fig, ax = plt.subplots()
y=template
sr=22050
M = librosa.feature.melspectrogram(y=y, sr=sr)
M_db = librosa.power_to_db(M, ref=np.max)
img = librosa.display.specshow(M_db, y_axis='mel', x_axis='time', ax=ax)
ax.set(title='Mel spectrogram display')
fig.colorbar(img, ax=ax, format="%+2.f dB")
#%%
# Construct a subplot grid with 3 rows and 1 column, sharing the x-axis)
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

# On the first subplot, show the original spectrogram
img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='STFT (log scale)')

# On the second subplot, show the mel spectrogram
img2 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=ax[1])
ax[1].set(title='Mel')

# On the third subplot, show the chroma features
img3 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma',
                                key='Eb:maj', ax=ax[2])
ax[2].set(title='Chroma')

# To eliminate redundant axis labels, we'll use "label_outer" on all subplots:
for ax_i in ax:
    ax_i.label_outer()

# And we can share colorbars:
fig.colorbar(img1, ax=[ax[0], ax[1]])

# Or have individual colorbars:
fig.colorbar(img3, ax=[ax[2]])

# We can then even do fancy things like zoom into a particular time and frequency
# region.  Since the axes are shared, this will apply to all three subplots at once.
ax[0].set(xlim=[1, 3])  # Zoom to seconds 1-3