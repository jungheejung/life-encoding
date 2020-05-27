# Import the necessary libraries
import matplotlib; matplotlib.use('Agg') # No pictures displayed
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os, pylab, librosa, librosa.display


ap = '/idata/DBIC/cara/life/data/audio/complete/life_part1.wav'
fp = '/idata/DBIC/cara/life/data/spectral/complete/life_part1.csv'

# Load sound file
y, sr = librosa.load(ap)

# Let's make and display a mel-scaled power (energy-squared) spectrogram
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

print(S.shape)

# make pictures name
save_path = 'part1_spectrogram.jpg'
#
S = pd.read_csv(fp)
filter_col = [col for col in S if col.startswith('mel')]
S_dat = S[filter_col]

print(S.onset)



# Convert to log scale (dB). We'll use the peak power as reference.
log_S = librosa.logamplitude(np.array(S_dat).T, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the spectrogram on a mel scale
# sample rate and hop length parameters are used to render the time axis
librosa.display.specshow(log_S, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title('mel power spectrogram')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

plt.show()
pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
pylab.close()
