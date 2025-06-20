# timbre_extraction.py

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_path = 'sample.wav'  # Replace with your file path
y, sr = librosa.load(audio_path)

# Extract features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
zero_crossings = librosa.zero_crossings(y, pad=False)

# Print summary
print("MFCC shape:", mfccs.shape)
print("Spectral Centroid (mean):", np.mean(spectral_centroid))
print("Spectral Bandwidth (mean):", np.mean(spectral_bandwidth))
print("Zero Crossing Rate (count):", sum(zero_crossings))

# Visualize MFCC
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
