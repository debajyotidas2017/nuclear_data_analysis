import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate

energySpectrum = np.loadtxt('Itoh_9_7.dat')

Ex_energy = energySpectrum[:, 0]
counts = energySpectrum[:, 1]