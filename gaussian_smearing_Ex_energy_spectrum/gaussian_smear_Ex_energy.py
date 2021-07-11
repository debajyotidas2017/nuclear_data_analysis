import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate

# energySpectrum = np.loadtxt('Itoh_9_7.dat')
energySpectrum = np.loadtxt('test_data.txt')

Ex_energy = energySpectrum[:, 0]
counts = energySpectrum[:, 1]

# binning of 400 keV and 80 keV --------------------------------------------

bin_width = 0.400                   # writing bin width of 400 keV in MeV

manual_bin_Positions = False
# if (manual_bin_positions == True) then give values for bin_startPosition and bin_endPosition
bin_startPosition = 3.000               # position (in MeV) from where binning will start
bin_endPosition = 35.400                # position (in MeV) at which binning will stop

binned_segments = []

if manual_bin_Positions:
    bin_x1 = bin_startPosition
    bin_x2 = 0
    while bin_x2 < bin_endPosition:
        binned_segments.append(bin_x1)
        bin_x2 = bin_x1 + bin_width
        bin_x1 = bin_x2
else:
    Ex_startValue = Ex_energy[0]        # starting value of Excitation energy
    Ex_endValue = Ex_energy[-1]         # ending value of Excitation energy
    bin_startPosition = np.around(Ex_startValue, decimals=1)
    if bin_startPosition > Ex_startValue:
        bin_startPosition = np.around(Ex_startValue, decimals=0)
    bin_endPosition = Ex_endValue
    bin_x1 = bin_startPosition
    bin_x2 = 0
    while bin_x2 < bin_endPosition:
        binned_segments.append(bin_x1)
        bin_x2 = bin_x1 + bin_width
        bin_x1 = bin_x2
    if Ex_endValue >= binned_segments[-1]:
        binned_segments.append(binned_segments[-1] + bin_width)

binned_segments = np.around(binned_segments, decimals=3)

binned_EX_energies = []
binned_counts = []

print('Binned Segments:', binned_segments)

# i = 0
# binned_count = 0
# for idx in range(len(energySpectrum)):
#     if binned_segments[i] < energySpectrum[idx][0] < binned_segments[i + 1]:
#         print(energySpectrum[idx][0])
#         binned_count = binned_count + energySpectrum[idx][1]
#     else:
#         print('Binning stopped. Shifting to next binned segment')
#         print(energySpectrum[idx][0])
#         EX_energy_per_seg = (binned_segments[i] + binned_segments[i+1]) / 2
#         binned_EX_energies.append(EX_energy_per_seg)
#         binned_counts.append(binned_count)
#         # print('Binned Counts:', binned_count)
#         i = i+1
#         binned_count = 0
#         binned_count = binned_count + energySpectrum[idx][1]


i = 0
binned_count = 0
for idx in range(len(energySpectrum)):
    if binned_segments[i] < energySpectrum[idx][0] < binned_segments[i + 1]:
        print(energySpectrum[idx][0])
        binned_count = binned_count + energySpectrum[idx][1]
        if energySpectrum[idx][0] == energySpectrum[-1][0]:
            EX_energy_per_seg = (binned_segments[i] + binned_segments[i + 1]) / 2
            binned_EX_energies.append(EX_energy_per_seg)
            binned_counts.append(binned_count)
    elif energySpectrum[idx][0] >= binned_segments[i + 1]:
        print('Binning stopped. Shifting to next binned segment')
        print(energySpectrum[idx][0])
        EX_energy_per_seg = (binned_segments[i] + binned_segments[i + 1]) / 2
        binned_EX_energies.append(EX_energy_per_seg)
        binned_counts.append(binned_count)
        i = i+1
        binned_count = 0
        binned_count = binned_count + energySpectrum[idx][1]
        if energySpectrum[idx][0] == energySpectrum[-1][0]:
            EX_energy_per_seg = (binned_segments[i] + binned_segments[i + 1]) / 2
            binned_EX_energies.append(EX_energy_per_seg)
            binned_counts.append(binned_count)





binned_EX_energies = np.around(binned_EX_energies, decimals=3)
binned_counts = np.around(binned_counts, decimals=5)

print('Binned Ex Energies:',binned_EX_energies)
print('Binned Counts:',binned_counts)

# print(len(binned_EX_energies))
# print(len(binned_counts))



# Plotting binned data and gaussian smeared + splined data ---------------------------------------------

binned_counts_smeared = gaussian_filter1d(binned_counts, sigma=0.5, mode='nearest')

binned_counts_smeared_tck = interpolate.splrep(binned_EX_energies, binned_counts_smeared, s=0)
binned_EX_energies_splined = np.arange(binned_EX_energies[0], binned_EX_energies[-1], 0.01)
binned_counts_smeared_splined = interpolate.splev(binned_EX_energies_splined, binned_counts_smeared_tck, der=0)


# source the _plot_master.py file which contains styles for plot -----------------------------------
exec(open('./_plot_master.py').read())


plt.plot(Ex_energy, counts, color='black', label = 'raw data')

plt.plot(binned_EX_energies, binned_counts, color='red', label='binned data')

# plt.plot(binned_EX_energies, binned_counts_smeared, color='blue', label='binned + smeared data')

plt.plot(binned_EX_energies_splined, binned_counts_smeared_splined, color='blue', label='binned + smeared + splined data')


plt.xlabel('Excitaion Energy [MeV]')
plt.ylabel('Counts/MeV')
plt.legend()
# plt.show()
