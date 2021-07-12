import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from datetime import datetime

# ---------------------------- user input part starts here ------------------------------

filename_input = 'EnergySpectrum.dat'         # Energy spectrum filename
filename_output_suffix = '_Smeared'     # suffix of the output file
filePath_output = './output/'            # path of output folder
plotPath_output = './outputPlots/'
plotname_output_suffix = '_Smeared'

bin_width = 0.400       # writing bin width of 400 keV in MeV

manual_bin_Positions = False
# if (manual_bin_positions == True) then give values for bin_startPosition and bin_endPosition
# position (in MeV) from where binning will start
bin_startPosition = 3.000
# position (in MeV) at which binning will stop. Please write
# intended bin_endPosition + .000001 as bin_endPosition
bin_endPosition = 35.400

sigma = 0.6                         # sigma of Gaussian smearing

outputPrecision_fmt = '%0.5e'       # write data upto 5 decimal points
# ----------------------------- user input part ends here -------------------------------


# ------------------------------ main program starts ------------------------------------

energySpectrum = np.loadtxt(filename_input)
Ex_energy = energySpectrum[:, 0]
counts = energySpectrum[:, 1]

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
# print('Binned Segments:', binned_segments)

binned_EX_energies = []
binned_counts = []

i = 0
binned_count = 0
for idx in range(len(energySpectrum)):
    if binned_segments[i] < energySpectrum[idx][0] < binned_segments[i + 1]:
        # print(energySpectrum[idx][0])
        binned_count = binned_count + energySpectrum[idx][1]
        if energySpectrum[idx][0] == energySpectrum[-1][0]:
            EX_energy_per_seg = (binned_segments[i] + binned_segments[i + 1]) / 2
            binned_EX_energies.append(EX_energy_per_seg)
            binned_counts.append(binned_count)
    elif energySpectrum[idx][0] >= binned_segments[i + 1]:
        # print('Binning stopped. Shifting to next binned segment')
        # print(energySpectrum[idx][0])
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
binned_counts = np.around(binned_counts, decimals=8)

# print('Binned Ex Energies:',binned_EX_energies)
# print('Binned Counts:',binned_counts)

# Gaussian Smear and Spline of data -----------------------------------------------------

binned_counts_smeared = gaussian_filter1d(binned_counts, sigma=sigma, mode='nearest')

binned_counts_smeared_tck = interpolate.splrep(binned_EX_energies, binned_counts_smeared, s=0)
binned_EX_energies_splined = np.arange(binned_EX_energies[0], binned_EX_energies[-1], 0.01)
binned_counts_smeared_splined = interpolate.splev(binned_EX_energies_splined, binned_counts_smeared_tck, der=0)


# writing smeared Ex. energy spectrum in output file ------------------------------------

smeared_Energy_spectrum = np.zeros(shape=(len(binned_counts_smeared),2), dtype=float)

smeared_Energy_spectrum[:,0] = binned_EX_energies
smeared_Energy_spectrum[:,1] = binned_counts_smeared
# print(smeared_Energy_spectrum)

filename_strlist = filename_input.split('.')
filename_outputfmt = filePath_output + filename_strlist[0] + filename_output_suffix + '_Sig{:0.2f}.txt'.format(sigma)
# print(filename_outputfmt)

np.savetxt(fname=filename_outputfmt, X=smeared_Energy_spectrum, fmt=outputPrecision_fmt, delimiter='    ', newline='\n')

# source the _plot_master.py file which contains styles for plot ------------------------
exec(open('./_plot_master.py').read())

plt.plot(Ex_energy, counts, color='black', label = 'raw data')
plt.plot(binned_EX_energies, binned_counts, color='red', label='binned data')
# plt.plot(binned_EX_energies, binned_counts_smeared, color='blue', label='binned + smeared data')
plt.plot(binned_EX_energies_splined, binned_counts_smeared_splined, color='blue', linewidth=1.5, label='binned + smeared + splined data')

plt.xlabel('Excitaion Energy [MeV]')
plt.ylabel('Counts/MeV')
plt.legend()


# Plot smeared Energy spectrum and save the plots ---------------------------------------

timestamp_now = datetime.now()
timestamp_str = timestamp_now.strftime("%Y%m%d_%H%M%S")
Plotname_outputfmt = plotPath_output + filename_strlist[0] + plotname_output_suffix \
                     + '_Sig{:0.2f}_'.format(sigma) +'BIN{:0.0f}keV_'.format(bin_width*1000) + timestamp_str
# print(Plotname_outputfmt)

plt.savefig(fname=Plotname_outputfmt+'.png', dpi=300)
plt.savefig(fname=Plotname_outputfmt+'.jpeg', dpi=300)
plt.savefig(fname=Plotname_outputfmt+'.pdf', dpi=300)

plt.show()


