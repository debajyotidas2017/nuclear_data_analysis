import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from numba import jit
from datetime import datetime
import multiprocessing

# ---------------------------- user input part starts here ------------------------------

filename_input = 'spec3.7.dat'              # input filename of Spectrum Data
filename_output_suffix = '_Smeared'         # suffix of the output file
filePath_output = './output/'               # path of output folder where smeared spectrum data to be written
plotPath_output = './outputPlots/'          # path where Plots will be generated
plotname_output_suffix = '_Smeared'


sigma = 0.15                        # Gaussian Kernel sigma value to smear energy spectrum
outputPrecision_fmt = '%0.5e'       # write data in Output file with upto 5 decimal points
freeCores = 2                       # the number of free CPU cores. Keep at least 1 CPU core free...
# ----------------------------- user input part ends here -------------------------------


# ------------------------------ main program starts ------------------------------------

SpectrumData = np.loadtxt(filename_input)            # read the Energy Spectrum Data
Xdata = SpectrumData[:, 0]
Ydata = SpectrumData[:, 1]
Xmin = Xdata[0]                     # lower x boundary of the spectrum
Xmax = Xdata[-1]                    # upper x boundary of the spectrum

# Interpolate Xdata and Ydata to generate an function named EvaluationFunction().
# This EvaluationFunction can be used to calculate Y-value for a given X-position in the NOT-Smeared spectrum
EvaluationFunction = interpolate.interp1d(Xdata, Ydata)

# Define an evaluate function
def evaluate(x, Xboundary_min=Xmin, Xboundary_max=Xmax):
    '''Given a value of x, this function will evaluate y-value. If x is within boundary then will
    return the corresponding y-value calculated from evaluation function. Esle will return 0.
    This is required because evaluation function can only evaluate y-value if x is within the
    boundary of Xdata. Otherwise this gives error.'''
    if Xboundary_min <= x <= Xboundary_max:
        return EvaluationFunction(x)
    else:
        return 0

@jit
def gauss(x, mean, sigma):
    '''This function calculates normalized gaussian probability distribution'''
    arg = (x-mean)/sigma
    res = np.exp(-0.5 * (arg**2))     # taking the square of the arg and the multilying with (-0.5) and then calculating the exponential
    norm_res = res / (np.sqrt(2*np.pi) * sigma)   # normalizing the result by dividing the res with (sqrt(2*pi) * sigma)
    return norm_res


def SmearingFunction(x, sigma=sigma, Xboundary_min=Xmin, Xboundary_max=Xmax):
    '''This function does gaussian smearing of spectrum.
       @param x: x-point at which gaussian smearing needs to be done
       @param sigma: sigma value of the gaussian kernel
       @param Xboundary_min: lower x boundary of the spectrum. Default value is Xmin
       @param Xboundary_max: upper x boundary of the spectrum. Default value is Xmax
       @return: returns smeared y-value
    '''
    # The smeared spectrum is F(x) = Integral of [ Gauss(t) * f(x-t) ] dt
    # Here f(x-t) is the not-Smeared spectrum
    # We will integrate in the limit from ( -3*sigma to +3*sigma )
    # Step size of Integration will be 0.001*sigma which we define as Mesh

    Mesh = 0.001 * sigma
    ThisY = 0.0
    SumY = 0.0
    SumGauss = 0.0

    # To make distinction what is x and where we do the Integration. We integrate at X_evaluate
    X_evaluate = x
    # we need this, if the x is near zero then the function will collapse
    if (x < 2 * sigma):
        X_evaluate = 2 * sigma

    # Integrate in the limit from ( -3*sigma to +3*sigma ). Here t is the variable of integral.
    for t in np.arange(start=-3.0 * sigma, stop=3.0 * sigma, step=Mesh):
        ThisY = gauss(t, 0.0, sigma)  # calculate gaussian at t position
        ThisY = ThisY * evaluate(X_evaluate - t)  # Gauss(t) * f(x-t); f(x-t): not-smeared_Counts(x-t) is calculated from evaluate function
        SumY = SumY + ThisY  # do the summation

        # If (X_evaluate - t) is outside Xboundary_max then do Gaussian integral in this way
        if ((X_evaluate - t) < Xboundary_min or (X_evaluate - t) > Xboundary_max):
            SumGauss = SumGauss + gauss(t, 0.0, sigma)

    # Multiply with Mesh to finish the Integral
    Integral = SumY * Mesh
    GaussIntegral = SumGauss * Mesh

    # Correct for the missing Gaussian surface near the end. Applicable only
    # when (X_evaluate - t) is outside Xboundary_max. Otherwise GaussIntegral is zero
    Smeared = Integral / (1.0 - GaussIntegral)

    # this is applicable for x >= 2.0*Sigma
    Answer = Smeared

    # But if x is in between 0 and 2.0*sigma then calculate smeared value in this way
    if (0.0 <= x < 2.0 * sigma):
        Ratio = Smeared / evaluate(2.0 * sigma)
        Value = evaluate(x)
        Answer = Value * Ratio

    # And if x < 0.0, then
    if (x < 0.0):
        Answer = 0.0

    # Calculation done and return Answer
    return Answer


if __name__ == "__main__":

    # start time
    timestamp_start = datetime.now()
    print("Gaussian smearing calculation of energy spectrum has started\n")

    # ------------------- Start calculation across a Pool of Workers -------------------------

    totalCores = multiprocessing.cpu_count()        # Detect total number of CPU cores
    use_cores = totalCores - freeCores              # use this much number of CPU cores
    print("Calculation running across %d out of %d CPU cores..."%(use_cores, totalCores))

    # define the Xdata array with the setpsize. Here Xdata is fine (depending on step) excitation energy
    eenergy = np.arange(start=Xmin, stop=Xmax, step=0.01)

    # set a pool of workers. The worker counts are the number of use_cores
    pool = multiprocessing.Pool(processes=use_cores)

    # map the Xdata with the Smearing function. This will calculate the smeared Ydata...
    # Here smeared Ydata is SmearedCounts
    SmearedCounts = pool.map(SmearingFunction, eenergy)

    # Creating a 2D ndarray SmearedSpectrum filled with zeros of the shape of (rows = len(eenergy), 3)
    SmearedSpectrum = np.zeros(shape=(len(eenergy), 3))
    # first column of SmearedSpectrum is fine excitation energy
    SmearedSpectrum[:,0] = eenergy
    # second column is NOT-Smeared Counts evaluated across the fine excitation energy
    SmearedSpectrum[:,1] = [evaluate(x) for x in eenergy]
    # Thied column is the SmearedCounts
    SmearedSpectrum[:,2] = SmearedCounts
    # --------------------------------- Calculation finished ---------------------------------

    # --------------------------- Write the SmearedSpectrum in file --------------------------
    filename_inpstr = filename_input[:-4]                   # if filename_input is filename5.6.txt, then filename_inpstr will be filename5.6
    filename_output = filename_inpstr + filename_output_suffix + '_Sig{:0.2f}.txt'.format(sigma)
    filename_outputWP = filePath_output + filename_output           # filename output With Path

    np.savetxt(fname=filename_outputWP, X=SmearedSpectrum, fmt=outputPrecision_fmt, delimiter='    ')

    # end time
    timestamp_end = datetime.now()
    Tot_time = timestamp_end - timestamp_start
    print("\nCalculation finished!")
    print("Total time taken to smear the spectrum:", Tot_time)
    print("Smeared spectrum data saved in file "+filename_outputWP)

    # ----------------------------- Plot the data and save plots -----------------------------

    # source the _plot_master.py file which contains styles for plot ----
    exec(open('./_plot_master.py').read())

    plt.plot(eenergy, SmearedSpectrum[:,1], label='NOT Smeared Spectrum', color='blue', linewidth=1.5)
    plt.plot(eenergy, SmearedCounts, label='Smeared Spectrum', color='red', linewidth=1.5)

    plt.xlabel('Excitaion Energy [MeV]')
    plt.ylabel('Counts / [MeV]')
    plt.legend()

    # plot output filename With Path
    timestamp_str = timestamp_start.strftime("%Y%m%d_%H%M%S")
    Plotname_outputWP = plotPath_output + filename_inpstr + plotname_output_suffix + '_Sig{:0.2f}_'.format(sigma) + timestamp_str
    # save plot files
    plt.savefig(fname=Plotname_outputWP+'.png', dpi=300)
    plt.savefig(fname=Plotname_outputWP+'.jpeg', dpi=300)
    plt.savefig(fname=Plotname_outputWP+'.pdf', dpi=300)

    plt.show()