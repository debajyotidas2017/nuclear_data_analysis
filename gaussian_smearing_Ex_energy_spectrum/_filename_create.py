# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 2021
@author: DEBAJYOTI DAS
"""
import numpy as np

filename_input = 'Itoh_9_7.dat'         # Energy spectrum filename
filename_output_suffix = '_smeared.txt'     # suffix of the output file
filePath_output = './output/'            # path of output folder

# filename analyse ---------------------------------------------

filename_strlist = filename_input.split('.')
print(filename_strlist)

filename_output = filePath_output + filename_strlist[0] + filename_output_suffix
print(filename_output)

data = np.loadtxt(filename_input)


smeared_Energy_spectrum = np.zeros(shape=(len(data),2), dtype=float)

# print(smeared_Energy_spectrum)

smeared_Energy_spectrum[:,0] = data[:,0]
smeared_Energy_spectrum[:,1] = data[:,1]

print(smeared_Energy_spectrum)
np.savetxt(fname=filename_output, X=smeared_Energy_spectrum, fmt='%.4e', delimiter='\t', newline='\n')

data2 = np.loadtxt(filepath_output)