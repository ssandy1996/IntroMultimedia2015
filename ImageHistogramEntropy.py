########################################################################
 # Copyright (C) 2015 S Sandeep Kumar (ee13b1025@iith.ac.in)
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program.  If not, see <http://www.gnu.org/licenses/>.
 # ---------------------------------------------------------------------
 # Function:
 #		Functions to compute Image histogram and entropy.
 # ---------------------------------------------------------------------
 # Usage:
 # 		Check the associated readme file in the repository.
 # ---------------------------------------------------------------------
 # Author:
 # 		S Sandeep Kumar
 # 		E-mail : ee13b1025@iith.ac.in
########################################################################

import math
import numpy as np

# Function to compute the Histogram of an input Image (2D array)
def hist(inputImage, nBins, noValues):
    histValues = np.zeros(noValues)
    histData = np.zeros(nBins)
    
    for i in range(0, (len(inputImage[:,0]))):
        for j in range(0, (len(inputImage[0]))):
            histValues[inputImage[i,j]] += 1
            
    for i in range(1, nBins):
        for j in range(int(math.floor((i-1)*(noValues/(nBins*1.0)))), int(math.ceil((i)*(noValues/(nBins*1.0))))):
            histData[i-1] += histValues[j]            
    
    return histData

# Function to compute the Entropy of an input Image (2D array)
def entropy(inputImage, noValues):
    histData = hist(inputImage, noValues, noValues)
    histData = histData/(inputImage.shape[0]*inputImage.shape[1])
    imageEntropy = 0
    
    for i in range(0, noValues):
        if histData[i] > 0:
            imageEntropy -= (histData[i])*(math.log(histData[i], 2))
        
    return imageEntropy            
 
########################################################################
 # End of ImageHistogramEntropy.py
########################################################################
     
