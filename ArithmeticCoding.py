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
 # ----------------------------------------------------------------------
 # Function:
 #		Arithmetic Coding of a string.  
 # ----------------------------------------------------------------------
 # Usage:
 # 		Check the associated readme file in the repository.
 # ----------------------------------------------------------------------
 # Author:
 # 		S Sandeep Kumar
 # 		E-mail : ee13b1025@iith.ac.in
########################################################################

import sys
import numpy as np

# Function to encode an input string using the inputPMF
def arithmeticEncode(data, inputPMF):
    inputPMF = sorted(inputPMF, key=lambda x: x[0], reverse=True)  
    # Sort the input PMF to ensure same Arithmetic code for different permutations 
    # of the list 
    
    intervalRange = np.zeros((len(inputPMF),2))
    initial = 0
    for i in range(0, len(inputPMF)):
        intervalRange[i,0] = initial
        initial = initial + inputPMF[i][0]
        intervalRange[i,1] = initial
    # Caluclate the lower and higher values of the intervals 
    
    if initial == 1: # Ensure it is a bonafide PMF
        h = 1 # Initialise h as 1 
        l = 0 # Initialise l as 0
        for i in range(0, len(data)): # Code each symbol using the interval ranges
            dataPoint = [x[1] for x in inputPMF].index(data[i])
            r = h - l # Calculate the range of interval
            h = l + r*intervalRange[dataPoint,1] # Find new higher value 
            l = l + r*intervalRange[dataPoint,0] # Find new lower value          
    else:
        sys.exit("Invalid Input - PMF doesn't sum up to 1")
    
    return l # Return the arithmetic code
    
# Function to decode the encoded string using the input PMF 
def arithmeticDecode(code, noChar, inputPMF):
    inputPMF = sorted(inputPMF, key=lambda x: x[0], reverse=True)  
    # Sort the input PMF to ensure same Arithmetic code for different permutations 
    # of the list 
    
    intervalRange = np.zeros((len(inputPMF),2))
    initial = 0
    for i in range(0, len(inputPMF)):
        intervalRange[i,0] = initial
        initial = initial + inputPMF[i][0]
        intervalRange[i,1] = initial 
    # Caluclate the lower and higher values of the intervals 
    
    data = ''
    if initial == 1: # Ensure it is a bonafide PMF
        r = code # Initialise r to encoded number
        for i in range(0, noChar): # Decode each character sequentially
            j = 0
            
            # Finding the interval where r lands
            while(r >= intervalRange[j,0] and j < (len(inputPMF)-1)):
                j += 1
            j -= 1
            
            data += inputPMF[j][1] # Update decoded string 
            
            # Compute new r - range
            r = (r-intervalRange[j,0])/(intervalRange[j,1] - intervalRange[j,0])          
    else:
        sys.exit("Invalid Input - PMF doesn't sum up to 1")
    
    return data # Return the decoded string 

# Source PMF
sourcePMF = [(0.25, 'a'), (0.25, 'e'), (0.2, 'i'), (0.15, 'o'), (0.15, 'u')];

# Input string 
data = "ieee"

# Number of characters to decode
noChar = 4
             
code = arithmeticEncode(data, sourcePMF) # Encode the input string

# Print the input string and the encoded string
print '\nArithmetic Coding'
print '-----------------' 
print "Input string for encoding = %s" % data
print "Encoded value of string   = %f" % code

decode = arithmeticDecode(code, noChar, sourcePMF) # Decode the encoded string

# Print the decoded string
print "Decoded string from value = %s" % decode

########################################################################
 # End of ArithmeticCoding.py
########################################################################
