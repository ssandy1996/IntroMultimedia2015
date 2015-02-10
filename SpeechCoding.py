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
 # 		Speech Coding using Linear Predictive Coding.
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
import scipy.io.wavfile
from numpy.linalg import pinv
# from scipy.linalg import toeplitz # Uncomment to use inbuilt function for speed
import matplotlib.pyplot as plt

# Function to read 10ms frames with 2ms overlap
def read10msFrame(inputAudio):    
    audioFrame = np.zeros((frameSize, nFrames), dtype=np.float32)
    # Store 10ms frames in a 2D matrix
    
    # Read 10 ms frames with 2ms overlap
    frameCount = 0
    for i in range(0, len(inputAudio[1]), ms8):
        if frameCount < nFrames-1:
            audioFrame[:, frameCount] = inputAudio[1][i:i+frameSize, 0]    
        frameCount += 1

    return audioFrame # Return the 10ms frames

# Function to calculate correlation for a frame
def timeDiffSum(signal, nPole): 
    rss = np.zeros((nPole+1, 1), dtype=np.float32)
    # Store the rss values 
    
    for j in range(0, nPole+1):
        for i in range(j, len(signal)):
            rss[j,0] += (1.0*signal[i])*signal[i-j]
    
    return rss # Return the rss vector

# Function to verify that Rss is indeed Toeplitz   
# This method to compute Rss is used only for one frame, since it is slow 
def verifyRssToeplitz(frame):
    Rss = np.zeros((nPole, nPole), dtype=np.float32)
    # Store the Rss matrix
    
    for i in range(0, nPole):
        for j in range(0, nPole):
            for k in range(0, len(frame)+min(i,j)):
                if (k-i)>=0 and (k-j)>=0:        
                    Rss[i,j] += 1.0*frame[k-i]*frame[k-j] 
            
    return Rss # Return the computed Rss matrix 

# Function to generate Rss matrix from rss values 
# For speed uncomment inbuilt function and comment previous block
def setRssMatrix(rss, nPole):
    Rss = np.zeros((nPole, nPole), dtype=np.float32)
    # Store Rss matrix for a given rss
    
    # Generate Rss matrix from rss
    temp = np.empty((nPole, 1), dtype=np.float32)
    temp.fill(rss[0,0])    
    Rss += np.diagflat(temp, 0)
    for i in range(1, nPole):
        temp = np.empty(((nPole-i), 1), dtype=np.float32)
        temp.fill(rss[i,0])
        Rss += np.diagflat(temp, i)
        Rss += np.diagflat(temp, -i)
        
    # Rss = toeplitz(rss[0:nPole]) # Inbuilt function to generate toeplitz matrix
    
    return Rss # Return Rss matrix
            
# Function to generate final audio vector
def constructFile(frameData):
    finalAudio = np.zeros((frameSize, 1), dtype=np.float32)
    
    # Construct final audio vector from 2D matrix
    finalAudio[:,0] = frameData[:,0]
    temp = np.zeros((ms8, 1), dtype=np.float32)
    for i in range(1, nFrames):
        temp[:,0] = frameData[(frameSize-ms8):,i]
        finalAudio = np.concatenate((finalAudio, temp), axis=0)    
    
    return finalAudio # Return the final audio vector to write to file

# Ensure input audio is in working directory   
audioFile = 'brian.wav'

# Read the input audio file
inputAudio = scipy.io.wavfile.read(audioFile)  

# Define the output audio file name
outputFile = 'hw4output.wav'

inputRate = inputAudio[0] # Input audio sampling rate

nPole = 10 # Number of poles in all pole system

# Samples for 8ms at input sampling rate
ms8 = int(math.ceil(0.008*inputRate)) 

# Samples for 10ms at input sampling rate
frameSize = int(math.ceil(0.010*inputRate))

# Number of complete 10ms frames
nFrames = len(range(0, len(inputAudio[1]), ms8)) - 1

# Store the 10ms frames from input audio
audioFrame = read10msFrame(inputAudio)

# Find and store ak for each 10ms frame
aVector = np.zeros((nPole, nFrames), dtype=np.float32)
for i in range(0, nFrames):    
    rss = timeDiffSum(audioFrame[:,i], nPole)
    # Find correlation values
    
    Rss = setRssMatrix(rss, nPole) # Compute Rss matrix
    rss = rss[1:] # Compute rss vector
    
    # Compute ak values for given frame using pseudo inverse
    aVector[:,i] = np.reshape(pinv(Rss)*np.mat(rss), nPole)

# Store shatn and en for each frame    
audioFrameConstruct = np.zeros((frameSize, nFrames), dtype=np.float32)
errorFrame = np.zeros((frameSize, nFrames), dtype=np.float32)

# Calculate shatn for each frame
for i in range(0, nFrames):
    for j in range(0, frameSize):
        for k in range(0, nPole):
            if (j-k) >= 0:
                audioFrameConstruct[j,i] += aVector[k,i]*audioFrame[(j-k),i]     
    
# Calculate en for each frame
errorFrame = audioFrame - audioFrameConstruct 

# Reconstruct audio frames using only en and ak
reconstructAudio = np.zeros((frameSize, nFrames), dtype=np.float32)

# For each frame first 10 error values are sn and rest can be computed using ak
for i in range(0, nFrames):
    reconstructAudio[0:10,i] = errorFrame[j,i]
    for j in range(10, frameSize):
        for k in range(0, nPole):
            reconstructAudio[j,i] += aVector[k,i]*errorFrame[(j-k),i]     

# Generate final audio to write to .wav file
finalAudio = constructFile(reconstructAudio)

# Store in int16 format   
finalAudioINT = finalAudio.astype(np.int16)

# Write reconstructed audio to output file
scipy.io.wavfile.write(outputFile, inputRate, finalAudioINT)

# N for 10ms frames at input sampling rate
print '\nSpeech Coding'
print '-------------'
print "Input Sampling Rate = %d" % inputRate
print "N for 10ms frames = %d " % frameSize

# Choose a specific frame
frameNo = 10

# Compute Rss for that frame 
verificationRss = verifyRssToeplitz(audioFrame[:,frameNo])
print '\nRss Matrix (only integer parts)'
print '----------'
print verificationRss

# Find sn, shatn and en for that frame
sn = audioFrame[:,frameNo]
shatn = audioFrameConstruct[:,frameNo] 
en = sn - shatn

# Plot sn, shatn and en for that frame
plt.figure('Speech Coding')

# Plot sn
plt.subplot(1,3,1)
plt.plot(sn)
plt.title("s[n]")

# Plot en
plt.subplot(1,3,2)
plt.plot(en)
plt.title("e[n]")
    
# Plot shatn
plt.subplot(1,3,3)
plt.plot(shatn)
plt.title("s^[n]")

plt.show()

########################################################################
 # End of SpeechCoding.py
########################################################################
