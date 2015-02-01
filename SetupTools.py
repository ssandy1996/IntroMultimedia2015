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
 #		Setup basic tools for Multimedia processing using Python.
 # ----------------------------------------------------------------------
 # Usage:
 # 		Check the associated readme file in the repository.
 # ----------------------------------------------------------------------
 # Author:
 # 		S Sandeep Kumar
 # 		E-mail : ee13b1025@iith.ac.in
########################################################################

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from hwFunc import *

# Ensure images are in working directory
aerialFile = '5.1.10.tiff'
airplaneFile = '5.1.11.tiff'
apcFile = '7.1.08.tiff'

# Read the grayscale images
aerial = cv2.imread(aerialFile, 0)
airplane = cv2.imread(airplaneFile, 0)
apc = cv2.imread(apcFile, 0)

# Display the images using imshow()
# imagesc() in python is shown in the Histogram plot
cv2.imshow('Aerial', aerial)
cv2.imshow('Airplane', airplane)
cv2.imshow('APC', apc)

# Maximum and Minimum pixel values 
print '\nMaximum pixel value of Images'
print '-----------------------------' 
print "Aerial Image Maximum Pixel = %d" % np.amax(aerial)
print "Airplane Image Maximum Pixel = %d" % np.amax(airplane) 
print "APC Image Maximum Pixel = %d" % np.amax(apc)

print '\nMinimum pixel value of Images'
print '-----------------------------' 
print "Aerial Image Minimum Pixel = %d" % np.amin(aerial) 
print "Airplane Image Minimum Pixel = %d" % np.amin(airplane)  
print "APC Image Minimum Pixel = %d" % np.amin(apc)  

# Calculate the resolution of images
aerialRes = [aerial.shape[0], aerial.shape[1]]
airplaneRes = [airplane.shape[0], airplane.shape[1]]
apcRes = [apc.shape[0], apc.shape[1]] 

# Print resolution of images
print '\nResolution of Images'
print '--------------------' 
print 'Resolution of Aerial Image = ' + str(aerialRes[0]) + ' X ' + str(aerialRes[1]) 
print 'Resolution of Airplane Image = ' + str(airplaneRes[0]) + ' X ' + str(airplaneRes[1]) 
print 'Resolution of APC Image = ' + str(apcRes[0]) + ' X ' + str(apcRes[1]) 

# Find the size of compressed images downloaded 
aerialSize = os.path.getsize(aerialFile)
airplaneSize = os.path.getsize(airplaneFile)
apcSize = os.path.getsize(apcFile)

# Print size of compressed images downloaded
print '\nSize of Compressed Images downloaded' 
print '------------------------------------'
print 'Size of Aerial Image = ' + str(aerialSize/1024.0) + ' KB'
print 'Size of Airplane Image = ' + str(airplaneSize/1024.0) + ' KB'
print 'Size of APC Image = ' + str(apcSize/1024.0) + ' KB'
  
# Efficiency of compressed images  
print '\nCompression efficiency of Images' 
print '--------------------------------'
print "Aerial Image Compression ratio = %.3f:1" % ((aerialRes[0]*aerialRes[1])/(aerialSize*1.0))
print "Airplane Image Compression ratio = %.3f:1" % ((airplaneRes[0]*airplaneRes[1])/(airplaneSize*1.0))
print "APC Image Compression ratio = %.3f:1" % ((apcRes[0]*apcRes[1])/(apcSize*1.0))

# Display each bitplane as a binary image
bitmask = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]

plt.figure('Aerial Image bitplanes')
for i in range(0,8):
    bitLayer = np.bitwise_and(aerial, bitmask[i])
    bitLayer = bitLayer/(bitmask[i])
    bitLayer = bitLayer*(0xFF)
    plt.subplot(2,4,i+1)
    plt.imshow(bitLayer, 'gray')
    plt.title("Bitplane %d" % i)
plt.tight_layout()

plt.figure('Airplane Image bitplanes')
for i in range(0,8):
    bitLayer = np.bitwise_and(airplane, bitmask[i])
    bitLayer = bitLayer/(bitmask[i])
    bitLayer = bitLayer*(0xFF)
    plt.subplot(2,4,i+1)
    plt.imshow(bitLayer, 'gray')
    plt.title("Bitplane %d" % i)
plt.tight_layout()

plt.figure('APC Image bitplanes')
for i in range(0,8):
    bitLayer = np.bitwise_and(apc, bitmask[i])
    bitLayer = bitLayer/(bitmask[i])
    bitLayer = bitLayer*(0xFF)
    plt.subplot(2,4,i+1)
    plt.imshow(bitLayer, 'gray')
    plt.title("Bitplane %d" % i)
plt.tight_layout()

# Sequentially set bitplanes to zero
bitzero = [0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80, 0x00]

plt.figure('Aerial Image bitplanes sequentially zero')
for i in range(0,8):
    bitLayer = np.bitwise_and(aerial, bitzero[i])
    plt.subplot(2,4,i+1)
    plt.imshow(bitLayer, 'gray')
    plt.title("Bitplane %d to 0 zero" % i)
plt.tight_layout()

plt.figure('Airplane Image bitplanes sequentially zero')
for i in range(0,8):
    bitLayer = np.bitwise_and(airplane, bitzero[i])
    plt.subplot(2,4,i+1)
    plt.imshow(bitLayer, 'gray')
    plt.title("Bitplane %d to 0 zero" % i)
plt.tight_layout()

plt.figure('APC Image bitplanes sequentially zero')
for i in range(0,8):
    bitLayer = np.bitwise_and(apc, bitzero[i])
    plt.subplot(2,4,i+1)
    plt.imshow(bitLayer, 'gray')
    plt.title("Bitplane %d to 0 zero" % i)
plt.tight_layout()

# Sequentially decrease the quantization level
plt.figure('Aerial Image Requantization')
plt.subplot(2,4,1)
plt.imshow(aerial, 'gray')
plt.title("Original Image 8 bits")
bitLayer = np.float32(aerial)
for i in range(1,8):
    bitLayer = np.floor((bitLayer+1)/2)
    bitPlot = bitLayer/(pow(2,8-i))
    plt.subplot(2,4,i+1)
    plt.imshow(bitPlot, 'gray')
    level = 8-i
    plt.title("Quantization to %d bits" % level)
plt.tight_layout()

plt.figure('Airplane Image Requantization')
plt.subplot(2,4,1)
plt.imshow(airplane, 'gray')
plt.title("Original Image 8 bits")
bitLayer = np.float32(airplane)
for i in range(1,8):
    bitLayer = np.floor((bitLayer+1)/2)
    bitPlot = bitLayer/(pow(2,8-i))
    plt.subplot(2,4,i+1)
    plt.imshow(bitPlot, 'gray')
    level = 8-i
    plt.title("Quantization to %d bits" % level)
plt.tight_layout()

plt.figure('APC Image Requantization')
plt.subplot(2,4,1)
plt.imshow(apc, 'gray')
plt.title("Original Image 8 bits")
bitLayer = np.float32(apc)
for i in range(1,8):
    bitLayer = np.floor((bitLayer+1)/2)
    bitPlot = bitLayer/(pow(2,8-i))
    plt.subplot(2,4,i+1)
    plt.imshow(bitPlot, 'gray')
    level = 8-i
    plt.title("Quantization to %d bits" % level)
plt.tight_layout()
    
# Computing the histogram of an image
aerialMyHist = hist(aerial, 256, 256)
airplaneMyHist = hist(airplane, 256, 256)
apcMyHist = hist(apc, 256, 256)

aerialHist = cv2.calcHist([aerial],[0],None,[256],[0,256])
airplaneHist = cv2.calcHist([airplane],[0],None,[256],[0,256])
apcHist = cv2.calcHist([apc],[0],None,[256],[0,256])
  
# Comparison with built-in histogram function
plt.figure('Aerial Image Histogram')
plt.subplot(1,3,1)
plt.imshow(aerial, 'gray')
plt.title('Aerial Image')

plt.subplot(1,3,2)
plt.plot(aerialMyHist)
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Histogram using my Function')

plt.subplot(1,3,3)
plt.plot(aerialHist)
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Histogram using OpenCV')
plt.tight_layout()

plt.figure('Airplane Image Histogram')
plt.subplot(1,3,1)
plt.imshow(airplane, 'gray')
plt.title('Airplane Image')

plt.subplot(1,3,2)
plt.plot(airplaneMyHist)
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Histogram using my Function')

plt.subplot(1,3,3)
plt.plot(airplaneHist)
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Histogram using OpenCV')
plt.tight_layout()

plt.figure('APC Image Histogram')
plt.subplot(1,3,1)
plt.imshow(apc, 'gray')
plt.title('APC Image')

plt.subplot(1,3,2)
plt.plot(apcMyHist)
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Histogram using my Function')

plt.subplot(1,3,3)
plt.plot(apcHist)
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.title('Histogram using OpenCV')
plt.tight_layout()

plt.show()
 
# Entropy of the images
print '\nEntropy of the Images' 
print '---------------------'
print "Aerial Image Entropy = %.4f" % entropy(aerial, 256)
print "Airplane Image Entropy = %.4f" % entropy(airplane, 256)
print "APC Image Entropy = %.4f" % entropy(apc, 256)

########################################################################
 # End of SetupTools.py
########################################################################
