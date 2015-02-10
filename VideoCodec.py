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
 #		Motion Estimation and Motion Compensation using Three Step 
 #		Search algorithm.
 # ---------------------------------------------------------------------
 # Usage:
 # 		Check the associated readme file in the repository.
 # ---------------------------------------------------------------------
 # Author:
 # 		S Sandeep Kumar
 # 		E-mail : ee13b1025@iith.ac.in
########################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read specific number of video frames and return a 3-D matrix
def readVideo(fileName, noFrame):
    captureVideo = cv2.VideoCapture(fileName)
    # Specify the video file to be read
    
    if noFrame > captureVideo.get(7):
        noFrame = captureVideo.get(7)
    # Limit the number of frames to total frames in the video
    
    inputVideo = np.zeros((captureVideo.get(4), captureVideo.get(3), noFrame), dtype=np.uint8)   
    # Store the video frames in a 3-D matrix
    
    count = 0
    while(count < noFrame): 
        retValue, frame = captureVideo.read()
        inputVideo[:,:,count] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        # Store video frames as grayscale images
        count += 1
        
    captureVideo.release() 
    # Release the video file to be read
    
    return inputVideo # Return the required video frames

# Function to divide the image into non-overlapping macroblocks
def imageBlocks(inputImage, blockSize):
    blockImage = np.zeros((blockSize, blockSize, total), dtype=np.uint8)
    # Store the non-overlapping blocks as a 3-D matrix
    
    blocks = 0
    for r in range(0,inputImage.shape[0], blockSize): 
        for c in range(0,inputImage.shape[1], blockSize):
            blockImage[:,:,blocks] = inputImage[r:r+blockSize,c:c+blockSize]
            blocks += 1
    
    return blockImage # Return the set of blocks

# Function to compute the Mean Absolute Difference (MAD) metric between two blocks
def madCalc(block1, block2):
    madValue = float(0)
    
    for r in range(0, blockSize):
        for c in range(0, blockSize):
            madValue += abs(int(block1[r,c] - block2[r,c]))
            
    retValue = ((madValue*1.0)/(blockSize*blockSize))
    
    return retValue # Return the computed MAD value

# Function to find the set of 9 coordinates spaced s distance apart for Three Step search
def neighbourCoord(coordXY, s):
    nearCoord = np.zeros((9, 2), dtype=np.int)
    # Store the i and j values for the set of 9 coordinates
    
    nearCoord[0,0] = coordXY[0]
    nearCoord[0,1] = coordXY[1]
    # Center coordinates
    
    nearCoord[1,0] = coordXY[0]
    nearCoord[1,1] = coordXY[1] + s
    # Top right coordinates
    
    nearCoord[2,0] = coordXY[0] - s
    nearCoord[2,1] = coordXY[1] + s
    # Top coordinates
    
    nearCoord[3,0] = coordXY[0] - s 
    nearCoord[3,1] = coordXY[1]
    # Top left coordinates
    
    nearCoord[4,0] = coordXY[0] - s
    nearCoord[4,1] = coordXY[1] - s
    # Left coordinates
    
    nearCoord[5,0] = coordXY[0]
    nearCoord[5,1] = coordXY[1] - s
    # Bottom left coordinates
    
    nearCoord[6,0] = coordXY[0] + s
    nearCoord[6,1] = coordXY[1] - s
    # Bottom coordinates
    
    nearCoord[7,0] = coordXY[0] + s
    nearCoord[7,1] = coordXY[1] 
    # Bottom right coordinates
    
    nearCoord[8,0] = coordXY[0] + s
    nearCoord[8,1] = coordXY[1] + s
    # Right coordinates
    
    return nearCoord # Return the set of 9 coordinates

# Function to find the top left corner coordinates of the macroblocks
def coordinates(inputImage, blockSize):
    coordXY = np.zeros((total, 2), dtype=np.uint8)
    # Store top left corner coordinates of the macroblocks
    
    blocks = 0
    for r in range(0,inputImage.shape[0], blockSize): 
        for c in range(0,inputImage.shape[1], blockSize):
            coordXY[blocks, 0] = r
            coordXY[blocks, 1] = c
            blocks += 1
    
    return coordXY # Return the top left coordinates of the macroblocks    

# Function to find motion vectors using the Three Step search algorithm
def threeStepSearch(frame1, frame2, cornerCoord):
    motionVector = np.zeros((total, 2), dtype=np.int)
    testBlock = np.zeros((blockSize, blockSize), dtype=np.uint8)
    # Store the motion vectors and macroblock from frame 1 for metric testing
    # Edge block neighbouring pixels not considered if outside the real pixel range of frame
    
    # Calculate motion vector for all macroblocks in frame 2
    for i in range(0, total):
        # Step 1: Calculate the MAD metric at center and neighbour coordinates 4 pixels apart
        nearCoord = neighbourCoord(cornerCoord[i,:], 4)
        minIndex = 0
        minValue = 0
        
        # Find minimum MAD value from valid neighbour coordinates
        for j in range(0, len(nearCoord)):
            if(nearCoord[j,0] >= 0 and nearCoord[j,1] >= 0 and nearCoord[j,0] <= frame1.shape[0]-blockSize and nearCoord[j,1] <= frame1.shape[1]-blockSize):   
                testBlock = frame1[nearCoord[j,0]:nearCoord[j,0]+blockSize, nearCoord[j,1]:nearCoord[j,1]+blockSize]
                madValue = madCalc(testBlock, frame2[:,:,i])
                if j == 0:
                    minValue = madValue
                else:
                    if madValue < minValue:
                        minValue = madValue
                        minIndex = j
                        
        # Step 2: Calculate MAD metric at previous best match and neighbour coordinates 2 pixels apart
        nearCoord = neighbourCoord(nearCoord[minIndex,:], 2)
        minIndex = 0
        minValue = 0
        
        # Find minimum MAD value from valid neighbour coordinates
        for j in range(0, len(nearCoord)):
            if(nearCoord[j,0] >= 0 and nearCoord[j,1] >= 0 and nearCoord[j,0] <= frame1.shape[0]-blockSize and nearCoord[j,1] <= frame1.shape[1]-blockSize):   
                testBlock = frame1[nearCoord[j,0]:nearCoord[j,0]+blockSize, nearCoord[j,1]:nearCoord[j,1]+blockSize]
                madValue = madCalc(testBlock, frame2[:,:,i])
                if j == 0:
                    minValue = madValue
                else:
                    if madValue < minValue:
                        minValue = madValue
                        minIndex = j
                        
        # Step 3: Calculate MAD metric at previous best match and neighbour coordinates 1 pixel apart
        nearCoord = neighbourCoord(nearCoord[minIndex,:], 1)
        minIndex = 0
        minValue = 0
        
        # Find minimum MAD value from valid neighbour coordinates
        for j in range(0, len(nearCoord)):
            if(nearCoord[j,0] >= 0 and nearCoord[j,1] >= 0 and nearCoord[j,0] <= frame1.shape[0]-blockSize and nearCoord[j,1] <= frame1.shape[1]-blockSize):   
                testBlock = frame1[nearCoord[j,0]:nearCoord[j,0]+blockSize, nearCoord[j,1]:nearCoord[j,1]+blockSize]
                madValue = madCalc(testBlock, frame2[:,:,i])
                if j == 0:
                    minValue = madValue
                else:
                    if madValue < minValue:
                        minValue = madValue
                        minIndex = j
                    
        # Store the motion vector for the current macroblock in the second frame
        motionVector[i,:] = nearCoord[minIndex,:] - cornerCoord[i,:]
                              
    return motionVector # Return the motion vectors in the second frame from the first

# Function to create the Motion Compensated Predicated frame using the reference 
# frame and the motion vectors
def motionCompensation(refFrame, motionVector, cornerCoord):
    mcFrame = np.zeros((refFrame.shape[0], refFrame.shape[1]), dtype=np.uint8)
    # Store the Motion Compensated Predicted frame
    
    blocks = 0
    for r in range(0,mcFrame.shape[0], blockSize):
        for c in range(0,mcFrame.shape[1], blockSize):
            row = motionVector[blocks,0] + cornerCoord[blocks,0] 
            col = motionVector[blocks,1] + cornerCoord[blocks,1]
            mcFrame[r:r+blockSize,c:c+blockSize] = frame1[row:row+blockSize,col:col+blockSize]
            blocks += 1
    
    return mcFrame # Return the Motion Compensated Predicted frame

# Ensure input video is in working directory
# QCIF - YUV format reference video in .avi container
videoFile = 'foreman.avi'

# Read first two frames from the reference video
inputVideo = readVideo(videoFile, 2)

# Macro block size 16
blockSize = 16

# Total number of macroblocks per frame
total = len(range(0,inputVideo.shape[0], blockSize))*len(range(0,inputVideo.shape[1], blockSize))

frame1 = inputVideo[:,:,0]
# Reference frame

frame2 = imageBlocks(inputVideo[:,:,1], blockSize)
# Finding the macroblocks of frame 2

testFrame = inputVideo[:,:,1]
# Frame 2 to find the Motion Compensated Predicted frame

tipCoord = coordinates(frame1, blockSize)
# Top left coordinates of the macroblocks

motionVector = threeStepSearch(frame1, frame2, tipCoord)
# Calculate motion vectors for macroblocks in second frame from first

compFrame = motionCompensation(frame1, motionVector, tipCoord)
# Generate the Motion Compensated Predicated frame using the motion vectors and first frame 

errorFrame = np.zeros((frame1.shape[0], frame1.shape[1]), dtype=np.int)
for i in range(0, frame1.shape[0]):
    for j in range(0, frame1.shape[1]):
        errorFrame[i,j] = int(testFrame[i,j]) - int(compFrame[i,j])  
# Compute the error between the second frame and its Motion Compensated Predicted version

constructFrame = errorFrame + compFrame
# Reconstruct the second frame from the error and Motion Compensated Predicted frame

# Compare the Motion Compensated Predicted frame with the input frame 
plt.figure('Video Codec')

# Video frame 1
plt.subplot(2,3,1)
plt.imshow(frame1, 'gray')
plt.title("Original Frame 1")

# Arrow plot the motion vectors
plt.subplot(2,3,2)
plt.quiver(tipCoord[:,1], tipCoord[:,0], motionVector[:,1], motionVector[:,0])
plt.gca().invert_yaxis()
plt.title("Motion Vectors")

# Video frame 2
plt.subplot(2,3,3)
plt.imshow(testFrame, 'gray')
plt.title("Original Frame 2")

# Motion Compensated Predicted frame
plt.subplot(2,3,4)
plt.imshow(compFrame, 'gray')
plt.title("Motion Compensated Predicted Frame")

# Error between the frames
plt.subplot(2,3,5)
plt.imshow(abs(errorFrame), 'gray')
plt.title("Error in MC Predicted Frame")

# Reconstructed frame
plt.subplot(2,3,6)
plt.imshow(constructFrame, 'gray')
plt.title("Reconstructed Frame")

plt.show()

########################################################################
 # End of VideoCodec.py
########################################################################
