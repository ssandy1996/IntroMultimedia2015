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
 #		JPEG like Image encoder and decoder.
 # ----------------------------------------------------------------------
 # Usage:
 # 		Check the associated readme file in the repository.
 # ----------------------------------------------------------------------
 # Author:
 # 		S Sandeep Kumar
 # 		E-mail : ee13b1025@iith.ac.in
########################################################################

import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import Queue

# Node object containing the left and right branch information
class treeNode(object):
    def __init__(self, left = None, right = None):
        self.left = left
        self.right = right
        
# Function to create the Huffman Tree for an input PMF list        
def generateTree(inputPMF):
    sumPMF = sum([pair[0] for pair in inputPMF]) # Calculate PMF Sum
    inputPMF = sorted(inputPMF, key=lambda x: x[0], reverse=True) 
    # Sort the input PMF to ensure same Huffman code for different permutations 
    # of the list 
    
    i = 0
    while(i < len(inputPMF) and inputPMF[i][0] > 0):
        i += 1 
    inputPMF = inputPMF[0:i]
    # Remove entries with zero probability
    
    if sumPMF == 1: # Ensure it is a bonafide PMF 
        tree = Queue.PriorityQueue() # Create empty priority queue for creating tree
        
        for i in inputPMF:     
            tree.put(i)
        # Add all PMF tuples to the queue       
    
        while tree.qsize() > 1: # Until we reach the top node         
            left = tree.get() # Remove the least probability node
            right = tree.get() # Remove the second least probability node 
            node = treeNode(left, right) # Create new node with left and right
            sumProb = left[0] + right[0]
            newNode = (sumProb, node) # Prbability of new node is sum of left and right
            tree.put(newNode) # Add the new node to the queue
    else:
        sys.exit("Invalid Input - PMF doesn't sum up to 1")
    
    return tree.get() # Return the root node               

# Function to create a list of Huffman Codes for the source symbols recursively           
def generateCodeList(node, code):
    if isinstance(node[1], treeNode) == 0: # Leaf node then add it to the list 
        newNode = (node[1], code)
        codeList.append(newNode)
    else:
        generateCodeList(node[1].left, code + '0') # Set left branch as '0'
        generateCodeList(node[1].right, code + '1') # Set right branch as '1'
        
# Function to encode AC stream using the code list
def codeACstream(codeList, stream):
    code = ''
    for i in range(0, len(stream)): # Code each symbol using the code list    
        index = [x[0] for x in codeList].index(str(stream[i]))
        code += codeList[index][1] 
    return code # Return encoded version of AC stream

# Function to decode an encoded AC stream using the Huffman tree
def decodeACstream(tree, code):
    node = tree
    decode = np.zeros(len(stream), dtype=np.uint8) # Store the decoded stream as a vector
    j = 0
    for i in code: # Traverse the tree using coded bits
        if i == '0':
            node = node[1].left
        else:
            node = node[1].right
        if isinstance(node[1], str): # Leaf node then symbol decoded
            decode[j] += int(node[1])
            j += 1
            node = tree # Return to root
    decodeStream = [('garbage', 0)] # Initialise AC stream list
    for i in range(0, len(stream), 2): # Create RLC AC stream 
        node = (stream[i], stream[i+1])
        decodeStream.append(node)
    del decodeStream[0] # Remove initial element       
     
    return decodeStream # Return the decoded stream vector

# Function to divide the image into non-overlapping square blocks
def imageBlocks(inputImage, blockSize):
    blockImage = np.zeros((blockSize, blockSize, total), dtype=np.uint8)
    # Store the non-overlapping blocks as a 3-D matrix
    
    blocks = 0
    for r in range(0,inputImage.shape[0], blockSize): 
        for c in range(0,inputImage.shape[1], blockSize):
            blockImage[:,:,blocks] = inputImage[r:r+blockSize,c:c+blockSize]
            blocks += 1
    
    return blockImage # Return the set of blocks

# Function to apply Type II DCT and quantize each block
def imageDCT(blockImage, qMatrix):
    cValue = np.ones(blockSize, dtype=np.float32)
    cValue[0] = (1/math.sqrt(2))
    # Create coefficients value array

    dctFactor = math.sqrt(4.0/(blockSize*blockSize))
    dctImage = np.zeros((blockSize, blockSize, len(blockImage[0,0,:])), dtype=np.int)
    # Initialise multiplication factor and DCT matrix
    
    for count in range(0, len(blockImage[0,0,:])): # Compute DCT block by block 
        for u in range(0, blockSize):
            for v in range(0, blockSize):
                dctValue = 0
                for i in range(0, blockSize):
                    for j in range(0, blockSize):
                        dctValue += blockImage[i,j,count]*math.cos(((2*i+1)*u*math.pi)/(2*blockSize))*math.cos(((2*j+1)*v*math.pi)/(2*blockSize))
                dctValue *= (cValue[u]*cValue[v])*dctFactor 
                dctImage[u,v,count] = int((dctValue/qMatrix[u,v])+0.5) 
                # Quantize using the qMatrix
    
    return dctImage # Return the quantized Type II DCT blocks

# Function to zig-zag scan and vectorize blocks
def zigzagScan(dctImage, zigzag): 
    dctVector = np.zeros((len(dctImage[0,0,:]), len(dctImage[:,0,0])*len(dctImage[0,:,0])), dtype=np.int)
    # Store the set of vectors as an array
    
    for i in range(0, len(dctImage[:,0,0])*len(dctImage[0,:,0])): # Vectorize blocks 
        index = np.where(zigzag == i)
        uindex = index[0]
        uindex = uindex[0,0]
        vindex = index[1]
        vindex = vindex[0,0]
        dctVector[:,i] = dctImage[uindex,vindex,:]
    
    return dctVector # Return the set of vectors

# Function to encode the DC coefficients and generate a DC stream
def encodeDPCM(dctVector):    
    streamDC = np.zeros(total, dtype=np.int)
    streamDC[0] = dctVector[0,0]
    for i in range(1, total):
        streamDC[i] = dctVector[i,0] - dctVector[i-1,0]
    
    return streamDC # Return the DC stream

# Function to RLC encode the AC coefficients and generate an AC stream 
def encodeRLC(dctVector):
    streamAC = [(0,0)] # Initialise the AC stream 
    for i in range(0, len(dctVector[:,0])): # Store skip and value pair
        skip = 0
        for j in range(1, blockSize*blockSize):
            if dctVector[i, j] == 0:
                skip += 1
            else:
                newNode = (skip, dctVector[i,j])
                streamAC.append(newNode)
                skip = 0
            if j == blockSize*blockSize-1:
                newNode = (skip, 0)
                streamAC.append(newNode)
                skip = 0
    del streamAC[0] # Remove initial element
    
    return streamAC # Return the AC stream

# Function to RLC decode the AC coefficients using the AC stream
def decodeRLC(streamAC): 
    idctVector = np.zeros((total, blockSize*blockSize), dtype=np.int)                                                        
    count = 0
    j = 1 

    for i in range(0, len(streamAC)):
        node = streamAC[i]
        if node == (0,0):
            count += 1
            j = 1
        elif node[1] == 0:
            idctVector[count,j:j+node[0]] = 0       
            count += 1
            j = 1
        else:
            idctVector[count, j+node[0]] = node[1]
            if node[0] == 0:
                j += 1
            else:
                idctVector[count, range(j, j+node[0])] = 0       
                j += node[0]+1
                
    return idctVector # Return the AC stream

# Function to decode the DC coefficients using the DC stream
def decodeDPCM(streamDC, idctVector):        
    idctVector[0,0] = streamDC[0]        
    for i in range(1, len(streamDC)):
        idctVector[i,0] = streamDC[i] + idctVector[i-1,0]
        
    return idctVector # Return the DC stream
    
# Function to recreate blocks from the coefficient vectors 
def zigzagOrder(idctVector):    
    idctImage = np.zeros((blockSize, blockSize, total), dtype=np.int)

    for i in range(0, blockSize*blockSize):
        index = np.where(zigzag8 == i)
        uindex = index[0]
        uindex = uindex[0,0]
        vindex = index[1]
        vindex = vindex[0,0]
        idctImage[uindex,vindex,:] = dctVector[:,i]
        
    return idctImage # Return the set of blocks

# Function to apply Typle II IDCT to each block and remove quantization
def imageIDCT(idctImage, qMatix):  
    cValue = np.ones(blockSize, dtype=np.float32)
    cValue[0] = (1/math.sqrt(2))
    # Create coefficients value array
    
    decodeImage = np.zeros((blockSize, blockSize, total), dtype=np.uint8)
    idctFactor = math.sqrt(4.0/(blockSize*blockSize))
    # Initialise decoded image blocks and multiplication factor  
      
    for count in range(0, total):
        for u in range(0, blockSize):
            for v in range(0, blockSize):
                idctImage[u,v,count] *= qMatrix[u,v] # Multiply with quantization matrix

    for count in range(0, total):
        for i in range(0, blockSize):
            for j in range(0, blockSize):
                idctValue = 0
                for u in range(0, blockSize):
                    for v in range(0, blockSize):
                        idctValue += (cValue[u]*cValue[v])*idctImage[u,v,count]*math.cos(((2*i+1)*u*math.pi)/(2*blockSize))*math.cos(((2*j+1)*v*math.pi)/(2*blockSize))
                idctValue *= idctFactor 
                decodeImage[i,j,count] = int(idctValue)

    return decodeImage # Return the set of blocks 
    
# Function to put sub-blocks together and generate decoded estimate of original image    
def formImage(decodeImage):
    finalImage = np.zeros((inputImage.shape[0], inputImage.shape[1]), dtype=np.uint8)
    blocks = 0
    
    for r in range(0,inputImage.shape[0], blockSize):
        for c in range(0,inputImage.shape[1], blockSize):
            finalImage[r:r+blockSize,c:c+blockSize] = decodeImage[:,:,blocks]
            blocks += 1
            
    return finalImage # Return the decoded estimate of original image

# Ensure input image is in working directory
imageFile = '5.1.10.tiff'
inputImage = cv2.imread(imageFile, 0) # Read input image

blockSize = 8 # Square sub-block of size 8
total = len(range(0,inputImage.shape[0], blockSize))*len(range(0,inputImage.shape[1], blockSize))
# Total number of blocks for input image

qMatrix = np.matrix([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                     [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                     [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                     [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])
# Initialise quantization matrix

zigzag8 = np.matrix([[0,1,5,6,14,15,27,28],[2,4,7,13,16,26,29,42],
                    [3,8,12,17,25,30,41,43],[9,11,18,24,31,40,44,53],
                    [10,19,23,32,39,45,52,54],[20,22,33,38,46,51,55,60],
                    [21,34,37,47,50,56,59,61],[35,36,48,49,57,58,62,63]]) 
# Zig-Zag order for vectorizing blocks

blockImage = imageBlocks(inputImage, blockSize) 
# Divide input image into non-overlapping blocks

dctImage = imageDCT(blockImage, qMatrix)
# Apply Type II DCT to the blocks and quantize

dctVector = zigzagScan(dctImage, zigzag8)
# Zig-zag scan to vectorize blocks

streamDC = encodeDPCM(dctVector)
# DPCM encode DC stream

streamAC = encodeRLC(dctVector)
# RLC encode AC stream

stream = np.zeros(2*len(streamAC), dtype=np.int) # Vectorizing AC stream list
for i in range(0, len(streamAC)):
    node = streamAC[i]
    stream[2*i] = node[0]
    stream[2*i+1] = node[1]

# Generate AC stream PMF
sumTotal = len(stream)
sourcePMF = [('garbage', 0)] # Initialise source PMF list
for i in range(min(stream), max(stream)+1):
    l = np.where(stream == i)
    l = len(l[0][:])
    newNode = (float((l*1.0)/sumTotal), str(i))
    sourcePMF.append(newNode)    
del sourcePMF[0] # Remove initial element 

# Entropy code the AC stream
huffmanTree = generateTree(sourcePMF) # Create Huffman tree for the source symbols  
codeList = [('garbage', 0)] # Initialise code list
generateCodeList(huffmanTree, '') # Create code list using the Huffman tree 
del codeList[0] # Remove initial element

code = codeACstream(codeList, stream) # Huffman encode the AC stream 

streamAC = decodeACstream(huffmanTree, code) # Huffman decode the AC stream

idctVector = decodeRLC(streamAC)
# RLC decode the AC stream

idctVector = decodeDPCM(streamDC, idctVector)
# DPCM decode the DC stream

idctImage = zigzagOrder(idctVector)
# Recreate blocks from the vectors

decodeImage = imageIDCT(idctImage, qMatrix)
# Remove quantization and apply Type II IDCT to the blocks 

finalImage = formImage(decodeImage)
# Put sub-blocks together and generate the decoded estimate of the original image

# Compare the input image with the decoded estimated of original image 
plt.figure('Image Codec')
plt.subplot(1,2,1)
plt.imshow(inputImage, 'gray')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(finalImage, 'gray')
plt.title("JPEG like Codec Image")
plt.tight_layout()

plt.show()

########################################################################
 # End of ImageCodec.py
########################################################################
