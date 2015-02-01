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
 #		Huffman Coding of a string and an image.
 # ----------------------------------------------------------------------
 # Usage:
 # 		Check the associated readme file in the repository.
 # ----------------------------------------------------------------------
 # Author:
 # 		S Sandeep Kumar
 # 		E-mail : ee13b1025@iith.ac.in
########################################################################

import sys
import os
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

# Function to encode an input string using the code list    
def codeString(codeList, stringData):
    code = ''
    for i in stringData: # Code each symbol using the code list 
        index = [x[0] for x in codeList].index(i)
        code += codeList[index][1] 
    return code # Return encoded version of string

# Function to decode an encoded string using the Huffman tree    
def decodeString(tree, code):
    node = tree
    decode = ''
    for i in code: # Traverse the tree using coded bits
        if i == '0':
            node = node[1].left
        else:
            node = node[1].right
        if isinstance(node[1], str): # Leaf node then symbol decoded 
            decode += node[1]
            node = tree # Return to root 
    return decode # Return the decoded string

# Function to encode an input image (2D array) using the code list
def codeImage(codeList, image):
    code = ''
    for i in range(0, image.shape[0]): # Encode pixel by pixel in each row 
        for j in range(0, image.shape[1]): # Code each symbol using the code list    
            index = [x[0] for x in codeList].index(str(image[i,j]))
            code += codeList[index][1] 
    return code # Return encoded version of image
    
# Function to decode an encoded image using the Huffman tree and image resolution
def decodeImage(tree, code, res):
    node = tree
    decode = np.zeros(res, dtype=np.uint8) # Store the decoded image as a vector
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
    return decode # Return the decoded image vector
    
# Function to compute entropy of source and average Huffman code length    
def entropyCalc(sourcePMF, codeList):
    entropy = 0
    average = 0
    for i in range(0, len(sourcePMF)): # Calculate entropy of source PMF 
        if sourcePMF[i][0] > 0:
            entropy -= (sourcePMF[i][0])*(math.log(sourcePMF[i][0], 2))
    
    for i in codeList: # Calculate average Huffman code length from code list
        index = [x[1] for x in sourcePMF].index(i[0])
        average += (sourcePMF[index][0])*len(i[1])
    return (entropy, average) # Return entropy of source and average Huffman code length 

# Source PMF
sourcePMF = [(0.25, 'a'), (0.25, 'e'), (0.2, 'i'), (0.15, 'o'), (0.15, 'u')];

# Input string
stringData = "aeiou"

huffmanTree = generateTree(sourcePMF) # Create Huffman tree for the source symbols  
codeList = [('garbage', 0)] # Initialise code list
generateCodeList(huffmanTree, '') # Create code list using the Huffman tree 
del codeList[0] # Remove initial element
code = codeString(codeList, stringData) # Encode the input string 

# Print the Huffman codes
print '\nHuffman Coding'
print '--------------' 
print 'Symbol\tHuffman Code'
for i in codeList:
    print "%s\t\t%s" % (i[0], i[1])

# Print the input string and encoded string
print "\nInput string for encoding = %s" % stringData
print "Encoded string            = %s" % code

decode = decodeString(huffmanTree, code) # Decode the encoded string 
compress = entropyCalc(sourcePMF, codeList) # Calculate entropy of source and 
                                            # average Huffman code length

# Print decoded string, entropy of source and average Huffman code length
print "Decoded string            = %s" % decode
print "\nEntropy of source           = %.3f" % compress[0]
print "Average Huffman code length = %.3f" % compress[1]

# Ensure input image is in working directory
imageFile = '5.1.10.tiff'
inputImage = cv2.imread(imageFile, 0) # Read input image
imageHist = cv2.calcHist([inputImage],[0],None,[256],[0,256])
sourceProb = imageHist/(inputImage.shape[0]*inputImage.shape[1])
# Compute the normalised image histogram

sourcePMF = [('garbage', 0)] # Initialise source PMF list
for i in range(0, len(sourceProb)): # Create the source PMF from normalised image histogram
    newNode = (float(sourceProb[i]), str(i))
    sourcePMF.append(newNode)    
del sourcePMF[0] # Remove initial element

# Create Huffman tree for source pixel values
huffmanTree = generateTree(sourcePMF)
codeList = [('garbage', 0)] # Initialise the code list
generateCodeList(huffmanTree, '') # Create code list using the Huffman tree
del codeList[0] # Remove initial element

code = codeImage(codeList, inputImage) # Encode the input image

# Print the image sizes
print "\nSize of Input Image file = %.3f KB" % (os.path.getsize(imageFile)/1024.0)
print "Size of Compressed Image (without header) = %.3f KB" % (len(code)/(8.0*1024)) 

# Decode the encoded image
decompress = decodeImage(huffmanTree, code, inputImage.shape[0]*inputImage.shape[1])

# Reshape the decoded image using the image resoultion
outputImage = np.reshape(decompress, (inputImage.shape[0], inputImage.shape[1]))

# Compare the input image with the Huffman encoded and decoded image 
plt.figure('Huffman Coding - Image')
plt.subplot(1,3,1)
plt.imshow(inputImage, 'gray')
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow((inputImage - outputImage), 'gray')
plt.title("Difference between Images (Lossless)")

plt.subplot(1,3,3)
plt.imshow(outputImage, 'gray')
plt.title("Huffman Codec Image")
plt.tight_layout()

plt.show()

########################################################################
 # End of HuffmanCoding.py
########################################################################
