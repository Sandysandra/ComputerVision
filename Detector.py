# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:24:39 2015

@author: SHE
"""

from scipy import signal
from scipy import linalg
import pylab
import skimage
import skimage.io
import numpy as NP
import math, pdb

import matplotlib.pyplot as plt
import matplotlib

def computeGaussianKernel(sigma, width):
    kernel = NP.zeros((width,width))
    sum = 0.0
    mean = width/2
    print "mean = ", mean
    for x in range(width):
        for y in range(width):
            kernel[x,y] = math.exp(-0.5 * ( math.pow((x-mean)/sigma, 2.0) + math.pow((y-mean)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            sum += kernel[x,y]
    # normalize        
    for x in range(width):
        for y in range(width):
            kernel[x,y] /= sum;      
    return kernel

def filteredGradient(filename,gaussianKernel):
    sobel_x = NP.array([[-1,0,1],[-2,0,2],[-1,0,1]]);
    sobel_y = NP.array([[1,2,1],[0,0,0],[-1,-2,-1]]);
    img = skimage.img_as_float(skimage.io.imread(filename))
    
    # convert img to grayscale
    IGrey = NP.zeros((img.shape[0], img.shape[1]),dtype='float')
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            IGrey[x,y] = 0.2989*img[x,y,0]+0.5870*img[x,y,1]+0.1140*img[x,y,2]
            
#    plt.imshow(IGrey, cmap = matplotlib.cm.Greys_r)
#    plt.show()
    # smooth the image
    IGrey = signal.convolve2d(IGrey,gaussianKernel,boundary='symm',mode='same')
    
    # compute the gradient (convolve with sobel)
    Fx = signal.convolve2d(IGrey,sobel_x,boundary='symm',mode='same')
    Fy = signal.convolve2d(IGrey,sobel_y,boundary='symm',mode='same')
    
    # compute the edge strength F
    F = NP.zeros(Fx.shape)
    maxF = 0.0
    minF = 100.0
    for x in range(F.shape[0]):
        for y in range(F.shape[1]):
            F[x,y] = math.sqrt(Fx[x,y]*Fx[x,y] + Fy[x,y]*Fy[x,y])
            maxF = max([maxF,F[x,y]])
            minF = min([minF,F[x,y]])
    '''
    # re-scale the edge strength to range 0-1
    for x in range(F.shape[0]):
        for y in range(F.shape[1]):
            F[x,y] = (F[x,y]-minF)/(maxF-minF)
    '''
            
    
    # compute the edge orientation D
    D = NP.zeros(IGrey.shape)
    atan2_vectorized = NP.vectorize(math.atan2)
    radians_to_degrees_vectorized = NP.vectorize(math.degrees)
    D = radians_to_degrees_vectorized(atan2_vectorized(Fy,Fx))
    return IGrey,F,D,Fx,Fy
    
def edgeDetector(filename,sigma, ThresH, ThresL):
    gaussianKernel = computeGaussianKernel(sigma,7)
    img,F,D,Fx,Fy = filteredGradient(filename,gaussianKernel)
    
    # nonmaximum suppression
    DStar = NP.zeros((D.shape[0], D.shape[1]))
    for x in range(D.shape[0]):
        for y in range(D.shape[1]):
            D[x,y] = D[x,y]/45.0
            DStar[x,y] = round(D[x,y])
            DStar[x,y] = DStar[x,y]*45

    I = NP.zeros(F.shape)
    for y in range(F.shape[0]):
        for x in range(F.shape[1]):
            I[y,x] = F[y,x]
            
    for y in range(D.shape[0]):
        for x in range(D.shape[1]):
            if DStar[y,x] == 0 or DStar[y,x] == 180 or DStar[y,x] == -180:
                if x+1 < D.shape[1] and F[y,x] < F[y,x+1]:
                        I[y,x] = 0
                if x-1 >= 0 and F[y,x] < F[y,x-1]:
                        I[y,x] = 0
            if DStar[y,x] == 45 or DStar[y,x] == -135:
                if x+1 < D.shape[1] and y-1 >= 0 and F[y,x] < F[y-1,x+1]:
                        I[y,x] = 0
                if x-1 >= 0 and y+1 < D.shape[0] and F[y,x] < F[y+1,x-1]:
                        I[y,x] = 0
            if DStar[y,x] == 90 or DStar[y,x] == -90:
                if y-1 >= 0 and F[y,x] < F[y-1,x]:
                        I[y,x] = 0
                if y+1 < D.shape[0] and F[y,x] < F[y+1,x]:
                        I[y,x] = 0
            if DStar[y,x] == 135 or DStar[y,x] == -45:
                if x-1 >= 0 and y-1 >= 0 and F[y,x] < F[y-1,x-1]:
                        I[y,x] = 0
                if x+1 < D.shape[1] and y+1 < D.shape[0] and F[y,x] < F[y+1,x+1]:
                        I[y,x] = 0
    # Hysteresis thresholding
    mask = NP.zeros(I.shape)
    output = NP.zeros(I.shape)
    stack = []
    for x in range(D.shape[0]):
        for y in range(D.shape[1]):
            if I[x,y] > ThresH and mask[x,y] == 0:
                mask[x,y] = 1
                output[x,y] = I[x,y]
                stack.append([x,y])
                
    while len(stack)!=0:
        [i,j] = stack.pop(0)
        if withinBound(I, i-1, j-1 ) and mask[i-1,j-1]==0 and I[i-1,j-1]>ThresL:
            stack.append([i-1,j-1])
            mask[i-1,j-1] = 1
            output[i-1,j-1] = I[i-1,j-1]
            if withinBound(I, i-1, j ) and mask[i-1,j]==0 and I[i-1,j]>ThresL:
                stack.append([i-1,j])
                mask[i-1,j] = 1
                output[i-1,j] = I[i-1,j]
            if withinBound(I, i-1, j+1 ) and mask[i-1,j+1]==0 and I[i-1,j+1]>ThresL:
                stack.append([i-1,j+1])
                mask[i-1,j+1] = 1
                output[i-1,j+1] = I[i-1,j+1]
            if withinBound(I, i, j-1 ) and mask[i,j-1]==0 and I[i,j-1]>ThresL:
                stack.append([i,j-1])
                mask[i,j-1] = 1
                output[i,j-1] = I[i,j-1]
            if withinBound(I, i, j+1 ) and mask[i,j+1]==0 and I[i,j+1]>ThresL:
                stack.append([i,j+1])
                mask[i,j+1] = 1
                output[i,j+1] = I[i,j+1]
            if withinBound(I, i+1, j-1 ) and mask[i+1,j-1]==0 and I[i+1,j-1]>ThresL:
                stack.append([i+1,j-1])
                mask[i+1,j-1] = 1
                output[i+1,j-1] = I[i+1,j-1]
            if withinBound(I, i+1, j ) and mask[i+1,j]==0 and I[i+1,j]>ThresL:
                stack.append([i+1,j])
                mask[i+1,j] = 1
                output[i+1,j] = I[i+1,j]
            if withinBound(I, i+1, j+1 ) and mask[i+1,j+1]==0 and I[i+1,j+1]>ThresL:
                stack.append([i+1,j+1])
                mask[i+1,j+1] = 1
                output[i+1,j+1] = I[i+1,j+1]
    
    inputname = filename[0:len(filename)-4] + '_edgeDetector_' + str(sigma) + '_' + str(ThresH) + '_' + str(ThresL)
#    skimage.io.imsave(inputname+'_Fx.png', Fx)
#    skimage.io.imsave(inputname+'_Fy.png', Fy)
#    skimage.io.imsave(inputname+'_Grey.png', img)
#    skimage.io.imsave(inputname+'_F.png', F)
#    skimage.io.imsave(inputname+'_DStar.png', DStar)
#    skimage.io.imsave(inputname+'_nonmaximum.png', I)      
    skimage.io.imsave(inputname+'_cannyEdge.png', mask)
    
        
def withinBound(I,i,j):
    return i >= 0 and j >= 0 and i < I.shape[0] and j < I.shape[1]

def cornerDetector(filename,sigma,numNeigh,thres):
    # filtered gradient
    gaussianKernel = computeGaussianKernel(sigma,7)
    img,F,D,Fx,Fy = filteredGradient(filename,gaussianKernel)
    
    # finding corners
    points = []
    eigs = []
    for y in range(F.shape[0]):
        for x in range(F.shape[1]):
            C = NP.zeros((2,2))
            for j in range(-numNeigh, numNeigh+1, 1):
                for i in range(-numNeigh, numNeigh+1, 1):
                    if withinBound(F, y+j, x+i):
                        C[0,0] += Fx[y+j,x+i]*Fx[y+j,x+i]
                        C[1,0] += Fx[y+j,x+i]*Fy[y+j,x+i]
                        C[0,1] += Fx[y+j,x+i]*Fy[y+j,x+i]
                        C[1,1] += Fy[y+j,x+i]*Fy[y+j,x+i]
            
            eigenvalues,principalComponents = linalg.eig(C)
            minEig = min(eigenvalues)
            if minEig > thres:
                eigs.append(minEig)
                points.append([y,x])
    
    # nonmaximum suppresion
    # sort points in decresing order of eigs
    if len(eigs) > 0:
        points, eigs = zip(*sorted(zip(points, eigs), reverse=True,
                         key=lambda x: x[1]))
    points = list(points)
    
    L = []
    mask = NP.zeros(F.shape)
    for i in range(len(points)):
        pos = points[i]
        if mask[pos[0],pos[1]] == 0:
            L.append(pos)
            for i in range(-2*(numNeigh+3), 2*(numNeigh+3)+1, 1):
                for j in range(-2*(numNeigh+3), 2*(numNeigh+3)+1, 1):
                    if withinBound(F, pos[0]+j, pos[1]+i):
                        mask[pos[0]+j, pos[1]+i] = 1

    for i in range(len(L)):
        pos = L[i]
        for x in range(-numNeigh, numNeigh+1, 1):
            for y in range(-numNeigh, numNeigh+1, 1):
                if x == -numNeigh or y == -numNeigh or x == numNeigh or y == numNeigh:
                    if withinBound(F, pos[0]+y, pos[1]+x):
                        img[pos[0]+y, pos[1]+x] = 1
                    
    inputname = filename[0:len(filename)-4] + '_cornerDetector_' + str(sigma) + '_' + str(numNeigh) + '_' + str(thres)
    skimage.io.imsave(inputname+'_corners.png', img)    
    
# O: number of octaves
# S: number of scales per octave
# thresPeak: threshold the local extrema
# thresEdge: threshold for eliminating edges
def sift(O,S,thresEdge,thresPeak):
    return O
    
def main():
    '''
    inputfilename = 'building.jpg'
    sigma = 2
    thresH = 3
    thresL = 20
    edgeDetector(inputfilename,sigma, thresH/10.0, thresL/100.0)
    
    inputfilename = 'checker.jpg'
    sigma = 2
    width = 7
    thres = 5
    cornerDetector(inputfilename,sigma,width,thres)
    '''

    '''
    inputfilename = 'building.jpg'
    for sigma in range (1, 8, 2):
        for thresH in range (3, 10, 2):
            for thresL in range (5, 25, 5):
                edgeDetector(inputfilename,sigma, thresH/10.0, thresL/100.0)
    '''
    
    inputfilename = 'checker.jpg'
    for sigma in range(1,8,2):
        for width in range(2,8,2):
            for thres in range(2,20,4):
                cornerDetector(inputfilename,sigma,width,thres)

    
if __name__ == "__main__": main()
    