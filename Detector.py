# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:24:39 2015

@author: SHE
"""

from scipy import signal
from scipy import linalg
import scipy
import pylab
import skimage
import skimage.io
import numpy as NP
import math, pdb

import matplotlib.pyplot as plt
import matplotlib

SIFT_MAX_INTERP_STEPS = 5
SIFT_IMG_BORDER = 1

# check if the index [i,j] is outof boundary interms of I.shape        
def withinBound(I,i,j):
    return i >= 0 and j >= 0 and i < I.shape[0] and j < I.shape[1]
    
# convert an rbg float image to a greyscale image
def greyScale(img):
    IGrey = NP.zeros((img.shape[0], img.shape[1]),dtype='float')
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            IGrey[x,y] = 0.2989*img[x,y,0]+0.5870*img[x,y,1]+0.1140*img[x,y,2]
    return IGrey

# compute gaussian kernel   
def computeGaussianKernel(sigma):
    width = 1 + 2*(int(3.0*sigma))
    width = 5
    kernel = NP.zeros((width,width))
    sum = 0.0
    mean = width/2
    for x in range(width):
        for y in range(width):
            kernel[x,y] = math.exp(-0.5 * ( math.pow((x-mean)/sigma, 2.0) + math.pow((y-mean)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            sum += kernel[x,y]
    # normalize        
    for x in range(width):
        for y in range(width):
            kernel[x,y] /= sum;      
    return kernel

# compute the greyscale image, the edge strength, the edge orientation and x,y gradient
def filteredGradient(filename,gaussianKernel):
    sobel_x = NP.array([[-1,0,1],[-2,0,2],[-1,0,1]]);
    sobel_y = NP.array([[1,2,1],[0,0,0],[-1,-2,-1]]);
    img = skimage.img_as_float(skimage.io.imread(filename))
    
    # convert img to grayscale
    IGreyOrigin = greyScale(img)
#    plt.imshow(IGrey, cmap = matplotlib.cm.Greys_r)
#    plt.show()
    # smooth the image
    IGrey = signal.convolve2d(IGreyOrigin,gaussianKernel,boundary='symm',mode='same')
    
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
    inputname = filename[0:len(filename)-4]
#    saveGradientImage(inputname+'_Fx.png',Fx)
#    saveGradientImage(inputname+'_Fy.png',Fy)
#    saveMagnituteImage(inputname+'_F.png',F)
    return IGreyOrigin,F,D,Fx,Fy

def saveGradientImage(name,F):
    I = NP.zeros((F.shape[0],F.shape[1],3))
    red = 0
    green = 1
    maxF = NP.amax(F)
    minF = NP.amin(F)
    print name, minF, maxF
    for y in range(F.shape[0]):
        for x in range(F.shape[1]):
            if F[y,x] >= 0:
                I[y,x][red] = F[y,x]/maxF#min(1.0,F[y,x])#F[y,x]/maxF
            if F[y,x] < 0:
                I[y,x][green] = math.fabs(F[y,x]/minF)#min(1.0,math.fabs(F[y,x]))#math.fabs(F[y,x]/minF)
    skimage.io.imsave(name, I)
    
def saveMagnituteImage(name,F):
    I = NP.zeros(F.shape)
    for y in range(F.shape[0]):
        for x in range(F.shape[1]):
            if F[y,x] > 1:
                I[y,x] = 1
            else:
                I[y,x] = F[y,x]
    skimage.io.imsave(name, I)
    

#*******************************************************#
#***********     Canny Edge Detector    ****************#  
#*******************************************************#     
def edgeDetector(filename,sigma, ThresH, ThresL):
    gaussianKernel = computeGaussianKernel(sigma)
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
    skimage.io.imsave(inputname+'_Fx.png', Fx)
    skimage.io.imsave(inputname+'_Fy.png', Fy)
#    skimage.io.imsave(inputname+'_Grey.png', img)
    skimage.io.imsave(inputname+'_F.png', F)
#    skimage.io.imsave(inputname+'_DStar.png', DStar)
    skimage.io.imsave(inputname+'_nonmaximum.png', I)      
    skimage.io.imsave(inputname+'_cannyEdge.png', mask)
    

    
    
#***************************************************#
#***********     Corner Detector    ****************#  
#***************************************************# 
def cornerDetector(filename,sigma,numNeigh,thres):
    # filtered gradient
    gaussianKernel = computeGaussianKernel(sigma)
    img,F,D,Fx,Fy = filteredGradient(filename,gaussianKernel)
    
    # finding corners
    points = []
    eigs = []
    for y in range(F.shape[0]):
        for x in range(F.shape[1]):
            C = NP.zeros((2,2))
            C[0,0] = sumforC(Fx,Fx,y-numNeigh,y+numNeigh+1,x-numNeigh,x+numNeigh+1)
            C[0,1] = sumforC(Fx,Fy,y-numNeigh,y+numNeigh+1,x-numNeigh,x+numNeigh+1)
            C[1,0] = C[0,1]
            C[1,1] = sumforC(Fy,Fy,y-numNeigh,y+numNeigh+1,x-numNeigh,x+numNeigh+1)
            '''
            for j in range(-numNeigh, numNeigh+1, 1):
                for i in range(-numNeigh, numNeigh+1, 1):
                    if withinBound(F, y+j, x+i):
                        C[0,0] += Fx[y+j,x+i]*Fx[y+j,x+i]
                        C[1,0] += Fx[y+j,x+i]*Fy[y+j,x+i]
                        C[0,1] += Fx[y+j,x+i]*Fy[y+j,x+i]
                        C[1,1] += Fy[y+j,x+i]*Fy[y+j,x+i]
            '''
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
    
def sumforC(Fx, Fy, y1, y2, x1, x2):
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if y2 >= Fx.shape[0]:
        y2 = Fx.shape[0]-1
    if x2 >= Fx.shape[0]:
        x2 = Fx.shape[0]-1
    FxNew = NP.asarray(Fx)
    FyNew = NP.asarray(Fy)
    mat1 = FxNew[y1:y2,x1:x2]
    mat2 = FyNew[y1:y2,x1:x2]
    mat2 = NP.transpose(mat2)
    mat = NP.dot(mat1,mat2)
    return sum(mat[i][i] for i in xrange(0, len(mat)))



#**************************************************#
#****************     SIFT    *********************#  
#**************************************************# 
   
# octaveNum: number of octaves
# scaleNum: number of scales per octave
# thresPeak: threshold the local extrema
# thresEdge: threshold for eliminating edges
def sift(filename,octaveNum,scaleNum,thresPeak,thresEdge):
    img = skimage.img_as_float(skimage.io.imread(filename))
    DoGPyramid = DoG(filename, octaveNum, scaleNum)
    
    # find keypoints
    # contains 4 steps: 
    # 1. find local extrema
    # 2. Taylor series to get the true location
    # 3. low contrast filtering by thresPeak
    # 4. rejecting strong edges by thresEdge
    pre_eliminate_thres = 0.5*thresPeak/scaleNum
    localExtremaList = []
    thresPeakList = []
    interpExtremaList = []
    edgeEliminatedList = []
    for o in range(octaveNum):
        for s in range(1,scaleNum+1,1):
            for y in range(1,DoGPyramid[o,s].shape[0]-1,1):
                for x in range(1,DoGPyramid[o,s].shape[1]-1,1):
                    # step 1: find local extrema
                    if is_extrema(DoGPyramid, o, s, y, x):
                        localExtremaList.append([o,s,y,x,0,0,0])
                        if math.fabs(DoGPyramid[o,s][y,x]) > pre_eliminate_thres:
                            thresPeakList.append([o,s,y,x,0,0,0])
                            # feature = [o,s,y,x,off_s,off_y,off_x]
                            feature = interpolate_extrema(DoGPyramid,o,s,y,x,scaleNum,thresPeak)
                            if len(feature) > 0:
                                interpExtremaList.append(feature)
                                [f_o, f_s, f_y, f_x, off_s, off_y, off_x] = feature
                                
                                if eliminatingEdgeResponse(DoGPyramid,f_o,f_s,f_y,f_x,thresEdge) == False:
                                    edgeEliminatedList.append(feature)
                            
    print "#localExtrema: ", len(localExtremaList)
    print "#thresPeakList: ", len(thresPeakList)
    print "#interpExtrema: ", len(interpExtremaList)
    print "#edgeEliminated: ", len(edgeEliminatedList)
    visualize(filename,img,localExtremaList,scaleNum,0)
    visualize(filename,img,thresPeakList,scaleNum,1)
    visualize(filename,img,interpExtremaList,scaleNum,2)
    visualize(filename,img,edgeEliminatedList,scaleNum,3)
    return octaveNum 

def visualize(filename, img, localExtremaList, scaleNum, index):
    magenta = [1,0,1]
    I = NP.zeros(img.shape)
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            I[y,x] = img[y,x]
    
    sigmaBase = 1.6
    for i in range(len(localExtremaList)):
        o,s,y,x,off_s,off_y,off_x = localExtremaList[i]
        sigma = sigmaBase*(2.0**(o+float(s+off_s)/float(scaleNum)))
        y_origin = (y + off_y) * (2.0**(o-1))
        x_origin = (x + off_x) * (2.0**(o-1))
        # draw circle
        for theata in range(0,360,5):
            newX = round(x_origin + sigma*math.cos(theata))
            newY = round(y_origin + sigma*math.sin(theata))
            if withinBound(I, newY, newX) and withinBound(I, y_origin, x_origin):
                I[newY, newX] = magenta
        
        '''
        numNeigh = int(round(sigma/2.0))
        # draw box
        for i in range(-numNeigh, numNeigh+1, 1):
            for j in range(-numNeigh, numNeigh+1, 1):
                if i == -numNeigh or j == -numNeigh or i == numNeigh or j == numNeigh:
                    if withinBound(I, y_origin+j, x_origin+i) and withinBound(I, y_origin, x_origin):
                        I[y_origin+j, x_origin+i] = magenta
        '''
    name = filename[0:len(filename)-4] + 'sift_keypoint_'+str(index)+'.png'
    skimage.io.imsave(name, I)
#        print '[',o,',',s,'] -- (',y, ',',x,')'
       

# SIFT - determind whether DoG[y,x] at #o octave and #s scale is local extrema by comparing [y,x] with its 26 neighbors   
def is_extrema(DoGPyramid, o, s, y, x):
    val = DoGPyramid[o,s][y,x]
    if val > 0:
        for k in range(-1,2,1):
            for j in range(-1,2,1):
                for i in range(-1,2,1):
                    if val < DoGPyramid[o,s+k][y+j,x+i]:
                        return False
    else:
        for k in range(-1,2,1):
            for j in range(-1,2,1):
                for i in range(-1,2,1):
                    if val > DoGPyramid[o,s+k][y+j,x+i]:
                        return False
    return True

# SIFT - perform the sub-pixel estimation
def interpolate_extrema(DoGPyramid, o, s, y, x, scaleNum,thresPeak):
    imgHeight = DoGPyramid[o,s].shape[0]
    imgWidth = DoGPyramid[o,s].shape[1]
    it = 0
    while it < SIFT_MAX_INTERP_STEPS:
        offset_x, offset_y, offset_s = interpolate_step(DoGPyramid, o, s, y, x)
        # if offset is larger than 0.5 in any dimension, then it means that the extremum lies cloaser to a different sample point
        # in this case the sample point is changed and the interpolation performed instead about that point
        if offset_x < 0.5 and offset_y < 0.5 and offset_s < 0.5:
            break
        else:
            x += round(offset_x)
            y += round(offset_y)
            s += round(offset_s)
            if s < 1 or s > scaleNum or y < SIFT_IMG_BORDER or y >= imgHeight - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= imgWidth - SIFT_IMG_BORDER:
                return []
        it += 1
    if it >= SIFT_MAX_INTERP_STEPS:
        return []
    offset = [offset_x, offset_y, offset_s]
    derivative = derivativeD(DoGPyramid, o, s, y, x)
    D_offset = DoGPyramid[o,s][y,x] + NP.dot(derivative,offset)*0.5
    # since D value is recomputed, so we should refilter it by the thresPeak
    if math.fabs(D_offset) < thresPeak/scaleNum:
        return []
    # contain x_origin, y_origin, x, y, o, s, offset_s
    # [y_origin, x_origin] is computed by upsample [y,x] to the origin image
    '''
    feature = {}
    feature['o'] = o
    feature['s'] = s
    feature['y'] = y
    feature['x'] = x
    feature['off_s'] = offset_s
    feature['y_origin'] = (y + offset_y) * (2.0**(o-1))
    feature['x_origin'] = (x + offset_x) * (2.0**(o-1))
    '''
    feature = [o,s,y,x,offset_s,offset_y,offset_x]
    return feature
        
    
# SIFT - take taylor series expansion, minimize to  get offset of extrema trueLoc X^ = - secondDerivative * Derivative
def interpolate_step(DoGPyramid, o, s, y, x):
    hessian = hessianMatrix(DoGPyramid, o, s, y, x)
    hessianInvert = linalg.inv(hessian)
    derivative = derivativeD(DoGPyramid, o, s, y, x)
    offset = NP.dot(hessianInvert,derivative)
    offset = offset*-1.0
    return offset[0], offset[1], offset[2]
    
# SIFT -     
def derivativeD(DoGPyramid, o, s, y, x):
    dx = (DoGPyramid[o,s][y,x+1] - DoGPyramid[o,s][y,x-1])/2.0
    dy = (DoGPyramid[o,s][y+1,x] - DoGPyramid[o,s][y-1,x])/2.0
    ds = (DoGPyramid[o,s+1][y,x] - DoGPyramid[o,s-1][y,x])/2.0
    return [dx,dy,ds]

def hessianMatrix(DoGPyramid, o, s, y, x):
    dxx = DoGPyramid[o,s][y,x+1] + DoGPyramid[o,s][y,x-1] - 2.0 * DoGPyramid[o,s][y,x]
    dyy = DoGPyramid[o,s][y+1,x] + DoGPyramid[o,s][y-1,x] - 2.0 * DoGPyramid[o,s][y,x]
    dss = DoGPyramid[o,s+1][y,x] + DoGPyramid[o,s-1][y,x] - 2.0 * DoGPyramid[o,s][y,x]
    dxy = (DoGPyramid[o,s][y+1,x+1]+DoGPyramid[o,s][y-1,x-1]-DoGPyramid[o,s][y-1,x+1]-DoGPyramid[o,s][y+1,x-1])/4.0
    dxs = (DoGPyramid[o,s+1][y,x+1]+DoGPyramid[o,s-1][y,x-1]-DoGPyramid[o,s-1][y,x+1]-DoGPyramid[o,s+1][y,x-1])/4.0
    dys = (DoGPyramid[o,s+1][y+1,x]+DoGPyramid[o,s-1][y-1,x]-DoGPyramid[o,s-1][y+1,x]-DoGPyramid[o,s+1][y-1,x])/4.0
    hessian = [[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]]
    return hessian

def eliminatingEdgeResponse(DoGPyramid, o, s, y, x, thresCurve):
    dxx = DoGPyramid[o,s][y,x+1] + DoGPyramid[o,s][y,x-1] - 2.0 * DoGPyramid[o,s][y,x]
    dyy = DoGPyramid[o,s][y+1,x] + DoGPyramid[o,s][y-1,x] - 2.0 * DoGPyramid[o,s][y,x]
    dxy = (DoGPyramid[o,s][y+1,x+1]+DoGPyramid[o,s][y-1,x-1]-DoGPyramid[o,s][y-1,x+1]-DoGPyramid[o,s][y+1,x-1])/4.0
    
    tr = dxx + dyy
    det = dxx*dyy - dxy*dxy
    
    if det <= 0:
        return True
    # when if statement is true, the keypoint won't be deleted
    if tr*tr/det < (thresCurve+1.0)*(thresCurve+1.0)/thresCurve:
        return False
    return True
    
def DoG(filename,octaveNum,scaleNum):
    sigma = 0.5
    inputname = filename[0:len(filename)-4] + '_sift'
    img = skimage.img_as_float(skimage.io.imread(filename))
    greyImg = greyScale(img)
    gaussianKernel = computeGaussianKernel(sigma)
    # upsample the image to get the -1 scale at first octave I (before upsample, blur the original image to denoise)
    greyImg = signal.convolve2d(greyImg,gaussianKernel,boundary='symm',mode='same')
    I = scipy.ndimage.interpolation.zoom(greyImg, 2.)
    skimage.io.imsave(inputname+'_grey.png', greyImg)
    skimage.io.imsave(inputname+'_upsampledGrey.png', I)
#    pdb.set_trace()
    
    
    sigmaBase = 1.6
    kBase = 2.0**(1.0/scaleNum)
    print 'kBase = ',kBase
    
    # compute the gaussian pyramid
    guassianPyramid = {}
    for o in range(octaveNum):
        for s in range(scaleNum+3):
            sigma = sigmaBase*(2.0**(o+float(s)/float(scaleNum)))
#            k = kBase**(o*scaleNum+s)
#            print '[',o,',',s,'] k = ', k
#            sigma = sigmaBase*k
            gaussianKernel = computeGaussianKernel(sigma)
            guassianPyramid[o,s] = signal.convolve2d(I,gaussianKernel,boundary='symm',mode='same')
            name = inputname + '_guassianPyramid_' + str(o) + '_' + str(s) + '.png'
            skimage.io.imsave(name, guassianPyramid[o,s])
            print '[',o,',',s,'] sigma = ',sigma
#        I = scipy.ndimage.interpolation.zoom(guassianPyramid[o,scaleNum],.5)
        I = scipy.ndimage.interpolation.zoom(I,.5)
    
    # compute the DoG pyramid
    DoGPyramid = {}
    for o in range(octaveNum):
        for s in range(scaleNum+2):
            DoGPyramid[o,s] = NP.subtract(guassianPyramid[o,s],guassianPyramid[o,s+1])
            name = inputname + '_DoGPyramid_' + str(o) + '_' + str(s) + '.png'
            skimage.io.imsave(name, DoGPyramid[o,s])
    return DoGPyramid
    
def main():
    
    inputfilename = 'building.jpg'
    sigma = 2
    thresH = 5
    thresL = 10
    edgeDetector(inputfilename,sigma, thresH/10.0, thresL/100.0)
    '''
    inputfilename = 'checker.jpg'
    sigma = 2
    width = 7
    thres = 5
    cornerDetector(inputfilename,sigma,width,thres)
    '''

    '''
    inputfilename = 'building.jpg'
    for sigma in range (1, 5, 1):
        for thresH in range (3, 10, 2):
            for thresL in range (5, 15, 5):
                edgeDetector(inputfilename,sigma, thresH/10.0, thresL/100.0)
    
    
    inputfilename = 'checker.jpg'
    for sigma in range(1,5,1):
        for width in range(2,8,2):
            for thres in range(1,10,2):
                print 'sigma = ', sigma, ' width = ', width, ' thres = ', thres
                cornerDetector(inputfilename,sigma,width,thres)
    '''           
    
    
    inputfilename = 'building.jpg'
    inputfilename = 'Lenna.png'
    inputfilename = 'mandrill.jpg'
#    sift(inputfilename,4,3,0.04,10)
    

    
if __name__ == "__main__": main()
    