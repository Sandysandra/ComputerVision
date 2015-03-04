# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 12:33:42 2015

@author: SHE
"""
from scipy import signal
from scipy import linalg
import scipy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import skimage,matplotlib,pdb
import skimage.io
import scipy, math, pdb, random
from random import randint

patchSize = 12

def withinBound(I,y,x):
    return y >= 0 and x >= 0 and y < I.shape[0] and x < I.shape[1]
    
def convertGifToPng(imgNames):
    for i in range(len(imgNames)):
        name = './images/' + imgNames[i]
        img = Image.open(name)
        im = img.convert('L')
        im.save(name[0:len(name)-4]+'.png', 'PNG')
    
def parseInput(filename):
    imgNames = np.loadtxt(filename, usecols=(0,), dtype="str")
    
    coorX = np.loadtxt(filename, usecols=(1, 3, 5, 7, 9, 11), dtype="float")
    coorY = np.loadtxt(filename, usecols=(2, 4, 6, 8, 10, 12), dtype="float")
    
    trainingSetImage = np.zeros((10*patchSize, 10*patchSize*2),dtype='float')
    trainX = np.zeros((200, patchSize*patchSize),dtype='float')
    trainY = np.zeros((200),dtype='float')
    testSetImage = np.zeros((10*patchSize, 10*patchSize),dtype='float')
    testX = np.zeros((200, patchSize*patchSize),dtype='float')
    testY = np.zeros((200))
    
    numTrainSet = 100
    
#    convertGifToPng(imgNames)
    faceBoundBox = {}
    padding = 4
    for i in range(numTrainSet):#len(imgNames)):
        # initialize training data (face)
        name = './images/' + imgNames[i][0:len(imgNames[i])-4]+'.png'
        img = skimage.img_as_float(skimage.io.imread(name))
        minX = min(coorX[i]) - padding
        maxX = max(coorX[i]) + padding
        minY = min(coorY[i]) - padding
        maxY = max(coorY[i]) + padding
        box = [minX,minY,maxX,maxY]
        I = img[minY:maxY+1,minX:maxX+1]
        yRatio = float(patchSize)/float(I.shape[0])
        xRatio = float(patchSize)/float(I.shape[1])
        imgI = scipy.ndimage.interpolation.zoom(I,(yRatio,xRatio))
        patchY = i/10
        patchX = i%10
        for y in range(patchSize):
            for x in range(patchSize):
                tmp = min(1.0,imgI[y,x])
                tmp = max(0.0,tmp)
                trainingSetImage[patchY*patchSize+y, patchX*patchSize+x] = tmp
                trainX[i,y*patchSize+x] = tmp
                trainY[i] = 1.0
        # create a dictionary associate image and faceBoundingboxs of faces in the image
        faceBoundBox.setdefault(name, []).append(box)

    for i in range(numTrainSet):
        # initialize test set
        name = './images/' + imgNames[i+numTrainSet][0:len(imgNames[i+numTrainSet])-4]+'.png'
        img = skimage.img_as_float(skimage.io.imread(name))
        minX = min(coorX[i+numTrainSet]) - padding
        maxX = max(coorX[i+numTrainSet]) + padding
        minY = min(coorY[i+numTrainSet]) - padding
        maxY = max(coorY[i+numTrainSet]) + padding
        box = [minX,minY,maxX,maxY]
        I = img[minY:maxY+1,minX:maxX+1]
        yRatio = float(patchSize)/float(I.shape[0])
        xRatio = float(patchSize)/float(I.shape[1])
        imgI = scipy.ndimage.interpolation.zoom(I,(yRatio,xRatio))
        patchY = i/10
        patchX = i%10
        for y in range(patchSize):
            for x in range(patchSize):
                tmp = min(1.0,imgI[y,x])
                tmp = max(0.0,tmp)
                testSetImage[patchY*patchSize+y, patchX*patchSize+x] = tmp
                testX[i,y*patchSize+x] = tmp
                testY[i] = 1
        faceBoundBox.setdefault(name, []).append(box)
        
    '''
    for i in range(numTrainSet+numTrainSet, len(imgNames), 1):
        name = './images/' + imgNames[i][0:len(imgNames[i])-4]+'.png'
        img = skimage.img_as_float(skimage.io.imread(name))
        minX = min(coorX[i]) - padding
        maxX = max(coorX[i]) + padding
        minY = min(coorY[i]) - padding
        maxY = max(coorY[i]) + padding
        box = [minX,minY,maxX,maxY]
        faceBoundBox.setdefault(name, []).append(box)
     '''   
    # extract non-face patches
    
    print len(faceBoundBox)
    random.seed()
    numNonFaceFromEachImg = 100/len(faceBoundBox) + 1
    for i in range(100):
        imgIndex = i/numNonFaceFromEachImg
        name = './images/' + imgNames[imgIndex][0:len(imgNames[imgIndex])-4]+'.png'
        img = skimage.img_as_float(skimage.io.imread(name))
        nonfaceY = randint(0,img.shape[0]-1-patchSize)
        nonfaceX = randint(0,img.shape[1]-1-patchSize)
        while overlapWithFace(nonfaceX,nonfaceY,faceBoundBox[name])==True:
            nonfaceY = randint(0,img.shape[0]-1-patchSize)
            nonfaceX = randint(0,img.shape[1]-1-patchSize)
        imgNonFace = img[nonfaceY:nonfaceY+patchSize+1,nonfaceX:nonfaceX+patchSize+1]
        patchY = (i)/10
        patchX = (i)%10
        for y in range(patchSize):
            for x in range(patchSize):
                trainingSetImage[patchY*patchSize+y, patchX*patchSize+x+10*patchSize] = imgNonFace[y,x]
                trainX[100+i,y*patchSize+x] = imgNonFace[y,x]
                trainY[100+i] = 0.0

    '''
    name = './images/harvard.png'
    img = skimage.img_as_float(skimage.io.imread(name))
    random.seed()
    for i in range(100):
        nonfaceY = randint(0,img.shape[0]-1-patchSize)
        nonfaceX = randint(0,img.shape[1]-1-patchSize)
        imgNonFace = img[nonfaceY:nonfaceY+patchSize+1,nonfaceX:nonfaceX+patchSize+1]
        patchY = (i)/10
        patchX = (i)%10
        for y in range(patchSize):
            for x in range(patchSize):
                trainingSetImage[patchY*patchSize+y, patchX*patchSize+x+10*patchSize] = imgNonFace[y,x]
                trainX[100+i,y*patchSize+x] = imgNonFace[y,x]
                trainY[100+i] = 0
    '''
    plt.imshow(trainingSetImage, cmap = matplotlib.cm.Greys_r)
    plt.show()
    skimage.io.imsave('trainingSet.png', trainingSetImage)
    skimage.io.imsave('testSet.png', testSetImage)
    skimage.io.imsave('trainX.png', trainX)
    return trainX, trainY, testX

def overlapWithFace(x,y,faceBoundBoxList):
    for i in range(len(faceBoundBoxList)):
        box = faceBoundBoxList[i]
        if x >= box[0]-patchSize and y >= box[1]-patchSize and x <= box[2] and y <= box[3]:
            return True
    return False

def computeGaussianModel(trainX):
    trainPos = trainX[0:100,:]
    meanPos = trainPos.mean(axis=0)
    # plot the mean face here
    meanFace = np.reshape(meanPos, (patchSize, patchSize))
    skimage.io.imsave('meanFace.png', meanFace)
    
    APos = [ trainX[i]-meanPos for i in range(0,100)]
    APos = np.asarray(APos)
    covPos = np.dot(np.transpose(APos), APos)
    UPos,SPos,VPos = np.linalg.svd(covPos,full_matrices=True)
    x = [i for i in range(20)]
    plt.plot(x, SPos[1:21], 'ro')
    plt.show()
    tau_index = 10
    UkPos = UPos[:,0:tau_index]
    covkPos = np.dot(np.dot(UkPos, scipy.linalg.inv(np.diag(SPos[0:tau_index]))), np.transpose(UkPos))
    detPos = np.linalg.det(np.diag(SPos[0:tau_index]))
    print 'detPos = ',detPos
    
    
    trainNeg = trainX[100:trainX.shape[0],:]
    meanNeg = trainNeg.mean(axis=0)
    meanNonFace = np.reshape(meanNeg, (patchSize, patchSize))
    skimage.io.imsave('meanNonFace.png', meanNonFace)
    ANeg = [ trainX[i]-meanNeg for i in range(100,trainX.shape[0])]
    ANeg = np.asarray(ANeg)
    covNeg = np.dot(np.transpose(ANeg), ANeg)
    UNeg,SNeg,VNeg = np.linalg.svd(covNeg,full_matrices=True)
    x = [i for i in range(20)]
    plt.plot(x, SNeg[1:21], 'ro')
    plt.show()
    tau_index = 10
    UkNeg = UNeg[:,0:tau_index]
    covkNeg = np.dot(np.dot(UkNeg, scipy.linalg.inv(np.diag(SNeg[0:tau_index]))), np.transpose(UkNeg))
    detNeg = np.linalg.det(np.diag(SNeg[0:tau_index]))
    print 'detNeg = ', detNeg
#    pdb.set_trace()
    print trainPos.shape, len(meanPos), APos.shape, ANeg.shape, covPos.shape
    return meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg

# decide if an input 12*12 patch is an face    
def gaussianDetectorPerPatch(facePatch, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg):
    try:
        facePatch.shape = [patchSize,patchSize]
    except:
        print "gaussianDetectorPerPatch: the face patch size should be ", [patchSize,patchSize]
        print 'Actuall size is ', facePatch.shape
    tauIndexPos = 10.0
    tauIndexNeg = 10.0
    face = facePatch.flatten()
    expValuePos = math.exp(-0.5*np.dot(np.dot(np.transpose(face-meanPos),covkPos),(face-meanPos)))
    print 'firstPart = ', 1.0/(math.pow(2.0*math.pi,tauIndexPos/2)*math.sqrt(detPos)), '| EPos = ', expValuePos
    probPos = 1.0/(math.pow(2.0*math.pi,tauIndexPos/2)*math.sqrt(detPos))*expValuePos
    expValueNeg = math.exp(-0.5*np.dot(np.dot(np.transpose(face-meanNeg),covkNeg),(face-meanNeg)))
    print 'firstPart = ', 1.0/(math.pow(2.0*math.pi,tauIndexNeg/2)*math.sqrt(detNeg)), '| ENeg = ', expValueNeg
    probNeg = 1.0/(math.pow(2.0*math.pi,tauIndexNeg/2)*math.sqrt(detNeg))*expValueNeg
    print probPos, probNeg
    if probPos >= probNeg:
        return True
    else:
        return False

# compute the gaussian pyramid for an input image        
def gaussianPyramid(filename,scaleNum):
    img = skimage.img_as_float(skimage.io.imread(filename))
    filtedImg = scipy.ndimage.filters.gaussian_filter(img,0.5)
    sigma = 0.5
    gaussPyr = []
    gaussPyr.append(filtedImg)
    for i in range(scaleNum):
        filtedImg = scipy.ndimage.interpolation.zoom(filtedImg,.5)
        gaussPyr.append(filtedImg)
        skimage.io.imsave(filename[0:len(filename)-4]+'_Pyramid_'+str(i)+'.png', filtedImg);
        #filtedImg = scipy.ndimage.filters.gaussian_filter(filtedImg,sigma)
    return gaussPyr

def gaussianDetector(gaussPyr, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg):
    faceList = []
    for i in range(len(gaussPyr)):
        I = gaussPyr[i]
        for y in range(I.shape[0]-patchSize):
            for x in range(I.shape[1]-patchSize):
                facePatch = I[y:y+patchSize, x:x+patchSize]
                if gaussianDetectorPerPatch(facePatch, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg) == True:
                    faceList.append([i,y,x])
                
def visualizeFace(testImgName, faceList):
    img = skimage.img_as_float(skimage.io.imread(testImgName))
    for i in range(len(faceList)):
        scale,posY,posX = faceList[i]
        for y in range(0, patchSize*int(math.pow(2,scale))+1, 1):
            for x in range(0, patchSize*int(math.pow(2,scale))+1, 1):
                if x == 0 or y == 0 or x == patchSize*int(math.pow(2,scale)) or y == patchSize*int(math.pow(2,scale)):
                    if withinBound(img, posY*int(math.pow(2,scale))+y, posX*int(math.pow(2,scale))+x):
                        img[posY*int(math.pow(2,scale))+y, posX*int(math.pow(2,scale))+x] = 1
    skimage.io.imsave(testImgName[0:len(testImgName)-4]+'_TestResult.png', img);

def computeG(w,x_i):
	tmp = -1.0*np.dot(np.transpose(w),x_i)
	tmp1 = np.exp(tmp)
	g = 1.0/(1.0+tmp1)
        return g

def logisticRegression(trainX, trainY, mu, convThres):
	w = np.zeros((patchSize*patchSize+1),np.dtype(np.float64))
        it = 0
	while it<1000:
		it += 1
		tmp = 0
		for i in range(trainX.shape[0]):
			tmp += np.dot((trainY[i] - computeG(w,trainX[i])),trainX[i])
		w = w + np.dot(mu,tmp)
	return w
	
def logisticDetectorPerPatch(face,w):
	g = computeG(w,face)
	if g >= 0.5:
		return True
	else:
		return False

def logisticDetector(gaussPyr,w):
    faceList = []
    for i in range(len(gaussPyr)):
        print 'pyramid ',i
        I = gaussPyr[i]
        for y in range(I.shape[0]-patchSize):
            for x in range(I.shape[1]-patchSize):
                facePatch = np.zeros((patchSize*patchSize+1),dtype='float')
                face = I[y:y+patchSize, x:x+patchSize]
                face = face.flatten()
                facePatch[0:patchSize*patchSize] = face
                facePatch[patchSize*patchSize] = 1
                if logisticDetectorPerPatch(facePatch, w) == True:
                    faceList.append([i,y,x])

def main():
    filename = 'faceList.txt'
    testImgName = './images/Argentina.png'
    
    scaleNum = 3
    trainX, trainY, testX = parseInput(filename)
    gaussPyr = gaussianPyramid(testImgName,scaleNum)
    
    
    # Gaussian Detector
    # detector proportional to P(x|face)*para / P(x|nonface)*(1-para)
    meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg = computeGaussianModel(trainX)
    #    faceList = gaussianDetector(gaussPyr, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg)
    #    visualizeFace(testImgName, faceList)
    #    testX[0:50] = trainX[100:150]
    
    for i in range(200):
        test = trainX[i]
        test = np.reshape(test, (patchSize, patchSize))
        if gaussianDetectorPerPatch(test, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg) == True:
            print i, 'Face'
        else:
            print i, 'nonFace'
    
    # Logistic Detector
    trainXL = np.zeros((200, patchSize*patchSize+1),dtype=np.dtype(np.float64))
    trainXL[:,0:patchSize*patchSize] = trainX
    trainXL[:,(patchSize*patchSize):(patchSize*patchSize+1)] = 1
    testXL = np.zeros((200, patchSize*patchSize+1),dtype=np.dtype(np.float64))
    testXL[:,0:patchSize*patchSize] = testX
    testXL[:,patchSize*patchSize:patchSize*patchSize+1] = 1
    print 'trainXL Shape ',trainXL.shape
    w = logisticRegression(trainXL, trainY, 0.5, 0.01)
    faceList = logisticDetector(gaussPyr,w)
    visualizeFace(testImgName,faceList)
    for i in range(200):
    	test = testXL[i]
    	if logisticDetectorPerPatch(test,w) == True:
    		print i, 'Face'
    	else:
    		print i, 'NonFace'
    		
if __name__ == "__main__": main()
