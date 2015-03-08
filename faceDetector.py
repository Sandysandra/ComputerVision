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
windowSize = 40
M = 15

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
    testSetImage = np.zeros((10*patchSize, 10*patchSize*2),dtype='float')
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
                testSetImage[patchY*patchSize+y, patchX*patchSize+x+10*patchSize] = imgNonFace[y,x]
                testX[100+i,y*patchSize+x] = imgNonFace[y,x]
                testY[100+i] = 0.0

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
    skimage.io.imsave('testX.png', testX)
    return trainX, trainY, testX

def overlapWithFace(x,y,faceBoundBoxList):
    for i in range(len(faceBoundBoxList)):
        box = faceBoundBoxList[i]
        if x >= box[0]-patchSize and y >= box[1]-patchSize and x <= box[2] and y <= box[3]:
            return True
    return False

def computeGaussianModel(trainX, tau_index):
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
    UkNeg = UNeg[:,0:tau_index]
    covkNeg = np.dot(np.dot(UkNeg, scipy.linalg.inv(np.diag(SNeg[0:tau_index]))), np.transpose(UkNeg))
    detNeg = np.linalg.det(np.diag(SNeg[0:tau_index]))
    print 'detNeg = ', detNeg
#    pdb.set_trace()
    print trainPos.shape, len(meanPos), APos.shape, ANeg.shape, covPos.shape
    return meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg

# decide if an input 12*12 patch is an face    
def gaussianDetectorPerPatch(tau_index, facePatch, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg):
    try:
        facePatch.shape = [patchSize,patchSize]
    except:
        print "gaussianDetectorPerPatch: the face patch size should be ", [patchSize,patchSize]
        print 'Actuall size is ', facePatch.shape
    face = facePatch.flatten()
    firstPartPos = 1.0/(math.pow(2.0*math.pi,tau_index/2)*math.sqrt(detPos))
    expValuePos = math.exp(-0.5*np.dot(np.dot(np.transpose(face-meanPos),covkPos),(face-meanPos)))
    probPos = expValuePos
    #print 'firstPart = ', firstPartPos, '| EPos = ', expValuePos
    firstPartNeg = 1.0/(math.pow(2.0*math.pi,tau_index/2)*math.sqrt(detNeg))
    expValueNeg = math.exp(-0.5*np.dot(np.dot(np.transpose(face-meanNeg),covkNeg),(face-meanNeg)))
    probNeg = expValueNeg
    #print 'firstPart = ', firstPartNeg, '| ENeg = ', expValueNeg
    #print probPos/probNeg
    if probPos >= probNeg:
        return [True,probPos-probNeg]
    else:
        return [False,probPos-probNeg]

# compute the gaussian pyramid for an input image        
def gaussianPyramid(filename,scaleNum):
    img = skimage.img_as_float(skimage.io.imread(filename))
    #filtedImg = scipy.ndimage.filters.gaussian_filter(img,0.5)
    filtedImg = img
    sigma = 0.5
    gaussPyr = []
    gaussPyr.append(filtedImg)
    for i in range(1,scaleNum):
        filtedImg = scipy.ndimage.interpolation.zoom(filtedImg,.5)
        gaussPyr.append(filtedImg)
        skimage.io.imsave(filename[0:len(filename)-4]+'_Pyramid_'+str(i)+'.png', filtedImg);
        #filtedImg = scipy.ndimage.filters.gaussian_filter(filtedImg,sigma)
    return gaussPyr

def gaussianDetector(tau_index, gaussPyr, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg):
    faceList = []
    probability = []
    for i in range(len(gaussPyr)):
    	print 'Pyramid ', i
        I = gaussPyr[i]
        for y in range(I.shape[0]-windowSize):
            for x in range(I.shape[1]-windowSize):
                facePatch = I[y:y+windowSize, x:x+windowSize]
                ratio = float(patchSize)/float(windowSize)
                face = scipy.ndimage.interpolation.zoom(facePatch,ratio)
                result = gaussianDetectorPerPatch(tau_index,face, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg)
                if result[0] == True:
                    faceList.append([y,x,i])
                    probability.append(result[1])
    finalFaceList = nonMaximumSuppression(faceList, probability, gaussPyr[0])
    return finalFaceList
                
def visualizeFace(typeOfClassifier, testImgName, faceList):
    img = skimage.img_as_float(skimage.io.imread(testImgName))
    for i in range(len(faceList)):
    	posY,posX,scale = faceList[i]
    	for y in range(0, windowSize*int(math.pow(2,scale))+1, 1):
    		for x in range(0, windowSize*int(math.pow(2,scale))+1, 1):
    			if x == 0 or y == 0 or x == windowSize*int(math.pow(2,scale)) or y == windowSize*int(math.pow(2,scale)):
    				if withinBound(img, posY*int(math.pow(2,scale))+y, posX*int(math.pow(2,scale))+x):
    					img[posY*int(math.pow(2,scale))+y, posX*int(math.pow(2,scale))+x] = 1
    				
    classifier = ''
    if typeOfClassifier == 0:
    	classifier = '_GaussianClassifier'
    else:
    	classifier = '_LogisticClassifier'
    
    skimage.io.imsave(testImgName[0:len(testImgName)-4]+classifier+'_TestResult.png', img);

def computeG(w,x_i):
	tmp = -1.0*np.dot(np.transpose(w),x_i)
	tmp1 = np.exp(tmp)
	g = 1.0/(1.0+tmp1)
        return g

def logisticRegression(trainX, trainY, mu, iters):
	w = np.zeros((patchSize*patchSize+1),np.dtype(np.float64))
        it = 0
	while it<iters:
		it += 1
		tmp = 0
		for i in range(trainX.shape[0]):
			tmp += np.dot((trainY[i] - computeG(w,trainX[i])),trainX[i])
		w = w + np.dot(mu,tmp)
	return w
	
def logisticDetectorPerPatch(face,w):
	g = computeG(w,face)
	#print 'g = ', g
	if g >= 0.5:
		return [True,g]
	else:
		return [False,g]

def logisticDetector(gaussPyr,w):
    faceList = []
    probability = []
    for i in range(len(gaussPyr)):
        print 'pyramid ',i
        I = gaussPyr[i]
        for y in range(I.shape[0]-windowSize):
            for x in range(I.shape[1]-windowSize):
                facePatch = np.zeros((patchSize*patchSize+1),dtype='float')
                face = I[y:y+windowSize, x:x+windowSize]
                ratio = float(patchSize)/float(windowSize)
                imgI = scipy.ndimage.interpolation.zoom(face,ratio)
                newface = imgI.flatten()
                facePatch[0:patchSize*patchSize] = newface
                facePatch[patchSize*patchSize] = 1
                result = logisticDetectorPerPatch(facePatch, w)
                if result[0] == True:
                    faceList.append([y,x,i])
                    probability.append(result[1])
    finalFaceList = nonMaximumSuppression(faceList, probability, gaussPyr[0])
    return finalFaceList
    
def nonMaximumSuppression(faceList, probability, I):
	if len(probability) > 0:
		faceList, probability = zip(*sorted(zip(faceList, probability), reverse=True, key=lambda x: x[1]))
	
	L = []
	mask = np.zeros(I.shape)
	for i in range(len(faceList)):
		pos = faceList[i]
		scale = pos[2]
		if mask[pos[0]*int(math.pow(2,scale)),pos[1]*int(math.pow(2,scale))] == 0:
			L.append(pos)
			for y in range(-windowSize-M*int(math.pow(2,scale)), 2*windowSize+M*int(math.pow(2,scale)), 1):
				for x in range( -windowSize-M*int(math.pow(2,scale)), 2*windowSize+M*int(math.pow(2,scale)), 1):
					if withinBound(I, pos[0]*int(math.pow(2,scale))+y, pos[1]*int(math.pow(2,scale))+x):
						mask[pos[0]*int(math.pow(2,scale))+y, pos[1]*int(math.pow(2,scale))+x] = 1
	return L

def detectConfilition( mask, pos, I ):
	scale = pos[2]
	for y in range(windowSize*int(math.pow(2,scale))):
		for x in range(windowSize*int(math.pow(2,scale))):
			if withinBound(I, pos[0]*int(math.pow(2,scale))+y, pos[1]*int(math.pow(2,scale))+x):
				if mask[pos[0]*int(math.pow(2,scale)),pos[1]*int(math.pow(2,scale))] == 1:
					return False
	return True

def testGaussian(tau_index, testX, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg):
	numCorrect = 0.0
	for i in range(200):
		test = testX[i]
		result = gaussianDetectorPerPatch(tau_index,test, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg)
		if result[0] == True:
			if i < 100:
				numCorrect += 1.0
		else:
			if i >= 100:
				numCorrect += 1.0
	return numCorrect/200.0
    
def testLogisitic(testX, w):
	numCorrect = 0.0
	for i in range(200):
		test = testX[i]
		result = logisticDetectorPerPatch(test,w)
		if result[0] == True:
			if i < 100:
				numCorrect += 1.0
		else:
			if i >= 100:
				numCorrect += 1.0
	return numCorrect/200.0
    
def main():
    filename = 'faceList.txt'
    testImgName = './images/Argentina.png'
    
    scaleNum = 3
    #trainX, trainY, testX = parseInput(filename)
    trainX = skimage.img_as_float(skimage.io.imread('trainX.png'))
    testX = skimage.img_as_float(skimage.io.imread('testX.png'))
    trainY = np.zeros((200), dtype = 'float')
    testY = np.zeros((200), dtype = 'float')
    trainY[0:100] = 1.0
    testY[0:100] = 1.0
    trainXL = np.zeros((200, patchSize*patchSize+1),dtype=np.dtype(np.float64))
    trainXL[:,0:patchSize*patchSize] = trainX
    trainXL[:,(patchSize*patchSize):(patchSize*patchSize+1)] = 1
    testXL = np.zeros((200, patchSize*patchSize+1),dtype=np.dtype(np.float64))
    testXL[:,0:patchSize*patchSize] = testX
    testXL[:,patchSize*patchSize:patchSize*patchSize+1] = 1
    gaussPyr = gaussianPyramid(testImgName,scaleNum)
    
    
    # Gaussian Detector
    # detector proportional to P(x|face)*para / P(x|nonface)*(1-para)
    gaussianAccuracy = []
    tau_index_start = 4
    tau_index_num = 20
    tau_index_maxAccuracy = tau_index_start
    gaussian_max_accuracy = 0.0
    x = []
    for i in range(tau_index_num):
    	tau_index = tau_index_start + 2*i
    	x.append(tau_index)
    	meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg = computeGaussianModel(trainX, tau_index)
    	accuracy = testGaussian(tau_index, testX, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg)
    	print 'Gaussian accuracy k = ', tau_index, ' | ', accuracy
    	gaussianAccuracy.append(accuracy)
    	if accuracy > gaussian_max_accuracy:
    		gaussian_max_accuracy = accuracy
    		tau_index_maxAccuracy = tau_index
    print 'max accuracy ', gaussian_max_accuracy, ' at tau_index=', tau_index_maxAccuracy
    plt.plot(x, gaussianAccuracy, 'ro')
    plt.show()
    meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg = computeGaussianModel(trainX, tau_index_maxAccuracy)
    faceList = gaussianDetector(tau_index, gaussPyr, meanPos, detPos, covkPos, meanNeg, detNeg, covkNeg)
    visualizeFace(0,testImgName, faceList)
    
    # Logistic Detector
    rate = 0.5
    logisticAccuracy = []
    iters_start = 50
    iters_num = 20
    iters_maxAccuracy = iters_start
    logistic_max_accuracy = 0.0
    y = []
    for i in range(iters_num):
    	iters = iters_start + 100*i
    	y.append(iters)
    	w = logisticRegression(trainXL, trainY, rate, iters)
    	accuracy = testLogisitic(testXL, w)
    	print 'Logistic accuracy it = ', iters, ' | ', accuracy
    	logisticAccuracy.append(accuracy)
    	if accuracy > logistic_max_accuracy:
    		logistic_max_accuracy = accuracy
    		iters_maxAccuracy = iters
    plt.plot(y, logisticAccuracy, 'ro')
    plt.show()
    print 'max accuracy ', logistic_max_accuracy, ' at tau_index=', iters_maxAccuracy
    w = logisticRegression(trainXL, trainY, rate, iters_maxAccuracy)
    faceList = logisticDetector(gaussPyr,w)
    visualizeFace(1,testImgName,faceList)
    
    		
if __name__ == "__main__": main()
