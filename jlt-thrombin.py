from jlt import *

import numpy
import random
from collections import Counter

def thrombinSubpsaceDistanceHistograms():
   import matplotlib.pyplot as plt
   from data import thrombin 
   train, labels = thrombin.load()
   
   for subspaceDim in [5000, 1000, 750, 500, 250, 100, 75, 50, 10, 5, 2]:
      newData = jlt(train, subspaceDim)
      
      plt.clf()
      plt.ylim(0,200000)
      plt.xlim(0,250)
      plt.hist(distances(newData), bins=100)
      plt.savefig('thrombin-animation/%05d.png' % subspaceDim, bbox_inches='tight')


def thrombinTheoreticalBoundReduction():
   import matplotlib.pyplot as plt
   from data import thrombin 
   train, labels = thrombin.load()
   
   numPoints = len(train)
   subspaceDim = theoreticalBound(numPoints, 0.2)
   ambientDim = len(train[0])
   print((subspaceDim, ambientDim))
   
   #newData = jlt(train, subspaceDim)
   allDistances = distances(train)
   plt.clf()
   plt.hist(allDistances, bins=100)
   plt.savefig('thrombin-original.png', bbox_inches='tight')

   projectedDistances = distances(jlt(train, subspaceDim))
   plt.clf()
   plt.hist(projectedDistances, bins=100)
   plt.savefig('thrombin-%05d.png' % subspaceDim, bbox_inches='tight')


def checkThrombin():
   from data import thrombin 
   train, labels = thrombin.load()
   
   numPoints = len(train)
   epsilon = 0.2
   subspaceDim = theoreticalBound(numPoints, epsilon)
   ambientDim = len(train[0])
   print((subspaceDim, ambientDim))
   
   newData = jlt(train, subspaceDim)

   print(checkTheorem(train, newData, epsilon))


# takes a while to run
def worstCaseDistancePlot():
   import matplotlib.pyplot as plt
   from data import thrombin 
   train, labels = thrombin.load()
         
   dims = [1000, 750, 500, 250, 100, 75, 50, 25, 10, 5, 2]
   epsilon = 0.1
   numTrials = 20

   dataPoints = []
   means = []
   stds = []

   for dim in dims:
      dataPoints = []

      for i in range(numTrials):
         print("%d, trial %d" % (dim, i))
         newData = jlt(train, dim)
         dataPoints.append(checkTheorem(train, newData, epsilon))
   
      means.append(numpy.mean(dataPoints))
      stds.append(numpy.std(dataPoints))

   plt.clf()
   plt.errorbar(dims, means, yerr=stds, fmt='-o')
   plt.savefig('thrombin-worst-case.png')


def randomSplit(data, labels, percent=0.2):
   n = len(data)
   permutation = list(range(n))
   for i in range(n):
      j = random.randint(i, n-1)
      permutation[i], permutation[j] = permutation[j], permutation[i]
 
   data = data[permutation] # numpy array
   labels = labels[permutation]
   
   splitIndex = int(percent * len(data))
   return data[splitIndex:], labels[splitIndex:], data[:splitIndex], labels[:splitIndex]


def nearestNeighborsAccuracy(data, labels, k=10):
   from sklearn.neighbors import NearestNeighbors
   trainData, trainLabels, testData, testLabels = randomSplit(data, labels)
   model = NearestNeighbors(n_neighbors=k).fit(trainData)
   distances, indices = model.kneighbors(testData)
   predictedLabels = []
   
   for x in indices:
      xLabels = [trainLabels[i] for i in x[1:]] 
      #predictedLabel = Counter(xLabels).most_common()[0][0]
      #predictedLabels.append(predictedLabel)

      # try max
      predictedLabel = max(xLabels) 
      predictedLabels.append(predictedLabel)
      

   totalAccuracy = sum(x == y for (x,y) in zip(testLabels, predictedLabels)) / len(testLabels)
   falsePositive = (sum(x == 0 and y == 1 for (x,y) in zip(testLabels, predictedLabels)) / 
      sum(x == 0 for x in testLabels))
   falseNegative = (sum(x == 1 and y == 0 for (x,y) in zip(testLabels, predictedLabels)) / 
      sum(x == 1 for x in testLabels))

   return totalAccuracy, falsePositive, falseNegative
      

def knnThrombinAccuracyPlot(trials=50):
   import matplotlib.pyplot as plt
   from data import thrombin
   train, labels = thrombin.load()

   dims = [1000, 750, 500, 250, 100, 75, 50, 25, 10, 5, 2]
   epsilon = 0.1

   print("original data")
   baseAccuracy, baseFP, baseFN = nearestNeighborsAccuracy(train, labels)
   print((baseAccuracy, baseFP, baseFN))

   accuracyMeans = []
   accuracyStds = []
   falsePosMeans = []
   falsePosStds = []
   falseNegMeans = []
   falseNegStds = []

   for dim in dims:
      accuracyPts = []
      falsePosPts = []
      falseNegPts = []

      for i in range(trials):
         print("dim %d, trial %d" % (dim, i))
         newData = jlt(train, dim)
         acc, fp, fn = nearestNeighborsAccuracy(newData, labels)
         accuracyPts.append(acc)
         falsePosPts.append(fp)
         falseNegPts.append(fn)
         #print((accuracies[-1], falsePos[-1], falseNeg[-1]))

      accuracyMeans.append(numpy.mean(accuracyPts))
      falsePosMeans.append(numpy.mean(falsePosPts))
      falseNegMeans.append(numpy.mean(falseNegPts))

      accuracyStds.append(numpy.std(accuracyPts))
      falsePosStds.append(numpy.std(falsePosPts))
      falseNegStds.append(numpy.std(falseNegPts))

   plt.clf()
   plt.errorbar(dims, accuracyMeans, yerr=accuracyStds, fmt='-o')
   plt.axhline(y=baseAccuracy)
   plt.savefig('thrombin-knn-accuracy.png')

   plt.clf()
   plt.errorbar(dims, falsePosMeans, yerr=falsePosStds, fmt='-o')
   plt.axhline(y=baseFP)
   plt.savefig('thrombin-knn-fp.png')

   plt.clf()
   plt.errorbar(dims, falseNegMeans, yerr=falseNegStds, fmt='-o')
   plt.axhline(y=baseFN)
   plt.savefig('thrombin-knn-fn.png')


if __name__ == "__main__":
   knnThrombinAccuracyPlot()
