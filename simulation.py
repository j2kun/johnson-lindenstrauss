import random
import math
from itertools import combinations

def randUnitCube(n):
   return [(random.random() - 0.5)*2 for _ in range(n)]

def sphereCubeRatio(n, numSamples):
   randomSample = [randUnitCube(n) for _ in range(numSamples)]
   return sum(1 for x in randomSample if sum(a**2 for a in x) <= 1) / numSamples

def sphereVolume(n):
   values = [0] * (n+1)
   for i in range(n+1):
      if i == 0:
         values[i] = 1
      elif i == 1:
         values[i] = 2
      else:
         values[i] = 2*math.pi / i * values[i-2]

   return values[-1]

def dist(x,y):
   return math.sqrt(sum((a-b)**2 for (a,b) in zip(x,y)))

def distancesRandomPoints(n, numSamples):
   randomSample = [randUnitCube(n) for _ in range(numSamples)]
   pairwiseDistances = [dist(x,y) for (x,y) in combinations(randomSample, 2)]
   return pairwiseDistances

if __name__ == "__main__":
   import matplotlib.pyplot as plt

   for i in range(2,100):
      plt.clf()
      plt.ylim(0,200000)
      plt.xlim(0,10)
      plt.hist(distancesRandomPoints(i, 1000))
      plt.savefig('distances-animation/%03d.png' % i, bbox_inches='tight')
