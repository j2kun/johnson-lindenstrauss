import math
import numpy
from itertools import combinations

def randomSubspace(subspaceDimension, ambientDimension):
   return numpy.random.normal(0, 1, size=(subspaceDimension, ambientDimension))

def project(v, subspace):
   subspaceDimension = len(subspace)
   return (1 / math.sqrt(subspaceDimension)) * subspace.dot(v)

def theoreticalBound(n, epsilon):
   return math.ceil(8*math.log(n) / (epsilon**2 - epsilon**3)) 

def distances(data):
   return numpy.array([numpy.linalg.norm(x-y) for (x,y) in combinations(data, 2)])

def checkTheorem(oldData, newData, epsilon):
   numBadPoints = 0

   for (x,y), (x2,y2) in zip(combinations(oldData, 2), combinations(newData, 2)):
      oldNorm = numpy.linalg.norm(x2-y2)**2 
      newNorm = numpy.linalg.norm(x-y)**2 

      if newNorm == 0 or oldNorm == 0:
         continue

      if abs(oldNorm / newNorm - 1) > epsilon:
         numBadPoints += 1

   return numBadPoints


def jlt(data, subspaceDimension, includeSubspace=False):
   ambientDimension = len(data[0])
   A = randomSubspace(subspaceDimension, ambientDimension) 
   transformed = (1 / math.sqrt(subspaceDimension)) * A.dot(data.T).T

   if includeSubspace:
      return transformed, A 

   return transformed


