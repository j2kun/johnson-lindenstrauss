'''
Data taken from KDD Cup 2001 
http://pages.cs.wisc.edu/~dpage/kddcup2001/
'''

import numpy

def normalizeRow(v):
   norm = numpy.linalg.norm(v)
   if norm == 0: 
      raise Exception("Norm is zero")

   return v / norm

def normalizeData(data):
   return numpy.array([normalizeRow(row) for row in data])

def load(normalize=False):
   trainFilename = 'data/thrombin.data'

   with open(trainFilename, 'r') as infile:
      dataAndLabels = [line.strip().split(',') for line in infile]
      labels = numpy.array([(1 if row[0] == 'A' else 0) for row in dataAndLabels])
      trainingData = numpy.array([[int(x) for x in row[1:]] for row in dataAndLabels])

   if normalize:
      return normalizeData(trainingData), labels

   return trainingData, labels
