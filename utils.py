import numpy as np

# For small datasets, perform dataset augmentation by adding more
# examples that are just noiser versions of existing data from
# the original dataset
def addNoisyExamplesToData(x, y, numNewExamples, mean, var):
  noisySamplesX = np.zeros((numNewExamples, ) + x.shape[1:])
  noisySamplesY = np.zeros(numNewExamples)

  for i in range(numNewExamples):
    idx = np.random.choice(x.shape[0])
    noisySamplesX[i] = x[idx] + np.random.normal(mean, var, x[idx].shape)
    noisySamplesY[i] = y[idx]

  return np.vstack((x, noisySamplesX)), np.hstack((y, noisySamplesY))