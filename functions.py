import torch
import torch.nn as nn
import torch.nn.functional as F




# A function that returns feature maps

def getFeatureMapActs(img,net):

  # initialize feature maps as a list
  featuremaps = []
  featurenames = []

  convLayerIdx = 0

  # loop through all layers in the "features" block
  for layernum in range(len(net.features)):

    # print out info from this layer
    # print(layernum,net.features[layernum])

    # process the image through this layer
    img = net.features[layernum](img)

    # store the image if it's a conv2d layer
    if 'Conv2d' in str(net.features[layernum]):
      featuremaps.append( img )
      featurenames.append( 'ConvLayer_' + str(convLayerIdx) )
      convLayerIdx += 1

  return featuremaps,featurenames







# A function that returns the Gram matrix of the feature activation map

def gram_matrix(F):

  # reshape to 2D
  _,chans,height,width = F.shape
  F = F.reshape(chans,height*width)

  # compute and return covariance matrix
  gram = torch.mm(F,F.t()) / (chans*height*width)
  return gram




