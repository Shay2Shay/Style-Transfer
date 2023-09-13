import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

from imageio import imread
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

import sys


from functions import *


def main(
  img4content = './images/content/pexels-photo-4906283.webp',
  img4style = './images/style/the-artist-margaret-schwartz.webp',
  styleScaling = 5e3 * 1.5,
  numepochs = 1000):
  # ==================================================

  alexnet = torchvision.models.alexnet(pretrained=True)

  # freeze all layers
  for p in alexnet.parameters():
      p.requires_grad = False
  alexnet.eval()


  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  alexnet.to(device)


  # ===================================================

  img4content = imread(img4content)
  img4style   = imread(img4style)

  img4target = np.random.randint(low=0,high=255,size=img4content.shape,dtype=np.uint8)

  ## These images are really large, which will make training take a long time.
  Ts = T.Compose([ T.ToTensor(),
                  T.Resize(256),
                  T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

  # apply them to the images ("unsqueeze" to make them a 4D tensor) and push to GPU
  img4content = Ts( img4content ).unsqueeze(0).to(device)
  img4style   = Ts( img4style ).unsqueeze(0).to(device)
  img4target  = Ts( img4target ).unsqueeze(0).to(device)


  # ====================================================

  featmaps,featnames = getFeatureMapActs(img4content,alexnet)
  contentFeatureMaps,contentFeatureNames = getFeatureMapActs(img4content,alexnet)

  fig,axs = plt.subplots(2,5,figsize=(18,6))
  for i in range(5):

    # average over all feature maps from this layer, and normalize
    pic = np.mean( contentFeatureMaps[i].cpu().squeeze().numpy() ,axis=0)
    pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))

    axs[0,i].imshow(pic,cmap='gray')
    axs[0,i].set_title('Content layer ' + str(contentFeatureNames[i]))


    ### now show the gram matrix
    pic = gram_matrix(contentFeatureMaps[i]).cpu().numpy()
    pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))

    axs[1,i].imshow(pic,cmap='gray',vmax=.1)
    axs[1,i].set_title('Gram matrix, layer ' + str(contentFeatureNames[i]))
    axs[1,i].set_xlabel(f"Size = {pic.shape}")

  plt.tight_layout()
  plt.show()


  styleFeatureMaps,styleFeatureNames = getFeatureMapActs(img4style,alexnet)


  # ====================================================

  # which layers to use
  layers4content = [ 'ConvLayer_0']#,'ConvLayer_1','ConvLayer_2','ConvLayer_3' ]
  layers4style   = [ 'ConvLayer_0','ConvLayer_1','ConvLayer_2','ConvLayer_3','ConvLayer_4' ]
  weights4style  = [      1       ,      .8      ,     .6      ,    .4      ,      .2      ]

  # ===================================================

  # make a copy of the target image and push to GPU
  target = img4content.clone()
  target.requires_grad = True
  target = target.to(device)
  styleScaling = styleScaling

  # number of epochs to train
  numepochs = numepochs

  # optimizer for backprop
  optimizer = torch.optim.RMSprop([target],lr=.001)


  for epochi in range(numepochs):
    sys.stdout.write('\r' + "EPOCH =>> " + str(epochi+1) + '/' + str(numepochs))
    # extract the target feature maps
    targetFeatureMaps,targetFeatureNames = getFeatureMapActs(target,alexnet)


    # initialize the individual loss components
    styleLoss = 0
    contentLoss = 0

    # loop over layers
    for layeri in range(len(targetFeatureNames)):


      # compute the content loss
      if targetFeatureNames[layeri] in layers4content:
        contentLoss += torch.mean( (targetFeatureMaps[layeri]-contentFeatureMaps[layeri])**2 )


      # compute the style loss
      if targetFeatureNames[layeri] in layers4style:

        # Gram matrices
        Gtarget = gram_matrix(targetFeatureMaps[layeri])
        Gstyle  = gram_matrix(styleFeatureMaps[layeri])

        # compute their loss (de-weighted with increasing depth)
        styleLoss += torch.mean( (Gtarget-Gstyle)**2 ) * weights4style[layers4style.index(targetFeatureNames[layeri])]


    # combined loss
    combiloss = styleScaling*styleLoss + contentLoss

    # finally ready for backprop!
    optimizer.zero_grad()
    combiloss.backward()
    optimizer.step()





  # ============================================================

  # the "after" pic
  fig,ax = plt.subplots(1,3,figsize=(13,9))

  pic = img4content.cpu().squeeze().numpy().transpose((1,2,0))
  pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
  ax[0].imshow(pic)
  ax[0].set_title('Content picture',fontweight='bold')
  ax[0].set_xticks([])
  ax[0].set_yticks([])

  pic = torch.sigmoid(target).cpu().detach().squeeze().numpy().transpose((1,2,0))
  ax[1].imshow(pic)
  ax[1].set_title('Target picture',fontweight='bold')
  ax[1].set_xticks([])
  ax[1].set_yticks([])

  pic = img4style.cpu().squeeze().numpy().transpose((1,2,0))
  pic = (pic-np.min(pic)) / (np.max(pic)-np.min(pic))
  ax[2].imshow(pic,aspect=.6)
  ax[2].set_title('Style picture',fontweight='bold')
  ax[2].set_xticks([])
  ax[2].set_yticks([])

  plt.show()