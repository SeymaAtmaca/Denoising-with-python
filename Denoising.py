from google.colab import drive
drive.mount('/content/drive/')

#______________________________________________________________________

import os
import cv2
import keras
import tensorflow
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, img_as_float
from skimage.metrics import peak_signal_noise_ratio

#______________________________________________________________________

pip install keras
#______________________________________________________________________

PATHv='drive/My Drive/Teknofest/Teknofest_Veriler/1.Oturum/Inme_Var/iskemi/pngli'
PATHy='drive/My Drive/Teknofest/Teknofest_Veriler/1.Oturum/Inme_Yok/Train'

IMG_SIZE=128

Labels=[]
Dataset=[]
i=0;

for path in os.listdir(PATHv):

  image=cv2.imread(PATHv+ '/' +path,1)

  if(image is not None):
    image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image= cv2.fastNlMeansDenoising(image,None,9,7,21)
    Labels.append("0")
    
    Dataset.append(image)

    print(i)

  i=i+1

for path in os.listdir(PATHy):

  image=cv2.imread(PATHy+'/'+path,1)

  if(image is not None):
    image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image= cv2.fastNlMeansDenoising(image,None,9,7,21)
    Labels.append("1")
    Dataset.append(image)

    print(i)

  i=i+1
#______________________________________________________________________
  
  
image_2=Dataset[5]
plt.imshow(image_2)
