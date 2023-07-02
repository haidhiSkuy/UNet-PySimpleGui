import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import backend as K
from keras.models import load_model


smooth = 100 

def dice_coef(y_true, y_pred):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  intersection = K.sum(y_true * y_pred) 
  coef = ((2 *intersection) / (K.sum(y_true) + K.sum(y_pred)))
  return coef 

def dice_loss(y_true, y_pred): 
  return 1 - dice_coef(y_true, y_pred)

def iou(y_true, y_pred): 
  intersection = K.sum(y_true * y_pred) 
  iou_score = intersection / ((K.sum(y_true + y_pred)) - intersection)
  return iou_score 

MODEL_PATH = "brain_mri.hdf5"
model = load_model(MODEL_PATH, 
                   custom_objects={'dice_loss':dice_loss, 
                                   'iou':iou, 
                                   'dice_coef':dice_coef})

im_width = 256
im_height = 256

def addChannel(image, b, g, r): 
  img = cv2.merge((image,image,image))
  B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]

  B[B==1] = b
  G[G==1] = g
  R[R==1] = r

  img2 = np.dstack((R,G,B)).astype(np.uint8)
  return img2

def predict(image): 
  sample = image / 255
  sample = sample[np.newaxis, :, :, :]
  pred = model.predict(sample)

  pred_img = np.squeeze(pred) > .5
  pred_img = pred_img.astype(int)
  pred_img_3C = addChannel(pred_img.astype(int), 0,0,255)

  segmentation_result = addChannel(pred_img.astype(int), 255,255,255)
  segmentation_result = cv2.imencode(".png", cv2.resize(segmentation_result,(200,200)))[1].tobytes()

  """
  pred_img will be reused for merging original image and segmentation result
  while segmentation_result will be showed in the GUI, 
  so we need to resize it to (200,200)
  """
  return pred_img_3C, segmentation_result


def merged(img1,img2,alpha=0.2): 
  """
  img1 is the original image, 
  and img2 is pred_img from previous function
  """
  out_img = np.zeros(img1.shape,dtype=img1.dtype)
  out_img[:,:,:] = (alpha * img1[:,:,:]) + ((1-alpha) * img2[:,:,:])
  out_img = cv2.resize(out_img,(200,200))
  return [cv2.imencode(".png", out_img)[1].tobytes(), out_img]

