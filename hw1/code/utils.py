import cv2
import numpy as np

def load_image(path):
  # load an image into RGB format
  img = cv2.imread(path)
  img = img[:, :, ::-1]  # BGR -> RGB
  return img

def save_image(path, img):
  # save an RGB image into a file
  img = img.copy()[:,:,::-1]
  return cv2.imwrite(path, img)

def resize_image(img, new_size, interpolation):
  # resize an image into new_size (w * h) using specified interpolation
  # opencv has a weird rounding issue & this is a hacky fix
  # ref: https://github.com/opencv/opencv/issues/9096
  mapping_dict = {cv2.INTER_NEAREST: cv2.INTER_NEAREST_EXACT}
  if interpolation in mapping_dict:
    img = cv2.resize(img, new_size,
                     interpolation=mapping_dict[interpolation])
  else:
    img = cv2.resize(img, new_size,
                     interpolation=interpolation)
  return img
