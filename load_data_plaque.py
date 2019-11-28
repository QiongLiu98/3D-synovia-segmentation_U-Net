
import csv
import os
import glob
import sys
import time
import random
# import ConfigParser
import shutil
import pickle

import numpy as np
import nibabel as nib
import image_utils

import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

from keras.utils.np_utils import to_categorical
import struct


switch = {
  "width:": 0,
  "height:": 1,
  "numframes:": 2,
  "xvoxelsize:": 3,
  "yvoxelsize:": 4,
  "zvoxelsize:": 5
}


def readnfo(nfofile):
  nfo = np.zeros(6)
  with open(nfofile, 'r') as file_to_read:
    while True:
      lines = file_to_read.readline()
      if not lines:
        break
      tmpstr = lines.split()
      try:
        key = switch[tmpstr[0]]
        nfo[key] = float(tmpstr[1])
      except KeyError:
        pass
  return nfo

def get_image_list(csv_file, shuffle = False):
  image_list, label_list, shape_list = [], [], []

  with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:

        image_name = row['filepath'].replace('\t','').strip()
        # print(imagename)
        image_list.append(image_name)
        pos = image_name.find('.bmp')
        label_name = image_name[0:pos] + '_mask.bmp'
        label_list.append(label_name)
        width = int(row['width'].replace('\t', '').strip())
        height = int(row['height'].replace('\t', '').strip())
        image_shape = (height, width)
        shape_list.append(image_shape)

  if shuffle:
    combinelist = list(zip(image_list, label_list, shape_list))
    random.shuffle(combinelist)
    image_list, label_list, shape_list = zip(*combinelist)


  data_list = {}
  data_list['image_filenames'] = image_list
  data_list['label_filenames'] = label_list
  data_list['shapes'] = shape_list

  return data_list

def readrawdata(filename, imgSize):

  fimage = open(filename, "rb")
  imagedata = np.zeros(imgSize)

  for k in range(imgSize[2]):
    for j in range(imgSize[0]):
      for i in range(imgSize[1]):
        data = fimage.read(1)
        elem = struct.unpack('B', data)[0]
        imagedata[j][i][k] = float(elem)
    # plt.imsave('output_%d.png'%(k), pic[:, :, k], format='png', cmap=plt.cm.gray)
  fimage.close()

  return imagedata

def imresize(inputdata, new_size):

  img = Image.fromarray(inputdata.astype('uint8')).convert('L')
  resized_img = img.resize((new_size[0], new_size[1]), Image.BICUBIC)
  outputdata = np.array(resized_img)
  return outputdata

def getperdata(datalist, index, shape, interval):

  image_name = datalist['image_filenames'][index]
  label_name = datalist['label_filenames'][index]
  imgSize = datalist['shapes'][index]


  imagedatalist = list()
  labeldatalist = list()

  imagedata = readrawdata(image_name, imgSize)
  labeldata = readrawdata(label_name, imgSize)

  mean_image =imagedata.mean()

  for i in range(0, imgSize[-1], interval):
    if i > imgSize[-1] - interval:

      tmpimage = imagedata[:, :, imgSize[-1] - 1]
      tmpimage = imresize(tmpimage, shape)
      tmpimage = tmpimage.astype('float32')
      tmpimage -= mean_image
      tmpimage /= 255
      imagedatalist.append(np.expand_dims(tmpimage, axis=3))

      tmplabel = labeldata[:, :, imgSize[-1] - 1]
      tmplabel = imresize(tmplabel, shape)
      tmplabel = tmplabel.astype('float32')
      labeldatalist.append(np.expand_dims(tmplabel, axis=3))
      break

    tmpimage = imagedata[:, :, i]
    tmpimage = imresize(tmpimage, shape)
    tmpimage = tmpimage.astype('float32')
    tmpimage -= mean_image
    tmpimage /= 255
    imagedatalist.append(np.expand_dims(tmpimage, axis=3))

    tmplabel = labeldata[:, :, i]
    tmplabel = imresize(tmplabel, shape)
    tmplabel = tmplabel.astype('float32')
    labeldatalist.append(np.expand_dims(tmplabel, axis=3))


  return imagedatalist, labeldatalist


def generate_arrays_from_file(batch_size, data_csv, shape):

  while True:
    ncount = 0
    imagelist = []
    labellist = []
    Datafilelist = get_image_list(data_csv)
    for index in range(len(Datafilelist['image_filenames'])):

      image_name = Datafilelist['image_filenames'][index]
      label_name = Datafilelist['label_filenames'][index]
      imgSize = Datafilelist['shapes'][index]
      print('Get data from %s' % (image_name))

      image = cv.imread(image_name)
      label = cv.imread(label_name)

      re_image = cv.resize(image, shape, interpolation=cv.INTER_LINEAR)
      re_label = cv.resize(label, shape, interpolation=cv.INTER_NEAREST)

      re_image = re_image.astype('float32')
      imagelist.append(re_image)

      re_label = re_label.astype('float32')
      labellist.append(re_label)
      ncount += 1

      if ncount == batch_size:
        ncount = 0
        yield (np.array(imagelist), np.array(labellist))
        imagelist = []
        labellist = []



def load_data(data_csv, trainpart, shape):

  imagelist = []
  labellist = []
  Datafilelist = get_image_list(data_csv)


  num_files = len(Datafilelist['image_filenames'])
  trainlocs = [i for i in range(int(num_files))]
  testlocs = [i for i in range(int(num_files* 0.67), num_files)]

  for index in trainlocs:
    image_name = Datafilelist['image_filenames'][index]
    label_name = Datafilelist['label_filenames'][index]
    imgSize = Datafilelist['shapes'][index]
    print('Get data from %s' % (image_name))

    image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)
    label = cv.imread(label_name, flags=cv.IMREAD_GRAYSCALE)

    re_image = cv.resize(image, shape, interpolation=cv.INTER_LINEAR)
    re_label = cv.resize(label, shape, interpolation=cv.INTER_NEAREST)

    re_image = re_image.astype('float32')
    re_image /= 255
    re_image -= np.mean(re_image)
    imagelist.append(re_image)

    re_label = re_label.astype('float32')
    re_label /= 255
    labellist.append(re_label)

  train_images = np.array(imagelist)
  train_images = np.expand_dims(train_images,axis=-1)
  train_labels = np.array(labellist)
  train_labels = np.expand_dims(train_labels,axis=-1)

  imagelist = []
  labellist = []

  for index in testlocs:
    image_name = Datafilelist['image_filenames'][index]
    label_name = Datafilelist['label_filenames'][index]
    imgSize = Datafilelist['shapes'][index]
    print('Get data from %s' % (image_name))

    image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)
    label = cv.imread(label_name, flags=cv.IMREAD_GRAYSCALE)

    re_image = cv.resize(image, shape, interpolation=cv.INTER_LINEAR)
    re_label = cv.resize(label, shape, interpolation=cv.INTER_NEAREST)

    re_image = re_image.astype('float32')
    re_image /= 255
    re_image -= np.mean(re_image)
    imagelist.append(re_image)

    re_label = re_label.astype('float32')
    re_label /= 255
    labellist.append(re_label)

  test_images = np.array(imagelist)
  test_images = np.expand_dims(test_images,axis=-1)
  test_labels = np.array(labellist)
  test_labels = np.expand_dims(test_labels,axis=-1)

  np.save('./npydata/Datafilelist.npy', Datafilelist)

  return train_images, train_labels, test_images, test_labels

def load_npydata(directory, shuffle=True):

  imgs_train = np.load(directory + '/imgs_train.npy')
  imgs_train = imgs_train.astype('float32')
  imgs_mask_train = np.load(directory + '/imgs_mask_train.npy')
  imgs_mask_train = imgs_mask_train.astype('float32')

  image_list = list(imgs_train)
  label_list = list(imgs_mask_train)

  if shuffle:
    combinelist = list(zip(image_list, label_list))
    random.shuffle(combinelist)
    image_list, label_list = zip(*combinelist)

  imgs_train = np.array(image_list)
  imgs_mask_train = np.array(label_list)


  imgs_test = np.load(directory + '/imgs_test.npy')
  imgs_test = imgs_test.astype('float32')
  test_mask = np.load(directory + '/test_mask.npy')
  test_mask = test_mask.astype('float32')

  image_test_list = list(imgs_test)
  label_test_list = list(test_mask)

  if shuffle:
    combinelist = list(zip(image_test_list, label_test_list))
    random.shuffle(combinelist)
    image_test_list, label_test_list = zip(*combinelist)

  imgs_test = np.array(image_test_list)
  test_mask = np.array(label_test_list)

  return imgs_train, imgs_mask_train, imgs_test, test_mask


def prior_probability(directory):
  imgs_mask_train = np.load(directory + '/imgs_mask_train.npy')
  imgs_mask_train = imgs_mask_train.astype('float32')
  imgsize = imgs_mask_train[0].shape

  probmap = np.zeros((imgsize[0], imgsize[1], 3), dtype='float32')
  for i in range(imgs_mask_train.shape[0]):
    tmpmask = imgs_mask_train[i,:]
    probmap[:, :, 0] = probmap[:, :, 0] + tmpmask[:, :, 0]
    probmap[:, :, 1] = probmap[:, :, 1] + tmpmask[:, :, 1]
    probmap[:, :, 2] = probmap[:, :, 2] + tmpmask[:, :, 2]


  for i in range(3):
    maxval = np.max(probmap[:, :, i])
    minval = np.min(probmap[:, :, i])
    probmap[:, :, i] = (probmap[:, :, i] - minval)/(maxval-minval)

  # np.save(directory+'/probmap.npy', probmap)

  return probmap


def gradient_image(image0):
  image = image0[:,:,0]
  gradx = cv.Sobel(image, cv.CV_32F, 1, 0)
  grady = cv.Sobel(image, cv.CV_32F, 0, 1)

  outimage = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)

   # cv.imshow("gradx", gradx)
  # cv.imshow("grady", grady)
  #
  # cv.imshow("image", image)
  # cv.imshow("Result", outimage)
  #
  # cv.waitKey(0)
  # cv.destroyAllWindows()

  return outimage

def save_gradient_data(npydata):
  imgs_train, imgs_mask_train, imgs_test, test_mask = load_npydata(npydata)
  gradientimages = list()
  gradienttests = list()
  for i in range(imgs_train.shape[0]):
    image = imgs_train[i, :, :, :]
    gradientimages.append(gradient_image(image))

  for i in range(imgs_test.shape[0]):
    image = imgs_test[i, :, :, :]
    gradienttests.append(gradient_image(image))

  gradientimages = np.array(gradientimages)
  gradientimages = np.expand_dims(gradientimages,axis=-1)

  gradienttests = np.array(gradienttests)
  gradienttests = np.expand_dims(gradienttests,axis=-1)

  np.save(npydata + '/imgs_train_gradient.npy', gradientimages)
  np.save(npydata + '/imgs_test_gradient.npy', gradienttests)

def load_gradient_data(directory):

  gradientimages = np.load(directory + '/imgs_train_gradient.npy')
  gradientimages = gradientimages.astype('float32')
  gradienttests = np.load(directory + '/imgs_test_gradient.npy')
  gradienttests = gradienttests.astype('float32')

  return gradientimages, gradienttests

def compact_prior2(directory):
  imgs_mask_train = np.load(directory + '/imgs_mask_train.npy')
  imgs_mask_train = imgs_mask_train.astype('uint8')
  image_shape = imgs_mask_train.shape

  priordata = np.ndarray(shape=image_shape, dtype='float32')
  for ind_images in range(image_shape[0]):
    tmpmask1 = 255 * imgs_mask_train[ind_images,:,:,0]
    tmpmask1 = 255 - cv.Canny(tmpmask1,100, 200)
    distmap1 = cv.distanceTransform(tmpmask1, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    priordata[ind_images, :, :, 0] = distmap1

  return priordata


def compact_prior_center(directory):
  imgs_mask_train = np.load(directory + '/imgs_mask_train.npy')
  imgs_mask_train = imgs_mask_train.astype('uint8')
  image_shape = imgs_mask_train.shape

  priordata = np.ndarray(shape=image_shape, dtype='float32')
  tmpmask1 = np.ones(shape=(image_shape[1], image_shape[2]), dtype='uint8')
  for ind_images in range(image_shape[0]):
    locs = np.where(imgs_mask_train[ind_images,:,:,0] > 0)
    tmpmask1[int(np.mean(locs[0])), int(np.mean(locs[1]))] = 0
    distmap1 = cv.distanceTransform(tmpmask1, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    priordata[ind_images, :, :, 0] = distmap1

  return priordata

if __name__=='__main__':

  imgs_train, imgs_mask_train, imgs_test, test_mask = load_data('./Datalist-2D_shuffle_small.csv', 1, (320, 224))


  savepath = './npydata_plaque_large'
  if not os.path.exists(savepath):
    os.mkdir(savepath)
    print("Directory ", savepath, " Created ")
  else:
    print("Directory ", savepath, " already exists")

  np.save(savepath + '/imgs_train.npy', imgs_train)
  np.save(savepath + '/imgs_mask_train.npy', imgs_mask_train)
  np.save(savepath + '/imgs_test.npy', imgs_test)
  np.save(savepath + '/test_mask.npy', test_mask)

  # probmap = prior_probability('./npydata')

  save_gradient_data(savepath)







