import csv
import  random, os
import cv2 as cv
import pandas as pd
from model_2DUNet import my2DUNet
from keras.preprocessing.image import array_to_img
from keras.optimizers import *
import scipy.io as sio
from LossMetrics import *
from Data_Augmentation2D import normalize, get_roirect_fix, get_image_list

import matplotlib.pyplot as plt
import scipy.misc

def takeSecond(elem):
    return elem[1]

if __name__ == '__main__':

    data_csv = '../Datalist_6patients_1025.csv'
    data_root = '../local'
    note = 'results_timepoints_6patients_UNet1025'
    modelpath = './log1010_aug_SPARCM/plaque_UNet.hdf5'
    lognoteDir = './' + note
    result_csv = lognoteDir + '/result.csv'
    savepath = lognoteDir + '/images'
    save_root_dir = './' + note
    shape = (96, 144)

    lognoteDir = './' + note
    if os.path.exists(lognoteDir) is True:
        print('Exist %s' % lognoteDir)
    else:
        os.mkdir(lognoteDir)

    if os.path.exists(savepath) is True:
        print('Exist %s' % savepath)
    else:
        os.mkdir(savepath)

    plaque_areas_label = list()
    error_plaque_areas = list()
    plaque_areas_pred = list()


    # mynet = VNet(shape[0], shape[1], 1)
    mynet = my2DUNet(None, None, 1)
    # mynet = FCN(img_rows, img_cols)
    #
    # model = mynet.model(classes=1, kernel_size=(5, 5))
    model = mynet.model(classes=1, stages = 4, base_filters = 64, activation_name='sigmoid', deconvolution = False)
    # model = mynet.FCN_Vgg16_8s(weight_decay=0.001, classes=1, activate='sigmoid')

    model.load_weights(modelpath)


    model.compile(optimizer=adam(lr=1e-4), loss=DSCLoss, metrics=[DSC2D, 'accuracy'])

    #get imagelist from csv file
    image_list, minx_list, miny_list, maxx_list, maxy_list, res_list = [], [], [], [], [], []

    with open(data_csv, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_name = row['filepath'].replace('\t', '').strip()
            print(image_name)
            image_list.append(image_name)
            pos = image_name.find('.bmp')

    image_name = os.path.join(data_root, image_list[1])
    print('Get data from %s' % (image_name))
    image = cv.imread(image_name, flags=cv.IMREAD_GRAYSCALE)
    rows, cols = image.shape
    image_roi = image[340:413, 339:493]
    pred = model.predict(image_roi, verbose=1)
    print('pred_shape is:'pred.shape)
    pred_resize = cv.resize(pred[0, :, :, 0], (image_roi.shape[1], image_roi.shape[0]), 0)
    ret, pred_binary = cv.threshold(pred_resize, 0.5, 1, cv.THRESH_BINARY)

    tmpimname = image_list[1]
    loc = tmpimname.rfind('/')
    saveimpath = os.path.join(savepath, '%s.bmp' % tmpimname[loc + 1: -4])
    savepredpath = os.path.join(savepath, '%s_pred.bmp' % tmpimname[loc + 1: -4])
    scipy.misc.imsave(saveimpath, image_roi, 'bmp')
    scipy.misc.imsave(savepredpath, pred_binary, 'bmp')