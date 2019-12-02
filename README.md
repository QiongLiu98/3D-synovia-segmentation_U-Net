# 3D-synovia-segmentation_U-Net
3D ultrasound synovia segmentation
#3D synovia segmentation
Project is located at /home/ran/MyProjs/3D Seg knee/
Please contact liuqiong1998@hust.edu.cn or qiongliu98@gmail.com if you have any questions

Raw_data: original tiff files
Data_prep: Matlab file: tiff2bmp.m is to transform tiff file into bmp images and save the image path into a csv file, in order to read the training and testing data in train_test folder.

train_test: (Run on GPU, activate it by 'conda activate tensroflow-gpu')
1.Data_Augmentation2D.py 
This code generates training and testing .npy data by reading the former csv file
Line 570 load_data(‘the csv file.csv’)
Line 582 savepath = ‘./npy data path’
Line 465 define the number of images as the training dataset and testing 		dataset
Line 340 define augmentation data by flipping and rotating or not
2.Train_plaque_unet.py
This code reads .npy data and train the U-Net. Weights are saved in the log folder.
Line 15 .npy data folder
Line 19 define the log folder
Line 38 define the learning rate(10-4)
3.test_predict.py
This code tests the model from the log folder on images of other patients. Cropped images, masks, predicted masks, and a csv file are saved in the result note folder.
Line 19 data_csv = ‘csv file containing test images roots.csv’
Line 20 note = ‘result note folder name’
Line 21 modelpath = ‘weights folder.hdf5’

Result_report
1. Qiong_sort_results.m
This code saves whole images, masks, and predicted masks from result note folder into patient number-based folder.
csv_name = ‘csv file path.csv’
Ori_folder = ‘result image folder’
Save_folder_name = ‘sorted image save path’
2. Qiong_generate_nii.m
This code generates .nii file for visualization
Image_folder = ‘sorted image folder’
Line 33 (0.058,0.058,0.333) is defined
There are many other .m file (Matlab functions generate .nii file) in this folder, please don’t delete them.
