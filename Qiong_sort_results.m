%this code generate whole area mask and pred from test results

clc
clear all
close all

csv_name = './result.csv';
ori_folder = './images';% crop images labels preds
save_folder_name = './images_sort';

csv_file = readtable(csv_name);
maxx = csv_file.maxx;
maxy = csv_file.maxy;
minx = csv_file.minx;
miny = csv_file.miny;
manual_area = csv_file.plaque_areas_label;
pred_area = csv_file.plaque_areas_pred;
image_file_name = csv_file.image_filenames;



for i = 1 : length(image_file_name)
    image_name = cell2mat(image_file_name(i));
    locs = strfind(image_name,'_');
    patient_name = image_name(locs(3)-2:locs(3)-1);
    save_folder = fullfile(save_folder_name,patient_name); % get patient name
    
    slice_name = image_name(locs(3)-2:locs(4)+5);
    label_name = [slice_name,'_label.bmp'];
    pred_name = [slice_name,'_pred.bmp'];
    create_folder = exist(save_folder,'dir');   %creat a new folder to sort out each patients
    if create_folder==0            
        mkdir(save_folder);   
    end
    
    image = imread(image_name);
    roi_label = imread(fullfile(ori_folder,label_name));
    roi_pred = imread(fullfile(ori_folder,pred_name));
    fr = (miny(i):maxy(i)-1);
    fc = (minx(i):maxx(i)-1);
    label = image*0;
    label(fr,fc) = im2uint8(roi_label);
    pred = image*0;
    pred(fr,fc) = im2uint8(roi_pred);
    
    imwrite(image,fullfile(save_folder,[slice_name,'.bmp']));
    imwrite(label,fullfile(save_folder,label_name));
    imwrite(pred,fullfile(save_folder,pred_name));
    
    
end

