clc
clear all
close all

csv_path = './datalist_25_patients.csv';
fib = fopen(csv_path,'a');
fprintf(fib,'%s,%s\r\n','filepath','labelpath');
data_raw_path = '../Raw_data/25/';
data_save_path = '/home/ran/MyProjs/3D Seg knee/Data_prep/US_data_25_patients/';
folder = dir(data_raw_path);
mask_seg = '_Rt-Supra-Pat-Mid_1-Segmentation.tiff';
image_seg = '_Rt-Supra-Pat-Mid_1.tiff';
if exist(data_save_path,'dir')==0
    mkdir(data_save_path);
end
for i_patient = 3:length(folder)
    
    patient_name = folder(i_patient).name;
    %patient_name = 'KHV18';
    patient_folder = fullfile(data_raw_path,patient_name);
    patient = dir(patient_folder);
    patient_number = patient_name(4:end);
    
    mask_name = [patient_number,mask_seg];
    image_name = [patient_number,image_seg];
    
    for i_pic=1:110   
        mask = imread(fullfile(data_raw_path,patient_name,mask_name),i_pic);   
        mask = uint8(mask);
        if max(max(mask))==255 % to identify whether this slice has the mask or not
            mask_save_path = [data_save_path,patient_number,'_',num2str(i_pic,'%04d'),'_mask.bmp'];
            imwrite(mask,mask_save_path);
            
            image = imread(fullfile(data_raw_path,patient_name,image_name),i_pic);   
            image = uint8(image);
            image_save_path = [data_save_path,patient_number,'_',num2str(i_pic,'%04d'),'_image.bmp'];
            imwrite(image,image_save_path);
            fprintf(fib,'%s,%s\r\n',image_save_path,mask_save_path);
        end
    end

end
