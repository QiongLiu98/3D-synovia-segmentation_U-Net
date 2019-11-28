clc
clear all
close all

image_folder = './images_sort';
image_nii_name = '1030_image_patients';
label_nii_name = '1030_label_patients';
pred_nii_name = '1030_pred_patients';

root = dir(image_folder);
for i_patient = 3:length(root)
    patient_name = root(i_patient).name;
    patient_folder = fullfile(image_folder,patient_name);
    slice = dir(patient_folder);
    image_nii_name_patient = [image_nii_name,patient_name,'.nii'];
    label_nii_name_patient = [label_nii_name,patient_name,'.nii'];
    pred_nii_name_patient = [pred_nii_name,patient_name,'.nii'];
    for i_slice = 3:length(slice)
        slice_name = slice(i_slice).name;
        image_name = fullfile(patient_folder,slice_name);
        image = imread(image_name);
        if rem(i_slice,3)==0 %image
            image_nii(:,:,fix(i_slice/3)) = image;
        else if rem(i_slice,3)==1 %label
                label_nii(:,:,fix(i_slice/3)) = image;
        else if rem(i_slice,3)==2%pred
                pred_nii(:,:,fix(i_slice/3)) = image;
            end
                
        end
        end  
    end
    image_nii_file = make_nii(image_nii,[0.058,0.058,0.333]);
    save_nii(image_nii_file,image_nii_name_patient);
    label_nii_file = make_nii(label_nii,[0.058,0.058,0.333]);
    save_nii(label_nii_file,label_nii_name_patient);
    pred_nii_file = make_nii(pred_nii,[0.058,0.058,0.333]);
    save_nii(pred_nii_file,pred_nii_name_patient);
end


