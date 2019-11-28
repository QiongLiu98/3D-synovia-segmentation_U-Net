clc
clear all
close all

path = '/home/ran/MyProjs/3D Seg knee/Data_prep/';
image_path = './25_roi_test';
csv_file = './roi_test.csv';
fid = fopen(csv_file, 'a');
fprintf(fid,'%s, %s, %s, %s, %s\r\n','filepath', 'minx','maxx','miny','maxy');
image_folder = dir(image_path);

for i = 3:length(image_folder)
    image_name = image_folder(i).name;
    image = imread(image_name);
    imshow(image)
    title(image_name);
    
    exit = input('is synovial exits?(input 0 if not)');
    if exit==0
        continue;
    end
    
    mouse=imrect;
    pos=getPosition(mouse);% x1 y1 w h
    minx = pos(1); miny = pos(2); maxx = pos(1)+pos(3); maxy = pos(2)+pos(4);
    filepath = fullfile(path,image_path,image_name);
    fprintf(fid,'%s, %d, %d, %d, %d\r\n',filepath, minx,maxx,miny,maxy);

end
