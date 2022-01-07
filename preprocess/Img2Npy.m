%Read tiff images and transfer them into npy files

close all

%load AR-OR image pairs
ARpatchdirs = '../../../Augdata/ARpatch/';  ORpatchdirs = '../../../Augdata/ORpatch/';
ARnpydirs =  '../../../Augdata/ARnpy/';  ORnpydirs = '../../../Augdata/ORnpy/';
img_path_list1 = dir( strcat(ARpatchdirs,'*.tiff')); 
img_path_list2 = dir( strcat(ORpatchdirs,'*.tiff'));                                             
img_num = length(img_path_list1); 

if img_num > 0
    for k=1:img_num
        %Read images
        img_name1 = img_path_list1(k).name; img_name2 = img_path_list2(k).name;
        img_dir1 = strcat(ARpatchdirs,img_name1); img_dir2 = strcat(ORpatchdirs,img_name2);
        ARImg = double(imread(img_dir1)); ORImg = double(imread(img_dir2));
        %0~1 Normalization
        ARImgNorm = (ARImg-min(ARImg(:)))./(max(ARImg(:))-min(ARImg(:)));
        ORImgNorm = (ORImg-min(ORImg(:)))./(max(ORImg(:))-min(ORImg(:)));
        %Save Npy 
        npy_dir1 = strcat(ARnpydirs, strrep(img_name1,'tiff','npy'));
        npy_dir2 = strcat(ORnpydirs, strrep(img_name2,'tiff','npy'));
        writeNPY(single(ARImgNorm), npy_dir1); 
        writeNPY(single(ORImgNorm), npy_dir2); 
    end
end