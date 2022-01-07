% Accurate alignment with reference to "deep learning microscopy"
% OR patch is used as a template to find the matching patch in AR image of highest correlation
% 鼠�?�血管PAM->基于灰度的模板匹配算�?

close all

%load AR-OR image pairs
ARdir = '../../Augdata/AR/';  ORdir = '../../Augdata/OR/';
ARpatchdirs = '../../Augdata/ARpatch/';  ORpatchdirs = '../../Augdata/ORpatch/';
img_path_list1 = dir( strcat(ARdir,'*.npy')); img_path_list2 = dir( strcat(ORdir,'*.npy'));                                             
img_num = length(img_path_list1); 
map = {};
%Create custom colormap
 mymap = colormap(hot(256));
Numcolor = size(mymap, 1);

if img_num > 0
    for k = 1:img_num
        % Read AR-OR mat files 
        img_name1 = img_path_list1(k).name; img_name2 = img_path_list2(k).name;
        if contains(img_name1, '02_018')
            img_dir1 = strcat(ARdir,img_name1); img_dir2 = strcat(ORdir,img_name2);
            ImgPatchdir1 = strrep(strcat(ARpatchdirs,img_name1),'.npy',''); 
            ImgPatchdir2  = strrep(strcat(ORpatchdirs,img_name2),'.npy','');

            OriARImg = readNPY(img_dir1); OriORImg = readNPY(img_dir2);
            ARMean = mean(OriARImg(:));

            %Zscore
            ARImg = (OriARImg-mean(OriARImg))./std(OriARImg,0); 
            ORImg = (OriORImg-mean(OriORImg))./std(OriORImg,0);
            [rows,cols] = size(ARImg); 

            % OR patch is used as a template to find the highest-correlation matching patch in the AR image
            % Gamma Transformation
            ARImgT = ARImg.^(0.6);  ARImgT = sign(ARImg).*abs(ARImgT);
            ORImgT = ORImg .^(0.6); ORImgT = sign(ORImg).*abs(ORImgT);

            % Extract patches
            patchH=384; patchW=384; overlap=64;
            [~,ORPatch,numH,numW] = extract_patch(ORImg,ORImgT,rows,cols,patchH,patchW,overlap);
            map(1:numH, 1:numW, k) = cell(numH,numW);

            %To avoid out-of-bounds index of ARImg
            BaseImg = zeros(rows+2*(patchH-1),cols+2*(patchW-1));
            BaseImg(patchH:rows+patchH-1,patchW:cols+patchW-1) = OriARImg(1:rows,1:cols);
            figure(k+1), h = imagesc(BaseImg); axis normal, colormap("hot");
            PatchCount=0;
            for i=1:numH
                for j =1:numW
                    Sect = ORPatch(:,:,(i-1)*max(numH,numW)+j);
                    crr = xcorr2(gpuArray(ARImgT),gpuArray(Sect));
                    crr = gather(crr);
                    [ij,ji] = find(crr==max(crr(:)),1,'first');  %find(crr==max(crr(:)));   
                    map{i,j,k}(1) = ij-patchH+1; map{i,j,k}(2) = ji-patchW+1;
                    ARaliPatch = BaseImg(ij:ij+patchH-1,ji:ji+patchW-1);
                    hold on
                    plot([ji ji ji+patchW-1 ji+patchW-1 ji],[ij ij+patchH-1 ij+patchH-1 ij ij],'w--')
                    PatchCount = PatchCount+1;

                    % Filter patches: large translation or low vessel density 
                    ARpatchMean = mean(ARaliPatch(:));
                    PairName = strrep(strrep(img_name1,'.npy',''),"AR","AROR");

%                     if abs(map{i,j,k}(1)-((i-1)*(patchH-overlap)+1))>10 || ...
%                               abs(map{i,j,k}(2)-((j-1)*(patchW-overlap)+1))>10 || ARpatchMean<0.5*ARMean
%                           continue
%                     else
                        ARpatchdir = strcat(ImgPatchdir1,sprintf('_%02d.png',(i-1)*numH+j)); 
                        ORpatchdir = strcat(ImgPatchdir2,sprintf('_%02d.png',(i-1)*numH+j)); 

                        % Inverse Gamma transformation & 0~1 normalization
                        SectR = Sect.^(1/0.6); SectR = sign(Sect).*abs(SectR);
                        ARaliPatch = (ARaliPatch-min(ARaliPatch(:)))./(max(ARaliPatch(:))-min(ARaliPatch(:)));                                          
                        SectR = (SectR-min(SectR(:)))./(max(SectR(:))-min(SectR(:)));
                        % Save patches 
                        %如果 A 是灰度图像或者属于数据类�? double �? single �? RGB 彩色图像�?
                        % �? imwrite 假设动�?�范围是 [0,1]，并在将其作�? 8 位�?�写入文件之前自动按 255 缩放数据
                        mappedARImg = uint8((ARaliPatch.* (Numcolor-1)).^0.95 );
%                         imwrite(mappedImg, mymap, sprintf("../../Augdata/AR_%d%d.png",i,j));
                        mappedORImg = uint8((SectR.* (Numcolor-1)).^0.95 );
                        imwrite(mappedARImg, mymap, ARpatchdir); 
                        imwrite(mappedORImg, mymap, ORpatchdir);                    
                        sprintf("%s, PatchCount: %d",PairName,PatchCount)
%                     end
                end
            end
        hold off
        figpath = sprintf("../../Augdata/%s.png", PairName); %/TempMatch
        SaveCenfig(h, figpath, cols, rows);
        end
    end
end



