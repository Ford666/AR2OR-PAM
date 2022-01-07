% Image patch extraction and alignment via correlation template matching

close all

%load AR-OR image pairs
ARdir = '../../../dataset/ear_AR_aug/';  ORdir = '../../../dataset/ear_OR_aug/';
ARpatchdirs = '../../../dataset/ARimgP_aug/new/';  ORpatchdirs = '../../../dataset/ORimgP_aug/new/';
img_path_list1 = dir( strcat(ARdir,'*.png')); img_path_list2 = dir( strcat(ORdir,'*.png'));                                             
img_num = length(img_path_list1); 
LocMap = {};  ShiftMap={};

if img_num > 0
    for k = 1:img_num
        % Read AR-OR mat files 
        img_name1 = img_path_list1(k).name; 
        if contains(img_name1, "AR_06")
            img_name2 = img_path_list2(k).name;

            img_dir1 = strcat(ARdir,img_name1); img_dir2 = strcat(ORdir,img_name2);
            ImgPatchdir1 = strrep(strcat(ARpatchdirs,img_name1),'.png',''); 
            ImgPatchdir2  = strrep(strcat(ORpatchdirs,img_name2),'.png','');

            OriARImg = im2double(imread(img_dir1)); 
            OriORImg = im2double(imread(img_dir2)); 

            %Zscore
            [P_y, P_x] = size(OriARImg);
            ARMean = mean(OriARImg(:));
            ARImg = (OriARImg-mean(OriARImg))./std(OriARImg,0); 
            ORImg = (OriORImg-mean(OriORImg))./std(OriORImg,0);

            % OR patch is used as a template to find the highest-correlation matching patch in the AR image
            % Gamma Transformation
            ARImgT = ARImg.^(0.6);  ARImgT = sign(ARImg).*abs(ARImgT);
            ORImgT = ORImg .^(0.6); ORImgT = sign(ORImg).*abs(ORImgT);
            ARImgT = gpuArray(ARImgT); ORImgT = gpuArray(ORImgT);

            % Extract patches
            patchH=390; patchW=390; 
            if min(P_y, P_x) >= 1980
                overlap=72;
            elseif min(P_y, P_x) >= 1940 && min(P_y, P_x) < 1980
                overlap = 80;
             elseif min(P_y, P_x) < 1940
                overlap = 98;
            end


            [ORPatch,~,~] = extract_patch1(0, OriORImg,P_y,P_x,patchH,patchW,overlap);   
            [ORTPatch,numH,numW] = extract_patch1(1, ORImgT,P_y,P_x,patchH,patchW,overlap);

            LocMap(1:numH, 1:numW, k) = cell(numH,numW);
            ShiftMap(1:numH, 1:numW, k) = cell(numH,numW);

            %To avoid out-of-bounds index of ARImg
            BaseImg = zeros(P_y+2*(patchH-1),P_x+2*(patchW-1));
            BaseImg(patchH:P_y+patchH-1,patchW:P_x+patchW-1) = OriARImg(1:P_y,1:P_x);
            fig=figure('position' ,[200, 300, 600, 500]); 
            imagesc(BaseImg); axis normal, colormap(hot);
            PatchCount=0;

            %Use only the first OR patch to determine if the alignment is ok
            ORhead = ORTPatch(:, :, 1);
            crr = xcorr2(ARImgT, ORhead );
            [ij,ji] = find(crr==max(crr(:)),1,'first');  %find(crr==max(crr(:))); 
            alignTF = ~(abs( ij-patchH)>40 || abs(ji-patchW)>40);
            s_y = ij; s_x = ji;
            ImgH = numH*patchH-(numH-1)*overlap;
            ImgW = numW*patchW-(numW-1)*overlap;

            for i=1:numH
                for j =1:numW
                    if alignTF
                        y0 = s_y+(i-1)*(patchH-overlap); y1 = s_y+i*patchH-(i-1)*overlap-1;
                        x0 = s_x+(j-1)*(patchW-overlap); x1 = s_x+j*patchW-(j-1)*overlap-1;
                        ARaliPatch = BaseImg(y0:y1, x0:x1);
                         LocMap{i,j,k}(1) = y0-patchH+1; LocMap{i,j,k}(2) = x0-patchW+1;    
                        hold on
                        plot([x0 x0 x1 x1 x0], [y0 y1 y1 y0 y0],'w--')
                    else
                        SectT = ORTPatch(:,:,(i-1)*max(numH,numW)+j);
                        crr = xcorr2(ARImgT, SectT);
                        [ij,ji] = find(crr==max(crr(:)),1,'first'); 
                        LocMap{i,j,k}(1) = ij-patchH+1; LocMap{i,j,k}(2) = ji-patchW+1;                     
                        ARaliPatch = BaseImg(ij:ij+patchH-1,ji:ji+patchW-1);
                        hold on
                        plot([ji ji ji+patchW-1 ji+patchW-1 ji],[ij ij+patchH-1 ij+patchH-1 ij ij],'w--')
                    end

                     PatchCount = PatchCount+1;

                    % Filter patches: large translation or low vessel density 
                    Sect = ORPatch(:,:,(i-1)*max(numH,numW)+j);
                    ARpatchMean = mean(ARaliPatch(:));
                    PairName = strrep(strrep(img_name1,'.png',''),"AR","AROR");

                    if  ARpatchMean<0.55*ARMean
                          continue
                    else
                        if  (abs(LocMap{i,j,k}(1)-((i-1)*(patchH-overlap)+1))>40 || ...
                              abs(LocMap{i,j,k}(2)-((j-1)*(patchW-overlap)+1))>40)
                          continue
                        else
    %                         % Futher registration between each AR-OR image patch (390x390).
    %                         SSIMs = {};
    %                         iter = 0; shifts = {}; 
    %                         for shiftX =-3:0.5:3
    %                             for shiftY=-3:0.5:3
    %                                 iter = iter+1; 
    %                                 shifts{iter}(1) = shiftX; shifts{iter}(2) = shiftY; 
    %                                 ARaliPatch_again = ApplyTrans(ARaliPatch,shiftX, shiftY);
    %                                 [SSIMs{iter}(1), ~] = ssim(ARaliPatch_again, Sect);
    %                             end
    %                         end
    % 
    %                         SSIMs=cell2mat(SSIMs);
    %                         best_iter = find(SSIMs==max(SSIMs)); 
    %                         ShiftMap{i,j,k}=shifts{best_iter};

                            % Futher registration between each AR-OR image patch (390x390).
                                PCCs = {};
                                iter = 0; shifts = {}; 
                                for shiftX =-3:0.5:3
                                    for shiftY=-3:0.5:3
                                        iter = iter+1; 
                                        shifts{iter}(1) = shiftX; shifts{iter}(2) = shiftY; 
                                        [ARaliPatch_again] = ApplyTrans(ARaliPatch,shiftX, shiftY);
                                        PCCs{iter}(1) = myPCC(ARaliPatch_again, Sect);
                                    end
                                end

                                PCCs=cell2mat(PCCs);
                                best_iter = find(PCCs==max(PCCs)); 
                                ShiftMap{i,j,k}=shifts{best_iter};
                                ARaliPatch_best = ApplyTrans(ARaliPatch,shifts{best_iter}(1), shifts{best_iter}(2));

                                % The registered images were cropped 3 pixels on each side to avoid registration artifacts
                                ARaliPatch_best = ARaliPatch_best(4:patchH-3, 4:patchW-3);
                                Sect_best = Sect(4:patchH-3, 4:patchW-3);

                                ARpatchdir = strcat(ImgPatchdir1,sprintf('_%02d.png',(i-1)*numW+j)); 
                                ORpatchdir = strcat(ImgPatchdir2,sprintf('_%02d.png',(i-1)*numW+j)); 

                                % Save patches 
                                 imwrite(uint8(255*ARaliPatch_best), ARpatchdir); 
                                 imwrite(uint8(255*Sect_best), ORpatchdir); 
                                 sprintf("%s, Count: %d, shifty: %2.1f, shiftx: %2.1f", ...
                                 PairName,PatchCount, shifts{best_iter}(1), shifts{best_iter}(2))
                        end
                    end
                end
            end
            hold off
            figpath = sprintf("../../../dataset/TempMatch/%s.png", PairName); 
            print(fig,  figpath, '-dpng', '-r300'); 
            close
        end
    end
end

function coeff=myPCC(i,j)
% pearson cross correlation coefficient between matrix i and j 
C = cov(i,j);
coeff = C(1,2) / sqrt(C(1,1) * C(2,2));
end

function [img_patch,numH,numW] = extract_patch1(GPUTF, img, rows, cols, patchH, patchW, overlap)
% Divide into image patches

numH =  idivide(int32(rows-overlap),int32(patchH-overlap),'floor');
numW = idivide(int32(cols-overlap),int32(patchW-overlap),'floor');
img_patch = zeros(patchH, patchW, numH*numW);

if GPUTF
    img_patch = gpuArray(img_patch);
end


%大坑！当图像高小于宽(numH<numW)时，语句(i-1)*numH+j会发生索引覆盖！ 
for i = 1:numH
    for j = 1:numW
        img_patch(:,:,(i-1)*max(numH,numW)+j) = img((i-1)*(patchH-overlap)+1:i*patchH-(i-1)*overlap, ...
                                    (j-1)*(patchW-overlap)+1:j*patchW-(j-1)*overlap);
    end
end

end


function [ali_patch] = ApplyTrans(img_patch,shiftX,shiftY)
% Only if shift error < tolerence, then apply the tranlation map to obtain 
% aligned pair of  image block

[H, W] = size(img_patch);
ali_patch = zeros(H, W);
shiftX_f = floor(shiftX); shiftY_f = floor(shiftY);
shiftX_c = ceil(shiftX); shiftY_c = ceil(shiftY);

for i=1:H
    for j=1:W
        %Pixel with a non-integral shift applied on its index is obtained by bilinear interpolation
        if 0<i+shiftX_f && i+shiftX_f<=H && 0< j+shiftY_f  && j+shiftY_f<=W && 0<i+shiftX_c && i+shiftX_c<=H && 0<j+shiftY_c &&j+shiftY_c<=W
            if shiftX_f~=shiftX_c && shiftY_f~=shiftY_c      
                ali_patch(i, j) = (shiftY_c-shiftY)*(shiftX_c-shiftX).*img_patch(i+shiftX_f, j+shiftY_f) + ...
                    (shiftY-shiftY_f)*(shiftX_c-shiftX).*img_patch(i+shiftX_f, j+shiftY_c) + ...
                    (shiftY_c-shiftY)*(shiftX-shiftX_f).*img_patch(i+shiftX_c, j+shiftY_f) + ...
                    (shiftY-shiftY_f)*(shiftX-shiftX_f).*img_patch(i+shiftX_c, j+shiftY_c);
            elseif shiftX_f == shiftX_c && shiftY_f~=shiftY_c 
                ali_patch(i, j) = (shiftY_c-shiftY).*img_patch(i+shiftX, j+shiftY_f)+(shiftY-shiftY_f).*img_patch(i+shiftX, j+shiftY_c);
            elseif shiftY_f == shiftY_c && shiftX_f~=shiftX_c
                 ali_patch(i, j) =  (shiftX_c-shiftX).*img_patch(i+shiftX_f, j+shiftY)+(shiftX-shiftX_f).*img_patch(i+shiftX_c, j+shiftY);
            else
                ali_patch(i, j) = img_patch(i+shiftX, j+shiftY);
            end
        else
            ali_patch(i, j) = img_patch(i ,j);
        end
    end
end
end
