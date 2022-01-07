
addpath('../utils')


%load AR-OR image pairs
ARpath = '../../../result/test/ear/AR_02.npy';  ORpath = '../../../result/test/ear/OR_02.npy';  
ARimg = readNPY(ARpath); ORimg = readNPY(ORpath);
Testdir = strrep(ARpath, 'AR_', ''); 
ARdir = strrep(Testdir, '.npy', '/x'); 
ORdir = strrep(Testdir, '.npy', '/y'); 


% Extract patches
[rows, cols] = size(ARimg);
patchH=384; patchW=384; 
if min(rows, cols) >= 1980
    overlap=64;
else
    overlap = 72;
end
extract_patch(ARimg, rows, cols, patchH, patchW, overlap, ARdir);
extract_patch(ORimg, rows, cols, patchH, patchW, overlap, ORdir);




function extract_patch(img, rows, cols, patchH, patchW, overlap, imgdir)
% Divide into image patches

numH =  idivide(int32(rows-overlap),int32(patchH-overlap),'floor');
numW = idivide(int32(cols-overlap),int32(patchW-overlap),'floor');

%大坑！当图像高小于宽(numH<numW)时，语句(i-1)*numH+j会发生索引覆盖！
idx = 1;
for i = 1:numH
    for j = 1:numW
        img_patch = img((i-1)*(patchH-overlap)+1:i*patchH-(i-1)*overlap, ...
                                    (j-1)*(patchW-overlap)+1:j*patchW-(j-1)*overlap);
        % Save patches 
         imgPpath = sprintf("%s/%02d.png", imgdir, idx);
         imwrite(uint8(255*img_patch), imgPpath); 
         idx = idx + 1;
    end
end



end