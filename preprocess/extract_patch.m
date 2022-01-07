function [img1_patch,img2_patch,numH,numW] = extract_patch(img1, img2, rows, cols, patchH, patchW, overlap)
% Divide into image patches

numH =  idivide(int32(rows-overlap),int32(patchH-overlap),'floor');
numW = idivide(int32(cols-overlap),int32(patchW-overlap),'floor');
img1_patch = zeros(patchH, patchW, numH*numW);
img2_patch = zeros(patchH, patchW, numH*numW);

%大坑！当图像高小于宽(numH<numW)时，语句(i-1)*numH+j会发生索引覆盖！ 
for i = 1:numH
    for j = 1:numW
        img1_patch(:,:,(i-1)*max(numH,numW)+j) = img1((i-1)*(patchH-overlap)+1:i*patchH-(i-1)*overlap, ...
                                    (j-1)*(patchW-overlap)+1:j*patchW-(j-1)*overlap);
        img2_patch(:,:,(i-1)*max(numH,numW)+j) = img2((i-1)*(patchH-overlap)+1:i*patchH-(i-1)*overlap, ...
                                    (j-1)*(patchW-overlap)+1:j*patchW-(j-1)*overlap);
    end
end

end


