addpath("../utils")
clear
close all
mymap = colormap(hot(256));
depths = [0, 700, 1300, 1700];  

for i = 4
    hair = readNPY(sprintf("../../../result/test/hair depth/depth-%d.npy", depths(i)));
%     imwrite(uint8(imresize(hair, [496, 496])*255), mymap, sprintf("../../../result/test/hair depth/depth-%d/AR.png", depths(i)));
    hairfilt = wiener2(hair, [5, 5]);
    figure, imshowpair(hair, hairfilt, 'montage'), colormap(hot)
%     writeNPY(hairfilt, sprintf("../../../result/test/hair depth/depth-%d_filt.npy", depths(i)));
end

hairfilt2 = wiener2(hairfilt, [5, 5]);
BW = imbinarize(hairfilt, 0.08); figure, imshow(BW)
BW = medfilt2(BW, [3,3]);
backgr = hair.*(1-BW);
backgrMean = mean(backgr(:))
backgrStd = std(backgr(:))


BW2 = medfilt2(BW, [3,3]); BW3 = medfilt2(BW2, [3,3]); BW4 = medfilt2(BW3, [3,3]);
hairfilt3 = hairfilt2.*BW;

hair2 = hair.*BW; 
hair3 = medfilt2(hair2, [3,3]);
figure, 
subplot(121), imagesc(hair), caxis([1.5*mean(hair(:)), 1])
subplot(122), imagesc(hair3), colormap(hot)

