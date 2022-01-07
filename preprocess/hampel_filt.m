function [filtTemp, abnorInx] = hampel_filt(realSig)

filtTemp = realSig;
filtSize = 10;
% figure,
while(filtSize <= 640)                  
    filtTemp = hampel(filtTemp, filtSize);
    filtSize = 2*filtSize;
%     plot(realSig), hold on, plot(filtTemp,'r-', 'LineWidth',1), hold off
end
abnorInx = find(abs(realSig-filtTemp) > 20);
end    