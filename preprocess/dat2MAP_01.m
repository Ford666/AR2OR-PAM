% ------------------------------------------------------------------------
%%A-line�������2ns����384��512���㣬��ȷ�Χ1.536mm; 2000*2000�ռ�����㣬����2.5um
%����ԭʼ3D A-line���ݣ����������λ����+��׼FOV��0~1��׼���õ�MAPͼ��

clear;
close all;

addpath('../utils')

%--------------------------------------------------------------------------
P_y=2000; P_x=2000;  
shifty = -4;        % y ������λ��
shiftx = -12;  % x������λ��

maindir = "../../../dataset/Raw data";

for groupidx = [1]
    %% Read the .dat files
    ARgroupdir = sprintf("%s/AR_%d.dat", maindir, groupidx);
    ORgroupdir = sprintf("%s/OR_%d.dat", maindir, groupidx);

    fid1 = fopen(ARgroupdir,'r'); fid2 = fopen(ORgroupdir,'r');
    data1 = fread(fid1, '*uint16');
    data2 = fread(fid2, '*uint16');
    ARdata=reshape(data1, [ ], P_x*P_y);
    ORdata=reshape(data2, [ ], P_x*P_y);
    P_t=size(ARdata,1);
    clear data1 data2


    %Adjust Aline data
    ARdata = reshape(ARdata, [], P_y, P_x);
    ORdata = reshape(ORdata, [], P_y, P_x);

    %ż����Aline�������ҷ�תfliplr
    evenIdx = 2:2:P_y;
    ARdata(:,:,evenIdx) = fliplr(ARdata(:,:,evenIdx));
    ORdata(:,:,evenIdx) = fliplr(ORdata(:,:,evenIdx));
     
    %������ɨ��ʱ���ܴ�λ��������ƽ�ƣ�����ͼ�񶶶�
    oddIdx = 1:2:1415;
    shiftY = 4;
    ARdata(:, (max(shiftY+1,1):min(P_y, P_y+shiftY))-shiftY, 1:2:P_x) = ...
                                        ARdata(:, max(shiftY+1,1):min(P_y, P_y+shiftY), 1:2:P_x);
    ORdata(:, (max(shiftY+1,1):min(P_y, P_y+shiftY))-shiftY, oddIdx) = ...
                                        ORdata(:, max(shiftY+1,1):min(P_y, P_y+shiftY), oddIdx);
    
    %�����ң�OR-PAM���ɨ���ڶദ������λ���ͼ�񶶶�
    %��λ1��ż�������ƣ�����������
     displace_X = 1415; shiftY = 6; 
     ORdata(:, (max(shiftY+1,1):min(P_y, P_y+shiftY))-shiftY, displace_X) = ...
                                    ORdata(:, max(shiftY+1,1):min(P_y, P_y+shiftY), displace_X);
     displace_X1 = 1416; displace_X1_end = 1467;
     shiftY_even = -23;  shiftY_odd = 23; 
     ORdata(:, (max(shiftY_odd+1,1):min(P_y, P_y+shiftY_odd))-shiftY_odd, displace_X1+1:2:displace_X1_end) = ...
                                ORdata(:, max(shiftY_odd+1,1):min(P_y, P_y+shiftY_odd), displace_X1+1:2:displace_X1_end);
                                
     ORdata(:, (max(shiftY_even+1,1):min(P_y, P_y+shiftY_even))-shiftY_even, displace_X1:2:displace_X1_end-1) = ...
                                    ORdata(:, max(shiftY_even+1,1):min(P_y, P_y+shiftY_even), displace_X1:2:displace_X1_end-1);
     
     %��λ2��ż�������ƣ�����������
     displace_X2 = 1468; displace_X2_end = 1500;
     shiftY_even = -27;  shiftY_odd = 27; 
     ORdata(:, (max(shiftY_odd+1,1):min(P_y, P_y+shiftY_odd))-shiftY_odd, displace_X2+1:2:displace_X2_end-1) = ...
                                ORdata(:, max(shiftY_odd+1,1):min(P_y, P_y+shiftY_odd), displace_X2+1:2:displace_X2_end-1);
                                
     ORdata(:, (max(shiftY_even+1,1):min(P_y, P_y+shiftY_even))-shiftY_even, displace_X2:2:displace_X2_end) = ...
                                    ORdata(:, max(shiftY_even+1,1):min(P_y, P_y+shiftY_even), displace_X2:2:displace_X2_end);
                             
%     ��λ3��ż�������ƣ�����������
     displace_X3 = 1501; displace_X3_end = P_x;
     shiftY_even = -40;  shiftY_odd = 32; 
     ORdata(:, (max(shiftY_odd+1,1):min(P_y, P_y+shiftY_odd))-shiftY_odd, displace_X3:2:displace_X3_end-1) = ...
                                ORdata(:, max(shiftY_odd+1,1):min(P_y, P_y+shiftY_odd), displace_X3:2:displace_X3_end-1);
                                
     ORdata(:, (max(shiftY_even+1,1):min(P_y, P_y+shiftY_even))-shiftY_even, displace_X3+1:2:displace_X3_end) = ...
                                    ORdata(:, max(shiftY_even+1,1):min(P_y, P_y+shiftY_even), displace_X3+1:2:displace_X3_end);
       
                                
     %     ��λ4��ż��������
     displace_X4 = 1502; displace_X4_end = 1508;
     shiftY_even = 12;                 
     ORdata(:, (max(shiftY_even+1,1):min(P_y, P_y+shiftY_even))-shiftY_even, displace_X4:2:displace_X4_end) = ...
                                    ORdata(:, max(shiftY_even+1,1):min(P_y, P_y+shiftY_even), displace_X4:2:displace_X4_end);
      
      %     ��λ5������������
     displace_X5 = 1510; 
     shiftY = 7;                 
     ORdata(:, (max(shiftY+1,1):min(P_y, P_y+shiftY_even))-shiftY, displace_X5:P_x) = ...
                                    ORdata(:, max(shiftY+1,1):min(P_y, P_y+shiftY), displace_X5:P_x);
                                                     
    %Maximum amplitude projection(MAP)
    AR_MAP = squeeze(max(ARdata, [], 1)-min(ARdata, [], 1)); 
    OR_MAP =squeeze(max(ORdata, [], 1)-min(ORdata, [], 1)); 
    
    figure('Position', [200, 200, 1200, 500]);
    [MAPssim, ~] = ssim(OR_MAP, AR_MAP);
    suptitle(sprintf("ssim between unaligned ORMAP & ARMAP: %.4f", MAPssim));
    subplot(121), imagesc(OR_MAP), colormap(hot(256));
    axis on, title('ORPAM:MAP')
    subplot(122), imagesc(AR_MAP), colormap(hot(256));
    axis on, title('ARPAM:MAP')
    pause(1)
   
    %crop off edge artifacts manually
    AR_MAP =   AR_MAP(34:2000, :);
    OR_MAP =   OR_MAP(34:2000, :);
    
    [P_y, P_x] = size(AR_MAP);
    
%     %% align FOV 
%     shifty = 0; shiftx = 0;
%     shift_error = min(P_y, P_x);
%     shiftYmin = -20; shiftYmax = 20;
%     shiftXmin = -20; shiftXmax = 20;
%     
%     mar = 40;
%     iter = 1; Maxiter = 20;
%     shiftYarray = zeros(1, 20); 
%     shiftXarray = zeros(1, 20);
%     
%     %iterative FOV alignment: PCC map w.r.t shifty and shiftx
%     while (shift_error > 2 && iter <=Maxiter)
%         iter 
%         %aligned FOV
%         newAR_MAP = AR_MAP(max(1,1+shifty):min(P_y,P_y+shifty), max(1,1+shiftx):min(P_x,P_x+shiftx));
%         newOR_MAP = OR_MAP(max(1,1-shifty):min(P_y,P_y-shifty), max(1,1-shiftx):min(P_x,P_x-shiftx));
%         
%         newP_y = P_y - abs(shifty); newP_x = P_x - abs(shiftx); 
%         ARlarge = zeros(newP_y+2*mar+1, newP_x+2*mar+1);
%         ARlarge(mar+1:mar+newP_y, mar+1:mar+newP_x) = newAR_MAP;
%         shiftpcc = zeros(shiftYmax-shiftYmin+1, shiftXmax-shiftXmin+1,'gpuArray');
%         tic
%         for x = shiftXmin : shiftXmax
%             for y = shiftYmin : shiftYmax
%                 AR_MAP_shift = ARlarge(y+mar+1:y+mar+newP_y, x+mar+1:x+mar+newP_x);
%                 shiftpcc(y-shiftYmin+1,x-shiftXmin+1) = calculate_pcc(gpuArray(newOR_MAP), gpuArray(AR_MAP_shift));
%             end
%         end
%         shiftpcc = gather(shiftpcc);
%         toc
%         
%     %     figure, surf(shiftXmin : shiftXmax, shiftYmin : shiftYmax, shiftpcc);
%     %     xlabel("shiftx"), ylabel("shifty"), zlabel("shift pcc")
%         [Indy, Indx] = find(shiftpcc==max(shiftpcc(:)));
%         
%         %Record shiftx & shifty per iter 
%         shiftYarray(iter) = Indy + shiftYmin - 1; shiftXarray(iter) = Indx(1) + shiftXmin - 1; 
%         
%         %Update shift amount
%         shifty = shifty + Indy(1) + shiftYmin - 1;
%         shiftx = shiftx + Indx(1) + shiftXmin - 1;
%     
%         %update boundary condition for shifting
%     %     shiftYmin = ceil(shiftYmin/2); shiftYmax = ceil(shiftYmax/2);
%     %     shiftXmin = ceil(shiftXmin/2); shiftXmax = ceil(shiftXmax/2);
%     
%         %Update shift error for stop criterion
%         shift_error = max(abs(Indy(1) + shiftYmin - 1), abs(Indx(1) + shiftXmin - 1))
%         
%         iter = iter + 1;
%     end
    shifty, shiftx

    new_Py = P_y - abs(shifty); new_Px = P_x - abs(shiftx); 
     
    newARMAP = AR_MAP(max(1,1+shifty):min(P_y,P_y+shifty), ...
                            max(1,1+shiftx):min(P_x,P_x+shiftx));                             
    newORMAP = OR_MAP(max(1,1-shifty):min(P_y,P_y-shifty), ...
                            max(1,1-shiftx):min(P_x,P_x-shiftx));  
 
    newARMAP = single(newARMAP)/65535;
    newORMAP = single(newORMAP)/65535;
    
    ARMAPpath = sprintf("%s/ear_MAP_processed_V1/AR_%02d.npy", maindir, groupidx);
    ORMAPpath = sprintf("%s/ear_MAP_processed_V1/OR_%02d.npy", maindir, groupidx);
    writeNPY(newARMAP, ARMAPpath);
    writeNPY(newORMAP, ORMAPpath);
    
    %-----Justify FOV alignment-----
    figure('Position', [200, 200, 1200, 500]);
    [MAPssim, ~] = ssim(newORMAP, newARMAP);
    suptitle(sprintf("ssim between aligned ORMAP & ARMAP: %.4f", MAPssim));
    subplot(121), imagesc(newORMAP), colormap(hot(256));
    axis on, title('ORPAM:MAP')
    subplot(122), imagesc(newARMAP), colormap(hot(256));
    axis on, title('ARPAM:MAP')
    pause (1)
end







