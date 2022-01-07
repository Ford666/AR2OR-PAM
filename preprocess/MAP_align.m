%根据PAM未发生电机错位而生成的MAP数据，包括2, 3, 5, 7,8,9,10,ear_1, ear_2, ear_3, ear_4
%做FOV配准，标准化得到0~1MAP图像
clear;
close all;

addpath('../utils')

%--------------------------------------------------------------------------
P_y=2000; P_x=2000;  

shiftys = [-1,164, 8, 1, 1, 3, 1, -2, -2, 2, 0];        % FOV配准y 方向移位量
shiftxs = [-2, 97, -15, -2, -4, -5, -8, 8, -10, -16, -40];  % FOV配准x方向移位量

maindir = "../../../dataset/Raw data";
varName = {2, 3, 5, 7, 8, 9, 10, 'ear_1', 'ear_2', 'ear_3', 'ear_4'};
for i = 1:length(varName)
    %% Read the .mat files
    groupidx = varName{i};
    if isnumeric(groupidx)
         ARgroupdir = sprintf("%s/AR_%02d.mat", maindir, groupidx);
         ORgroupdir = sprintf("%s/OR_%02d.mat", maindir, groupidx);
    elseif  isa(groupidx, 'char')
         ARgroupdir = sprintf("%s/AR_%s.mat", maindir, groupidx);
         ORgroupdir = sprintf("%s/OR_%s.mat", maindir, groupidx);
    end
    AR_MAP_name = load(ARgroupdir); OR_MAP_name = load(ORgroupdir);
    AR_MAP = AR_MAP_name.AR; OR_MAP = OR_MAP_name.OR;
    
      
    figure('Position', [200, 200, 1200, 500]);
    [MAPssim, ~] = ssim(OR_MAP, AR_MAP);
    suptitle(sprintf("ssim between unaligned ORMAP & ARMAP: %.4f", MAPssim));
    subplot(121), imagesc(OR_MAP), colormap(hot(256));
    axis on, title('ORPAM:MAP')
    subplot(122), imagesc(AR_MAP), colormap(hot(256));
    axis on, title('ARPAM:MAP')
    pause(1)
    
      %normalization
    AR_MAP = single(AR_MAP)/65535;
    OR_MAP = single(OR_MAP)/65535;
    
    [P_y, P_x] = size(AR_MAP);
    
%    %% align FOV 
%     mar = 40;
%     
%     shifty = 0; shiftx = 0;
%     shift_error = min(P_y, P_x);
%     shiftYmin = -20; shiftYmax = 20;
%     shiftXmin = -20; shiftXmax = 20;
%     
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
%     shifty, shiftx

    shifty = shiftys(i); shiftx = shiftxs(i);
    
    new_Py = P_y - abs(shifty), new_Px = P_x - abs(shiftx) 
     
    newARMAP = AR_MAP(max(1,1+shifty):min(P_y,P_y+shifty), ...
                            max(1,1+shiftx):min(P_x,P_x+shiftx));                             
    newORMAP = OR_MAP(max(1,1-shifty):min(P_y,P_y-shifty), ...
                            max(1,1-shiftx):min(P_x,P_x-shiftx)); 
                        
    %-----Justify FOV alignment-----
    figure('Position', [200, 200, 1200, 500]);
    [MAPssim, ~] = ssim(newORMAP, newARMAP);
    suptitle(sprintf("ssim between aligned ORMAP & ARMAP: %.4f", MAPssim));
    subplot(121), imagesc(newORMAP), colormap(hot(256));
    axis on, title('ORPAM:MAP')
    subplot(122), imagesc(newARMAP), colormap(hot(256));
    axis on, title('ARPAM:MAP')
    pause (1)
    
    if isnumeric(groupidx)
          ARMAPpath = sprintf("%s/ear_MAP_processed_V1/AR_%02d.npy", maindir, groupidx);
          ORMAPpath = sprintf("%s/ear_MAP_processed_V1/OR_%02d.npy", maindir, groupidx);
    elseif  isa(groupidx, 'char')
         ARMAPpath = sprintf("%s/ear_MAP_processed_V1/AR_%s.npy", maindir, groupidx);
         ORMAPpath = sprintf("%s/ear_MAP_processed_V1/OR_%s.npy", maindir, groupidx);
    end
    writeNPY(newARMAP, ARMAPpath);
    writeNPY(newORMAP, ORMAPpath);
    
end

