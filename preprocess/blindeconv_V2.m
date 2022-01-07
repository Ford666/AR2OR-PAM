% AR-PAM (blurred), OR-PAM (ground truth), blind deconv to estimate initial PSF and object.
% clear
close all

addpath("../../utils")
mymap = colormap(hot(256));
Numcolor = size(mymap, 1);

% load aligned AR-OR image pairs
% eardir = 'I:/Research Projects/AROR_PAM/dataset/Raw data/brain_out-of-focus_V2/test/';  
eardir = 'I:/Research Projects/AROR_PAM/result/test/hair depth/';  
img_path_list = dir( strcat(eardir,'*.npy'));                                   
img_num = length(img_path_list); 

if img_num > 0
    for k = 1:img_num
        img_name = img_path_list(k).name; 
%         if contains(img_name,'AR')
            ARname = img_name;

            AR_path = strcat(eardir, ARname); 
            AR = readNPY(AR_path);   
            [ysize, xsize] = size(AR);
            
            %generate initial PSF first using MLA blind deconv
           [J, psfi] = deconvblind(AR, ones(ysize, xsize) , 30);
           OTF = psf2otf(psfi);
           
            maxIters = 100;
           Afun   = @(x) opAx(AR, OTF);  % function handles
           Atfun  = @(x) opAtx(AR, OTF);
           
           %% Run RL deconv
%            [deconv_RL, r, t] = RL(Afun, Atfun, AR, 0, 1, maxIters, false);
%            figure(); imagesc(deconv_RL); colormap(hot);
%            
%            %save super-resolved image
%             SRimg_path = strcat(eardir, strrep(strrep(ARname, 'AR', 'SRdeconv_RL'), 'npy', 'png')); 
%             imwrite(uint8(255*deconv_RL), mymap, SRimg_path);
%             SRnpy_path = strcat(eardir, strrep(ARname, 'AR', 'SRdeconv_RL')); 
%             writeNPY(deconv_RL, SRnpy_path); 
            
            %% Run ADMM deconv
           rho = 0.21;
           [deconv_ADMM, r, t] = ADMM(rho, AR, maxIters, OTF);
           figure(); imagesc(deconv_ADMM); colormap(hot); caxis([0.02 1])
           
           %save super-resolved image
            SRimg_path = strcat(eardir, strrep(strrep(ARname, 'AR', 'SRdeconv_ADMM'), 'npy', 'png')); 
            imwrite(uint8(255*deconv_ADMM), mymap, SRimg_path);
            SRnpy_path = strcat(eardir, strrep(ARname, 'AR', 'SRdeconv_ADMM')); 
            writeNPY(deconv_ADMM, SRnpy_path); 
           
%         end
    end
end


% run the Richardson-Lucy updates

function [x,r,t] = RL(Afun, Atfun, b, lb, ub,maxIters, bQuiet)

    if nargout>1
        r = nan(maxIters,1);
    end
    
    % initial guess
    AtONE = Atfun(ones(size(b)));           
    x = b;
    for k=1:maxIters
        tic
        % multiplicative update
        div = b./Afun(x);
        div(isnan(div))=0;
        x = Atfun(div).*x./AtONE ;
                   
        % Clip output
        if ~isempty(lb)            
            x(x<lb) = lb; 
        end
        if ~isempty(ub)
            x(x>ub) = ub;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if (nargout>1) && ~bQuiet
        
            % compute residual
            bb          = (b-Afun(x));                       
            residual    = real(sum(bb(:).^2));
            r(k)        = residual;
            t(k) = toc; 
            % plot current residual
            if ~bQuiet
                disp(['  RL iter ' num2str(k) ' | ' num2str(maxIters) ', residual: ' num2str(residual)]);
            end
        end
    
    end
   
end

%Implements the function A'*x = C'*P'*x
% 
%Input:  fs is a 3D focal stack
%Output: vol is a 3D volume

function vol = opAtx(fs,OTF)
    vol = ifftn(fftn(fs) .* conj(OTF));    
end



%Implements the function A*vol = P*C*vol 
%
%Input:  vol is a 3D volume.
%Output: fs is a 3D focal stack.

function fs = opAx(vol, OTF)  
    fs = ifftn(fftn(vol) .* OTF);
end

           
% Run ADMM for 3D deconvolution

function [x,r,t] = ADMM(rho, b, maxIters, OTF)
    imageSize = size(b);
    
    x = b;
    p2o = @(x) psf2otf(x, imageSize);

    
    % precompute OTFs 
    cFT     = OTF;
    cTFT    = conj(OTF);

    
    % initialize intermediate variables
    z = zeros([imageSize,2]);
    u = zeros([imageSize,2]);
    x_denominator = (cTFT.*cFT) + 1;
    
    
    % start ADMM iterations
    for iter=1:maxIters    
        tic 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % x update
        v = z-u;
        x_numerator = (cTFT.*p2o(v(:,:,1))) + p2o(v(:,:,2));
        x = otf2psf(x_numerator./x_denominator);

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % z1 update
        Ax = otf2psf(cFT.*p2o(x));
        v(:,:,1) = Ax + u(:,:,1);

        t1 = -(1 - rho.*v(:,:,1))./(2*rho);
        t21 = (-t1).^2; t22 = b./rho;
        t2 = sqrt(t21+t22);
        z(:,:,1) = t1 + t2;

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % z2 update
        v(:,:,2) = x + u(:,:,2);
        z(:,:,2) = max(0,v(:,:,2));

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % u update
        Kx(:,:,1) = Ax;
        Kx(:,:,2) = x;
        u = u + Kx - z;
        
        % compute residual     
        r(iter) = real(sum( (b(:)-Ax(:)).^2));
        t(iter) = toc;
        
        
        % display status
        disp(['  ADMM iter ' num2str(iter) ' | ' num2str(maxIters) ', residual: ' num2str(r(iter))]);
    end
end            
            
            