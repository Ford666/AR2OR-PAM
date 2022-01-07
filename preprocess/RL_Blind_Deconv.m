function [obj, AER] = RL_Blind_Deconv(I, psfi, maxIters)
    
    %initial estimation for object and psf
    obj = I; 
    psf = psfi; 
    
    AER = gpuArray(zeros(maxIters, 1)); %absolute error ratio  
    
    for i = 1: maxIters    
        div = I  ./ (imfilter(obj, psf)+eps);
        div(isnan(div)) = 0;
        
        % update estimated psf
        new_psf = imfilter(div, obj) .* psf;
        new_div = I  ./ (imfilter(obj, new_psf)+eps);
        new_div(isnan(new_div)) = 0;
        
        % update estimated obj
        new_obj = imfilter(new_div, new_psf) .* obj;
        new_obj(new_obj<0) = 0; % clip obj estimation
        new_obj(new_obj>1) = 1;
        
        obj = new_obj; psf = new_psf;
        
        %evaluate consistency
        residual = sum(I - imfilter(obj, psf), 'all');
        AER(i) = real(residual(:).^2/sum(I(:).^2));
    end

end
    