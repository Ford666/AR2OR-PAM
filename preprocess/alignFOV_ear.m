
clear


%load AR-OR image pairs
groups =  ["2nd","3rd","5th","7th","8th","9th","10th","ear_1","ear_2","ear_3","ear_4"];


fid = fopen('../../../datasplit/test/ear/alignFOV.txt', 'w+');

for j = 1:length(groups)
    % Read AR-OR NPY files 
    Testdir = sprintf('../../../datasplit/test/ear/%s', groups(j));

    ARdir =sprintf( '%s/AR.npy' , Testdir);
    ORdir =sprintf( '%s/OR.npy' , Testdir);

    ARimg = readNPY(ARdir);  ORimg = readNPY(ORdir);
    [P_y, P_x] = size(ARimg);
    %PCC map w.r.t shifty and shiftx
    shifty = 0; shiftx = 0;
    shift_error = min(P_y, P_x);
    shiftYmin = -20; shiftYmax = 20;
    shiftXmin = -20; shiftXmax = 20;

    mar = 40;
    iter = 1; Maxiter = 20;
    shiftYarray = zeros(1, 20); 
    shiftXarray = zeros(1, 20);

    %iterative FOV alignment workflow
    while (shift_error > 2 && iter <=Maxiter)
        iter 
        %aligned FOV
        newAR_img = ARimg(max(1,1+shifty):min(P_y,P_y+shifty), max(1,1+shiftx):min(P_x,P_x+shiftx));
        newOR_img = ORimg(max(1,1-shifty):min(P_y,P_y-shifty), max(1,1-shiftx):min(P_x,P_x-shiftx));

        newP_y = P_y - abs(shifty); newP_x = P_x - abs(shiftx); 
        ARlarge = zeros(newP_y+2*mar+1, newP_x+2*mar+1);
        ARlarge(mar+1:mar+newP_y, mar+1:mar+newP_x) = newAR_img;
        shiftpcc = zeros(shiftYmax-shiftYmin+1, shiftXmax-shiftXmin+1,'gpuArray');
        tic
        for x = shiftXmin : shiftXmax
            for y = shiftYmin : shiftYmax
                ARimg_shift = ARlarge(y+mar+1:y+mar+newP_y, x+mar+1:x+mar+newP_x);
                shiftpcc(y-shiftYmin+1,x-shiftXmin+1) = calculate_pcc(gpuArray(newOR_img), gpuArray(ARimg_shift));
            end
        end
        shiftpcc = gather(shiftpcc);
        toc

%             figure, surf(shiftXmin : shiftXmax, shiftYmin : shiftYmax, shiftpcc);
%             xlabel("shiftx"), ylabel("shifty"), zlabel("shift pcc")
        [Indy, Indx] = find(shiftpcc==max(shiftpcc(:)));

        %Record shiftx & shifty per iter 
        shiftYarray(iter) = Indy(1) + shiftYmin - 1; shiftXarray(iter) = Indx(1) + shiftXmin - 1; 

        %Update shift amount
        shifty = shifty + Indy(1) + shiftYmin - 1;
        shiftx = shiftx + Indx(1) + shiftXmin - 1;

        %update boundary condition for shifting
    %     shiftYmin = ceil(shiftYmin/2); shiftYmax = ceil(shiftYmax/2);
    %     shiftXmin = ceil(shiftXmin/2); shiftXmax = ceil(shiftXmax/2);

        %Update shift error for stop criterion
        shift_error = max(abs(Indy(1) + shiftYmin - 1), abs(Indx(1) + shiftXmin - 1))

        iter = iter + 1;
    end
    shifty, shiftx
     fprintf(fid,  '%s: shifty=%d, shiftx=%d\n' , groups(j), shifty, shiftx);
end

fclose(fid);
