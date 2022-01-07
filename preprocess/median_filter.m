function [ img ] = median_filter( image, m )
%----------------------------------------------
%��ֵ�˲�
%���룺
%image��ԭͼ
%m��ģ��Ĵ�С3*3��ģ�壬m=3

%�����
%img����ֵ�˲��������ͼ��
%----------------------------------------------
    n = m;
    [ height, width ] = size(image);
    x1 = double(image);
    x2 = x1;
    mea = mean(x1(:)); st = std(x1(:));
    for i = 1: height-n+1
        for j = 1:width-n+1
            if x1( i+(n-1)/2,  j+(n-1)/2 ) > mea+3*st
                mb = x1( i:(i+n-1),  j:(j+n-1) );
                mb = mb(:);
                mm = median(mb);
                x2( i+(n-1)/2,  j+(n-1)/2 ) = mm;
            end
        end
    end

    img = x2;


end