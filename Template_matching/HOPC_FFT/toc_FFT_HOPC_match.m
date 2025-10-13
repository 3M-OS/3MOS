function [CP_Ref,CP_Sen] = toc_FFT_HOPC_match(im_Ref,im_Sen,cellsize,nCells,orbin)

% detect the tie points using HOPC  (histogram of oriented phase
% congruency) by the template matching strategy.As the matlab cannot handle
% the georeference information of images, we use some control points ('CP_Check_file')
% to determine the search region
% 
% If you use this implementation please cite:
% Yuanxin Ye et al.: "HOPC: A NOVEL SIMILARITY METRIC BASED ON GEOMETRIC STRUCTURAL 
% PROPERTIES FOR MULTI-MODEL REMOTE SENSING IMAGE MATCHING"
% ISPRS XXIII congress (2016)
% 
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
% ----------------------------------------------------------------------
% 
% 
% Contact: yeyuanxin110@163.com
% input Parameters: 
%                 im_Ref: the reference image
%                 im_Sen: the sensed image
%                 CP_Check_file: the file path of check points. they are used to determine the searchRegion, and judge if the matching points are correct
%                 disthre: the threshold of the errors of matching point pair, if the error of the point pair is less than dis, they are regrad as the correct match                 
%                 tranFlag:  the geometric transformation between two images. 0:affine, 1: projective, 2: Quadratic polynomial,3: cubic polynomial,the default is 3
%                 templateSize: the template size, its minimum value must be more than 20, the default is 100
%                 searchRad: the radius of searchRegion, the default is 10.the maxinum should be less than 20
%                 sFlag: 1: compute the NCC of the hopc descriptors; 0:compute the SSD of the hopc descriptors (more fast).the default is 1
%                 
%                 
%                 
%                
%                
%                 

%return vaules:                
%                 CP_Ref: the coordinates ([x,y]) in the reference image
%                 CP_Sen: the coordinates ([x,y]) in the sensed image


if nargin < 2
    disp('the number input parameters must be >= 3 ');
    return;
end
if nargin < 3
    cellsize = 4;
end
if nargin < 4
    nCells = 3;
end
if nargin < 5
    orbin = 8;
end


[h1,w1,c1] = size(im_Ref);
[h2,w2,c2] = size(im_Sen);
% tranfer the rgb to gray
if c1 == 3
    im_Ref = rgb2gray(im_Ref);
end
im_Ref = double(im_Ref);

if c2 == 3
    im_Sen = rgb2gray(im_Sen);
end
im_Sen = double(im_Sen);


%the interest points in Reference Image
points1 = [1, 1];

%caculate the CFOG 
% tic;

%caculate the dense block-HOPC descriptor for each pixel
% cellsize=4; % the pixel number (cellsize*cellsize) in a cell
% orbin = 8;  % the number of orientation bin
% nCells=3;  % the cell number (nCell*nCell) of in a block 
% blockoverlay = 0.5;% the overlay degree between adjacent block
im_Refh = h1 - cellsize*nCells + 1; im_Refw = w1 - cellsize*nCells + 1;
im_Senh = h2 - cellsize*nCells + 1; im_Senw = w2 - cellsize*nCells + 1;

%caculate the dense block-HOPC descriptor 
blockHOPC_Ref = denseBlockHOPC(single(im_Ref),cellsize,orbin,nCells);
blockHOPC_Sen = denseBlockHOPC(single(im_Sen),cellsize,orbin,nCells);
blockHOPC_Ref = blockHOPC_Ref(1:im_Refh,1:im_Refw,:);
blockHOPC_Sen = blockHOPC_Sen(1:im_Senh,1:im_Senw,:);

% fprintf('the dense block-HOPC descriptor time for two images is %fs\n',toc);

% tic
corr_fft = Ncc3_fft(blockHOPC_Ref, blockHOPC_Sen);
% imagesc(corr_fft);

%   找到相关系数最大的位置
maxCorr = max(corr_fft(:));
[Y_match, X_match] = find(corr_fft == maxCorr);% 最大相关索引
if(size(X_match,1)>1)
    X_match = X_match(1);
    Y_match = Y_match(1);
end
% t = toc;
% fprintf('the FFT matching time for two images is %fs\n',toc);

CP_Ref = points1;
CP_Sen = [X_match,Y_match];
if isempty(CP_Sen)
    CP_Sen = [unidrnd(length(corr_fft)),unidrnd(length(corr_fft))];
end
