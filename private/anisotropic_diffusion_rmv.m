function [img_pre, img_bck] = anisotropic_diffusion_rmv(img,lambda,K,numIter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB code for pre-processing PIV image using anisotropic diffusion    %
% filter to eliminate light reflections                                   %
% Authors: Sagar Adatrao and Andrea Sciacchitano                          %
% Date: 11 September 2018                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:                                                                 %
%   - PIV image in 2D                                                     %
%   - Parameter representing rate of diffusion (lambda)                   %
%   - Threshold parameter (K)                                             %
%   - Number of iterations                                                %
% Outputs:                                                                %
%   - Pre-processed image                                                 %
%   - Background image containing reflections                             %
% Pseudo code:                                                            %
%   - Compute normalized intensity (function evaluateIn)                  %
%   - Estimate diffusion coefficients along four directions               %
%   - Diffuse the pixel intensities in the image following anisotropic    %
%     diffusion equation in each iteration                                %
%   - Obtain background image which is the diffused image after the       %
%     specified number of iterations                                      %
%   - Generate pre-processed image by subtracting the background image    %
%     from the original image                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Processing by anisotropic diffusion
% img = double(img);
[j,i,~]=size(img);
img0=img; 
imgTot=zeros(j,i,numIter); 
imgTot(:,:,1)=img(:,:,1);
%%
for t=2:numIter
    img=squeeze(imgTot(:,:,t-1));
    imgTot(:,:,t)=img;
    
    % Computing In (intensity normalized with respect to local mean)
    In=intensity_normalization(img); 
           
    imgC=padarray(img(2:j-1,2:i-1),[1,1]);
    
    % Neighbors
    imgN=padarray(imgC(3:j,2:i-1),[1,1]);
    imgS=padarray(imgC(1:j-2,2:i-1),[1,1]);
    imgE=padarray(imgC(2:j-1,1:i-2),[1,1]); 
    imgW=padarray(imgC(2:j-1,3:i),[1,1]);
    
    % Difference in the intensities of neighboring pixels 
    dImgN=imgN-imgC; 
    dImgS=imgS-imgC; 
    dImgE=imgE-imgC; 
    dImgW=imgW-imgC;

    % Computing diffusion coefficients
    cN=(1+(dImgN./(K.*In)).^2).^-1;
    cS=(1+(dImgS./(K.*In)).^2).^-1;
    cE=(1+(dImgE./(K.*In)).^2).^-1;
    cW=(1+(dImgW./(K.*In)).^2).^-1;
    
    % Updating the image
    imgNew=img+lambda*(cN.*dImgN+cS.*dImgS+cE.*dImgE+cW.*dImgW);
    
    % Edges
    imgNew(:,1)=img(:,1); imgNew(:,i)=img(:,i);
    imgNew(1,:)=img(1,:); imgNew(j,:)=img(j,:);
    
    imgNew(isnan(imgNew))=0;
    imgTot(:,:,t)=imgNew;

end

img_pre = (img0-imgNew);
img_bck = (imgNew);