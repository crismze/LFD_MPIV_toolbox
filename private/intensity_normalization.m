function In = intensity_normalization(img)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB function for estimating normalized intensity                     %
% Authors: Sagar Adatrao and Andrea Sciacchitano                          %
% Date: 11 September 2018                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input: PIV image                                                        %                                                %
% Output: Normalized intensity (In)                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Neighbors
imgN=circshift(img,[-1,0]);
imgS=circshift(img,[1,0]);
imgE=circshift(img,[0,1]);
imgW=circshift(img,[0,-1]);

imgNN=circshift(imgN,[-1,0]);
imgSS=circshift(imgS,[1,0]);
imgEE=circshift(imgE,[0,1]);
imgWW=circshift(imgW,[0,-1]);

imgNE=circshift(img,[-1,1]);
imgNW=circshift(img,[-1,-1]);
imgSE=circshift(img,[1,1]);
imgSW=circshift(img,[1,-1]);
      
imgMean=(imgN+imgS+imgW+imgE+imgNE+imgNW+imgSE+imgSW+imgNN+imgSS+imgEE+...
    imgWW)/12;
imgMean(imgMean==0)=10E-05;   
In=img./imgMean;
        
In(:,1)=0; In(1,:)=0; In(:,2)=0; In(2,:)=0;
In(:,end)=0; In(end,:)=0; In(:,end-1)=0; In(end-1,:)=0;
        
% Normalizing
medIn=median(In(:)); 
In=In/medIn; In(In==0)=10E-05;
   
end