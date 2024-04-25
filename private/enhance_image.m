function enhanced_bundle = enhance_image(image_bundle)
%ENHANCE_IMAGE Remove some motion blur from images
%   Enhance images by using a radial filter to sharpened the particles.
%   This function use GPU Accelaration. 
%   

enhanced_bundle = zeros(size(image_bundle), class(image_bundle));
try
    image_bundle = gpuArray(squeeze(image_bundle));
catch
    image_bundle = squeeze(image_bundle);
    disp('Using CPU')
end
n = size(image_bundle,3);
%% radial kernel
for it = 1:n
    current_image = image_bundle(:,:,it);
%     H = fspecial('disk',10);
%     blurred_gray = imfilter(current_image,H,'replicate');
%     k=2; % filter gain
%     gray_enhanced = current_image + k.*(current_image - blurred_gray);
    K = 2;
    L = fspecial('gaussian',4,4);
    H = fspecial('gaussian',2,2);
    D = fspecial('disk',.5);
    back_image = wiener2(current_image, [4,4]);
    back_image = imfilter(back_image,L,'replicate');
    high_image = current_image - imfilter(current_image,H,'replicate');
    gray_enhanced = K*high_image + back_image/K;
    gray_enhanced = imfilter(gray_enhanced,D,'replicate');
    if gpuDeviceCount; enhanced_bundle(:,:,it) = gather(gray_enhanced);
    else, enhanced_bundle(:,:,it) = gray_enhanced; end
    try
        enhanced_bundle(:,:,it) = gather(gray_enhanced);
    catch
        enhanced_bundle(:,:,it) = (gray_enhanced);
    end
end
end