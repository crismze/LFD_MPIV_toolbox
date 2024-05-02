function enhanced_bundle = enhance_image(image_bundle)
%ENHANCE_IMAGE Remove some motion blur from images
%   Enhance images by using a radial filter to sharpened the particles.
%   This function use GPU Accelaration. 
%   
enhanced_bundle = zeros(size(image_bundle), class(image_bundle));
if gpuDeviceCount
    try
        image_bundle = gpuArray(squeeze(image_bundle));
        isgpu = 1;
    catch
        fprintf(2,'GPU cannot be used!')
        isgpu = 0;
    end
else 
    image_bundle = squeeze(image_bundle);
    disp('Using CPU to process')
    isgpu = 0;
end
n = size(image_bundle,3);
%% radial kernel
%     H = fspecial('disk',10);
%     H = fspecial('gaussian',4,4);
%     blurred_gray = imfilter(current_image,H,'replicate');
%     k=2; % filter gain
%     gray_enhanced = current_image + k.*(current_image - blurred_gray);
%     gray_enhanced = current_image - blurred_gray;
%% Low Filter
    K = 2;
    L = fspecial('gaussian',4,4);
    H = fspecial('gaussian',2,2);
    D = fspecial('disk',.5);
for it = 1:n
    current_image = image_bundle(:,:,it);
    if isgpu
        back_image = wiener2(gather(current_image), [4,4]);
        back_image = imfilter(gpuArray(back_image),L,'replicate');
        high_image = current_image - imfilter(current_image,H,'replicate');
        gray_enhanced = K*high_image + back_image/K;
        gray_enhanced = imfilter(gray_enhanced,D,'replicate');
        enhanced_bundle(:,:,it) = gather(gray_enhanced);
    else
        back_image = wiener2(current_image, [4,4]);
        back_image = imfilter(back_image,L,'replicate');
        high_image = current_image - imfilter(current_image,H,'replicate');
        gray_enhanced = K*high_image + back_image/K;
        gray_enhanced = imfilter(gray_enhanced,D,'replicate');
        enhanced_bundle(:,:,it) = gray_enhanced; 
    end
end
end