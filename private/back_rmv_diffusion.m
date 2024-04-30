function preproc_bundle = back_rmv_diffusion(image_bundle)
    preproc_bundle = zeros(size(image_bundle), class(image_bundle));
    try
        image_bundle = gpuArray(squeeze(image_bundle));
    catch
        image_bundle = squeeze(image_bundle);
        disp('Using CPU')
    end
    n = size(image_bundle,3);
    lambda=0.20; % lambda determines rate of diffusion (0 < lambda <= 1)
    K=10; % Threshold parameter
    numIter=300; % No. of iterations
    %% radial kernel
    for it = 1:n
        fprintf('Removing backgroung in img %d\n', it)
        current_image = squeeze(image_bundle(:,:,it));
        preproc_enhanced = anisotropic_diffusion_rmv(current_image, lambda, K, numIter);
        if gpuDeviceCount 
            preproc_bundle(:,:,it) = gather(preproc_enhanced);
        else
            preproc_bundle(:,:,it) = preproc_enhanced; 
        end
    end
end