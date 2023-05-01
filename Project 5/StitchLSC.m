% Load images.
images = fullfile('Ruggles50', {'img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg', 'img6.jpg'});
numImages = length(images);

% Display images to be stitched.
% montage(buildingScene.Files)

% Read the first image from the image set.
I = imread(images{1});
cameraParams = cameraParameters;
% Initialize features for I(1)
%  J = im2gray(I);
%  grayImage = imadjust(J, [0.6 1]);
% k = [2916.60474  0 2027.30484; 0 2915.53386 1470.13067; 0 0 1];
% 
% radialDistortion = [0.17659 -0.45672 0]; 
% tangentialDistortion = [-0.00152 -0.00089];
% 
% cameraParams = cameraParameters("K",k,"RadialDistortion",radialDistortion, "TangentialDistortion", tangentialDistortion);

% J2 = undistortImage(I,cameraParams,'OutputView','full');
% J = undistortImage(I,cameraParams,'OutputView','full');
grayImage = im2gray(I);

% points = detectORBFeatures(grayImage);
%{
mharris_points = detectHarrisFeatures(grayImage);
%}

%{%
[y, x, m] = harris(grayImage, 1500, 'tile', [2 2], 'disp');
location = [x y];
points = cornerPoints(location, Metric=m);
%}

% imshow(I); hold on;
% plot(points);

[features, points] = extractFeatures(grayImage, points);

% Initialize all the transformations to the identity matrix. Note that the
% projective transormation is used here because the building images are fairly
% close to the camera. For scenes captured from a further distance, you can use
% affine transformations.
tforms(numImages) = projtform2d;

% Initialize variable to hold image sizes.
imageSize = zeros(numImages, 2);

% Iterate over remaining image pairs
for n = 2:numImages
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    
    % Read I(n).
    I = imread(images{n});
    
    % Convert image to grayscale.
%      J1 = undistortImage(I,cameraParams);
     grayImage = im2gray(I);    
%      J = im2gray(I);
%      grayImage = imadjust(J, [0.6 1]);
    
    % Save image size.
    imageSize(n,:) = size(grayImage);

%     points = detectORBFeatures(grayImage);
    %{%
    [y, x, m] = harris(grayImage, 1500, 'tile', [2 2], 'disp');
    location = [x y];
    points = cornerPoints(location, Metric=m);
    %}
    [features, points] = extractFeatures(grayImage, points);
  
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
       
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
    
    % Estimate the transformation between I(n) and I(n-1).
    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 50.9, 'MaxNumTrials', 5000);
    
    % Compute T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).A = tforms(n-1).A * tforms(n).A; 
end

% Compute the output limits for each transformation.
for i = 1:numel(tforms)  
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
% I = [1 0 0; 0 1 0; 0 0 1];
% Tinv = I \ tforms(centerImageIdx);
for i = 1:numel(tforms)    
    tforms(i).A = Tinv.A * tforms(i).A;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    I = imread(images{i}); % readimage(buildingScene, i);   
   
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1), size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)