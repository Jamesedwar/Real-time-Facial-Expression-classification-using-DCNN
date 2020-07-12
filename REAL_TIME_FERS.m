clc;
cnt=0;
[path,~]=imgetfile();
img=imread(path);

FaceDetect = vision.CascadeObjectDetector; 
FaceDetect.MergeThreshold = 7 ;
BB = step(FaceDetect, img); 

for i = 1 : size(BB,1)     
  rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
end 
for i = 1 : size(BB, 1) 
  J = imcrop(img, BB(i, :)); 
    
  F2 = imresize(J,[50 50]);
  
end
FacialExpressionClassificationSystem_train();
imdstrain = imageDatastore('C:\FEB 2020\CNN\Code\train', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
% imdstest = imageDatastore('C:\FEB 2020\CNN\Code\Data', ...
% 'IncludeSubfolders',true, ...
% 'LabelSource','foldernames');

numTrainFiles = 70;
inputSize = [50 50 3];
numClasses = 7;


layers = [
imageInputLayer(inputSize)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride', 2)
convolution2dLayer(3,64,'Padding','same')
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,'Stride', 2)
convolution2dLayer(3,128,'Padding','same')
batchNormalizationLayer
reluLayer
fullyConnectedLayer(7)
softmaxLayer
classificationLayer];

options = trainingOptions('sgdm', ...
'MaxEpochs',30, ...
'ValidationData',imdstrain, ...
'ValidationFrequency',10, ...
'Verbose',false, ...
'Plots','training-progress');
net = trainNetwork(imdstrain,layers,options);
YPred = classify(net,F2);
 
figure;
title('Test Image-F2');
imshow(F2);
disp(YPred);

