clc;
cnt=0;

imdstrain = imageDatastore('C:\FEB 2020\CNN\Code\train', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
imdstest = imageDatastore('C:\FEB 2020\CNN\Code\test', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');

numTrainFiles = 70;
inputSize = [144 176 3];
numClasses = 7;

layers = [
imageInputLayer(inputSize)
convolution2dLayer(5,50)
batchNormalizationLayer
reluLayer
fullyConnectedLayer(numClasses)
softmaxLayer
classificationLayer];

options = trainingOptions('sgdm', ...
'MaxEpochs',10, ...
'ValidationData',imdstrain, ...
'ValidationFrequency',30, ...
'Verbose',false, ...
'Plots','training-progress');
net = trainNetwork(imdstrain,layers,options);




YPred = classify(net,imdstest);
YValidation = imdstest.Labels;
accuracy = mean(YPred == YValidation)

