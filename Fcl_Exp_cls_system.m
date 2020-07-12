clc;
cnt=0;
FacialExpressionClassificationSystem_train();
FacialExpressionClassificationSystem_test();
imdstrain = imageDatastore('C:\FEB 2020\CNN\Code\train', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
imdstest = imageDatastore('C:\FEB 2020\CNN\Code\test', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');

numTrainFiles = 70;
inputSize = [50 50 3];
numClasses = 7;

layers = [
imageInputLayer(inputSize)
convolution2dLayer(5,64)
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
disp(imdstrain);



YPred = classify(net,imdstest);
YValidation = imdstest.Labels;
accuracy = mean(YPred == YValidation)
