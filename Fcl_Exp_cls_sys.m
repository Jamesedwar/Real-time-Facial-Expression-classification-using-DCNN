clc;
cnt=0;
FacialExpressionClassificationSystem_train();
FacialExpressionClassificationSystem_test();
imdstrain = imageDatastore('C:\FEB 2020\CNN\Code\train', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
disp(imdstrain);
imdstest = imageDatastore('C:\FEB 2020\CNN\Code\test', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');

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




YPred = classify(net,imdstest);
YValidation = imdstest.Labels;
accuracy = mean(YPred == YValidation);
disp(YPred);

disp('Accuracy :');
disp(accuracy*100);
