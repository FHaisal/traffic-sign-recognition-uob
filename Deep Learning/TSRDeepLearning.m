imds = imageDatastore('Images', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds);

[imdsTrain,imdsValidation] = splitEachLabel(imds, 0.70, 'randomize');

layers = [
    imageInputLayer([32 32 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(43)
    softmaxLayer
    classificationLayer];

% Max epoch is 34
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',214, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
ConfusionMatrix = confusionmat(YValidation, YPred);