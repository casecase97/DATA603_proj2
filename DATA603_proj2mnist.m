data = load('mnist.mat');
imgs_train = data.imgs_train;
imgs_test = data.imgs_test;
labels_train = data.labels_train;
labels_test = data.labels_test;

[d1,d2,num_train] = size(imgs_train);
x_train = zeros(d1,d2,1,num_train);
for i = 1:num_train
    x_train(:,:,1,i) = imgs_train(:,:,i);
end

[d1,d2,num_test] = size(imgs_test);
x_test = zeros(d1,d2,1,num_test);
for i = 1:num_test
    x_test(:,:,1,i) = imgs_test(:,:,i);
end

% Network #1 attempted
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1) 
%     reluLayer 
%     
%     % not needed because we dont have channels
%     crossChannelNormalizationLayer(5) 
%     maxPooling2dLayer(2, 'Stride', 2)
%     reluLayer
%     crossChannelNormalizationLayer(5)
%     reluLayer
%     maxPooling2dLayer(3) 
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% Network #2 attempted
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1) 
%     reluLayer 
%     batchNormalizationLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(2, 'Stride', 2) 
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% Network #3 attempted
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1) 
%     reluLayer 
%     batchNormalizationLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(2, 'Stride', 2) 
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% Network #4 attempt
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1) 
%     reluLayer 
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3) 
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% Network #4 attempt
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1) 
%     reluLayer 
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3) 
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% Final Network yielding highest accuracy
layers = [ ...
    imageInputLayer([d1,d2,1])
    convolution2dLayer(5,16,'Padding',1) 
    reluLayer
    batchNormalizationLayer 
    maxPooling2dLayer(3)
    reluLayer
    reluLayer
    maxPooling2dLayer(3) 
    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch');

net = trainNetwork(x_train, labels_train, layers, options);

%% Classify test data

test_label_pred = classify(net, x_test);
%numel returns the number of elements in the array
accuracy = sum(test_label_pred == labels_test)/numel(labels_test)
