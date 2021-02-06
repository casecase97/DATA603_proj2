data_p = load('pose.mat');
pose = data_p.pose;
[d1, d2,pose_num,subject_num] = size(pose);

%% Divide the data into Train and Test sets

% Use first 10 poses for training and last 3 for test
train_num = 10;
test_num = pose_num - train_num;
x_train = zeros(d1,d2,1,subject_num*train_num);
for subject_i = 1:subject_num
    start = (subject_i -1)*train_num;
    for pose_i = 1:train_num
        x_train(:,:,1,start+pose_i) = pose(:,:,pose_i,subject_i);
    end
end

train_label = categorical((kron(1:subject_num,ones(1,train_num)))');

x_test = zeros(d1,d2,1,subject_num*test_num);
for subject_i = 1: subject_num
    start = (subject_i-1)*test_num;
    for pose_i = 1:test_num
        x_test(:,:,1,start+pose_i) = pose(:,:,train_num+pose_i,subject_i);
    end
end

test_label = categorical((kron(1:subject_num,ones(1,test_num)))');

%% Set up and Train Neural Network

% from Matlab's help on trainNetwork
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,20)
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     fullyConnectedLayer(subject_num)
%     softmaxLayer
%     classificationLayer];

% reached 59% accuracy
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(3,16,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(subject_num)
%     softmaxLayer
%     classificationLayer];

% accuracy of 60% with 60 epochs
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,12,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(subject_num)
%     softmaxLayer
%     classificationLayer];

% Network Attempt
% layers = [ ...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(2,'Stride',2)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(2,'Stride',2) 
%     fullyConnectedLayer(512)
%     reluLayer
%     batchNormalizationLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(subject_num)
%     softmaxLayer
%     classificationLayer];

% Alexnet 69.61%
layers = [ ...
    imageInputLayer([d1,d2,1])
    convolution2dLayer(5,16,'Padding',1)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(3)
    groupedConvolution2dLayer(3,24,2,'Padding','same')
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(3) 
    fullyConnectedLayer(512)
    reluLayer
    batchNormalizationLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(subject_num)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(x_train, train_label, layers, options);

%% Classify test data

test_label_pred = classify(net, x_test);
%numel returns the number of elements in the array
accuracy = sum(test_label_pred == test_label)/numel(test_label)