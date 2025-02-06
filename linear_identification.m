clear;
clc;
load(fullfile('bwdataset', 'bw_matlab'))
nx = 3;

%% Train linear model
train_ds = iddata(y, u, 1.0);
%model = ssest(train_ds, nx,'Ts',1,'Feedthrough',1,'Focus','simulation'); 
model = n4sid(train_ds, nx);

%% Test linear model
% On the same training data
y_train_hat = compare(model, train_ds).OutputData;
train_rmse = rms(train_ds.OutputData - y_train_hat) * 1e5; % 16.67
train_fit = (1 - rms(train_ds.OutputData - y_train_hat)/rms(train_ds.OutputData))*100; 

% On the test set
load(fullfile('bwdataset', 'uval_multisine.mat'))
load(fullfile('bwdataset', 'yval_multisine.mat'))
yval_multisine = yval_multisine(:);
uval_multisine = uval_multisine(:);

test_ds = iddata(yval_multisine, uval_multisine, 1.0);
y_test_hat = compare(model, test_ds).OutputData;
test_rmse = rms(test_ds.OutputData - y_test_hat) * 1e5; % 16.67
test_fit = (1 - rms(test_ds.OutputData - y_test_hat)/rms(test_ds.OutputData))*100; 