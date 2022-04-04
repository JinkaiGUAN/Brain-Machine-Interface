function [label] = FCNN(Data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
load('weights9.mat')
layer1w = layer1_weight;
layer1b = layer1_bias;

layer2w = layer2_weight;
layer2b = layer2_bias;

layer3w = layer3_weight;
layer3b = layer3_bias;

layer4w = layer4_weight;
layer4b = layer4_bias;

layer1b = layer1b';
layer2b = layer2b';
layer3b = layer3b';
layer4b = layer4b';

x = layer1w*Data + layer1b;
x(x<0) = 0;

x = layer2w*x + layer2b;
x(x<0) = 0;

x = layer3w*x + layer3b;
x(x<0) = 0;

x = layer4w*x + layer4b;
% layer4(find(layer4<0)) = 0;

label = find(x==max(x));
label = label(1);
%disp(size(label))
end