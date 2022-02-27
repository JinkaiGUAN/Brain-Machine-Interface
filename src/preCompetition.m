clc;
clear;
close all;

load("monkeydata_training.mat");

%% 1. Plot population raster plot

%%% In this section, only one trail is being processed to plot population raster
%%% plot.

trialIdx = 1;
angleIdx = 1;
for unitIdx = 1: 98
    dataSingleUnit = trial(trialIdx, angleIdx).spikes(unitIdx, :);
    nonZeroTime = find(dataSingleUnit == 1);
    labelData = ones(1, length(nonZeroTime)) * unitIdx;
    scatter(nonZeroTime, labelData, 'blue');
    if unitIdx == 1
        hold on;
    end
end
hold off;
timeSeries = 1:length(dataSingleUnit);
xlim([timeSeries(1) timeSeries(end)]);
xlabel("Time [ms]");
ylabel("Neuro unit number [-]");
title("Raster plot for single trial");

%% 2. Plot raster plot for one unit over several trial

