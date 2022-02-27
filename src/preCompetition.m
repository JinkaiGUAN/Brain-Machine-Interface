clc;
clear;
close all;
load("monkeydata_training.mat");
%% 1. Plot population raster plot

%%% In this section, only one trail is being processed to plot population raster
%%% plot.

trialIdx = 1;  % The trail number index
angleIdx = 1;  % The reaching angle index

for unitIdxLoop = 1: 98
    dataSingleUnit = trial(trialIdx, angleIdx).spikes(unitIdxLoop, :);
    nonZeroTime = find(dataSingleUnit == 1);
    labelData = ones(1, length(nonZeroTime)) * unitIdxLoop;
    scatter(nonZeroTime, labelData, 'blue');

    if unitIdxLoop == 1
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
%%% In this section, we are going to plot the neuro idx of 1 at the first
%%% reaching angle. 

neuroIdx = 1;               % The index of a specificed neuro
totralTrailNumber = 100;    % How many trials you wanna plot
maxTimeIdx = 0;             % The max time index along these trials w.r.t. reaching angle
for trailIdxLoop = 1 : totralTrailNumber
    dataSingleTrial = trial(trailIdxLoop, angleIdx).spikes(neuroIdx, :);
    nonZeroTime = find(dataSingleTrial == 1);
    labelData = ones(1, length(nonZeroTime)) * trailIdxLoop;
    scatter(nonZeroTime, labelData);
    
    maxTimeIdx = max(maxTimeIdx, length(dataSingleTrial));
    if trailIdxLoop == 1
        hold on;
    end
end

hold off;
xlabel("Time [ms]");
ylabel("Trail number [-]");
title(["Raster plot for neuro number of ", num2str(neuroIdx), " over ", ...
    num2str(totralTrailNumber), " trails"], 'Interpreter','latex');

%% 3. Plot PSTHs (peri-stimulus time histograms) for different neuros
%%% PSTH is a histogram of the times at which neurons fire. 
%%% Plotting proceudre:
%%%     1. Only Plot the PSTH for one neuro
%%%     2. Concate the data over 100 (the trial number should be considered)
%%%       trails.
%%%     3. Using histogram to plot the PSTH

neuroIdx = 1;                       % Neuro unit index, for example
totralTrailNumber = 100;            % Trial number that would be counted
dataCount = zeros(1, maxTimeIdx);   % Store the spike number
for trailIdxLoop = 1: totralTrailNumber
    dataSingleTrial = trial(trailIdxLoop, angleIdx).spikes(neuroIdx, :);
    nonZeroTime = find(dataSingleTrial == 1);
    dataCount(nonZeroTime) = dataCount(nonZeroTime) + dataSingleTrial(nonZeroTime); 
end

% The sliding window counting the spike number, which also acts like the`nbins`.
deltaT = 10;                       
data = [];
xTicks = [];
for t = (1 + deltaT):deltaT:(maxTimeIdx)
    if (t - 1) <= maxTimeIdx 
        xTicks = [xTicks, t- deltaT];
        subData = sum(dataCount((t - deltaT) :(t-1)));
        data = [data, subData];
    end
end

bar(xTicks, data);
xlabel("Time [s]");
ylabel("Spike number [-]");
title("Peri-stimulus Time Histograms");


%% 4. Plot hand position for differnt trials.
%%% Procedure:
%%%     1. Retrieve single piece of data for one neuro and one single trial.
%%%     2. Plot for x and y position information, respectively.

neuroIdx = 1;
angleIdx = 1;
totalTrialNumber = 5;
positionIdx = 2;            % 1 represents x axis, 2 represents y axis;
maxTimeIdx = 0;             % Index referring to the max time when plotting

for trialIdxLoop = 1 : totalTrialNumber
    dataSingleTrial = trial(trialIdxLoop, angleIdx).handPos(positionIdx, :);
    maxTimeIdx = max(maxTimeIdx, length(dataSingleTrial));
    plot(dataSingleTrial);

    if trialIdxLoop == 1
        hold on;
    end
end

hold off;
xlabel('Time [s]');
ylabel("Position [mm]");
titleStr = "";
if positionIdx == 1
    titleStr = "X position deviation";
elseif positionIdx == 2
    titleStr = "Y position deviation";
end
title(titleStr);

%% 5. Plot turning curves for movement direction






