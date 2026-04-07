% Load your npy files
eegData = readNPY('C:\Users\gl_pc\Desktop\data\LFP\op1\LFP_Output\lfp_data.npy');
timeVec = readNPY('C:\Users\gl_pc\Desktop\data\LFP\op1\LFP_Output\lfp_timestamps.npy');
motion = readNPY('C:\Users\gl_pc\Desktop\data\LFP\op1\LFP_Output\theta_delta_ratio.npy');

% Define parameters
eegFS = 1000;  % change to your actual sampling rate
eegData = eegData'; 
motion=double(motion');
% Build rawEeg cell array (max 3 channels)
% Pick which channels to use, e.g. channels 1, 2, 3
chsToUse = [7,16,19];  % change to your desired channel indices

inputData.Chs = chsToUse;
inputData.eegFS = eegFS;

% Fill rawEeg cell array
inputData.rawEeg = {};
for i = 1:length(chsToUse)
    inputData.rawEeg{i} = single(eegData(chsToUse(i), :));
end

% Motion - simplest option is to set to none
inputData.MotionType = 'File';
inputData.motion = motion;

% Run StateEditor
baseName = 'HM_neurons';
TheStateEditor(baseName, inputData);