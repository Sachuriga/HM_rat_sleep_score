function Sleep_score_HM_neuron()
% Sleep Score Setup GUI
% Select an LFP folder, enter 3 channel numbers, auto-detect EMG,
% choose an output folder, then launch TheStateEditor.

%% ---------- Add npy-matlab tool to path ----------
scriptDir = fileparts(mfilename('fullpath'));
npyToolPath = fullfile(scriptDir, 'tool', 'npy-matlab-master', ...
    'npy-matlab-master', 'npy-matlab');
if isfolder(npyToolPath)
    addpath(npyToolPath);
end

%% ---------- Shared state ----------
lfpFolder = '';
emgFile   = '';
outFolder = '';

%% ---------- Figure ----------
fig = figure( ...
    'Name',        'Sleep Score Setup', ...
    'NumberTitle', 'off', ...
    'MenuBar',     'none', ...
    'ToolBar',     'none', ...
    'Position',    [300 200 520 460], ...
    'Resize',      'off', ...
    'Color',       [0.94 0.94 0.94]);

%% ---------- LFP Folder ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'LFP Output Folder', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 415 200 22]);

lfpFolderEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '', 'HorizontalAlignment', 'left', ...
    'Enable', 'inactive', 'BackgroundColor', 'white', ...
    'Position', [20 390 400 26]);

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Browse...', ...
    'Position', [430 390 70 26], 'Callback', @selectLFPFolder);

%% ---------- Channel Numbers ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Channel Numbers (1–32)', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 358 220 22]);

chLabels = {'Ch 1:', 'Ch 2:', 'Ch 3:'};
chEdits  = zeros(1, 3);
xPos     = [20, 185, 350];
for k = 1:3
    uicontrol('Parent', fig, 'Style', 'text', 'String', chLabels{k}, ...
        'HorizontalAlignment', 'left', ...
        'BackgroundColor', [0.94 0.94 0.94], ...
        'Position', [xPos(k) 333 50 22]);
    chEdits(k) = uicontrol('Parent', fig, 'Style', 'edit', ...
        'String', num2str(k), ...
        'BackgroundColor', 'white', ...
        'Position', [xPos(k)+52 333 80 26]);
end

%% ---------- Motion / EMG ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Motion / EMG File', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 300 200 22]);

emgFileEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '', 'HorizontalAlignment', 'left', ...
    'Enable', 'inactive', 'BackgroundColor', 'white', ...
    'Position', [20 275 400 26]);

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Browse...', ...
    'Position', [430 275 70 26], 'Callback', @selectEMGFile);

emgAutoLabel = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', '', 'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0 0.5 0], ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 255 500 18]);

%% ---------- Output Folder ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Output / Save Folder', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 225 200 22]);

outFolderEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '', 'HorizontalAlignment', 'left', ...
    'Enable', 'inactive', 'BackgroundColor', 'white', ...
    'Position', [20 200 400 26]);

uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Browse...', ...
    'Position', [430 200 70 26], 'Callback', @selectOutFolder);

%% ---------- Parameters ----------
uicontrol('Parent', fig, 'Style', 'text', 'String', 'Sampling Rate (Hz):', ...
    'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 165 150 22]);
fsEdit = uicontrol('Parent', fig, 'Style', 'edit', 'String', '1000', ...
    'BackgroundColor', 'white', 'Position', [175 165 80 24]);

uicontrol('Parent', fig, 'Style', 'text', 'String', 'Session Name:', ...
    'HorizontalAlignment', 'left', 'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [290 165 110 22]);
baseNameEdit = uicontrol('Parent', fig, 'Style', 'edit', 'String', 'HM_neurons', ...
    'BackgroundColor', 'white', 'Position', [405 165 95 24]);

%% ---------- Status + Launch ----------
uicontrol('Parent', fig, 'Style', 'text', 'String', '', ...
    'BackgroundColor', [0.7 0.7 0.7], 'Position', [20 148 480 1]);

statusText = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Ready. Select an LFP folder to begin.', ...
    'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0.3 0.3 0.3], ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 110 480 32]);

uicontrol('Parent', fig, 'Style', 'pushbutton', ...
    'String', 'Launch TheStateEditor', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.18 0.55 0.18], 'ForegroundColor', 'white', ...
    'Position', [145 30 230 55], 'Callback', @runStateEditor);

%% ================================================================
%  CALLBACKS
%% ================================================================

    % ---- Select LFP Folder ----------------------------------------
    function selectLFPFolder(~, ~)
        folder = uigetdir('', 'Select LFP Output Folder');
        if isequal(folder, 0), return; end
        lfpFolder = folder;
        set(lfpFolderEdit, 'String', folder);

        % Check lfp_data.npy exists
        if ~isfile(fullfile(folder, 'lfp_data.npy'))
            setStatus('Warning: lfp_data.npy not found in this folder.', [0.8 0.4 0]);
        else
            setStatus('Folder loaded. Enter channel numbers, then click Launch.', [0 0 0]);
        end

        % Auto-detect EMG / motion file
        emgCandidates = {'emg_rms.npy', 'emg_data.npy', 'theta_delta_ratio.npy', 'awakeness.npy'};
        for k = 1:length(emgCandidates)
            candidate = fullfile(folder, emgCandidates{k});
            if isfile(candidate)
                emgFile = candidate;
                set(emgFileEdit, 'String', candidate);
                set(emgAutoLabel, 'String', ['Auto-detected: ' emgCandidates{k}], ...
                    'ForegroundColor', [0 0.5 0]);
                break;
            end
        end
        if isempty(emgFile)
            set(emgAutoLabel, 'String', 'No EMG file auto-detected — browse manually.', ...
                'ForegroundColor', [0.8 0.4 0]);
        end

        % Default output = same folder
        if isempty(outFolder)
            outFolder = folder;
            set(outFolderEdit, 'String', folder);
        end
    end

    % ---- Select EMG file manually ---------------------------------
    function selectEMGFile(~, ~)
        startPath = lfpFolder;
        if isempty(startPath), startPath = pwd; end
        [f, p] = uigetfile('*.npy', 'Select Motion / EMG File', startPath);
        if isequal(f, 0), return; end
        emgFile = fullfile(p, f);
        set(emgFileEdit, 'String', emgFile);
        set(emgAutoLabel, 'String', 'Motion file set manually.', 'ForegroundColor', [0 0 0.6]);
    end

    % ---- Select Output Folder ------------------------------------
    function selectOutFolder(~, ~)
        folder = uigetdir('', 'Select Output / Save Folder');
        if isequal(folder, 0), return; end
        outFolder = folder;
        set(outFolderEdit, 'String', folder);
    end

    % ---- Launch TheStateEditor ------------------------------------
    function runStateEditor(~, ~)

        %% Validate inputs
        if isempty(lfpFolder)
            setStatus('Error: Please select an LFP folder.', [0.8 0 0]); return;
        end
        if isempty(emgFile)
            setStatus('Error: Please select a motion/EMG file.', [0.8 0 0]); return;
        end
        if isempty(outFolder)
            setStatus('Error: Please select an output folder.', [0.8 0 0]); return;
        end

        eegFS = str2double(get(fsEdit, 'String'));
        if isnan(eegFS) || eegFS <= 0
            setStatus('Error: Sampling rate must be a positive number.', [0.8 0 0]); return;
        end

        % Parse channel numbers
        chsToUse = zeros(1, 3);
        for k = 1:3
            val = str2double(strtrim(get(chEdits(k), 'String')));
            if isnan(val) || val < 1 || val ~= round(val)
                setStatus(sprintf('Error: Ch %d must be a whole number ≥ 1.', k), [0.8 0 0]); return;
            end
            chsToUse(k) = val;
        end
        if length(unique(chsToUse)) < 3
            setStatus('Error: All 3 channel numbers must be different.', [0.8 0 0]); return;
        end

        baseName = strtrim(get(baseNameEdit, 'String'));
        if isempty(baseName), baseName = 'session'; end

        %% Load lfp_data.npy
        lfpFile = fullfile(lfpFolder, 'lfp_data.npy');
        if ~isfile(lfpFile)
            setStatus(['Error: lfp_data.npy not found in ' lfpFolder], [0.8 0 0]); return;
        end

        setStatus('Loading lfp_data.npy (this may take a moment)...', [0 0 0.7]); drawnow;
        try
            eegData = readNPY(lfpFile);       % shape: [samples x channels]
            eegData = double(eegData)';       % transpose → [channels x samples]
        catch ME
            setStatus(['Error loading LFP: ' ME.message], [0.8 0 0]); return;
        end

        nCh = size(eegData, 1);
        for k = 1:3
            if chsToUse(k) > nCh
                setStatus(sprintf('Error: Channel %d does not exist (file has %d channels).', ...
                    chsToUse(k), nCh), [0.8 0 0]); return;
            end
        end

        inputData.Chs    = chsToUse;
        inputData.eegFS  = eegFS;
        inputData.rawEeg = {};
        for i = 1:3
            inputData.rawEeg{i} = single(eegData(chsToUse(i), :));
        end
        eegData = [];  % free memory

        %% Load and downsample motion signal
        setStatus('Loading motion file...', [0 0 0.7]); drawnow;
        try
            motion = double(readNPY(emgFile));
            motion = motion(:);   % flatten to column (N x 1)

            % TheStateEditor needs 1 value per spectrogram bin.
            % Its formula: nFFTChunks = max(1, round((nSamples - eegFS) / eegFS))
            nSamples  = length(inputData.rawEeg{1});
            targetLen = max(1, round((nSamples - eegFS) / eegFS));

            if length(motion) > targetLen
                % Full-rate signal — average into 1-second bins
                usable = targetLen * eegFS;
                if length(motion) >= usable
                    motion = motion(1:usable);
                else
                    motion = [motion; zeros(usable - length(motion), 1)];
                end
                motion = mean(reshape(motion, eegFS, targetLen), 1);  % 1 x targetLen
            else
                % Already ~1 Hz — just resample to exact target length
                origX   = linspace(0, 1, length(motion));
                targetX = linspace(0, 1, targetLen);
                motion  = interp1(origX, motion', targetX, 'linear');
            end

            inputData.MotionType = 'File';
            inputData.motion     = motion;
        catch ME
            setStatus(['Error loading motion: ' ME.message], [0.8 0 0]); return;
        end

        %% Launch TheStateEditor from output folder
        prevDir = pwd;
        cd(outFolder);
        setStatus('Launching TheStateEditor...', [0 0.45 0]); drawnow;
        try
            TheStateEditor(baseName, inputData);
        catch ME
            cd(prevDir);
            setStatus(['StateEditor error: ' ME.message], [0.8 0 0]); return;
        end
        cd(prevDir);
        setStatus('TheStateEditor closed. Results saved to output folder.', [0 0.4 0]);
    end

    % ---- Helper ---------------------------------------------------
    function setStatus(msg, color)
        set(statusText, 'String', msg, 'ForegroundColor', color);
        drawnow;
    end

end