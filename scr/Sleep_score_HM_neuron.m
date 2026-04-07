function Sleep_score_HM_neuron()
% Sleep Score Setup GUI
% Select an LFP folder, pick 3 channels, auto-detect EMG, choose output folder,
% then launch TheStateEditor.

%% ---------- Add npy-matlab tool to path ----------
scriptDir = fileparts(mfilename('fullpath'));
npyToolPath = fullfile(scriptDir, 'tool', 'npy-matlab-master', ...
    'npy-matlab-master', 'npy-matlab');
if isfolder(npyToolPath)
    addpath(npyToolPath);
end

%% ---------- State ----------
lfpFolder       = '';
emgFile         = '';
outFolder       = '';
availableChNums = [];   % numeric channel indices found in folder

%% ---------- Figure ----------
fig = figure( ...
    'Name',        'Sleep Score Setup', ...
    'NumberTitle', 'off', ...
    'MenuBar',     'none', ...
    'ToolBar',     'none', ...
    'Position',    [300 150 540 560], ...
    'Resize',      'off', ...
    'Color',       [0.94 0.94 0.94]);

%% ---------- Section: LFP Folder ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'LFP Output Folder', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 510 200 22]);

lfpFolderEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '', ...
    'HorizontalAlignment', 'left', ...
    'Enable', 'inactive', ...
    'BackgroundColor', 'white', ...
    'Position', [20 485 410 26]);

uicontrol('Parent', fig, 'Style', 'pushbutton', ...
    'String', 'Browse...', ...
    'Position', [440 485 80 26], ...
    'Callback', @selectLFPFolder);

%% ---------- Section: Channel Selection ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Select Exactly 3 Channels', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 452 250 22]);

% Hold Ctrl/Cmd to multi-select
channelListBox = uicontrol('Parent', fig, 'Style', 'listbox', ...
    'String', {'(load a folder first)'}, ...
    'Min', 0, 'Max', 3, ...
    'Value', [], ...
    'BackgroundColor', 'white', ...
    'Position', [20 310 240 140], ...
    'Callback', @onChannelSelect);

channelInfoText = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Load a folder to see available channels.', ...
    'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0.4 0.4 0.4], ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [270 310 250 140]);

%% ---------- Section: Motion / EMG ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Motion / EMG File', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 278 200 22]);

emgFileEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '', ...
    'HorizontalAlignment', 'left', ...
    'Enable', 'inactive', ...
    'BackgroundColor', 'white', ...
    'Position', [20 253 410 26]);

uicontrol('Parent', fig, 'Style', 'pushbutton', ...
    'String', 'Browse...', ...
    'Position', [440 253 80 26], ...
    'Callback', @selectEMGFile);

emgAutoLabel = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', '', ...
    'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0 0.5 0], ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 233 500 18]);

%% ---------- Section: Output Folder ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Output / Save Folder', ...
    'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 210 200 22]);

outFolderEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '', ...
    'HorizontalAlignment', 'left', ...
    'Enable', 'inactive', ...
    'BackgroundColor', 'white', ...
    'Position', [20 185 410 26]);

uicontrol('Parent', fig, 'Style', 'pushbutton', ...
    'String', 'Browse...', ...
    'Position', [440 185 80 26], ...
    'Callback', @selectOutFolder);

%% ---------- Section: Parameters ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Sampling Rate (Hz):', ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 152 150 22]);

fsEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', '1000', ...
    'BackgroundColor', 'white', ...
    'Position', [175 152 80 24]);

uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Session Name:', ...
    'HorizontalAlignment', 'left', ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [290 152 110 22]);

baseNameEdit = uicontrol('Parent', fig, 'Style', 'edit', ...
    'String', 'HM_neurons', ...
    'BackgroundColor', 'white', ...
    'Position', [405 152 115 24]);

%% ---------- Divider + Status ----------
uicontrol('Parent', fig, 'Style', 'text', ...
    'String', '', ...
    'BackgroundColor', [0.7 0.7 0.7], ...
    'Position', [20 135 500 1]);

statusText = uicontrol('Parent', fig, 'Style', 'text', ...
    'String', 'Ready. Select an LFP folder to begin.', ...
    'HorizontalAlignment', 'left', ...
    'ForegroundColor', [0.3 0.3 0.3], ...
    'BackgroundColor', [0.94 0.94 0.94], ...
    'Position', [20 100 500 30]);

%% ---------- Launch Button ----------
uicontrol('Parent', fig, 'Style', 'pushbutton', ...
    'String', 'Launch TheStateEditor', ...
    'FontSize', 12, 'FontWeight', 'bold', ...
    'BackgroundColor', [0.18 0.55 0.18], ...
    'ForegroundColor', 'white', ...
    'Position', [155 30 230 50], ...
    'Callback', @runStateEditor);

%% ================================================================
%  CALLBACKS
%% ================================================================

    % ---- Select LFP Folder ----------------------------------------
    function selectLFPFolder(~, ~)
        folder = uigetdir('', 'Select LFP Output Folder');
        if isequal(folder, 0), return; end
        lfpFolder = folder;
        set(lfpFolderEdit, 'String', folder);

        % Scan channels_npy subfolder first, then root
        chDir = fullfile(folder, 'channels_npy');
        if ~isfolder(chDir)
            chDir = folder;
        end
        files = dir(fullfile(chDir, 'lfp_nt*_ch01.npy'));

        nums = [];
        for k = 1:length(files)
            tok = regexp(files(k).name, 'lfp_nt(\d+)_ch01\.npy', 'tokens');
            if ~isempty(tok)
                nums(end+1) = str2double(tok{1}{1}); %#ok<AGROW>
            end
        end
        nums = sort(nums);
        availableChNums = nums;

        if isempty(nums)
            set(channelListBox, 'String', {'(no channel files found)'}, 'Value', []);
            set(channelInfoText, 'String', ...
                sprintf('No lfp_ntXX_ch01.npy files found in:\n%s', chDir), ...
                'ForegroundColor', [0.8 0 0]);
        else
            labels = arrayfun(@(n) sprintf('Channel %02d', n), nums, 'UniformOutput', false);
            set(channelListBox, 'String', labels, 'Value', []);
            set(channelInfoText, 'String', ...
                sprintf('%d channels available.\n\nHold Ctrl (Win/Linux)\nor Cmd (Mac) to\nselect multiple.\n\nSelect exactly 3.', ...
                length(nums)), ...
                'ForegroundColor', [0.3 0.3 0.3]);
        end

        % Auto-detect EMG / motion file (priority order)
        emgCandidates = {'emg_rms.npy', 'emg_data.npy', 'theta_delta_ratio.npy', 'awakeness.npy'};
        found = '';
        foundName = '';
        for k = 1:length(emgCandidates)
            candidate = fullfile(folder, emgCandidates{k});
            if isfile(candidate)
                found = candidate;
                foundName = emgCandidates{k};
                break;
            end
        end
        if ~isempty(found)
            emgFile = found;
            set(emgFileEdit, 'String', found);
            set(emgAutoLabel, 'String', ...
                ['Auto-detected: ' foundName], ...
                'ForegroundColor', [0 0.5 0]);
        else
            emgFile = '';
            set(emgFileEdit, 'String', '');
            set(emgAutoLabel, 'String', ...
                'No EMG file auto-detected — please browse manually.', ...
                'ForegroundColor', [0.8 0.4 0]);
        end

        % Default output folder = same as LFP folder
        if isempty(outFolder)
            outFolder = folder;
            set(outFolderEdit, 'String', folder);
        end

        setStatus('Folder loaded. Select 3 channels, then click Launch.', [0 0 0]);
    end

    % ---- Channel listbox selection feedback -----------------------
    function onChannelSelect(~, ~)
        sel = get(channelListBox, 'Value');
        n   = length(sel);
        if n == 3
            setStatus('3 channels selected. Ready to launch.', [0 0.45 0]);
        elseif n > 3
            % Force back to first 3
            set(channelListBox, 'Value', sel(1:3));
            setStatus('Maximum 3 channels. Selection trimmed to first 3.', [0.7 0.4 0]);
        else
            setStatus(sprintf('%d / 3 channels selected.', n), [0 0 0]);
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
        set(emgAutoLabel, 'String', 'Motion file set manually.', ...
            'ForegroundColor', [0 0 0.6]);
    end

    % ---- Select Output Folder ------------------------------------
    function selectOutFolder(~, ~)
        folder = uigetdir('', 'Select Output / Save Folder');
        if isequal(folder, 0), return; end
        outFolder = folder;
        set(outFolderEdit, 'String', folder);
    end

    % ---- Run TheStateEditor ---------------------------------------
    function runStateEditor(~, ~)

        %% Validate
        if isempty(lfpFolder)
            setStatus('Error: Please select an LFP folder.', [0.8 0 0]); return;
        end

        sel = get(channelListBox, 'Value');
        if length(sel) ~= 3
            setStatus('Error: Please select exactly 3 channels.', [0.8 0 0]); return;
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

        baseName = strtrim(get(baseNameEdit, 'String'));
        if isempty(baseName), baseName = 'session'; end

        chsToUse = availableChNums(sel);

        %% Load EEG channels
        chDir = fullfile(lfpFolder, 'channels_npy');
        if ~isfolder(chDir), chDir = lfpFolder; end

        inputData.Chs   = chsToUse;
        inputData.eegFS = eegFS;
        inputData.rawEeg = {};

        for i = 1:3
            setStatus(sprintf('Loading channel %02d (%d/3)...', chsToUse(i), i), [0 0 0.7]);
            drawnow;
            fname = fullfile(chDir, sprintf('lfp_nt%02d_ch01.npy', chsToUse(i)));
            if ~isfile(fname)
                setStatus(['Error: File not found: ' fname], [0.8 0 0]); return;
            end
            try
                chData = readNPY(fname);
                inputData.rawEeg{i} = single(chData(:)');
            catch ME
                setStatus(['Error loading channel: ' ME.message], [0.8 0 0]); return;
            end
        end

        %% Load Motion and resample to match spectrogram bins
        setStatus('Loading motion file...', [0 0 0.7]); drawnow;
        try
            motion = double(readNPY(emgFile));

            % If 2D matrix (e.g. N x channels or channels x N), collapse to 1D
            if ~isvector(motion)
                if size(motion, 1) <= size(motion, 2)
                    motion = mean(motion, 1);  % average across rows → 1 x N
                else
                    motion = mean(motion, 2)'; % average across cols → 1 x N
                end
            end
            motion = motion(:)';  % ensure 1 x N row vector

            % TheStateEditor uses: nFFTChunks = max(1, round((nSamples - WinLength) / winstep))
            % where WinLength = eegFS and winstep = eegFS  (see mtchglongIn call in TheStateEditor)
            nSamples  = length(inputData.rawEeg{1});
            targetLen = max(1, round((nSamples - eegFS) / eegFS));

            if length(motion) ~= targetLen
                origX   = linspace(0, 1, length(motion));
                targetX = linspace(0, 1, targetLen);
                motion  = interp1(origX, motion, targetX, 'linear');
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
