 %% ================================================================
%  Time Production Experiment ? 5-Session Protocol
%  v9.0 ? Optimized | Triggers fixed | Timer in probe | RT display
%  Pure MATLAB, no Psychtoolbox
%% ================================================================

clear; clc; close all;

%% ===================== EVENT CODES ==============================
EVT.start       = 1;
EVT.behav       = 2;   % Behavioral probe (90s no response)
EVT.sched       = 3;   % Scheduled final probe
EVT.manual      = 4;   % Manual probe
EVT.endprob     = 5;
EVT.endsess     = 6;
EVT.pause_evt   = 7;
EVT.resume_evt  = 8;
EVT.correct     = 9;   % Undo / correction
EVT.respBtn     = 10;   % Response button press
EVT.resetBtn    = 11;   % Reset button press
EVT.respProbe   = 12;   % Probe: response streak (Nap2)
EVT.resetProbe  = 13;   % Probe: reset streak    (Nap2)
EVT.safetyProbe = 14;   % Probe: safety 15 min   (Nap2)

%% ===================== SIMULATION MODE ==========================
SIMULATION_MODE = false;  % true = no trigger box | false = COM3

%% ===================== SUBJECT PARAMETERS =======================
resp = inputdlg({'Subject ID:'}, 'Experiment Setup', 1, {''});
if isempty(resp) || isempty(resp{1}); error('Cancelled.'); end
subID = strtrim(resp{1});

sessionOptions = {'Session 0 (Test    ? 1 min)',  ...
                  'Session 1 (Wake 1  ? 15 min)', ...
                  'Session 2 (Nap 1   ? 30 min)', ...
                  'Session 3 (Nap 2   ? 90 min)', ...
                  'Session 4 (Wake 2  ? 15 min)'};
startSession = menu('Choose starting session:', sessionOptions);
if startSession == 0; error('Cancelled.'); end
fprintf('Starting from session %d\n', startSession - 1);

%% ===================== PATHS ====================================
main_path  = 'C:\Users\MATLAB.PSL5216024\Documents\TickToz_IloSara';
audio_path = fullfile(main_path, 'Audio_questionnaire');
for d = {main_path, audio_path}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end
cd(main_path);

% Live detection flags
respFlagPath  = fullfile(main_path, 'live_detect', 'resp_flag.txt');
resetFlagPath = fullfile(main_path, 'live_detect', 'reset_flag.txt');

data_path = fullfile(main_path, 'data');
if ~exist(data_path,'dir'), mkdir(data_path); end
subFolder = fullfile(data_path, sprintf('sub%s', subID));
if ~exist(subFolder,'dir'), mkdir(subFolder); end

%% ===================== AUDIO FILES ==============================
audio_questions = {
    '1.1_mental_content_Gertrud.mp3'
    '2.1_Chiffre_Gertrud.mp3'
    '3.1_Temps_ecoule_Gertrud.mp3'
    '4.1_Perception_Gertrud.mp3'
    '5.1_Bizarrerie_Gertrud.mp3'
    '6.1_Spontaneite_Gertrud.mp3'
    '7.1_Fluidite_Gertrud.mp3'
    '8.1_Plaisir_Gertrud.mp3'
    '9.1_Rapidite_Temps_Gertrud.mp3'
    '10.1_Bouton_reponse_Gertrud.mp3'
    '11.1_Bouton_reset_Gertrud.mp3'
    '12.1_endormi_Gertrud.mp3'
    '13.1_parti_Gertrud.mp3'
    '14.1_merci_Gertrud.mp3'
};
audio_last_end   = '14.1_merci_Gertrud.mp3';
audio_last_final = '15.1_merci_fin_Gertrud.mp3';
audio_alarm      = 'alarm.mp3';
audio_start      = 'debut.mp3';

% --- Check files ---
allFiles = [audio_questions; {audio_last_end}; {audio_last_final}; ...
            {audio_alarm}; {audio_start}];
missing = {};
for i = 1:numel(allFiles)
    if ~exist(fullfile(audio_path, allFiles{i}), 'file')
        missing{end+1} = allFiles{i}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    fprintf(2, '\n*** MISSING FILES:\n');
    for i = 1:numel(missing), fprintf(2,'   - %s\n', missing{i}); end
    ans_ = questdlg(sprintf('%d missing file(s). Continue anyway?', numel(missing)), ...
        'Missing Files','Continue','Cancel','Cancel');
    if strcmp(ans_,'Cancel'); error('Missing audio files.'); end
end

% --- Preload sounds ---
if exist(fullfile(audio_path, audio_alarm), 'file')
    [alarmY, alarmFs] = audioread(fullfile(audio_path, audio_alarm));
else
    alarmFs = 44100; alarmY = zeros(44100,1);
    warning('alarm.mp3 not found, using silence');
end
if exist(fullfile(audio_path, audio_start), 'file')
    [startY, startFs] = audioread(fullfile(audio_path, audio_start));
else
    startFs = 44100; startY = zeros(44100,1);
    warning('debut.mp3 not found, using silence');
end

%% ===================== SESSION PARAMETERS =======================
% { name, duration_min, type }
% Probe logic per type:
%   test / wake1 / wake2 : probe after 90s, session continues
%   nap1                 : probe after 90s, session ends
%   nap2                 : probe after 90s OR streaks OR 15-min safety
sessions = {
    {'session0_test',   1, 'test' }
    {'session1_wake',  15, 'wake1'}
    {'session2_nap1',  30, 'nap1' }
    {'session3_nap2',  90, 'nap2' }
    {'session4_wake',  15, 'wake2'}
};
totalSessions = numel(sessions);

NO_RESP_TIMEOUT_S = 90;       % seconds without button press before probe
SAFETY_INTERVAL_S = 15 * 60;  % Nap2: safety probe if no probe for 15 min

%% ===================== TRIGGER BOX ==============================
if SIMULATION_MODE
    SerialPortObj = [];
    fprintf('\n=========================================\n');
    fprintf('       SIMULATION MODE  (no triggers)\n');
    fprintf('=========================================\n\n');
else
    instrreset;
    try
        SerialPortObj = serial('COM3','Timeout',1);
        fopen(SerialPortObj);
        flushinput(SerialPortObj);
        flushoutput(SerialPortObj);
        fprintf('Trigger box connected on COM3\n');
    catch ME
        warning('COM3 unavailable, switching to simulation. Error: %s', ME.message);
        SIMULATION_MODE = true;
        SerialPortObj   = [];
    end
end

%% ===================== AUDIO CONFIG =============================
aCfg.path      = audio_path;
aCfg.questions = audio_questions;
aCfg.lastEnd   = audio_last_end;
aCfg.lastFinal = audio_last_final;
aCfg.alarmY    = alarmY;
aCfg.alarmFs   = alarmFs;
aCfg.fsRec     = 44100;
aCfg.nBits     = 16;
aCfg.nCh       = 2;
aCfg.devID     = -1;
aCfg.nQ        = numel(audio_questions);

%% ===================== TRIGGER TEST (optional) ==================
if ~SIMULATION_MODE
    choix = questdlg('Test EEG triggers before starting?', ...
        'Trigger Test','Test','Skip','Skip');
    if strcmp(choix,'Test')
        evtFields = fieldnames(EVT);
        fprintf('=== TRIGGER TEST ===\n');
        for i = 1:numel(evtFields)
            fprintf('  %-15s -> %d\n', evtFields{i}, EVT.(evtFields{i}));
            trigSend(SerialPortObj, EVT.(evtFields{i}));
            pause(0.4);
        end
        rep = questdlg('Were all triggers received correctly?', ...
            'Test Result','Yes ? Continue','No ? Abort','Yes ? Continue');
        if strcmp(rep,'No ? Abort')
            try; fclose(SerialPortObj); catch; end
            error('Trigger test failed.');
        end
        fprintf('Trigger test OK!\n\n');
    end
end

%% ===================== MAIN SESSION LOOP ========================
hFig = [];

for sessIdx = startSession:totalSessions 

    sessName    = sessions{sessIdx}{1};
    sessDur_min = sessions{sessIdx}{2};
    sessType    = sessions{sessIdx}{3};
    sessDur_s   = sessDur_min * 60;
    isNap2      = strcmp(sessType, 'nap2');

    fprintf('\n=========================================\n');
    fprintf('  SESSION %d/%d: %s (%s) ? %d min\n', ...
        sessIdx, totalSessions, sessName, sessType, sessDur_min);
    fprintf('=========================================\n');

    %% -- Directories --
    sessFolder  = fullfile(subFolder, sessName);
    logDir      = fullfile(sessFolder, 'logs');
    dataDir     = fullfile(sessFolder, 'data');
    audioRecDir = fullfile(sessFolder, 'audio_recordings');
    for d = {sessFolder, logDir, dataDir, audioRecDir}
        if ~exist(d{1},'dir'), mkdir(d{1}); end
    end

    %% -- Log + data files --
    ts      = datestr(now, 'yyyymmdd_HHMMSS');
    logPath  = fullfile(logDir,  sprintf('sub%s_%s_log_%s.csv',  subID, sessName, ts));
    dataPath = fullfile(dataDir, sprintf('sub%s_%s_data_%s.csv', subID, sessName, ts));
    fidLog  = fopen(logPath,  'w');
    fidData = fopen(dataPath, 'w');
    fprintf(fidLog,  'WallTime,Session_s,Event,Details\n');
    fprintf(fidData, ['SubjectID,SessionIdx,SessionType,EventN,EventType,' ...
        'WallTime,Session_s,ConsecResp,ConsecReset,' ...
        'RespThreshold,ResetThreshold,ProbeCount\n']);

    %% -- Start confirmation (sessions 2+) --
    if sessIdx > 1
        ans_ = questdlg(sprintf('Start Session %d  (%s)?', sessIdx, sessName), ...
            'Next Session','Start','Cancel','Start');
        if strcmp(ans_,'Cancel')
            fclose(fidLog); fclose(fidData);
            fprintf('Experiment cancelled.\n');
            break
        end
    end

    %% -- Build GUI --
    [hFig, ui] = buildGUI(sessName, sessDur_min, isNap2);
    figure(hFig); drawnow;

    %% -- AppData init --
    setappdata(hFig, 'mode',         'monitor');
    setappdata(hFig, 'resp',          0);        % counter, not boolean
    setappdata(hFig, 'reset_evt',     0);        % counter, not boolean
    setappdata(hFig, 'manualProbe',   false);
    setappdata(hFig, 'endSession',    false);
    setappdata(hFig, 'spacePressed',  false);
    setappdata(hFig, 'togglePause',   false);
    setappdata(hFig, 'isPaused',      false);
    setappdata(hFig, 'undoReq',       false);
    setappdata(hFig, 'emergencyExit', false);

    % Streak counters (R/E thresholds relevant for Nap2 only)
    cnt = struct(...
        'consec_resp',     0, ...
        'consec_reset',    0, ...
        'resp_threshold',  3, ...   % Nap2: escalates 3->5->8
        'reset_threshold', 3, ...   % Nap2: escalates 3->5->8, resets on other probes
        'last_probe_tic',  tic());
    setappdata(hFig, 'cnt',     cnt);
    setappdata(hFig, 'lastEvt', []);

    %% -- Session start triggers --
    switch sessIdx
        case 1, trigSend(SerialPortObj, EVT.start);
        case 2, trigSend(SerialPortObj, EVT.start);
        case 3, trigSend(SerialPortObj, EVT.start);
        case 4, trigSend(SerialPortObj, EVT.start);
        case 5, trigSend(SerialPortObj, EVT.start);
    end
    trigSend(SerialPortObj, EVT.start);

    % Start sound at every session
    try; playblocking(audioplayer(startY, startFs)); catch; end

    %% -- Session state variables --
    sessionTic      = tic();
    lastBtnTic      = tic();    % no-response timer
    lastRespTic     = tic();    % inter-response time tracker
    respTimes       = [];       % stores all inter-response intervals (R only)
    firstResp       = true;     % skip IRT calculation on very first R press
    totalPaused_s   = 0;
    pausedSinceBtnS = 0;
    isPaused        = false;
    pauseStartTic   = [];
    frozenEl_s      = 0;
    probeCount      = 0;
    eventCount      = 0;
    skipFinalProbe  = false;
    sessionEnded    = false;
    lastRespFlagSeen  = readFlagCounter(respFlagPath);   % ignore counts from before this session
    lastResetFlagSeen = readFlagCounter(resetFlagPath);

    writeLog(fidLog, 0, 'SESSION_START', sprintf('sub%s %s', subID, sessType));
    writeData(fidData, subID, sessIdx, sessType, eventCount, 'SESSION_START', 0, cnt, probeCount);

    set(ui.status, 'String', sprintf('Monitoring [%s]...', sessName), ...
        'ForegroundColor', [.3 .95 .3]);
    fprintf('=== Session %d started | expected end ~%s ===\n', ...
        sessIdx, datestr(datetime('now') + minutes(sessDur_min), 'HH:MM'));

    %% ==================== MONITORING LOOP ====================
    while isvalid(hFig) && ~sessionEnded

        if ~isequal(get(0,'CurrentFigure'), hFig), figure(hFig); end

        %% -- Emergency exit --
        if getappdata(hFig, 'emergencyExit')
            fprintf('>>> Emergency exit (ESC or window closed)\n');
            skipFinalProbe = true;
            sessionEnded   = true;
            break
        end

        %% -- Pause / Resume --
        if getappdata(hFig, 'togglePause')
            setappdata(hFig, 'togglePause', false);
            if ~isPaused
                isPaused      = true;
                pauseStartTic = tic();
                frozenEl_s    = toc(sessionTic) - totalPaused_s;
                setappdata(hFig, 'isPaused', true);
                trigSend(SerialPortObj, EVT.pause_evt);
                writeLog(fidLog, frozenEl_s, 'PAUSE', '');
                set(ui.pauseBtn,'String','RESUME (P)','BackgroundColor',[.15 .55 .20]);
                set(ui.status,  'String','PAUSED',    'ForegroundColor',[1 .9 .2]);
                enableBtns(ui, false);
                set(ui.pauseBtn,'Enable','on');
            else
                pd              = toc(pauseStartTic);
                totalPaused_s   = totalPaused_s   + pd;
                pausedSinceBtnS = pausedSinceBtnS + pd;
                isPaused        = false;
                setappdata(hFig,'isPaused',false);
                trigSend(SerialPortObj, EVT.resume_evt);
                writeLog(fidLog, frozenEl_s, 'RESUME', sprintf('pause %.0fs', pd));
                set(ui.pauseBtn,'String','PAUSE (P)','BackgroundColor',[.52 .42 .10]);
                enableBtns(ui, true);
                set(ui.status,'String',sprintf('Monitoring [%s]...', sessName), ...
                    'ForegroundColor',[.3 .95 .3]);
            end
        end

        if isPaused
            lastRespFlagSeen  = readFlagCounter(respFlagPath);
            lastResetFlagSeen = readFlagCounter(resetFlagPath);

            pDur = toc(pauseStartTic);
            set(ui.sessTime,'String',sprintf('%s / %s', fmtTime(frozenEl_s), fmtTime(sessDur_s)));
            set(ui.status,  'String',sprintf('PAUSED (%s)', fmtTime(pDur)), ...
                'ForegroundColor',[1 .9 .2]);
            pause(0.1); drawnow limitrate;
            continue
        end

        %% -- Current times --
        el_s    = toc(sessionTic) - totalPaused_s;
        noBtn_s = toc(lastBtnTic) - pausedSinceBtnS;
        cnt     = getappdata(hFig, 'cnt');

        %% -- Auto-end --
        if el_s >= sessDur_s, break; end

        %% -- Manual end --
        if getappdata(hFig, 'endSession')
            setappdata(hFig,'endSession',false);
            ans_ = questdlg('End session and launch final probe?', ...
                'Confirmation','Yes','No','No');
            if strcmp(ans_,'Yes'), break; end
            continue
        end

        %% -- GUI update --
        updateGUI(ui, el_s, noBtn_s, NO_RESP_TIMEOUT_S, sessDur_s, cnt, isNap2);
        drawnow limitrate;

        %% -- UNDO (Z or button) --
        if getappdata(hFig, 'undoReq')
            setappdata(hFig,'undoReq',false);
            lastEvt = getappdata(hFig,'lastEvt');
            if ~isempty(lastEvt)
                oldType = lastEvt.type;
                cnt     = lastEvt.pre_cnt;
                if strcmp(oldType,'RESPONSE')
                    newType          = 'RESET';
                    cnt.consec_reset = lastEvt.pre_cnt.consec_reset + 1;
                    cnt.consec_resp  = 0;
                    trigSend(SerialPortObj, EVT.resetBtn);
                else
                    newType          = 'RESPONSE';
                    cnt.consec_resp  = lastEvt.pre_cnt.consec_resp + 1;
                    cnt.consec_reset = 0;
                    trigSend(SerialPortObj, EVT.respBtn);
                end
                trigSend(SerialPortObj, EVT.correct);
                eventCount = eventCount + 1;
                writeLog(fidLog, el_s, 'UNDO', ...
                    sprintf('%s -> %s  (timer unchanged)', oldType, newType));
                writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                    sprintf('UNDO_%s_%s', oldType, newType), el_s, cnt, probeCount);
                setappdata(hFig,'cnt',cnt);
                lastEvt.type = newType;
                setappdata(hFig,'lastEvt',lastEvt);
                set(ui.status,'String', ...
                    sprintf('UNDO: %s -> %s  (timer unchanged)', oldType, newType), ...
                    'ForegroundColor',[1 1 0]);
                drawnow; pause(1.2);
            else
                set(ui.status,'String','Nothing to undo','ForegroundColor',[1 1 0]);
                drawnow; pause(0.8);
            end
            set(ui.status,'String',sprintf('Monitoring [%s]...', sessName), ...
                'ForegroundColor',[.3 .95 .3]);
            continue
        end

        %% -- Live detection flags --
        newResp = readFlagCounter(respFlagPath);   % returns 0 if file missing/unreadable
        if newResp > lastRespFlagSeen
            setappdata(hFig,'resp', getappdata(hFig,'resp') + (newResp - lastRespFlagSeen));
            lastRespFlagSeen = newResp;
        end
        newReset = readFlagCounter(resetFlagPath);
        if newReset > lastResetFlagSeen
            setappdata(hFig,'reset_evt', getappdata(hFig,'reset_evt') + (newReset - lastResetFlagSeen));
            lastResetFlagSeen = newReset;
        end

        %% -- RESPONSE button (R) --
        if getappdata(hFig,'resp') > 0
            setappdata(hFig,'resp', getappdata(hFig,'resp') - 1);
            setappdata(hFig,'lastEvt', struct('type','RESPONSE','pre_cnt',cnt,'time_s',el_s));

            % ---- Inter-response time display ----
            if ~firstResp
                irt_s = toc(lastRespTic);
                respTimes(end+1) = irt_s; %#ok<AGROW>
                fprintf('  [R] IRT = %5.2f s  |  Mean = %5.2f s  (n=%d)\n', ...
                    irt_s, mean(respTimes), numel(respTimes));
            else
                firstResp = false;
                fprintf('  [R] First response at t = %.2f s\n', el_s);
            end
            lastRespTic = tic();
            % -----------------------------------------

            cnt.consec_resp  = cnt.consec_resp + 1;
            cnt.consec_reset = 0;
            lastBtnTic       = tic();
            pausedSinceBtnS  = 0;
            eventCount = eventCount + 1;
            trigSend(SerialPortObj, EVT.respBtn);
            writeLog(fidLog, el_s, 'RESPONSE', sprintf('streak_R=%d', cnt.consec_resp));
            writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                'RESPONSE', el_s, cnt, probeCount);
            setappdata(hFig,'cnt',cnt);

            % Nap2: response streak -> probe
            if isNap2 && cnt.consec_resp >= cnt.resp_threshold
                probeCount = probeCount + 1;
                writeLog(fidLog, el_s, 'PROBE_TRIGGER', ...
                    sprintf('RESP_STREAK %d/%d -> probe #%d', ...
                    cnt.consec_resp, cnt.resp_threshold, probeCount));
                trigSend(SerialPortObj, EVT.respProbe);
                doProbe(probeCount, 'resp_streak', subID, aCfg, hFig, ui, ...
                    false, audioRecDir, sessionTic, totalPaused_s, sessDur_s);
                trigSend(SerialPortObj, EVT.endprob);
                lastRespFlagSeen  = readFlagCounter(respFlagPath);   % drop presses made during the probe
                lastResetFlagSeen = readFlagCounter(resetFlagPath);

                % Threshold escalation: 3 -> 5 -> 8
                if     cnt.resp_threshold < 5,  cnt.resp_threshold = 5;
                elseif cnt.resp_threshold < 8,  cnt.resp_threshold = 8;
                end
                cnt.consec_resp     = 0;
                cnt.consec_reset    = 0;
                cnt.reset_threshold = 3;   % another probe type breaks reset streak
                cnt.last_probe_tic  = tic();
                lastBtnTic = tic(); pausedSinceBtnS = 0;
                setappdata(hFig,'cnt',cnt); setappdata(hFig,'lastEvt',[]);
                set(ui.probeN,'String',num2str(probeCount));
                el_now = toc(sessionTic) - totalPaused_s;
                writeLog(fidLog, el_now, 'PROBE_END', ...
                    sprintf('#%d (resp_streak, new_thr=%d)', probeCount, cnt.resp_threshold));
                writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                    'PROBE_END_RESP', el_now, cnt, probeCount);
                enableBtns(ui, true);
            end
            continue
        end

        %% -- RESET button (E) --
        if getappdata(hFig,'reset_evt') > 0
            setappdata(hFig,'reset_evt', getappdata(hFig,'reset_evt') - 1);
            setappdata(hFig,'lastEvt', struct('type','RESET','pre_cnt',cnt,'time_s',el_s));

            cnt.consec_reset = cnt.consec_reset + 1;
            cnt.consec_resp  = 0;
            lastBtnTic       = tic();
            pausedSinceBtnS  = 0;
            eventCount = eventCount + 1;
            trigSend(SerialPortObj, EVT.resetBtn);
            writeLog(fidLog, el_s, 'RESET', sprintf('streak_E=%d', cnt.consec_reset));
            writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                'RESET', el_s, cnt, probeCount);
            setappdata(hFig,'cnt',cnt);

            % Nap2: reset streak -> probe
            if isNap2 && cnt.consec_reset >= cnt.reset_threshold
                probeCount = probeCount + 1;
                writeLog(fidLog, el_s, 'PROBE_TRIGGER', ...
                    sprintf('RESET_STREAK %d -> probe #%d', cnt.reset_threshold, probeCount));
                trigSend(SerialPortObj, EVT.resetProbe);
                doProbe(probeCount, 'reset_streak', subID, aCfg, hFig, ui, ...
                    false, audioRecDir, sessionTic, totalPaused_s, sessDur_s);
                trigSend(SerialPortObj, EVT.endprob);
                lastRespFlagSeen  = readFlagCounter(respFlagPath);   % drop presses made during the probe
                lastResetFlagSeen = readFlagCounter(resetFlagPath);

                % Threshold escalation: 3 -> 5 -> 8 (stays elevated in streak)
                if     cnt.reset_threshold < 5,  cnt.reset_threshold = 5;
                elseif cnt.reset_threshold < 8,  cnt.reset_threshold = 8;
                end
                cnt.consec_resp    = 0;
                cnt.consec_reset   = 0;
                % NOTE: reset_threshold NOT reset here (stays elevated within streak)
                cnt.last_probe_tic = tic();
                lastBtnTic = tic(); pausedSinceBtnS = 0;
                setappdata(hFig,'cnt',cnt); setappdata(hFig,'lastEvt',[]);
                set(ui.probeN,'String',num2str(probeCount));
                el_now = toc(sessionTic) - totalPaused_s;
                writeLog(fidLog, el_now, 'PROBE_END', ...
                    sprintf('#%d (reset_streak, new_thr=%d)', probeCount, cnt.reset_threshold));
                writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                    'PROBE_END_RESET', el_now, cnt, probeCount);
                enableBtns(ui, true);
            end
            continue
        end

        %% -- No-response timeout (90s ? all sessions) --
        noBtn_s = toc(lastBtnTic) - pausedSinceBtnS;
        if noBtn_s >= NO_RESP_TIMEOUT_S
            lastBtnTic = tic(); pausedSinceBtnS = 0;
            probeCount = probeCount + 1;
            eventCount = eventCount + 1;

            % For nap1: this probe ends the session (plays goodbye audio)
            isFinalProbeNow = strcmp(sessType, 'nap1');

            writeLog(fidLog, el_s, 'PROBE_TRIGGER', ...
                sprintf('NO_RESP_90s -> probe #%d  [%s]', probeCount, sessType));
            trigSend(SerialPortObj, EVT.behav);
            doProbe(probeCount, 'no_resp', subID, aCfg, hFig, ui, ...
                isFinalProbeNow, audioRecDir, sessionTic, totalPaused_s, sessDur_s);
            trigSend(SerialPortObj, EVT.endprob);
            lastRespFlagSeen  = readFlagCounter(respFlagPath);   % drop presses made during the probe
            lastResetFlagSeen = readFlagCounter(resetFlagPath);

            cnt.consec_resp     = 0;
            cnt.consec_reset    = 0;
            cnt.reset_threshold = 3;
            cnt.last_probe_tic  = tic();
            lastBtnTic = tic(); pausedSinceBtnS = 0;
            setappdata(hFig,'cnt',cnt); setappdata(hFig,'lastEvt',[]);
            set(ui.probeN,'String',num2str(probeCount));
            el_now = toc(sessionTic) - totalPaused_s;
            writeLog(fidLog, el_now, 'PROBE_END', sprintf('#%d (no_resp)', probeCount));
            writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                'PROBE_END_NORESP', el_now, cnt, probeCount);
            enableBtns(ui, true);

            if strcmp(sessType,'nap1')
                fprintf('Nap1: no-resp probe -> ending session.\n');
                skipFinalProbe = true;
                sessionEnded   = true;
            end
            continue
        end

        %% -- Manual probe (M or button) --
        if getappdata(hFig,'manualProbe')
            setappdata(hFig,'manualProbe',false);
            probeCount = probeCount + 1;
            eventCount = eventCount + 1;
            writeLog(fidLog, el_s, 'MANUAL_PROBE', sprintf('#%d', probeCount));
            trigSend(SerialPortObj, EVT.manual);
            doProbe(probeCount, 'manual', subID, aCfg, hFig, ui, ...
                false, audioRecDir, sessionTic, totalPaused_s, sessDur_s);
            trigSend(SerialPortObj, EVT.endprob);
            lastRespFlagSeen  = readFlagCounter(respFlagPath);   % drop presses made during the probe
            lastResetFlagSeen = readFlagCounter(resetFlagPath);

            cnt.consec_resp     = 0;
            cnt.consec_reset    = 0;
            cnt.reset_threshold = 3;
            cnt.last_probe_tic  = tic();
            lastBtnTic = tic(); pausedSinceBtnS = 0;
            setappdata(hFig,'cnt',cnt); setappdata(hFig,'lastEvt',[]);
            set(ui.probeN,'String',num2str(probeCount));
            el_now = toc(sessionTic) - totalPaused_s;
            writeLog(fidLog, el_now, 'PROBE_END', sprintf('#%d (manual)', probeCount));
            writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                'PROBE_END_MANUAL', el_now, cnt, probeCount);
            enableBtns(ui, true);
            continue
        end

        %% -- Nap2: 15-min safety probe --
        if isNap2
            cnt = getappdata(hFig,'cnt');
            if toc(cnt.last_probe_tic) >= SAFETY_INTERVAL_S
                probeCount = probeCount + 1;
                eventCount = eventCount + 1;
                writeLog(fidLog, el_s, 'PROBE_TRIGGER', ...
                    sprintf('SAFETY_15MIN -> probe #%d', probeCount));
                trigSend(SerialPortObj, EVT.safetyProbe);
                doProbe(probeCount, 'safety', subID, aCfg, hFig, ui, ...
                    false, audioRecDir, sessionTic, totalPaused_s, sessDur_s);
                trigSend(SerialPortObj, EVT.endprob);
                lastRespFlagSeen  = readFlagCounter(respFlagPath);   % drop presses made during the probe
                lastResetFlagSeen = readFlagCounter(resetFlagPath);

                cnt.consec_resp     = 0;
                cnt.consec_reset    = 0;
                cnt.reset_threshold = 3;
                cnt.last_probe_tic  = tic();
                lastBtnTic = tic(); pausedSinceBtnS = 0;
                setappdata(hFig,'cnt',cnt); setappdata(hFig,'lastEvt',[]);
                set(ui.probeN,'String',num2str(probeCount));
                el_now = toc(sessionTic) - totalPaused_s;
                writeLog(fidLog, el_now, 'PROBE_END', ...
                    sprintf('#%d (safety_15min)', probeCount));
                writeData(fidData, subID, sessIdx, sessType, eventCount, ...
                    'PROBE_END_SAFETY', el_now, cnt, probeCount);
                enableBtns(ui, true);
                continue
            end
        end

        pause(0.05);

    end  % monitoring loop

    %% -- Final probe --
    if isvalid(hFig) && ~skipFinalProbe
        probeCount = probeCount + 1;
        el_final   = toc(sessionTic) - totalPaused_s;
        writeLog(fidLog, el_final, 'FINAL_PROBE', sprintf('#%d', probeCount));
        trigSend(SerialPortObj, EVT.sched);
        doProbe(probeCount, 'final', subID, aCfg, hFig, ui, ...
            true, audioRecDir, sessionTic, totalPaused_s, sessDur_s);
        trigSend(SerialPortObj, EVT.endprob);
        el_now = toc(sessionTic) - totalPaused_s;
        writeLog(fidLog, el_now, 'PROBE_END', sprintf('#%d (final)', probeCount));
        writeData(fidData, subID, sessIdx, sessType, eventCount, ...
            'PROBE_END_FINAL', el_now, cnt, probeCount);
    end

    %% -- Session end --
    el_end = toc(sessionTic) - totalPaused_s;
    trigSend(SerialPortObj, EVT.endsess);
    writeLog(fidLog, el_end, 'SESSION_END', ...
        sprintf('probes=%d  paused=%.0fs', probeCount, totalPaused_s));
    writeData(fidData, subID, sessIdx, sessType, eventCount, ...
        'SESSION_END', el_end, cnt, probeCount);

    % ---- Response time summary ----
    fprintf('\n--- Session %d  RT Summary ---\n', sessIdx);
    if numel(respTimes) > 0
        fprintf('  N responses : %d\n',              numel(respTimes));
        fprintf('  Mean IRT    : %.2f s\n',          mean(respTimes));
        fprintf('  Median IRT  : %.2f s\n',          median(respTimes));
        fprintf('  Min / Max   : %.2f / %.2f s\n',   min(respTimes), max(respTimes));
        fprintf('  Std dev     : %.2f s\n',           std(respTimes));
    else
        fprintf('  No responses recorded.\n');
    end
    fprintf('-----------------------------\n\n');

    fclose(fidLog);
    fclose(fidData);
    if isvalid(hFig)
        set(hFig,'CloseRequestFcn','closereq');
        close(hFig);
    end

    fprintf('=== Session %d done | %d probes\n', sessIdx, probeCount);
    fprintf('    Log : %s\n', logPath);
    fprintf('    Data: %s\n\n', dataPath);

    %% -- Inter-session break --
    if sessIdx < totalSessions
        ans_ = questdlg( ...
            sprintf('Session %d complete. Continue to session %d?', sessIdx, sessIdx+1), ...
            'Break','Continue','Stop','Continue');
        if strcmp(ans_,'Stop')
            fprintf('Experiment stopped after session %d.\n', sessIdx);
            break
        end
        pause(2);
    end

end  % session loop

%% ===================== FINAL CLEANUP ============================
if exist('hFig','var') && ~isempty(hFig) && isvalid(hFig)
    set(hFig,'CloseRequestFcn','closereq');
    close(hFig);
end
if ~isempty(SerialPortObj)
    try; fclose(SerialPortObj); catch; end
    try; delete(SerialPortObj); catch; end
end
try; instrreset; catch; end
clear SerialPortObj;

fprintf('\n=========================================\n');
fprintf('   ALL SESSIONS COMPLETE!\n');
fprintf('   Participant : sub%s\n', subID);
fprintf('   Data folder : %s\n', data_path);
fprintf('   Subject     : %s\n', subFolder);
fprintf('=========================================\n\n');

%% ================================================================
%                       LOCAL FUNCTIONS
%% ================================================================

% -----------------------------------------------------------------
function doProbe(probeNum, probeType, subID, aCfg, hFig, ui, ...
                 isFinal, outDir, sessionTic, totalPaused_s, sessDur_s)
% Alarm + continuous recording + questionnaire
% isFinal = true  -> plays lastFinal audio on last question
% isFinal = false -> plays lastEnd   audio on last question

    setappdata(hFig,'mode','probe');
    enableBtns(ui, false);

    tag = upper(probeType);
    set(ui.status,'String',sprintf('PROBE #%d [%s] ? alarm...', probeNum, tag), ...
        'ForegroundColor',[1 0.4 0.2]);
    drawnow;

    % Alarm
    try; playblocking(audioplayer(aCfg.alarmY, aCfg.alarmFs)); catch; beep; beep; end
    pause(2);

    % Recording
    if ~exist(outDir,'dir'), mkdir(outDir); end
    recObj  = audiorecorder(aCfg.fsRec, aCfg.nBits, aCfg.nCh, aCfg.devID);
    aborted = false;
    record(recObj);

    % Question loop
    for q = 1:aCfg.nQ
        if ~isvalid(hFig) || getappdata(hFig,'endSession')
            aborted = true; break;
        end
        isLast = (q == aCfg.nQ);
        if isLast && isFinal,  qFile = aCfg.lastFinal;
        elseif isLast,         qFile = aCfg.lastEnd;
        else,                  qFile = aCfg.questions{q};
        end
        [~, qName] = fileparts(qFile);
        set(ui.status,'String', ...
            sprintf('Probe #%d [%s]  ?  Q%d/%d   %s   (SPACE)', ...
            probeNum, tag, q, aCfg.nQ, strrep(qName,'_',' ')), ...
            'ForegroundColor',[1 .65 .1]);
        drawnow;

        % waitSpace updates session timer display while waiting
        stopNow = waitSpace(hFig, ui, sessionTic, totalPaused_s, sessDur_s);
        if stopNow; aborted = true; break; end

        try
            [qY, qFs] = audioread(fullfile(aCfg.path, qFile));
            playblocking(audioplayer(qY, qFs));
        catch
            warning('Cannot play: %s', qFile);
        end
    end

    try; stop(recObj); catch; end
    y = getaudiodata(recObj, 'double');

    suffix  = ternary(aborted, '_ABORTED', '');
    outFile = fullfile(outDir, sprintf('sub%s_probe%02d_%s%s.wav', ...
        subID, probeNum, probeType, suffix));
    try
        audiowrite(outFile, y, aCfg.fsRec, 'BitsPerSample', 16);
        fprintf('  Recording saved -> %s\n', outFile);
    catch ME
        warning('Audio save failed: %s', ME.message);
    end
    setappdata(hFig,'mode','monitor');
end

% -----------------------------------------------------------------
function stopNow = waitSpace(hFig, ui, sessionTic, totalPaused_s, sessDur_s)
% Wait for SPACE; keeps session timer running on screen
    stopNow = false;
    setappdata(hFig,'spacePressed',false);
    figure(hFig); drawnow;
    while ~getappdata(hFig,'spacePressed')
        pause(0.05); drawnow;
        el = toc(sessionTic) - totalPaused_s;
        set(ui.sessTime,'String', ...
            sprintf('%s / %s', fmtTime(el), fmtTime(sessDur_s)));
        drawnow limitrate;
        if ~isvalid(hFig) || getappdata(hFig,'endSession')
            stopNow = true; return;
        end
    end
end

% -----------------------------------------------------------------
function keyHandler(hFig, evt)
% R=response  E=reset  Z=undo  M=manual probe  P=pause  SPACE=next question
    mode   = getappdata(hFig,'mode');
    paused = getappdata(hFig,'isPaused');
    if strcmpi(evt.Key,'p') && strcmp(mode,'monitor')
        setappdata(hFig,'togglePause',true); return;
    end
    switch mode
        case 'monitor'
            if paused, return; end
            switch lower(evt.Key)
                case 'r', setappdata(hFig,'resp',      getappdata(hFig,'resp')      + 1);
                case 'e', setappdata(hFig,'reset_evt', getappdata(hFig,'reset_evt') + 1);
                case 'z', setappdata(hFig,'undoReq',       true);
                case 'm', setappdata(hFig,'manualProbe',   true);
                case 'escape', setappdata(hFig,'emergencyExit', true);
            end
        case 'probe'
            if strcmpi(evt.Key,'space')
                setappdata(hFig,'spacePressed',true);
            end
    end
end

% -----------------------------------------------------------------
function [hFig, ui] = buildGUI(sessName, sessDur_min, isNap2)
    W = 560; H = 570;
    ss = get(0,'ScreenSize');
    bg = [0.10 0.10 0.16];
    fg = [0.85 0.85 0.92];

    hFig = figure('Name',sprintf('TickToz ? %s', sessName), ...
        'NumberTitle','off','MenuBar','none','ToolBar','none', ...
        'Resize','off','Color',bg, ...
        'Position',[(ss(3)-W)/2, (ss(4)-H)/2, W, H], ...
        'CloseRequestFcn', @(src,~) setappdata(src,'emergencyExit',true), ...
        'KeyPressFcn',     @(src,evt) keyHandler(src,evt));

    mk = @(pos,str) uicontrol('Parent',hFig,'Style','text','String',str,...
        'Units','normalized','Position',pos,'FontSize',11,...
        'ForegroundColor',fg,'BackgroundColor',bg,'HorizontalAlignment','left');

    mk([.04 .91 .36 .055], 'Session time:');
    ui.sessTime = mk([.40 .91 .56 .055], sprintf('00:00:00 / %02d:00:00', sessDur_min));
    set(ui.sessTime,'FontSize',14,'FontWeight','bold','ForegroundColor',[.4 1 .4]);

    mk([.04 .83 .36 .055], 'No response:');
    ui.noResp = mk([.40 .83 .22 .055], '00:00');
    set(ui.noResp,'FontSize',14,'FontWeight','bold','ForegroundColor',[1 1 1]);

    ui.barBg = uicontrol('Style','text','Parent',hFig,'Units','normalized',...
        'Position',[.65 .84 .30 .033],'BackgroundColor',[.25 .25 .30]);
    ui.bar   = uicontrol('Style','text','Parent',hFig,'Units','normalized',...
        'Position',[.65 .84 .001 .033],'BackgroundColor',[.3 .9 .3]);

    mk([.04 .75 .36 .055], 'Probes sent:');
    ui.probeN = mk([.40 .75 .15 .055], '0');
    set(ui.probeN,'FontSize',12,'FontWeight','bold','ForegroundColor',[1 .85 .3]);

    if isNap2
        ui.streakInfo = mk([.04 .67 .92 .055], 'streak:  R=0/3   E=0/3');
        set(ui.streakInfo,'FontSize',10,'ForegroundColor',[.7 .85 1], ...
            'HorizontalAlignment','center');
    else
        ui.streakInfo = mk([.04 .67 .92 .055], '');
        set(ui.streakInfo,'ForegroundColor',bg,'BackgroundColor',bg);
    end

    ui.status = mk([.04 .59 .92 .055], 'Ready');
    set(ui.status,'FontSize',12,'FontWeight','bold', ...
        'HorizontalAlignment','center','ForegroundColor',[.3 .95 .3]);

    ui.resp = uicontrol('Parent',hFig,'Style','pushbutton','String','RESPONSE  (R)',...
        'Units','normalized','Position',[.04 .44 .44 .12],'FontSize',13,'FontWeight','bold',...
        'BackgroundColor',[.18 .58 .28],'ForegroundColor',[1 1 1],...
        'Callback',@(~,~) setappdata(hFig,'resp', getappdata(hFig,'resp') + 1));

    ui.reset = uicontrol('Parent',hFig,'Style','pushbutton','String','RESET  (E)',...
        'Units','normalized','Position',[.52 .44 .44 .12],'FontSize',13,'FontWeight','bold',...
        'BackgroundColor',[.58 .48 .12],'ForegroundColor',[1 1 1],...
        'Callback',@(~,~) setappdata(hFig,'reset_evt', getappdata(hFig,'reset_evt') + 1));

    ui.undoBtn = uicontrol('Parent',hFig,'Style','pushbutton', ...
        'String','UNDO LAST (Z)  ?  timer unchanged',...
        'Units','normalized','Position',[.04 .31 .92 .10],'FontSize',11,'FontWeight','bold',...
        'BackgroundColor',[.65 .22 .65],'ForegroundColor',[1 1 1],...
        'Callback',@(~,~) setappdata(hFig,'undoReq',true));

    ui.pauseBtn = uicontrol('Parent',hFig,'Style','pushbutton','String','PAUSE  (P)',...
        'Units','normalized','Position',[.04 .19 .92 .09],'FontSize',11,'FontWeight','bold',...
        'BackgroundColor',[.52 .42 .10],'ForegroundColor',[1 1 1],...
        'Callback',@(~,~) setappdata(hFig,'togglePause',true));

    ui.manual = uicontrol('Parent',hFig,'Style','pushbutton','String','MANUAL PROBE  (M)',...
        'Units','normalized','Position',[.04 .03 .44 .13],'FontSize',10,'FontWeight','bold',...
        'BackgroundColor',[.65 .22 .22],'ForegroundColor',[1 1 1],...
        'Callback',@(~,~) setappdata(hFig,'manualProbe',true));

    ui.endBtn = uicontrol('Parent',hFig,'Style','pushbutton','String','END SESSION',...
        'Units','normalized','Position',[.52 .03 .44 .13],'FontSize',10,'FontWeight','bold',...
        'BackgroundColor',[.48 .12 .12],'ForegroundColor',[1 1 1],...
        'Callback',@(~,~) setappdata(hFig,'endSession',true));
end

% -----------------------------------------------------------------
function updateGUI(ui, el_s, noBtn_s, timeout_s, sessDur_s, cnt, isNap2)
    set(ui.sessTime,'String',sprintf('%s / %s', fmtTime(el_s), fmtTime(sessDur_s)));
    set(ui.noResp,  'String',sprintf('%02d:%02d', floor(noBtn_s/60), floor(mod(noBtn_s,60))));

    frac = min(noBtn_s / timeout_s, 1);
    bp   = get(ui.barBg,'Position');
    set(ui.bar,'Position',[bp(1) bp(2) max(bp(3)*frac, 0.001) bp(4)]);

    if noBtn_s > timeout_s * 0.83
        set(ui.noResp,'ForegroundColor',[1 .15 .15]); set(ui.bar,'BackgroundColor',[1  .2  .2]);
    elseif noBtn_s > timeout_s * 0.67
        set(ui.noResp,'ForegroundColor',[1  .6 .15]); set(ui.bar,'BackgroundColor',[1  .6 .15]);
    else
        set(ui.noResp,'ForegroundColor',[1  1   1 ]); set(ui.bar,'BackgroundColor',[.3 .9  .3]);
    end

    if isNap2 && ~isempty(cnt)
        set(ui.streakInfo,'String', ...
            sprintf('streak:   R = %d / %d      E = %d / %d', ...
            cnt.consec_resp,  cnt.resp_threshold, ...
            cnt.consec_reset, cnt.reset_threshold));
    end
end

% -----------------------------------------------------------------
function enableBtns(ui, state)
    v = ternary(state,'on','off');
    set([ui.resp ui.reset ui.manual ui.endBtn ui.undoBtn],'Enable',v);
    set(ui.pauseBtn,'Enable','on');   % PAUSE always available
end

% -----------------------------------------------------------------
function trigSend(s, code)
% Send trigger pulse: value -> 0 (clean pulse, EEG can distinguish repeats)
    if isempty(s)
        fprintf('[SIMU] trigger %d\n', code);
        return;
    end
    try
        fwrite(s, code, 'uint8');
        pause(0.010);
        fwrite(s, 0, 'uint8');
        pause(0.010)
    catch ME
        warning('Trigger %d failed: %s', code, ME.message);
    end
end

% -----------------------------------------------------------------
function writeLog(fid, t, evt, det)
    fprintf(fid, '%s,%.1f,%s,%s\n', datestr(now,'HH:MM:SS'), t, evt, det);
    fprintf('[%s | %s]  %-26s  %s\n', datestr(now,'HH:MM:SS'), fmtTime(t), evt, det);
end

% -----------------------------------------------------------------
function writeData(fid, subID, sessIdx, sessType, evtN, evtType, t, cnt, probeN)
    fprintf(fid, '%s,%d,%s,%d,%s,%s,%.1f,%d,%d,%d,%d,%d\n', ...
        subID, sessIdx, sessType, evtN, evtType, ...
        datestr(now,'HH:MM:SS'), t, ...
        cnt.consec_resp, cnt.consec_reset, ...
        cnt.resp_threshold, cnt.reset_threshold, probeN);
end

% -----------------------------------------------------------------
function s = fmtTime(sec)
    s = sprintf('%02d:%02d:%02d', ...
        floor(sec/3600), floor(mod(sec,3600)/60), floor(mod(sec,60)));
end

% -----------------------------------------------------------------
function out = ternary(cond, yes, no)
    if cond, out = yes; else, out = no; end
end

% -----------------------------------------------------------------
function n = readFlagCounter(p)
    n = 0;
    try
        fid = fopen(p, 'r');
        if fid ~= -1
            n = str2double(fgetl(fid));
            fclose(fid);
            if isnan(n), n = 0; end
        end
    catch
        n = 0;
    end
end
