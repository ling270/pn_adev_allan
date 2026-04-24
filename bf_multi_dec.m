clc; clear; close all;

%% ===================== 0. 文件 =====================
filename = 'datasource4/phase_file/phi34.dat'; % ⭐ 输入：相位文件

if ~exist(filename, 'file')
    error('找不到文件: %s', filename);
end

%% ===================== 1. 参数 =====================
FS = 133e6;
CIC_R = 64;
Fs_start = FS / CIC_R;

DecFactor = 10;
TotalStages = 4;

% ⭐ 自定义保存级数（最后 m 级）
saveLastNStages = 4;

%% 分块
chunkSizeBytes = 100 * 1024 * 1024;
chunkSizePoints = floor(chunkSizeBytes / 8);

%% 输出目录
outDir = 'datasource4/phase_decimated';
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ===================== 2. 滤波器瞬态测量 =====================
fprintf('测量滤波器参数...\n');

impulse_len = 6000;
impulse = [1; zeros(impulse_len-1,1)];
y = MULTISPEED(impulse);

[~, peak] = max(abs(y));
group_delay = peak - 1;

threshold = 1e-6 * max(abs(y));
impulse_len2 = find(abs(y) > threshold,1,'last');

TransientGuard = impulse_len2 + group_delay;

fprintf('群延迟: %d\n', group_delay);
fprintf('瞬态保护区: %d\n', TransientGuard);

%% ===================== 3. 文件打开 =====================
fid_in = fopen(filename,'rb');

fseek(fid_in, 0, 'eof');
fileSize = ftell(fid_in);
fseek(fid_in, 0, 'bof');

startSaveStage = TotalStages - saveLastNStages + 1;

fid_out = cell(1,TotalStages);
for i = 1:TotalStages
    if i >= startSaveStage
        fname = fullfile(outDir,...
            sprintf('phi34_stage_%d_%.0fHz.dat', i, Fs_start/(DecFactor^i)));
        fid_out{i} = fopen(fname,'wb');
        fprintf('保存 Stage %d → %s\n', i, fname);
    end
end

%% ===================== 4. 初始化 =====================
dataBuffer = zeros(TransientGuard,1);

% ⭐ 绘图缓存
plotData = cell(1,TotalStages);

block = 0;

fprintf('\n开始处理...\n');

%% ===================== 5. 主循环 =====================
while ~feof(fid_in)

    block = block + 1;

    data = fread(fid_in, chunkSizePoints, 'double');
    if isempty(data), break; end

    % 拼接
    current = [dataBuffer; data];

    temp = current;

    for st = 1:TotalStages

        % ⭐ 多级滤波（直接调用，不用clone）
        y = MULTISPEED(temp);

        % ⭐ 丢瞬态
        if length(y) > TransientGuard
            y_valid = y(TransientGuard+1:end);
        else
            y_valid = [];
        end

        % ⭐ 写文件
        if st >= startSaveStage && ~isempty(y_valid)
            fwrite(fid_out{st}, y_valid, 'double');
        end

        % ⭐ 缓存用于绘图（自动抽样）
        if ~isempty(y_valid)
            step = max(1, floor(length(y_valid)/5000));
            plotData{st} = [plotData{st}; y_valid(1:step:end)];
        end

        temp = y_valid;
    end

    % 更新buffer
    if length(current) >= TransientGuard
        dataBuffer = current(end-TransientGuard+1:end);
    else
        dataBuffer = current;
    end

    % 进度
    if mod(block,10)==0 || feof(fid_in)
        progress = ftell(fid_in)/fileSize*100;
        fprintf('进度: %.1f%% (块 %d)\n', progress, block);
    end
end

%% ===================== 6. 收尾 =====================
fclose(fid_in);
for i = 1:TotalStages
    if ~isempty(fid_out{i}), fclose(fid_out{i}); end
end

fprintf('\n数据处理完成\n');

%% ===================== 7. 绘图 =====================
figure('Color','w','Name','各级相位波动');

valid = 0;
for i = 1:TotalStages
    if ~isempty(plotData{i}), valid = valid + 1; end
end

idx = 1;
for i = 1:TotalStages
    if isempty(plotData{i}), continue; end

    fs = Fs_start / (DecFactor^i);
    t = (0:length(plotData{i})-1)/fs;

    subplot(valid,1,idx);
    plot(t, plotData{i}, 'LineWidth',1);
    grid on;
    title(sprintf('Stage %d 相位 (Fs=%.2f Hz)',i,fs));
    xlabel('Time (s)');
    ylabel('Phase');

    idx = idx + 1;
end

fprintf('🎉 全部完成！\n');