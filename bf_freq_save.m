clc; clear; close all;

%% ===================== 1. 参数设置 =====================
filePath = "same1_FDR_two_10M_Ref_10M_DUT_20260421_204638.dat";

FS = 133000000;
CIC_R = 64;
Fs_base = FS / CIC_R; % 2,078,125 Hz

% ⭐ 修改点：定义绘图窗口（仅用于预览）
plot_time_start = 4;   
plot_time_end   = 50;  

% 计算绘图窗口对应的采样点索引
plot_start_idx = plot_time_start * Fs_base + 1; 
plot_end_idx   = plot_time_end * Fs_base;

% ⭐ 修改点：不再设置处理上限，处理整个文件
% maxFrames 变量被移除，循环将直到文件结束

% 分块大小：每次处理约 50MB，适应大文件
chunkInt32 = 12500000;  % 12.5M 个 int32 ≈ 50MB
chunkInt32 = floor(chunkInt32/4)*4;  % 确保是4的倍数

% 输出目录
outDir = "datasource4\freq_file";
if ~exist(outDir,'dir'), mkdir(outDir); end

df12_file = fullfile(outDir,"df12_smae1.dat"); % 全量数据文件
df34_file = fullfile(outDir,"df34_smae1.dat");

%% ===================== 2. 文件信息 =====================
fid = fopen(filePath,"rb");
if fid < 0, error("无法打开文件 %s", filePath); end

fseek(fid,0,"eof");
fileSizeBytes = ftell(fid);
fileSizeGB = fileSizeBytes / (1024^3);
fprintf("📂 文件大小: %.2f GB\n", fileSizeGB);

totalInt32 = floor((fileSizeBytes/4)/4)*4;
totalFrames = totalInt32/4;
fseek(fid,0,"bof");

fprintf("📊 总帧数: %d\n", totalFrames);
fprintf("👁️ 预览范围: %.1fs - %.1fs (索引: %d - %d)\n", plot_time_start, plot_time_end, plot_start_idx, plot_end_idx);

%% ===================== 3. 主循环：读取、处理、写入 & 绘图采集 =====================
% 预分配绘图内存 (只存 4-50s 的数据)
plot_duration = plot_time_end - plot_time_start;
plot_samples_max = plot_duration * Fs_base;
% 为了绘图流畅，这里依然采用降采样显示，或者直接存全量（如果内存够）
% 这里我们存全量用于绘图，因为46秒的数据约 95MB，内存完全吃得消
df12_plot = zeros(plot_samples_max, 1);
df34_plot = zeros(plot_samples_max, 1);

% 打开输出文件
fd12 = fopen(df12_file,"wb");
fd34 = fopen(df34_file,"wb");
if fd12 < 0 || fd34 < 0, error("无法创建输出文件"); end

done = 0;         % 当前已处理的总点数
plot_idx = 1;     % 绘图数组的索引
tprint = tic;

while ~feof(fid)
    % 读取数据块
    raw = fread(fid, chunkInt32, "int32=>int32", 0, "b");
    if isempty(raw), break; end

    % 确保数据是4的倍数
    n = floor(numel(raw)/4)*4;
    raw = raw(1:n);

    % 字节交换和重塑
    B1 = swapbytes(reshape(raw,1,[]));
    CC = reshape(B1,4,[]);
    M = size(CC,2);

    % 提取4个通道的数据
    % 注意：这里一次性转为double计算，精度足够
    CH1 = double(CC(4,:))'/(2^32)*Fs_base;
    CH2 = double(CC(3,:))'/(2^32)*Fs_base;
    CH3 = double(CC(2,:))'/(2^32)*Fs_base;
    CH4 = double(CC(1,:))'/(2^32)*Fs_base;
    
    % 计算频率差
    df12 = CH2 - CH1;
    df34 = CH3 - CH4;

    %% ⭐ 关键逻辑：写入文件（全量写入）
    fwrite(fd12, df12, "double");
    fwrite(fd34, df34, "double");

    %% ⭐ 关键逻辑：采集绘图数据（仅 4s-50s）
    % 计算当前块在绘图范围内的交集
    % 当前块的索引范围: [done+1, done+M]
    % 绘图目标范围: [plot_start_idx, plot_end_idx]
    
    block_start = done + 1;
    block_end = done + M;
    
    % 如果当前块与绘图范围有交集
    if block_start <= plot_end_idx && block_end >= plot_start_idx
        % 计算交集的起始和结束位置（在当前块 df12 中的索引）
        intersect_start = max(1, plot_start_idx - done);
        intersect_end = min(M, plot_end_idx - done);
        
        % 提取数据存入绘图数组
        len_to_copy = intersect_end - intersect_start + 1;
        if len_to_copy > 0
            % 确保不超出绘图数组边界（防止万一计算错误）
            if plot_idx + len_to_copy - 1 <= plot_samples_max
                df12_plot(plot_idx : plot_idx + len_to_copy - 1) = df12(intersect_start : intersect_end);
                df34_plot(plot_idx : plot_idx + len_to_copy - 1) = df34(intersect_start : intersect_end);
                plot_idx = plot_idx + len_to_copy;
            end
        end
    end

    done = done + M;
    
    % 进度显示
    if toc(tprint) > 2
        progress = 100 * done / totalFrames;
        fprintf("处理进度: %.1f%% (%.1f GB / %.1f GB)\n", progress, done*4*4/1024^3, fileSizeGB);
        tprint = tic;
    end
end

% 关闭文件句柄
fclose(fid); 
fclose(fd12); 
fclose(fd34);

% 裁剪绘图数组到实际大小（去除预分配的空余部分）
df12_plot = df12_plot(1 : plot_idx-1);
df34_plot = df34_plot(1 : plot_idx-1);

%% ===================== 4. 绘图验证 =====================
% 生成时间轴（从 4秒 开始）
tt = ((1:length(df12_plot))' - 1) / Fs_base + plot_time_start;

figure('Position', [100, 100, 1200, 800], 'Color', 'w');

subplot(2,1,1);
plot(tt, df12_plot, 'b-', 'LineWidth', 0.8);
grid on;
title(sprintf('CH2 - CH1 频率差 (预览: %.0fs - %.0fs)', plot_time_start, plot_time_end), 'FontSize', 14, 'FontWeight', 'bold');
xlabel('时间 (s)');
ylabel('频率差 (Hz)');

subplot(2,1,2);
plot(tt, df34_plot, 'r-', 'LineWidth', 0.8);
grid on;
title(sprintf('CH3 - CH4 频率差 (预览: %.0fs - %.0fs)', plot_time_start, plot_time_end), 'FontSize', 14, 'FontWeight', 'bold');
xlabel('时间 (s)');
ylabel('频率差 (Hz)');

fprintf("\n✅ 任务完成！\n");
fprintf("📂 全量数据已保存至:\n");
fprintf("   - %s\n", df12_file);
fprintf("   - %s\n", df34_file);
fprintf("👁️ 预览图显示范围: %.1fs - %.1fs (共 %d 个点)\n", plot_time_start, plot_time_end, length(df12_plot));