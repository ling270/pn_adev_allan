clc; clear; close all;

%% ===================== 1. 参数设置 =====================
filePath = "same1_FDR_two_10M_Ref_10M_DUT_20260421_204638.dat";

FS = 133000000;
CIC_R = 64;
Fs_base = FS / CIC_R;

chunkInt32 = 12500000;
chunkInt32 = floor(chunkInt32/4)*4;

outDir = "datasource4\phase_file";
if ~exist(outDir,'dir'), mkdir(outDir); end

phi12_file = fullfile(outDir,"phi12.dat");
phi34_file = fullfile(outDir,"phi34.dat");

%% ===================== 2. 文件信息 =====================
fid = fopen(filePath,"rb");
if fid < 0, error("无法打开文件"); end

fseek(fid,0,"eof");
fileSizeBytes = ftell(fid);
totalInt32 = floor((fileSizeBytes/4)/4)*4;
totalFrames = totalInt32/4;
fseek(fid,0,"bof");

fprintf("📊 总帧数: %d\n", totalFrames);

%% ===================== 3. 初始化 =====================
fd12 = fopen(phi12_file,"wb");
fd34 = fopen(phi34_file,"wb");

done = 0;
tprint = tic;

% ⭐ 用于去直流（滑动均值）
mean_df12 = 0;
mean_df34 = 0;
alpha = 1e-6;   % 慢速更新均值（避免低频被误删）

% ⭐ 相位连续
phi_last12 = 0;
phi_last34 = 0;

%% ===================== 4. 主循环 =====================
while ~feof(fid)

    raw = fread(fid, chunkInt32, "int32=>int32", 0, "b");
    if isempty(raw), break; end

    n = floor(numel(raw)/4)*4;
    raw = raw(1:n);

    B1 = swapbytes(reshape(raw,1,[]));
    CC = reshape(B1,4,[]);
    M = size(CC,2);

    % ===== 4通道 =====
    CH1 = double(CC(4,:))'/(2^32)*Fs_base;
    CH2 = double(CC(3,:))'/(2^32)*Fs_base;
    CH3 = double(CC(2,:))'/(2^32)*Fs_base;
    CH4 = double(CC(1,:))'/(2^32)*Fs_base;

    % ===== 频率差 =====
    df12 = CH2 - CH1;
    df34 = CH3 - CH4;

    %% ⭐ 去直流（关键修改）
    % 使用“慢均值”，避免把低频当成DC删掉
    mean_df12 = (1-alpha)*mean_df12 + alpha*mean(df12);
    mean_df34 = (1-alpha)*mean_df34 + alpha*mean(df34);

    df12 = df12 - mean_df12;
    df34 = df34 - mean_df34;

    %% ⭐ 转相位（连续积分）
    dphi12 = 2*pi*df12 / Fs_base;
    dphi34 = 2*pi*df34 / Fs_base;

    phi12 = phi_last12 + cumsum(dphi12);
    phi34 = phi_last34 + cumsum(dphi34);

    phi_last12 = phi12(end);
    phi_last34 = phi34(end);

    %% ⭐ 保存（不做 detrend）
    fwrite(fd12, phi12, "double");
    fwrite(fd34, phi34, "double");

    done = done + M;

    %% 进度
    if toc(tprint) > 2
        fprintf("进度: %.1f%%\n", 100*done/totalFrames);
        tprint = tic;
    end
end

fclose(fid);
fclose(fd12);
fclose(fd34);

fprintf("\n✅ 相位数据已保存：\n");
fprintf(" - %s\n", phi12_file);
fprintf(" - %s\n", phi34_file);