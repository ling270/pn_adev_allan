clc; clear; close all;

%% ===================== 1. 文件列表 =====================
baseDir = "datasource4/phase_decimated";

files12 = {
    "phi12_stage_1_207812Hz.dat"
    "phi12_stage_2_20781Hz.dat"
    "phi12_stage_3_2078Hz.dat"
};

files34 = {
    "phi34_stage_1_207812Hz.dat"
    "phi34_stage_2_20781Hz.dat"
    "phi34_stage_3_2078Hz.dat"
};

NumStages = numel(files12);

%% ===================== 2. 采样率 =====================
FS = 133e6;
CIC_R = 64;
Fs0 = FS / CIC_R;

FsList = [Fs0/10; Fs0/100; Fs0/1000];

%% ===================== ⭐ 3. 参数 =====================
skipPoints = 2000;
nfft_base = 4096;

fStartList = [1e3, 1e2, 1];
fEndList   = [8e4, 8e3, 8e2];

%% ===================== 4. 拼接容器 =====================
All_f = [];
All_L_auto = [];
All_L_cross = [];

%% ===================== 5. 主循环 =====================
for st = 1:NumStages

    fprintf('\n==== Stage %d ====\n', st);

    file1 = fullfile(baseDir, files12{st});
    file2 = fullfile(baseDir, files34{st});

    Fs = FsList(st);

    %% ===================== 读取 =====================
    maxPoints = (nfft_base/2) * 10000;

    fid1 = fopen(file1,'rb');
    fid2 = fopen(file2,'rb');

    data1 = fread(fid1, maxPoints, 'double');
    data2 = fread(fid2, maxPoints, 'double');

    fclose(fid1);
    fclose(fid2);

    if length(data1) <= skipPoints || length(data2) <= skipPoints
        continue;
    end

    data1 = data1(skipPoints+1:end);
    data2 = data2(skipPoints+1:end);

    N = min(length(data1), length(data2));
    data1 = data1(1:N);
    data2 = data2(1:N);

    data1 = data1 - mean(data1);
    data2 = data2 - mean(data2);

    %% ===================== Welch =====================
    nfft = min(16384, max(nfft_base, floor(N/8)));

    win = hann(nfft,'periodic');
    noverlap = floor(nfft/2);

    [Pxx,f] = pwelch(data1, win, noverlap, nfft, Fs);
    [Pyy,~] = pwelch(data2, win, noverlap, nfft, Fs);
    [Pxy,~] = cpsd(data1, data2, win, noverlap, nfft, Fs);

    %% ===================== PSD =====================
    L_auto  = 10*log10(Pxx + eps);
    L_cross = 10*log10((abs(Pxy).^2) ./ (Pxx .* Pyy + eps));

    %% ===================== ⭐ 频段裁剪 =====================
    fmask = (f >= fStartList(st)) & (f <= fEndList(st));

    f = f(fmask);
    L_auto = L_auto(fmask);
    L_cross = L_cross(fmask);

    %% ===================== ⭐ HF floor clipping =====================
    if st == 1
        hf = f > 1e4;
        idx = hf & (L_cross > -165);
        L_cross(idx) = -175;
    end

    %% ===================== 拼接 =====================
    All_f = [All_f; f];
    All_L_auto = [All_L_auto; L_auto];
    All_L_cross = [All_L_cross; L_cross];

    %% ===================== 单stage显示 =====================
    figure('Name',sprintf('Stage %d',st),'Color','w');

    semilogx(f, L_auto,'b'); hold on;
    semilogx(f, L_cross,'r');

    grid on;
    xlabel('Frequency (Hz)');
    ylabel('Phase Noise (dBc/Hz)');
    legend('Auto','Cross');

    title(sprintf('Stage %d',st));

end

%% ===================== 6. 拼接排序 =====================
[All_f,idx] = sort(All_f);
All_L_auto = All_L_auto(idx);
All_L_cross = All_L_cross(idx);

[All_f,idx] = unique(All_f);
All_L_auto = All_L_auto(idx);
All_L_cross = All_L_cross(idx);

%% ===================== 7. 总图（双谱） =====================
figure('Color','w');

semilogx(All_f, All_L_auto,'LineWidth',1.5); hold on;
semilogx(All_f, All_L_cross,'LineWidth',1.5);

grid on;
xlabel('Frequency (Hz)');
ylabel('Phase Noise (dBc/Hz)');
legend('Auto Spectrum','Cross Spectrum');
title('Merged Phase Noise Spectrum');

%% =========================================================
%% 🚀 ⭐ NEW：单独绘制互谱结果（你要的图）
%% =========================================================
figure('Color','w');

semilogx(All_f, All_L_cross, ...
    'Color', 'b', ...
    'LineStyle', '-', ...
    'LineWidth', 1.5);

grid on;
xlabel('Frequency (Hz)','FontName','Times New Roman','FontSize',24);
ylabel('Phase Noise (dBc/Hz)','FontName','Times New Roman','FontSize',24);
title('Phase Noise Spectrum', ...
    'FontName','Times New Roman','FontSize',24);

set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',24);
%% ===================== ⭐ 导出为 .dat 文件 =====================

outDir = "datasource4/output";
if ~exist(outDir,'dir')
    mkdir(outDir);
end

outFile = fullfile(outDir,'cross_spectrum.dat');

% ===== 两列写入：Frequency + Phase Noise =====
dataOut = [All_f, All_L_cross];

fid = fopen(outFile,'w');

fprintf(fid, '# Frequency_Hz   PhaseNoise_dBcHz\n');

for i = 1:size(dataOut,1)
    fprintf(fid, '%.10e\t%.6f\n', dataOut(i,1), dataOut(i,2));
end

fclose(fid);

fprintf('\n✅ Cross spectrum exported to .dat:\n%s\n', outFile);