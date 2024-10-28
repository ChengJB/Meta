% 清除工作区和关闭所有图形窗口
clear all;
close all;

% 创建文件夹
mainFolder = 'Channel_estimation';
if ~exist(mainFolder, 'dir')
    mkdir(mainFolder);
end

% 使用随机生成数据模拟仿真数据
SNR_Matrix = rand(16, 10); % 生成随机的 SNR 矩阵
BLER_Matrix = rand(16, 10); % 生成随机的 BLER 矩阵
BER_Matrix = rand(16, 10); % 生成随机的 BER 矩阵
tao_range = 0:0.1:1; % 生成时延范围
Channel_MSE_total_mean_Matrix = rand(16, 10); % 生成随机的信道均方误差总均值矩阵
chan_type_range = {'Type1', 'Type2'}; % 信道类型范围
chest_method_range = {'Method1', 'Method2'}; % 信道估计方法范围
CIR_Thr_range = 0:0.1:1; % CIR 阈值范围

% 生成唯一的文件名（方便测试）
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['OMA_Data_' timestamp '.mat'];

% 创建新的文件夹以保存数据
dataFolder = fullfile(mainFolder, ['OMA_Data_' timestamp]);
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end

% 保存数据
save(fullfile(dataFolder, filename), 'SNR_Matrix', 'BLER_Matrix', 'BER_Matrix', 'tao_range', 'Channel_MSE_total_mean_Matrix', 'chan_type_range', 'chest_method_range', 'CIR_Thr_range');

% 显示保存信息
disp(['Data saved to: ' fullfile(dataFolder, filename)]);
