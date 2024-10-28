% ����������͹ر�����ͼ�δ���
clear all;
close all;

% �����ļ���
mainFolder = 'Channel_estimation';
if ~exist(mainFolder, 'dir')
    mkdir(mainFolder);
end

% ʹ�������������ģ���������
SNR_Matrix = rand(16, 10); % ��������� SNR ����
BLER_Matrix = rand(16, 10); % ��������� BLER ����
BER_Matrix = rand(16, 10); % ��������� BER ����
tao_range = 0:0.1:1; % ����ʱ�ӷ�Χ
Channel_MSE_total_mean_Matrix = rand(16, 10); % ����������ŵ���������ܾ�ֵ����
chan_type_range = {'Type1', 'Type2'}; % �ŵ����ͷ�Χ
chest_method_range = {'Method1', 'Method2'}; % �ŵ����Ʒ�����Χ
CIR_Thr_range = 0:0.1:1; % CIR ��ֵ��Χ

% ����Ψһ���ļ�����������ԣ�
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = ['OMA_Data_' timestamp '.mat'];

% �����µ��ļ����Ա�������
dataFolder = fullfile(mainFolder, ['OMA_Data_' timestamp]);
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end

% ��������
save(fullfile(dataFolder, filename), 'SNR_Matrix', 'BLER_Matrix', 'BER_Matrix', 'tao_range', 'Channel_MSE_total_mean_Matrix', 'chan_type_range', 'chest_method_range', 'CIR_Thr_range');

% ��ʾ������Ϣ
disp(['Data saved to: ' fullfile(dataFolder, filename)]);
