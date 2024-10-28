clear all; 
close all; % 关闭所有图形窗口
loadFile = 0; % 初始化 loadFile 变量，表示是否加载文件

if loadFile==1 % 如果 loadFile 为 1
    load fileName.mat filenameTmp; % 加载指定的文件，并将其内容赋值给 filenameTmp
    filenameTmp; % 显示 filenameTmp 的内容
else % 如果 loadFile 不为 1
    filenameTmp='ANC_Data_20240706_224407.mat'; % 使用默认的文件名
    filenameTmp=filenameTmp(1:end-4); % 去掉文件名的后缀 .mat
end

% 构建加载字符串
loadStr = strcat('.\ANCdata\', filenameTmp); % 构建文件路径
loadStr1 = strcat(loadStr, '\'); % 添加斜杠
loadStr = strcat(loadStr1, filenameTmp); % 构建完整路径
loadStr = strcat(loadStr, '.mat'); % 添加 .mat 后缀
loadStr = mystrcat('load ', loadStr); % 调用自定义函数 mystrcat 生成最终的加载字符串
eval(loadStr); % 评估加载字符串，加载文件

SelectMatrix = []; % 初始化选择矩阵
display(SNR_Matrix); % 显示 SNR 矩阵
display(BLER_Matrix); % 显示 BLER 矩阵
display(BER_Matrix); % 显示 BER 矩阵
display(tao_range); % 显示 tao 范围
display(sqrt(Channel_MSE_total_mean_Matrix)); % 显示信道 MSE 总均值矩阵的平方根
display(size(Channel_MSE_total_mean_Matrix)); % 显示信道 MSE 总均值矩阵的大小
display(size(unique(Channel_MSE_total_mean_Matrix))); % 显示信道 MSE 总均值矩阵中唯一值的大小

% 配置绘图参数
NumofPLots = min(1*4*2, size(SNR_Matrix, 1)); % 计算每个图中的绘图数量
StepPlot = 1; % 设置步长
StepPlot_shift  = 0; % 设置步长偏移
PlotRatio = 1; % 设置绘图比例
plotStartPoiont = 1; % 设置绘图起始点
Endplotpoint = 0; % 设置绘图结束点
LineMode = 22; % 设置线条模式
SubFigNum = 1*1; % 设置子图数量

PointSelectSort = size(SNR_Matrix, 2); % 选择排序点
IndexAll = 0; % 初始化索引
Semilog = 1; % 设置半对数模式

% 开始绘图循环
for i = 1:floor(size(SNR_Matrix, 1) / NumofPLots / PlotRatio) % 按照绘图数量和比例计算循环次数
    selectedIndex = [(i-1)*NumofPLots+1 : 1 : i*NumofPLots]; % 计算当前批次的索引
    selectedIndex_New = [1+StepPlot_shift : StepPlot : NumofPLots]; % 计算新的索引
    SelectMatrix(i, :) = selectedIndex_New + (i-1)*NumofPLots; % 更新选择矩阵

    SNR_MatrixPlot = SNR_Matrix(SelectMatrix(i, :), :); % 提取当前批次的 SNR 数据
    DataPlot = Channel_MSE_total_mean_Matrix(SelectMatrix(i, :), :); % 提取当前批次的 MSE 数据

    RowNum = size(DataPlot, 1) / SubFigNum; % 计算每个子图的行数
    for k = 1:SubFigNum % 对每个子图进行处理
        DataTmp = DataPlot((k-1)*RowNum+1 : k*RowNum, :); % 提取当前子图的数据
        IndexAll = IndexAll + 1; % 更新索引
        [SortedIndexValue(IndexAll, :) SortedIndex(IndexAll, :)] = sort(DataTmp(:, PointSelectSort)); % 对数据进行排序
    end

    DataPlot = DataPlot(:, plotStartPoiont:end-Endplotpoint, :); % 调整数据范围
    SNR_MatrixPlot = SNR_MatrixPlot(:, plotStartPoiont:end-Endplotpoint); % 调整 SNR 数据范围

    figure; % 创建新图形窗口
    dd = SubPlotFigures(Semilog, i, SNR_MatrixPlot, DataPlot, SubFigNum, LineMode); % 调用绘图函数 SubPlotFigures
    % legend('MirDFT','DFT-MMSE', 'MMSE1D1D',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS');
    % figure;  plot_snr_bler(SNR_Matrix( SelectMatrix(i,:),:),BLER_Matrix( SelectMatrix(i,:),:));%
    % legend('LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS');
    %
    % ylabel('BLER')
end

% 显示额外的信息
display(chan_type_range); % 显示信道类型范围
display(SelectMatrix); % 显示选择矩阵
display(chest_method_range); % 显示信道估计方法范围
display(CIR_Thr_range); % 显示 CIR 阈值范围
display(SortedIndex); % 显示排序索引

% 下面是一些被注释掉的代码，用于调整数据
% if i==1
%     DataPlot(3,1:2) = DataPlot(3,1:2) * 0.93;
%     DataPlot(5,1:2) = DataPlot(5,1:2) * 0.83;
%     DataPlot(7,1:2) = DataPlot(7,1:2) * 0.93;
%     DataPlot(7,1:2) = DataPlot(7,1:2) * 0.83;
%     DataPlot(5,4:7) = DataPlot(5,4:7) * 0.9125;
% end
% plot_snr_bler(SNR_Matrix(selectedIndex, :), Rawber_Matrix(selectedIndex, :)); % 绘制 SNR 和原始 BER 的图形

function t = mystrcat(varargin)
% STRCAT Concatenate strings.
%   T = STRCAT(S1,S2,S3,...) horizontally concatenates corresponding
%   rows of the character arrays S1, S2, S3 etc. All input arrays must 
%   have the same number of rows (or any can be a single string). When 
%   the inputs are all character arrays, the output is also a character 
%   array.
%
%   When any of the inputs is a cell array of strings, STRCAT returns 
%   a cell array of strings formed by concatenating corresponding 
%   elements of S1, S2, etc. The inputs must all have the same size 
%   (or any can be a scalar). Any of the inputs can also be character 
%   arrays.
%
%   Trailing spaces in character array inputs are ignored and do not 
%   appear in the output. This is not true for inputs that are cell 
%   arrays of strings. Use the concatenation syntax [S1 S2 S3 ...] 
%   to preserve trailing spaces.
%
%   Example
%       strcat({'Red','Yellow'},{'Green','Blue'})
%   returns
%       'RedGreen'    'YellowBlue'
%
%   See also STRVCAT, CAT, CELLSTR.

% 检查输入参数的数量，如果少于1个，则抛出错误
if nargin < 1
    error('Not enough input arguments.');
end

% 初始化 rows 和 twod 数组，分别存储每个输入数组的行数和是否为二维数组
for i = nargin:-1:1
    rows(i) = size(varargin{i}, 1); % 获取每个输入数组的行数
    twod(i) = ndims(varargin{i}) == 2; % 检查每个输入数组是否为二维数组
end

% 如果有任何输入数组不是二维的，则抛出错误
if ~all(twod)
    error('All the inputs must be two dimensional.');
end

% 移除空输入
k = (rows == 0); % 找到所有行数为0的输入
varargin(k) = []; % 移除对应的输入
rows(k) = []; % 移除对应的行数

% 标量扩展
for i = 1:length(varargin)
    % 如果当前输入的行数为1且小于最大行数，则进行标量扩展
    if rows(i) == 1 && rows(i) < max(rows)
        varargin{i} = varargin{i}(ones(1, max(rows)), :); % 将输入扩展到最大行数
        rows(i) = max(rows); % 更新行数
    end
end


% 如果所有输入的行数不一致，则抛出错误
if any(rows ~= rows(1))
    error('All the inputs must have the same number of rows or a single row.');
end

% 获取最大行数
n = rows(1);
t = ''; % 初始化输出字符串

% 逐行拼接字符串
for i = 1:n
    s = varargin{1}(i, :); % 获取当前行的第一个输入字符串
    for j = 2:length(varargin)
        s = [s deblank(varargin{j}(i, :))]; % 拼接去掉尾随空格后的其他输入字符串
    end
    t = strvcat(t, s); % 将拼接结果添加到输出字符串中
end
end

function [d] = SubPlotFigures(Semilog,PlotAllNum,SNR_Matrix,DataPlot_Matrix,PlotNum,LineMode)
    % SubPlotFigures: 根据提供的信噪比 (SNR) 矩阵和数据矩阵绘制多个子图。
    % 
    % 输入参数：
    % Semilog       - 0: 普通线性图，1: 对数图。
    % PlotAllNum    - 控制标题显示，1 或 2。
    % SNR_Matrix    - 信噪比 (SNR) 数据矩阵。
    % DataPlot_Matrix - 要绘制的数据矩阵 (如误比特率，BLER)。
    % PlotNum       - 子图数量。
    % LineMode      - 线型模式，用于控制不同的线条样式。
    % 
    % 输出参数：
    % d             - 恒为 1，用于函数返回。
RowsNum =size(SNR_Matrix,1)/PlotNum;

 % 根据子图数量确定行和列数
if PlotNum>4
    RowsPlot =floor(sqrt(PlotNum));    ColumnsPlot = ceil(sqrt(PlotNum));
else
    RowsPlot=1;    ColumnsPlot =PlotNum;
end
for i = 1:PlotNum
    SNR_MatrixPlot = SNR_Matrix((i-1)*RowsNum+1:i*RowsNum,:)
    DataPlot= DataPlot_Matrix((i-1)*RowsNum+1:i*RowsNum,:)
    subplot(RowsPlot, ColumnsPlot, i);
    % figure; plot_snr_bler(SNR_MatrixPlot,DataPlot);%
    plot_snr_bler(Semilog,SNR_MatrixPlot,DataPlot,LineMode);%
    %legend( 'LS','Proposed Method','MMSE time domain (Ideal PDP) [1]','MMSE time domain (Mac CP) [1]','DFT-MMSE',  'MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS');
    % 设置y轴标签
    ylabel('MSE');
    % legend('Proposed Method, 4 用户','提供文献 [1] (Ideal PDP)，4 用户','Proposed Method, 3 用户','提供文献 [1] (Ideal PDP)，3 用户','Proposed Method, 2 用户','提供文献 [1] (Ideal PDP)，2 用户','Proposed Method, 1 用户','提供文献 [1] (Ideal PDP)，1 用户');
    legend('Proposed Method,           TDL-A','提供文献 [1] (Ideal PDP), TDL-A',...,
        'Proposed Method,            TDL-B','提供文献 [1] (Ideal PDP), TDL-B',...,
        'Proposed Method,            TDL-C','提供文献 [1] (Ideal PDP),  TDL-C',...,
        'Proposed Method, 1 用户','提供文献 [1] (Ideal PDP)，1 用户');

    %legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28')
    if PlotAllNum==1     title('4用户, 25PRB, FFT size = 1024');    end
    if PlotAllNum==2     title('TDL-B, 3km/h, 25PRB, FFT size = 1024');    end
    h  =gcf;
    % 根据PlotNum设置字体和线条参数
    if PlotNum<2
        MarkerSize=12; YLabelFontSize =25; FontSize =25; LineWidth =2; LegendFontSize =25; TitleFontSize =25;
    else
        MarkerSize=12; YLabelFontSize =15; FontSize =10; LineWidth =2; LegendFontSize =15; TitleFontSize =15;
    end
     % 调用myboldify函数加粗图形
    myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize);    set(h, 'WindowState', 'maximized');
end
d=1;
end

function  plot_snr_bler(Semilog,SNR_Matrix,BLER_Matrix,LineMode)
LineStyles='-bs -gs -bv -gv';
LineStyles='-bs -gv -rp -ko -m< -cd -y> --bs --gv --rp --co --m< --kd --y> -bs -gv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y>';
%'-bo -go -ro -co -mo -yo -ko -bd -gd -rd -cd -md -yd -kd -bp -gp -rp -cp -mp -yp -kp -bh -gh -rh -ch -mh -yh -kh -b> -g> -r> -c> -m> -y> -k> -bs -gs -rs -cs -ms -ys -ks -bo -go -ro -co -mo -yo -ko';
%       b     blue          .     point              -     solid
%       g     green         o     circle             :     dotted
%       r     red           x     x-mark             -.    dashdot
%       c     cyan          +     plus               --    dashed
%       m     magenta       *     star             (none)  no line
%       y     yellow        s     square
%       k     black         d     diamond
%                           v     triangle (down)
%                           ^     triangle (up)
%                           <     triangle (left)
%                           >     triangle (right)
%                           p     pentagram
%                           h     hexagram equivalent
if LineMode==22
    LineStyles='-bs -gs -bd -gd -bv -gv -b^ -g^ -b< -g< -b> -g> -bp -gp -bh -gh -m< --bs --gv --rp --ko --m< --kd --y> -bs -gv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y> --bs --gv --rp --co --m< --kd --y>';
end
if LineMode==23
    LineStyles='-bs -gv -rp -ko -m< --bs --gv --rp --ko --m< --kd --y> -bs -gv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y> --bs --gv --rp --co --m< --kd --y>';
end
if LineMode==24
    LineStyles='-bs -gv -rp -ko -m< --bs --gv --rp --ko --m< --kd --y> -bs -gv -rp -co -m< -kd -y> --bs --gv --rp --co --m< --kd --y> --bs --gv --rp --co --m< --kd --y>';
end

% if solidLine==3
%     MarkerFaeColor='b g r b g r b g r b g r b g r';
%     MarkerFaceColor=parse(MarkerFaeColor);
%     ColorStyles='b g r b g r b g r b g r b g r';
%     MarkerFaeColor='b g r b g r b g r b g r b g r';
%     ColorStyles='b g r b g r b g r b g r b g r';
%     LineStyles='-bo -gv --ro --bo --gv --ro -bo -gv -ro';
%     MarkerFaeColor='b g r b g r b g r b g r b g r';
%     ColorStyles='b g r b g r b g r b g r b g r';
% end
%     LineStyles='-bs -bo -gs --gs -go --go -rs --rs -ro --ro -ks --ks -ko --ko';
Rows=size(BLER_Matrix,1);LineStyles=parse(LineStyles);MarkerSize =12; LineWidth = 2;
if Semilog==1
    for i=1:Rows
        %i
        semilogy(SNR_Matrix(i,:),BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
        %           plot(SNR_Matrix(i,:),BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
        hold on;    grid on;
    end
end

if Semilog==0
    for i=1:Rows
        %i
        plot(SNR_Matrix(i,:),BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
        %           plot(SNR_Matrix(i,:),BLER_Matrix(i,:),LineStyles(i,:),'LineWidth',LineWidth,'MarkerSize',MarkerSize);
        hold on;    grid on;
    end
end

SNR=SNR_Matrix;   DataMatrix =BLER_Matrix;
if min(min(DataMatrix))>0
    axis([min(min(SNR))*1 max(max(SNR))*1 min(min(DataMatrix))*0.8 max(max(DataMatrix))*1.1]);
else
    axis([min(min(SNR))*1 max(max(SNR))*1 min(min(DataMatrix))*1.1 max(max(DataMatrix))*1.1]);
end
% legend('new','old')
% %     title(' SCMA  1Tx2Rx  4PRB  QPSK  1/2  6users ')
% ylabel('BER'); xlabel('SNR (dB)');
%      legend('EP','EP-外')

YLabelFontSize =26;FontSize =26;LegendFontSize =26;TitleFontSize =26;
h  =gcf;
MarkerSize=12; YLabelFontSize =25; FontSize =25; LineWidth =2; LegendFontSize =25; TitleFontSize =25;
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

end

function myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
    % myboldify: 使当前图形的线条和文本加粗；适用于句柄为 h 的图形。
    % 
    % 输入参数：
    % h              - 图形句柄。如果未提供，将使用当前图形句柄 (gcf)。
    % MarkerSize     - 标记大小。默认值为 9。
    % YLabelFontSize - Y 轴标签字体大小。默认值为 24。
    % FontSize       - 字体大小。默认值为 24。
    % LineWidth      - 线条宽度。默认值为 2。
    % LegendFontSize - 图例字体大小。默认值为 36。
    % TitleFontSize  - 标题字体大小。默认值为 24。
%h  =gcf; % 获得当前figure 句柄，大家需要用这个模板来画图，仔细调整写出的文章才
%FontSize= 24 ; LineWidth = 3;TitleFontSize = 20; LegendFontSize = 36; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
%  myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize)
% myboldify: make lines and text bold;  boldifies the current figure; applies to the figure with the handle h
if nargin < 1
    h = gcf; 
    MarkerSize=9; YLabelFontSize =24; FontSize= 24 ; 
    LineWidth = 2;TitleFontSize =24; LegendFontSize =36; axis_ratio=1.5; %myboldify(h,FontSize,LineWidth,LegendFontSize,TitleFontSize);
end
ha = get(h, 'Children'); % the handle of each axis
for i = 1:length(ha)    
    if strcmp(get(ha(i),'Type'), 'axes') % 如果子对象是轴
            % 设置刻度标记和框架的格式
        set(ha(i), 'FontSize', LegendFontSize);      % tick mark and frame format
        set(ha(i), 'LineWidth', LineWidth);
        % 设置字体大小
        set(get(ha(i),'XLabel'), 'FontSize', YLabelFontSize);
        %set(get(ha(i),'XLabel'), 'VerticalAlignment', 'top');
        set(get(ha(i),'YLabel'), 'FontSize', YLabelFontSize);
        %set(get(ha(i),'YLabel'), 'VerticalAlignment', 'baseline');
        set(get(ha(i),'ZLabel'), 'FontSize', FontSize);
        %set(get(ha(i),'ZLabel'), 'VerticalAlignment', 'baseline');
        set(get(ha(i),'Title'), 'FontSize', TitleFontSize);
        %set(get(ha(i),'Title'), 'FontWeight', 'Bold');
    end    
    hc = get(ha(i), 'Children'); % the objects within an axis
    for j = 1:length(hc)
        chtype = get(hc(j), 'Type');
        if strcmp(chtype(1:length(chtype)), 'text')
            set(hc(j), 'FontSize', LegendFontSize); % 14 pt descriptive labels
        elseif strcmp(chtype(1:length(chtype)), 'line')
            set(hc(j), 'LineWidth', LineWidth);
            set(hc(j), 'MarkerSize', MarkerSize);
        elseif strcmp(chtype, 'hggroup')
            hcc = get(hc(j), 'Children');
            if strcmp(get(hcc, 'Type'), 'hggroup')
                hcc = get(hcc, 'Children');
            end
            for k = 1:length(hcc) % all elements are 'line'
                set(hcc(k), 'LineWidth', LineWidth);
                set(hcc(k), 'MarkerSize', LegendFontSize);
            end            
        end
    end
end
    
end



% function myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
% % myboldify: make lines and text bold
% %   myboldify boldifies the current figure
% %   myboldify(h) applies to the figure with the handle h
% if nargin < 1
%     h = gcf;
% end
% ha = get(h, 'Children'); % the handle of each axis
% for i = 1:length(ha)
%     if strcmp(get(ha(i),'Type'), 'axes') % axis format
%         set(ha(i), 'FontSize', FontSize);      % tick mark and frame format
%         set(ha(i), 'LineWidth', LineWidth);
%
%         set(get(ha(i),'XLabel'), 'FontSize', YLabelFontSize);
%         %set(get(ha(i),'XLabel'), 'VerticalAlignment', 'top');
%
%         set(get(ha(i),'YLabel'), 'FontSize', YLabelFontSize);
%         %set(get(ha(i),'YLabel'), 'VerticalAlignment', 'baseline');
%
%         set(get(ha(i),'ZLabel'), 'FontSize', FontSize);
%         %set(get(ha(i),'ZLabel'), 'VerticalAlignment', 'baseline');
%
%         set(get(ha(i),'Title'), 'FontSize', TitleFontSize);
%         %set(get(ha(i),'Title'), 'FontWeight', 'Bold');
%     end
%
%     hc = get(ha(i), 'Children'); % the objects within an axis
%     for j = 1:length(hc)
%         chtype = get(hc(j), 'Type');
%         if strcmp(chtype(1:4), 'text')
%             set(hc(j), 'FontSize', LegendFontSize); % 14 pt descriptive labels
%         elseif strcmp(chtype(1:4), 'line')
%             set(hc(j), 'LineWidth', LineWidth);
%             set(hc(j), 'MarkerSize', MarkerSize);
%         elseif strcmp(chtype, 'hggroup')
%             hcc = get(hc(j), 'Children');
%             if strcmp(get(hcc, 'Type'), 'hggroup')
%                 hcc = get(hcc, 'Children');
%             end
%             for k = 1:length(hcc) % all elements are 'line'
%                 set(hcc(k), 'LineWidth', LineWidth);
%                 set(hcc(k), 'MarkerSize', LegendFontSize);
%             end
%         end
%     end
% end
% end
function [x] = parse(inStr)
sz=size(inStr);
strLen=sz(2);
x=blanks(strLen);
%x=blanks(strLengthMax);2002/5/12 modify
wordCount=1;
last=0;
for i=1:strLen
    if inStr(i) == ' '
        wordCount = wordCount + 1;
        x(wordCount,:)=blanks(strLen);
        %x(wordCount,:)=blanks(strLengthMax); 2002/5/12 modify
        last=i;
    else
        x(wordCount,i-last)=inStr(i);
    end
end
% h  =gcf;
%    MarkerSize=12; YLabelFontSize =25; FontSize =25; LineWidth =2; LegendFontSize =25; TitleFontSize =25;
%    myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
end