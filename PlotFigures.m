clear all; 
close all; % �ر�����ͼ�δ���
loadFile = 0; % ��ʼ�� loadFile ��������ʾ�Ƿ�����ļ�

if loadFile==1 % ��� loadFile Ϊ 1
    load fileName.mat filenameTmp; % ����ָ�����ļ������������ݸ�ֵ�� filenameTmp
    filenameTmp; % ��ʾ filenameTmp ������
else % ��� loadFile ��Ϊ 1
    filenameTmp='ANC_Data_20240706_224407.mat'; % ʹ��Ĭ�ϵ��ļ���
    filenameTmp=filenameTmp(1:end-4); % ȥ���ļ����ĺ�׺ .mat
end

% ���������ַ���
loadStr = strcat('.\ANCdata\', filenameTmp); % �����ļ�·��
loadStr1 = strcat(loadStr, '\'); % ���б��
loadStr = strcat(loadStr1, filenameTmp); % ��������·��
loadStr = strcat(loadStr, '.mat'); % ��� .mat ��׺
loadStr = mystrcat('load ', loadStr); % �����Զ��庯�� mystrcat �������յļ����ַ���
eval(loadStr); % ���������ַ����������ļ�

SelectMatrix = []; % ��ʼ��ѡ�����
display(SNR_Matrix); % ��ʾ SNR ����
display(BLER_Matrix); % ��ʾ BLER ����
display(BER_Matrix); % ��ʾ BER ����
display(tao_range); % ��ʾ tao ��Χ
display(sqrt(Channel_MSE_total_mean_Matrix)); % ��ʾ�ŵ� MSE �ܾ�ֵ�����ƽ����
display(size(Channel_MSE_total_mean_Matrix)); % ��ʾ�ŵ� MSE �ܾ�ֵ����Ĵ�С
display(size(unique(Channel_MSE_total_mean_Matrix))); % ��ʾ�ŵ� MSE �ܾ�ֵ������Ψһֵ�Ĵ�С

% ���û�ͼ����
NumofPLots = min(1*4*2, size(SNR_Matrix, 1)); % ����ÿ��ͼ�еĻ�ͼ����
StepPlot = 1; % ���ò���
StepPlot_shift  = 0; % ���ò���ƫ��
PlotRatio = 1; % ���û�ͼ����
plotStartPoiont = 1; % ���û�ͼ��ʼ��
Endplotpoint = 0; % ���û�ͼ������
LineMode = 22; % ��������ģʽ
SubFigNum = 1*1; % ������ͼ����

PointSelectSort = size(SNR_Matrix, 2); % ѡ�������
IndexAll = 0; % ��ʼ������
Semilog = 1; % ���ð����ģʽ

% ��ʼ��ͼѭ��
for i = 1:floor(size(SNR_Matrix, 1) / NumofPLots / PlotRatio) % ���ջ�ͼ�����ͱ�������ѭ������
    selectedIndex = [(i-1)*NumofPLots+1 : 1 : i*NumofPLots]; % ���㵱ǰ���ε�����
    selectedIndex_New = [1+StepPlot_shift : StepPlot : NumofPLots]; % �����µ�����
    SelectMatrix(i, :) = selectedIndex_New + (i-1)*NumofPLots; % ����ѡ�����

    SNR_MatrixPlot = SNR_Matrix(SelectMatrix(i, :), :); % ��ȡ��ǰ���ε� SNR ����
    DataPlot = Channel_MSE_total_mean_Matrix(SelectMatrix(i, :), :); % ��ȡ��ǰ���ε� MSE ����

    RowNum = size(DataPlot, 1) / SubFigNum; % ����ÿ����ͼ������
    for k = 1:SubFigNum % ��ÿ����ͼ���д���
        DataTmp = DataPlot((k-1)*RowNum+1 : k*RowNum, :); % ��ȡ��ǰ��ͼ������
        IndexAll = IndexAll + 1; % ��������
        [SortedIndexValue(IndexAll, :) SortedIndex(IndexAll, :)] = sort(DataTmp(:, PointSelectSort)); % �����ݽ�������
    end

    DataPlot = DataPlot(:, plotStartPoiont:end-Endplotpoint, :); % �������ݷ�Χ
    SNR_MatrixPlot = SNR_MatrixPlot(:, plotStartPoiont:end-Endplotpoint); % ���� SNR ���ݷ�Χ

    figure; % ������ͼ�δ���
    dd = SubPlotFigures(Semilog, i, SNR_MatrixPlot, DataPlot, SubFigNum, LineMode); % ���û�ͼ���� SubPlotFigures
    % legend('MirDFT','DFT-MMSE', 'MMSE1D1D',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS');
    % figure;  plot_snr_bler(SNR_Matrix( SelectMatrix(i,:),:),BLER_Matrix( SelectMatrix(i,:),:));%
    % legend('LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS','MirDFT 11', 'DFT MMSE',  'LS');
    %
    % ylabel('BLER')
end

% ��ʾ�������Ϣ
display(chan_type_range); % ��ʾ�ŵ����ͷ�Χ
display(SelectMatrix); % ��ʾѡ�����
display(chest_method_range); % ��ʾ�ŵ����Ʒ�����Χ
display(CIR_Thr_range); % ��ʾ CIR ��ֵ��Χ
display(SortedIndex); % ��ʾ��������

% ������һЩ��ע�͵��Ĵ��룬���ڵ�������
% if i==1
%     DataPlot(3,1:2) = DataPlot(3,1:2) * 0.93;
%     DataPlot(5,1:2) = DataPlot(5,1:2) * 0.83;
%     DataPlot(7,1:2) = DataPlot(7,1:2) * 0.93;
%     DataPlot(7,1:2) = DataPlot(7,1:2) * 0.83;
%     DataPlot(5,4:7) = DataPlot(5,4:7) * 0.9125;
% end
% plot_snr_bler(SNR_Matrix(selectedIndex, :), Rawber_Matrix(selectedIndex, :)); % ���� SNR ��ԭʼ BER ��ͼ��

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

% �������������������������1�������׳�����
if nargin < 1
    error('Not enough input arguments.');
end

% ��ʼ�� rows �� twod ���飬�ֱ�洢ÿ������������������Ƿ�Ϊ��ά����
for i = nargin:-1:1
    rows(i) = size(varargin{i}, 1); % ��ȡÿ���������������
    twod(i) = ndims(varargin{i}) == 2; % ���ÿ�����������Ƿ�Ϊ��ά����
end

% ������κ��������鲻�Ƕ�ά�ģ����׳�����
if ~all(twod)
    error('All the inputs must be two dimensional.');
end

% �Ƴ�������
k = (rows == 0); % �ҵ���������Ϊ0������
varargin(k) = []; % �Ƴ���Ӧ������
rows(k) = []; % �Ƴ���Ӧ������

% ������չ
for i = 1:length(varargin)
    % �����ǰ���������Ϊ1��С���������������б�����չ
    if rows(i) == 1 && rows(i) < max(rows)
        varargin{i} = varargin{i}(ones(1, max(rows)), :); % ��������չ���������
        rows(i) = max(rows); % ��������
    end
end


% ������������������һ�£����׳�����
if any(rows ~= rows(1))
    error('All the inputs must have the same number of rows or a single row.');
end

% ��ȡ�������
n = rows(1);
t = ''; % ��ʼ������ַ���

% ����ƴ���ַ���
for i = 1:n
    s = varargin{1}(i, :); % ��ȡ��ǰ�еĵ�һ�������ַ���
    for j = 2:length(varargin)
        s = [s deblank(varargin{j}(i, :))]; % ƴ��ȥ��β��ո������������ַ���
    end
    t = strvcat(t, s); % ��ƴ�ӽ����ӵ�����ַ�����
end
end

function [d] = SubPlotFigures(Semilog,PlotAllNum,SNR_Matrix,DataPlot_Matrix,PlotNum,LineMode)
    % SubPlotFigures: �����ṩ������� (SNR) ��������ݾ�����ƶ����ͼ��
    % 
    % ���������
    % Semilog       - 0: ��ͨ����ͼ��1: ����ͼ��
    % PlotAllNum    - ���Ʊ�����ʾ��1 �� 2��
    % SNR_Matrix    - ����� (SNR) ���ݾ���
    % DataPlot_Matrix - Ҫ���Ƶ����ݾ��� (��������ʣ�BLER)��
    % PlotNum       - ��ͼ������
    % LineMode      - ����ģʽ�����ڿ��Ʋ�ͬ��������ʽ��
    % 
    % ���������
    % d             - ��Ϊ 1�����ں������ء�
RowsNum =size(SNR_Matrix,1)/PlotNum;

 % ������ͼ����ȷ���к�����
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
    % ����y���ǩ
    ylabel('MSE');
    % legend('Proposed Method, 4 �û�','�ṩ���� [1] (Ideal PDP)��4 �û�','Proposed Method, 3 �û�','�ṩ���� [1] (Ideal PDP)��3 �û�','Proposed Method, 2 �û�','�ṩ���� [1] (Ideal PDP)��2 �û�','Proposed Method, 1 �û�','�ṩ���� [1] (Ideal PDP)��1 �û�');
    legend('Proposed Method,           TDL-A','�ṩ���� [1] (Ideal PDP), TDL-A',...,
        'Proposed Method,            TDL-B','�ṩ���� [1] (Ideal PDP), TDL-B',...,
        'Proposed Method,            TDL-C','�ṩ���� [1] (Ideal PDP),  TDL-C',...,
        'Proposed Method, 1 �û�','�ṩ���� [1] (Ideal PDP)��1 �û�');

    %legend('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28')
    if PlotAllNum==1     title('4�û�, 25PRB, FFT size = 1024');    end
    if PlotAllNum==2     title('TDL-B, 3km/h, 25PRB, FFT size = 1024');    end
    h  =gcf;
    % ����PlotNum�����������������
    if PlotNum<2
        MarkerSize=12; YLabelFontSize =25; FontSize =25; LineWidth =2; LegendFontSize =25; TitleFontSize =25;
    else
        MarkerSize=12; YLabelFontSize =15; FontSize =10; LineWidth =2; LegendFontSize =15; TitleFontSize =15;
    end
     % ����myboldify�����Ӵ�ͼ��
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
%      legend('EP','EP-��')

YLabelFontSize =26;FontSize =26;LegendFontSize =26;TitleFontSize =26;
h  =gcf;
MarkerSize=12; YLabelFontSize =25; FontSize =25; LineWidth =2; LegendFontSize =25; TitleFontSize =25;
myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

end

function myboldify(h,MarkerSize,YLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
    % myboldify: ʹ��ǰͼ�ε��������ı��Ӵ֣������ھ��Ϊ h ��ͼ�Ρ�
    % 
    % ���������
    % h              - ͼ�ξ�������δ�ṩ����ʹ�õ�ǰͼ�ξ�� (gcf)��
    % MarkerSize     - ��Ǵ�С��Ĭ��ֵΪ 9��
    % YLabelFontSize - Y ���ǩ�����С��Ĭ��ֵΪ 24��
    % FontSize       - �����С��Ĭ��ֵΪ 24��
    % LineWidth      - ������ȡ�Ĭ��ֵΪ 2��
    % LegendFontSize - ͼ�������С��Ĭ��ֵΪ 36��
    % TitleFontSize  - ���������С��Ĭ��ֵΪ 24��
%h  =gcf; % ��õ�ǰfigure ����������Ҫ�����ģ������ͼ����ϸ����д�������²�
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
    if strcmp(get(ha(i),'Type'), 'axes') % ����Ӷ�������
            % ���ÿ̶ȱ�ǺͿ�ܵĸ�ʽ
        set(ha(i), 'FontSize', LegendFontSize);      % tick mark and frame format
        set(ha(i), 'LineWidth', LineWidth);
        % ���������С
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