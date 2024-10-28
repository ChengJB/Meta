clc;clear; close all;
%%


%% 
% 次级通道Sw估计完成之后，使用FXLMS算法估计控制器Wc的滤波器系数，
% 使得参考麦克风采集到的初级噪声xn通过Wc系统之后，
% 能够产生一个与xn通过初级通道Pw之后的信号dn完全相同的抵消声波信号yn
lengthx = 1000; %噪声源长度
xn=randn(1,lengthx); %随机噪声源
Index=0;  

% N_order_range：Sw的阶数；
% Norder_esti_range：Sw估计的阶数；
% orderWc_range：Wc控制器的阶数
% NLMS_range：是否使用NLMS；
% SwTrue_range：是否使用理想Sw；
% mu_range：步长的范围；
% same_FxLMS：是否和FxLMS算法输出结果相同
% vsFlag_range：是否采用变步长算法

N_order_range=[60];Norder_esti_range=[60];orderWc_range= [80];   
NLMS_range=[ 0 1 3 4];SwTrue_range =[0 ];mu_range =[1]/1000;same_FxLMS=[0];
vsFlag_range=[0 ];
tic

for N_order=N_order_range   for Norder_esti=Norder_esti_range
    
    Sw=Sw_generate(N_order);%生成一个N_order阶的滤波器
    Sw_estimate=estimateSw(Norder_esti,Sw);%利用LMS算法估计Sw
    Pw = 1 * Sw;  %初级信道系数
    dn   = filter(Pw,1,xn)            ;%噪声源经过初级信道
    Rf_real    = filter(Sw,1,xn)                 ;%噪声源经过次级信道Sw
    Rf_esti    = filter(Sw_estimate,1,xn)             ;%噪声源经过次级估计信道Sw_estimate,用以FXLMS更新输入
figure;
    for orderWc=orderWc_range  for NLMS=NLMS_range  for SwTrue=SwTrue_range  for vsFlag=vsFlag_range  
    for mu=mu_range
            
        if SwTrue==0  
            [dn, Sy, en, Cw,x_n] = FXLMS_my(xn, Sw, Sw_estimate, orderWc, mu,NLMS,vsFlag,same_FxLMS);
            
            %[Er,y_t,Wc]  = FxLMS(orderWc, dn, Rf_real,Rf_esti,mu,NLMS);
            
            [en1, yk, W]  = myFDAF(dn,Rf_real,mu,mu, orderWc,0);
        end
       
        if SwTrue==1  
           % [dn, Sy, en, Cw] = FXLMS_my(xn, Sw, Sw.', orderWc, mu,NLMS,vsFlag,same_FxLMS);
            %[Er,y_t,Wc]  = FxLMS(orderWc, dn, Rf_real,Rf_real,mu,NLMS);
        end
              
       %Index=Index+1; ErrAll(Index,:)=sum(abs(en));
        %Index=Index+1; ErrAll(Index,:)=sum(abs(Er));
       Index=Index+1; ErrAll(Index,:)=sum(abs(en));
       ErrAll1(Index,:)=sum(abs(en1));
%         Index=Index+1; ErrAll(Index,:)=sum(abs(en-Er.'));
%plot(Er);
plot(en);
plot(en1);
legend("AutoUpdate","on")
 title("FXLMS 函数")
 hold on

         

                end;end;end;end;end;end;end  
toc
ErrAll(1:end,:)
ErrAll1(1:end,:)
Index








%%
function [Sw]=Sw_generate(N_order)


Fs     = 4e3;  % 采样率 8 kHz
%N      = 3;   % 800 个采样点数，总共时间 0.1 seconds
N=N_order;
Flow   = 160;  % 低频边界: 160 Hz
Fhigh  = 2000; % 最高频率边界: 2000 Hz
delayS = 2;
Ast    = 20;   % 20 dB 阻带衰减
Nfilt  = 8;    % 滤波器的阶数

% Design bandpass filter to generate bandlimited impulse response
filtSpecs = fdesign.bandpass('N,Fst1,Fst2,Ast',Nfilt,Flow,Fhigh,Ast,Fs);
bandpass = design(filtSpecs,'cheby2','FilterStructure','df2tsos',  'SystemObject',true);

% Filter noise to generate impulse response
%secondaryPathCoeffsActual = bandpass([zeros(delayS,1); log(0.99*rand(N-delayS,1)+0.01).* sign(randn(N-delayS,1)).*exp(-0.01*(1:N-delayS)')]);
secondaryPathCoeffsActual = [zeros(delayS,1); log(0.99*rand(N-delayS,1)+0.01).* sign(randn(N-delayS,1)).*exp(-0.01*(1:N-delayS)')];
secondaryPathCoeffsActual = secondaryPathCoeffsActual/norm(secondaryPathCoeffsActual);
Sw=secondaryPathCoeffsActual/5;   % 真实的次级通道传递函数系数

% figure(1)
% t = (1:N)/Fs;
% plot(t,secondaryPathCoeffsActual,'b');
% xlabel('Time [sec]');
% ylabel('Coefficient value');
% title('True Secondary Path Impulse Response');

end

function[Sw_estimate]=estimateSw(Norder_esti,Sw)
%% 对产生的次级通道Sw进行估计
%order = 3; % 我们所估计的次级通道的传递函数的阶数

order = Norder_esti; % 我们所估计的次级通道的传递函数的阶数
lengthSimSignal = 50000;  % 用来进行测试次级通道的白噪声输入信号的采样点数
x_iden=randn(1, lengthSimSignal); %generating white noise signal to estimate Sw
% x_iden 为估计Sw所需要用到的输入信号
% send it to the actuator, and measure it at the sensor position,
y_iden=filter(Sw, 1, x_iden);  % x_iden送入系统Sw，产生经过真实的次级路径之后的输出信号y__iden

% Then, start the identification process
% 开始进行次级通道识别估计
statex=zeros(1,order);     % 估计过程中输入信号的此刻的状态缓存
Sw_estimate=zeros(1,order);     % 估计出来的次级通道传递函数的系数
err_iden=zeros(1, lengthSimSignal);   %   每次估计迭代中的识别误差

%LMS 算法估计次级通道Sw
% 相当于用白噪声信号作为输入，白噪声经过系统Sw之后的输出信号作为LMS算法的期望信号，从而估计Sw系统
mu=0.001;

for k=order : (length(x_iden))

    statex = x_iden(k: -1: (k - order +1));        %   此时系统的输入信号，u(n), u(n-1), u(n-1),.....,u(n-16+1)
    y_iden_estimate=sum(statex.*Sw_estimate);	  %  经过我们估计出的有误差的次级通道之后的模拟输出信号
    err_iden(k)=y_iden(k)-y_iden_estimate;               %   y_iden(k)为通过真实的次级通道之后的期望信号
    Sw_estimate=Sw_estimate+mu*err_iden(k)*statex;
end

% figure
% subplot(2,1,1)
% plot(1: length(x_iden)-order+1, err_iden(order: length(x_iden)))
% ylabel('每次迭代产生的误差');
% xlabel('迭代次数');
% title('每次迭代估计的次级通道的系数和真实预设的次级通道的系数之间的误差');
% 
% subplot(2,1,2)
% stem(Sw)
% hold on
% stem(Sw_estimate, 'r*');
% ylabel('系数大小');xlabel('滤波器系数索引');legend('真实的次级通道系数 S(z)', 'lms算法估计出来的次级通道系数 Sestimate(z)')


end

%% 函数化FXLMS_my算法
function [dn, Sy, en, Cw,x_n] = FXLMS_my(xn, Sw, Sw_estimate, orderW, mu,NLMS,vsFlag,same_FxLMS)
lengthx = length(xn);

% 经过初级路径之后到达误差麦克风的信号d(n)
Pw = 1 * Sw;
dn = filter(Pw, 1, xn);

orderS_real = length(Sw);
orderS_esti = length(Sw_estimate);

Cw = zeros(1, orderW);  % 控制器的自适应滤波器的权重系数
Sy = zeros(1, lengthx); % Sw输出
en = zeros(1, lengthx); % 误差传感器进行噪声消除之后的误差
Cy = zeros(1, lengthx); % 控制器的输出
x_n = zeros(1, lengthx);% xn经过Sw_estimate后的输出，用以FxLMS的更新输入
Cx= zeros(1,orderW) ;   %  控制器输入缓存
Sx=zeros(1,orderS_real);% Sw输入缓存
Xin=zeros(1,orderS_esti);% xn输入缓存
xn_state=zeros(1,orderW);
Pd=1; Pyhat=1; Pe=1;alpha=0.995;eta=0.5;


for k = 1:lengthx
    %Cx = xn(k:-1:k-orderW+1);           % 控制器接收到的输入信号
    Cx=[xn(k) Cx(1:end-1)];
%% 如果想和FxLMS算法一样，用以下Sx，若如此做，则没有用到之前时刻更新的滤波器系数
if same_FxLMS ==1
   Cy=filter(Cw,1,xn);
    if k<=orderS_real
        Cy_state=Cy(1:k);
        Sx=[Cy_state(end:-1:1) zeros(1,orderS_real-k)];
    elseif k>orderS_real
        Sx=Cy(k:-1:k-orderS_real+1);
    end
end
%% 以下只用当前时刻滤波器系数计算当前时刻的控制器输出，Sx用到了之前时刻更新的滤波器系数
if same_FxLMS ==0
    Cy(k) = sum(Cx .* Cw);              % 控制器的输出
    %Sx = Cy(k:-1:k-orderS_real+1);      
    Sx=[Cy(k) Sx(1:end-1)];             %控制器的输出信号也要经过次级路径才能到达误差传感器，这是即将进入次级路径的控制器的输出信号
end
%%
    Sy(k) = sum(Sx .* Sw');
    en(k) = dn(k) - Sy(k);              % 控制器的输出信号经过次级路径之后，与参考麦克风接收的参考信号经过初级路径传播之后生成的dn信号相减，计算误差
    Xin=[xn(k) Xin(1:end-1)];
    %x_n(k) = sum(xn(k:-1:k-orderS_esti+1) .* Sw_estimate);
    x_n(k) = sum(Xin .* Sw_estimate);
    %xn_state = x_n(k:-1:k-orderW+1);
    xn_state=[x_n(k) xn_state(1:end-1)];
       
    if NLMS==0
        Cw = Cw + mu * en(k) * xn_state;    % 更新控制器的滤波器系数
       
    elseif NLMS==1
        if vsFlag==1  %是否变步长
            Pd=alpha*Pd+(1-alpha)*dn(k)*dn(k);
            Pyhat=alpha*Pyhat+(1-alpha)*Sy(k)*Sy(k);
            Pe=alpha*Pe+(1-alpha)*en(k)*en(k);
            mu=1-eta*Pyhat/Pd;
            if mu>1
                mu=1;
            elseif mu<0
                mu=0;
            end
        end
        % 使用NLMS算法更新自适应滤波器系数
        % 计算归一化因子
        norm_factor = dot(xn_state, xn_state);
        % 归一化步长因子
        alpha = mu / norm_factor;
        % 更新控制器的滤波器系数
        Cw = Cw + alpha * en(k) * xn_state;
    elseif NLMS==2
        alpha=1;
        p=1;
        L=1;
        K=2;
        lamda=1-1/(K*L);
        norm_factor = dot(xn_state, xn_state);
        seita2_v=0;
        seita2_u=dot(xn_state, xn_state)/L;
        if k>1
        seita2_e=lamda*seita2_e+(1-lamda)*en(k)^2;
        else
            seita2_e=en(k)^2;
        end
        g=1-exp(alpha*seita2_v/(seita2_u*p*seita2_e-seita2_v));
        v_esti=g*en(k);
        if en(k)>0
        muw=(en(k).^2-en(k)*v_esti)./(en(k).^2*norm_factor);
        Cw = Cw + muw * en(k) * xn_state;
        end

    elseif NLMS==3
        
        % 生成示例输入信号
        signal =xn; % 生成长度为1000的随机信号
        % 计算自相关矩阵
        R = xcorr(signal, 'biased'); % 计算自相关序列
        R = toeplitz(R(length(signal):end)); % 生成自相关矩阵
        % 求取自相关矩阵的特征值
        eigenvalues = eig(R); % 计算特征值
        max_eigenvalue = max(eigenvalues); % 找到最大特征值
        mu_min=0;
        mu_max=1/max_eigenvalue;
        alpha=0.97;gamma=4.8*10^-4;
        mu=alpha* mu+gamma *en(k)^2;
        if mu>mu_max  mu=mu_max  
        end
          if mu<mu_min  mu=mu_min;  
          end
        Cw = Cw + mu* en(k) * xn_state;  
   elseif NLMS==4
           beta=0.5;
           alpha=0.97;gamma=4.8*10^-4;
           if k==1
               PP=zeros(size(xn_state));
         
           else 
                PP=beta*PP+(1-beta)*xn_state*en(k);
           end
           mu=alpha* mu+gamma*norm(PP)^2*en(k)^2;
           Cw = Cw + mu* en(k) * xn_state; 
    

            

end

 

end
 
end

%% single-channel FxLMS algorithm
function [Er,y_t,Wc] = FxLMS(Len_Filter, dn, Rf_real, Rf_esti,muw,NLMS)

N   = Len_Filter ;
Wc  = zeros(N,1); %控制器的自适应滤波器的权重系数
Rf_i  = zeros(N,1) ; %Rf_real输入缓存
Rf_ii  = zeros(N,1) ;%Rf_esti输入缓存，用以更新LMS输入
Er  = zeros(length(Rf_real),1);
lengthx = length(Rf_real);
y_t = zeros(lengthx,1);


for tt = 1:length(Rf_real)

    Rf_ii= [Rf_esti(tt);Rf_ii(1:end-1)];
    Rf_i  = [Rf_real(tt);Rf_i(1:end-1)];

    y_t(tt)  = Wc'*Rf_i    ;
    e    = dn(tt)-y_t(tt) ;
    Er(tt) = e         ;
    %Wc_false     = Wc + muw*e*Rf_i;
   
    if NLMS==0
        Wc     = Wc + muw*e*Rf_ii;    % 更新控制器的滤波器系数
     
    elseif NLMS==1
        % 使用NLMS算法更新自适应滤波器系数
        % 计算归一化因子
        norm_factor = dot(Rf_ii, Rf_ii);
        % 归一化步长因子
        alpha = muw/ norm_factor;
        % 更新控制器的滤波器系数
        Wc     = Wc + alpha*e*Rf_ii;
    end

end

%  figure;plot(1: lengthx, dn'-y_t);legend("残余噪声信号")
%  title("FXLMS 函数")
%  hold on
end
% 参考：https://github.com/CharlesThaCat/acoustic-interference-cancellation
function [en, yk, W] = myFDAF(d,x,mu,mu_unconst, M, select)
% 参数:
% d                输入信号(麦克风语音)
% x                远端语音
% mu                约束 FDAF的步长
% mu_unconst        不受约束的 FDAF的步长
% M                 滤波器阶数
% select;            选择有约束或无约束FDAF算法
%
% 参考:        
% S. Haykin, Adaptive Filter Theory, 4th Ed, 2002, Prentice Hall
% by Lee, Gan, and Kuo, 2008
% Subband Adaptive Filtering: Theory and Implementation
% Publisher: John Wiley and Sons, Ltd

x_new = zeros(M,1);     % 将新块的M个样本初始化为0
x_old = zeros(M,1);     % 将旧块的M个样本初始化为0

AdaptStart = 2*M;       % 在获得2M个样本块后开始自适应
W = zeros(2*M,1);       % 将2M个滤波器权重初始化为0
d_block = zeros(M,1);   % 将期望信号的M个样本初始化为0

power_alpha = 0.5;        % 常数以更新每个frequency bin的功率
power = zeros(2*M,1);   % 将每个bin的平均功率初始化为0
d_length = length(d);             % 输入序列的长度
en = [];                       % 误差向量
window_save_first_M = [ones(1,M), zeros(1,M)]';  % 设置向量以提取前M个元素 (2M,1)

for k = 1:d_length
    x_new = [x_new(2:end); x(k)];         % 开始的输入信号块 (2M,1)
    d_block = [d_block(2:end); d(k)];     % 开始的期望信号快 (M,1)
    if mod(k,M)==0                        % If iteration == block length, 
        x_blocks = [x_old; x_new];        % 2M样本的输入信号样本块 (2M,1)
        x_old = x_new;
        if k >= AdaptStart                % 频域自适应滤波器

            % 将参考信号转换到频域
            Xk = fft(x_blocks);     % (2M,1)
            % FFT[old block; new block]
            % Old block 包含M个先前的输入样本 (u_old)
            % New 包含M个新的输入样本 (u_new)

            % 计算滤波器估计信号
            Yk = Xk.*W;                  % 输入和权重向量的乘积(2M,1)*(2M,1)=(2M,1)
            temp = real(ifft(Yk));            % IFFT 输出的实部 (2M,1)
            yk = temp(M+1:2*M);               % 抛弃前M个元素，保留后M个元素 (M,1)

            % 计算误差信号
            error = d_block-yk;              % 误差信号块 (M,1)
            Ek = fft([zeros(1,M),error']');   % 在FFT之前插入零块以形成2M块(2M,1)

            % 更新信号功率估算
            power = (power_alpha.*power) + (1 - power_alpha).*(abs(Xk).^2); % (2M,1)
 
            norm_factor = dot(x_blocks,x_blocks);
%             norm_factor = dot(x_old,x_old);
            % 归一化步长因子
            mu = mu/ norm_factor;
            % 计算频域中的梯度和权重更新
            if select == 1
                gradient = real(ifft((1./power).*conj(Xk).* Ek));   % (2M,1)
                gradient = gradient.*window_save_first_M;   % 去除后一个数据块，并且补零 (2M,1)
                W = W + mu.*(fft(gradient));    % 权重是频域的 (2M,1)
            else
                gradient = conj(Xk).* Ek;   %  (2M,1)
                W = W + mu_unconst.*gradient;    % (2M,1)
            end
            
            en = [en; error];             % 更新误差块
        end
    end
end
end



