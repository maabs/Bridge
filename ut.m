format long
fprintf('Program started at');
clock;


setenv('SLURM_JOB_ID', '34234');
setenv('SLURM_PROCID', '5');
%job_id = getenv('SLURM_JOB_ID');
%proc_id = getenv('SLURM_PROCID');
folder_read = '';
%folder_write = sprintf('%s%s', job_id, '/');
%results_filename = sprintf('%sL_%s_%s.txt', folder_write, job_id,proc_id);
%rng_seed = mod(sum(clock)*mod(str2num(job_id),10000)*(str2num(proc_id)+1), 10000000);
%rng(rng_seed);


global DataTimes;
global kangaroo_Data;
global basic_delta_t;
%kangaroo_Data =readmatrix('Kangaroo Data.txt');
kangaroo_Data = readmatrix(sprintf('%s%s', folder_read,'Kangaroo Data.txt'));
DataTimes = kangaroo_Data(:,3);
%DataTimes=zeros(2,1);
%disp(DataTimes);
%DataTimes(1,1)=DataTimes_or(1,1);
%DataTimes( 2,1)=DataTimes_or(end-1,1);
%DataTimes( 3,1)=DataTimes_or(end-1,1)+30;
%disp(DataTimes);
Y_data = kangaroo_Data(:,1:2);
%Y_data=zeros(2,2);
%disp(Y_data);
%Y_data(1,:)=Y_data_or(1,:);
%Y_data(2,:)=Y_data_or(end-1,:);
%Y_data(3,:)=Y_data_or(end,:);
%disp(Y_data)
% MLUSMA parameters
L_max = 9;
%L_start = 3;
L_min = 8;

p_max = 0;      % useless now
p_min = 0;      % p_min is always 1.
N_0 =5;
particleCount = 500;
%iterCount = 300;                % replicates count
iterCount = 10;
%theta0 = [0.397, 4.429e-03, 0.840, 17.631];
%theta0 = [1.45754903e+00, 2.72797500e-03, 6.71310500e-01, 1.86623609e+01];
%theta0_p = [log(2.397), log(4.429e-03), log(0.840), log(17.631)];
theta0_p=log(theta0(:));
%theta0_p = [1.860247644028022,  -5.675069792663762 , -0.257493022287784  , 2.924471348097306];
%X0 = log(Y_data(1,1))/theta0(3);
X0 = gamrnd(2*theta0(1)/theta0(3)^2,theta0(3)^2/(2*theta0(2)),1);
Cost_MSA_try_iter = zeros(iterCount, 1);
Time_MSA_try_iter = zeros(iterCount, 1);
used_L_try_iter =   zeros(iterCount, 1);
used_p_try_iter = zeros(iterCount, 1);
used_p_prob_try_iter = zeros(iterCount, 1);
used_L_prob_try_iter = zeros(iterCount, 1);
theta_iter = zeros(iterCount, 4);
theta_weighted_iter = zeros(iterCount, 4);
%iterCounts = zeros(1, L_max - L_start + 1);
theta_trace1 = cell(iterCount, 1);
theta_trace2 = cell(iterCount, 1);
theta_p_trace1 = cell(iterCount, 1);
theta_p_trace2 = cell(iterCount, 1);
theta_trace = cell(iterCount, 1);
theta_p_trace = cell(iterCount, 1);
thetas=zeros(iterCount,2*N_0-1,4);
hs=zeros(iterCount,2*N_0-1,4);
ch_me=zeros(iterCount);
start_0=tic;
X_in=zeros(particleCount,1)+X0;
%disp(X0);
%for i=1:1
%      X=generate_discrete_Kang(2^(-10), 10, X_in, theta);
%end
end_0=toc(start_0);
tic;
X_sm=zeros(iterCount,N_0,length(DataTimes));
for i=1:iterCount
    %tic
    [L, L_density] = sample_l(L_max, L_min);
    delta_t = 2^(-L);
    %[p, p_density] = sample_p(L, L_max, p_max);
    %[p, p_density] = sample_p_given_l(L, L_max, p_max);
    [p, p_density] = sample_p_given_l(L-2, L_max-2, p_max);
    N = N_0 * 2^p-1;
    used_p_try_iter(i) = p;
    used_L_try_iter(i) = L;
    used_p_prob_try_iter(i) = p_density(p-p_min+1);
    used_L_prob_try_iter(i) = L_density(L-L_min+1);
    theta_level = 0;
    approx_DataTimes = round((DataTimes-DataTimes(1))/delta_t/2)*delta_t*2;
    disp(max(log2(approx_DataTimes)));
    disp(min(log2(approx_DataTimes)));
    start_t=tic;
  
    if mod(i,1) == 0
        disp(['i = ', num2str(i), ', p = ', num2str(p), ', L = ', num2str(L), ', N = ', num2str(N)]);
    end
    if L == L_min
        iter_time_start = tic;
        theta_p = zeros(N+1,4);
        theta_p(1,:) = theta0_p;
        Xs = generate_discrete_Kang(delta_t, approx_DataTimes(end) - approx_DataTimes(1), X0, theta0);
        for n=1:N
            
            gamma = get_gamma(n,L);
            theta = [exp(theta_p(n,1)), exp(theta_p(n,2)), exp(theta_p(n,3)), exp(theta_p(n,4))];
            [Xs,X] = Conditional_Particle_Filter_2(delta_t, approx_DataTimes, particleCount, Y_data, theta, Xs);
            X_sm(i,n,:)=X;
            
            h = H_p_2(Y_data, X, Xs, theta_p(n,:), delta_t, approx_DataTimes);
            %theta(n+1) = theta(n) - gamma * (h/(sqrt(h^2 + 1)));`
            %theta(n+1) = theta(n) - gamma * max(min(1,h),-1);
            %gamma(1)=0;
            gamma(4)=1e5*gamma(4);
            theta_p(n+1,:) = theta_p(n,:) ;%+ 0.00001*gamma .* h;
            %disp(gamma);
            %.disp(exp(theta_p(n+1,:)));
            hs(i,n,:)=h;
            thetas(i,n,:)=exp(theta_p(n+1,:));
            %if any(abs(gamma.*h) > 0.1)
            %    theta_p(n+1,:) = theta_p(n,:);
            %end
            %h_values(i,tryIndex,n) = h;
        end
        theta_p_level =  theta_p(end,:);
        
        if p > p_min
            %N = Calculate_N_p(p-1, L_max_local);
            %theta_level =  theta_level - theta(N);
            theta_p_level =  theta_p_level - theta_p(round(N/2),:);
            %theta_level =  theta_level - mean(theta(round(N/2)+1:(N+1)));
            %theta_level =  theta_level - mean(theta(1:(N+1)));
        end  
        iter_time_end = toc(iter_time_start);
        Time_MSA_try_iter(i) = iter_time_end;
        Cost_MSA_try_iter(i) = N/delta_t;
        theta_p_trace{i,1} = theta_p;
    else
        iter_time_start = tic;
        theta_p_1 = zeros(N+1,4);
        theta_p_2 = zeros(N+1,4);
        theta_p_1(1,:) = theta0_p;
        theta_p_2(1,:) = theta0_p;
        [Xs1,Xs2] = generate_discrete_coupled_Kang(delta_t, approx_DataTimes(end) - approx_DataTimes(1), X0, X0, theta0, theta0);
        for n=1:N
            theta1 = [exp(theta_p_1(n,1)), exp(theta_p_1(n,2)), exp(theta_p_1(n,3)), exp(theta_p_1(n,4))];
            theta2 = [exp(theta_p_2(n,1)), exp(theta_p_2(n,2)), exp(theta_p_2(n,3)), exp(theta_p_2(n,4))];
            [Xs1,X1,Xs2,X2] = Coupled_Conditional_Particle_Filter_3(delta_t, approx_DataTimes, particleCount, ...
                Y_data, theta1, theta2, Xs1, Xs2);
            gamma = get_gamma(n,L);
            h1 = H_p(Y_data, X1, Xs1, theta_p_1(n,:), delta_t, approx_DataTimes);
            %theta1(n+1) = theta1(n) - gamma * (h/(sqrt(h^2 + 1)));
            %theta1(n+1) = theta1(n) - gamma * max(min(1,h),-1);
            theta_p_1(n+1,:) = theta_p_1(n,:) + gamma .* h1;
            %theta1(n+1) = theta1(n) - 1e-3*gamma * h;
            h2 = H_p(Y_data, X2, Xs2, theta_p_2(n,:), 2*delta_t);
            %theta2(n+1) = theta2(n) - gamma * (h/(sqrt(h^2 + 1)));
            %theta2(n+1) = theta2(n) - gamma * max(min(1,h),-1);
            theta_p_2(n+1,:) = theta_p_2(n,:) + gamma .* h2;
            %6thetas(i,n,:)=exp(theta_p_2(n+1));
            if any(abs(gamma.*h1) > 0.1) || any(abs(gamma.*h2) > 0.1)
                theta_p_1(n+1,:) = theta_p_1(n,:);
                theta_p_2(n+1,:) = theta_p_2(n,:);
            end
        end
        %Cost_MSA_try_iter(i,tryIndex,level) = 1.5*Cost_MSA_try_iter(i,tryIndex,level);
        theta_p_level = theta_p_1(end,:) - theta_p_2(end,:);

        %theta_level = mean(theta1(round(N/2)+1:(N+1))) - mean(theta2(round(N/2)+1:(N+1)));
        %theta_level = mean(theta1(1:(N+1))) - mean(theta2(1:(N+1)));
        if p > p_min
            %N = Calculate_N_p(p-1, L_max_local);
            %theta_level = theta_level - (theta1(N) - theta2(N));
            theta_p_level = theta_p_level - (theta_p_1(round(N/2),:) - theta_p_2(round(N/2),:));
        end
        iter_time_end = toc(iter_time_start);
        Time_MSA_try_iter(i) = iter_time_end;
        Cost_MSA_try_iter(i) = 3/2*N*(1/delta_t);
        theta_p_trace1{i,1} = theta_p_1;
        theta_p_trace2{i,1} = theta_p_2;
    end
    %end_t=toc(start_t)
    

    theta_iter(i,:) = theta_p_level; 
    theta_weighted_iter(i,:) = theta_p_level/(L_density(L-L_min+1)*p_density(p-p_min+1)); 
    %toc
end
toc

estimated_theta_p = mean(theta_weighted_iter, 1);
estimated_theta = [estimated_theta_p(1), exp(estimated_theta_p(2)), exp(estimated_theta_p(3)), exp(estimated_theta_p(4))];

X_flat = reshape(X_sm, [], size(X_sm, 3));  % or reshape(A, [], 1) for full vector

%writematrix(X_flat, 'Observationsdata/X_sm_4.txt')

X_mean=mean(X_sm,[1,2]);
X_est=mean(X_sm,2);
X_var=var(X_est,1);
X_var;
%disp(X_var(1,1,:))
figure
plot(squeeze(DataTimes), squeeze(X_mean(1,1,:)))
hold on
plot(squeeze(DataTimes), squeeze(Y_data(:,1)))
hold on 
plot(squeeze(DataTimes), squeeze(Y_data(:,2)))
hold off
legend ("sm","data 0", "data 1")
legend show 
%plot(squeeze(DataTimes),squeeze(X_var))

par_1=1;
par_2=2;
%figure
h_mean=mean(hs,[1,2]);
%disp(mean(exp(hs),[1,2]));
%disp(var(exp(hs),0,2));
%disp(h_mean);
%disp(thetas(2,:,1));
%disp(thetas(1,:,par_1));
%plot(thetas(1,:,par_1),thetas(1,:,par_2))
%for i=2:iterCount
%    hold on
%    plot(thetas(i,:,par_1),thetas(i,:,par_2))
    
%end
%hold off

%results_filename = sprintf('%stheta_iter_%s_%s.txt', folder_write, job_id,proc_id);
%writematrix(theta_iter, results_filename);
%results_filename = sprintf('%stheta2_iter_%s_%s.txt', folder_write, job_id,proc_id);
%writematrix(theta_weighted_iter, results_filename);
%results_filename = sprintf('%scost_%s_%s.txt', folder_write, job_id,proc_id);
%writematrix(Cost_MSA_try_iter, results_filename);
%results_filename = sprintf('%stime_%s_%s.txt', folder_write, job_id,proc_id);
%writematrix(Time_MSA_try_iter, results_filename);















% functions
% ================================================================================


function [l, density] = sample_l(L_max, L_min)
    %density = 2.^(-1*(L_min:L_max));
    %density = 2.^(-1.5*(L_min:L_max));
    %density = ones(1, L_max - L_min + 1);
    density = 2.^(-1*(L_min:L_max)).*((L_min:L_max)+1).*log2((L_min:L_max)+2).^2;
    density = density / sum(density);
    l = randsample(L_max - L_min + 1, 1, true, density);
    l = l + L_min - 1;
end


function [p, density] = sample_p(l, L, p_max)
    density = 2.^(-1*(1:p_max));
    %density = 2.^(-1*(1:p_max)) .* (1:p_max <= L - l + 1);
    %density = 2.^(-1*(1:p_max)).*((1:p_max)+1).*log2((1:p_max)+2).^2 .* (1:p_max <= L - l + 1);
    %density = ones(1,p_max);
    density = density / sum(density);
    p = randsample(p_max, 1, true, density);
end

function [p, density] = sample_p_given_l(l, L, p_max)
    density = zeros(1,L);
    for i = 1:min(4, L-l+1)
        density(i) = 2^(4-i);
    end
    for i = 5:(L-l+1)
        density(i) = 2^(-i)*i*(log(i))^2;
    end
    density = density / sum(density);
    p = randsample(1:L, 1, true, density);
end

function gamma = get_gamma(n,L)
    n_threhold = 1;
    if n < n_threhold
        n = n_threhold;
    end
    C = 1*1e-3;
    %gamma =  C * 1/(n+100);
    %gamma =  C * 1/((n - (n_threhold-1))^(3/4));
    %gamma =  C * 1/((n - (n_threhold-1))^(1/2 + 0.1));
    %gamma =  C * 1/((n - (n_threhold-1))^(1 - 0.2));
    %gamma =  C * 1/((n - (n_threhold-1) + 100)^(1 - 0.0));
    gamma = zeros(1,4);
    gamma(1) =  5*10*C * 1/((n - (n_threhold-1) + 100)^(0.6));
    gamma(2) =  1*100*C * 1/((n - (n_threhold-1) + 100)^(0.6));
    gamma(3) =  1*10e-2* C * 1/((n - (n_threhold-1) + 100)^(0.6));
    gamma(4) =  1*C * 1/((n - (n_threhold-1) + 100)^(0.6));
end



function Xs = generate_discrete_Kang(delta_t, T, X0, theta)
    N = T/delta_t;
    dW = sqrt(delta_t)*randn(length(X0), N);
    Xs = zeros(length(X0), N+1);
    Xs(:,1) = X0;

    for i=2:N+1
        Xs(:,i) = Xs(:,i-1) + (theta(3)^2/2+theta(1) - theta(2)*Xs(:,i-1)).*Xs(:,i-1)*delta_t + theta(3)*Xs(:,i-1).*dW(:,i-1);
    end
end

% X0_1 and X0_2 have the same length.
function [Xs1, Xs2] = generate_discrete_coupled_Kang(delta_t, T, X0_1, X0_2, theta1, theta2)
    N = T/delta_t;      % N should be even
    particleCount = length(X0_1);

    Xs1 = zeros(length(X0_1), N+1);
    Xs1(:,1) = X0_1;
    Xs2 = zeros(length(X0_1), N/2+1);
    Xs2(:,1) = X0_2;
    dWc = zeros(particleCount, 1);

    for i=2:N+1
        dWf = sqrt(delta_t)*randn(particleCount, 1);
        dWc = dWc + dWf;
        Xs1(:,i) = Xs1(:,i-1) + (theta1(1)/theta1(3) - theta1(2)/theta1(3)*exp(theta1(3)*Xs1(:,i-1)))*delta_t + dWf;

        if mod(i,2)==1
            Xs2(:,(i+1)/2) = Xs2(:,(i-1)/2) + (theta2(1)/theta2(3) - theta2(2)/theta2(3)*exp(theta2(3)*Xs2(:,(i-1)/2)))*2*delta_t + dWc;
            dWc = zeros(particleCount, 1);
        end
        
    end
end







function [Xs, X] = Conditional_Particle_Filter_2(delta_t, times, particleCount, Y, theta, ref_path)
    
    N = (times(end)-times(1))/delta_t;
    Xs = zeros(particleCount,N+1);
    X = zeros(particleCount,length(times));
    ref_path_at_unit_times = ref_path((times - times(1))/delta_t+1);

    X0 = gamrnd(2*theta(1)/theta(3)^2,theta(3)^2/(2*theta(2)),particleCount,1);
    
    X(:,1) = X0;
    Xs(:,1) = X0;
    log_W = zeros(particleCount,1);


    
    for i=1:length(times)
        % resampling first for the previous step
        r = theta(4);
        p = r./(r+X(:,i));
        %log_w = log(nbinpdf(Y(i,1), r, p)) + log(nbinpdf(Y(i,2), r, p));
        log_w = log_nb_pdf(Y(i,1), theta(4), X(:,i)) + log_nb_pdf(Y(i,2), theta(4), X(:,i));
        log_W = log_W + log_w;
        %w_1 = exp(log_w - max(log_w));
        W = exp(log_W - max(log_W));
        W = W/sum(W);
   
        %W = W.*w_1;
        if i == length(times)
            index = randsample(particleCount, 1, true, W);
            Xs = Xs(index,:);
            X = X(index,:);
            return;
        end
       
        %if 1/sum(W.^2) < particleCount/3 || i==length(times)
        if true
            %log_like = log_like + log(mean(W)) + max(log_w);
            %log_like = log_like + log(sum(w_1)) + max(log_w) + log(1/particleCount);
            %I = randsample(particleCount, particleCount, true, W/sum(W));
            I = resampleSystematic(W);
            Xs = Xs(I,:);
            X = X(I,:);
            Xs(end,:) = ref_path;
            X(end,:) = ref_path_at_unit_times;
            %W = ones(particleCount,1)/particleCount;
            log_W = zeros(particleCount,1);
        end

        A = generate_discrete_Kang(delta_t, times(i+1)-times(i), X(:,i), theta);
        A(end,:) = ref_path(((times(i)-times(1))/delta_t+1):((times(i+1)-times(1))/delta_t+1));
        Xs(:,((times(i)-times(1))/delta_t+1):((times(i+1)-times(1))/delta_t+1)) = A;
        X(:,i+1) = A(:,end);
    end    
end


function [Xs1, X1, Xs2, X2] = Coupled_Conditional_Particle_Filter_3(delta_t, times, particleCount, Y, theta1, theta2, ref_path1, ref_path2)
    
    N1 = (times(end)-times(1))/delta_t;
    N2 = (times(end)-times(1))/delta_t/2;
    Xs1 = zeros(particleCount,N1+1);
    X1 = zeros(particleCount,length(times));
    Xs2 = zeros(particleCount,N2+1);
    X2 = zeros(particleCount,length(times));

    ref_path_1_at_unit_times = ref_path1((times - times(1))/delta_t+1);
    ref_path_2_at_unit_times = ref_path2((times - times(1))/delta_t/2+1);

    X0_1 = 10/theta1(3)*randn(particleCount,1) + 5/theta1(3);
    X0_2 = 10/theta2(3)*randn(particleCount,1) + 5/theta2(3);

    X1(:,1) = X0_1;
    Xs1(:,1) = X0_1;
    X2(:,1) = X0_2;
    Xs2(:,1) = X0_2;
    log_W1 = zeros(particleCount,1);
    log_W2 = zeros(particleCount,1);
    for i=1:length(times)
        % resampling first for the previous step
        r1 = theta1(4);
        p1 = r1./(r1+exp(theta1(3)*X1(:,i)));
        r2 = theta2(4);
        p2 = r2./(r2+exp(theta2(3)*X2(:,i)));
        %log_w1 = log(nbinpdf(Y(i,1), r1, p1)) + log(nbinpdf(Y(i,2), r1, p1));
        log_w1 = log_nb_pdf(Y(i,1), theta1(4), exp(theta1(3)*X1(:,i))) + log_nb_pdf(Y(i,2), theta1(4), exp(theta1(3)*X1(:,i)));
        log_W1 = log_W1 + log_w1;
        W1 = exp(log_W1 - max(log_W1));
        W1 = W1/sum(W1);
        %log_w2 = log(nbinpdf(Y(i,1), r2, p2)) + log(nbinpdf(Y(i,2), r2, p2));
        log_w2 = log_nb_pdf(Y(i,1), theta2(4), exp(theta2(3)*X2(:,i))) + log_nb_pdf(Y(i,2), theta2(4), exp(theta2(3)*X2(:,i)));
        log_W2 = log_W2 + log_w2;
        W2 = exp(log_W2 - max(log_W2));
        W2 = W2/sum(W2);


        if i == length(times)
            alpha = sum(min(W1,W2));
            U = rand();
            if U < alpha
                index1 = randsample(particleCount, 1, true, min(W1,W2)/alpha);
                index2 = index1;
            else
                index1 = randsample(particleCount, 1, true, (W1-min(W1,W2))/(1-alpha));
                index2 = randsample(particleCount, 1, true, (W2-min(W1,W2))/(1-alpha));
            end
            Xs1 = Xs1(index1,:);
            X1 = X1(index1,:);
            Xs2 = Xs2(index2,:);
            X2 = X2(index2,:);
            return;
        end

        %if 1/sum(W1.^2) < particleCount/3 || 1/sum(W2.^2) < particleCount/3 || i == length(times)
        if true
            I1 = zeros(particleCount,1);
            I2 = zeros(particleCount,1);
            alpha = sum(min(W1,W2));
            U = rand();
            if U < alpha
                I1 = resampleSystematic(min(W1,W2)/alpha);
                I2 = I1;
            else
                I1 = resampleSystematic((W1-min(W1,W2))/(1-alpha));
                I2 = resampleSystematic((W2-min(W1,W2))/(1-alpha));
            end

            Xs1 = Xs1(I1,:);
            X1 = X1(I1,:);
            Xs1(end,:) = ref_path1;
            X1(end,:) = ref_path_1_at_unit_times;
            Xs2 = Xs2(I2,:);
            X2 = X2(I2,:);
            Xs2(end,:) = ref_path2;
            X2(end,:) = ref_path_2_at_unit_times;
            log_W1 = zeros(particleCount,1);
            log_W2 = zeros(particleCount,1);
        end
        if i == length(times)
            break
        end

        
        [A1,A2] = generate_discrete_coupled_Kang(delta_t, times(i+1)-times(i), X1(:,i), X2(:,i), theta1, theta2);

        A1(end,:) = ref_path1(((times(i)-times(1))/delta_t+1):((times(i+1)-times(1))/delta_t+1));
        Xs1(:,((times(i)-times(1))/delta_t+1):((times(i+1)-times(1))/delta_t+1)) = A1;
        X1(:,i+1) = A1(:,end);

        A2(end,:) = ref_path2(((times(i)-times(1))/delta_t/2+1):((times(i+1)-times(1))/delta_t/2+1));
        Xs2(:,((times(i)-times(1))/delta_t/2+1):((times(i+1)-times(1))/delta_t/2+1)) = A2;
        X2(:,i+1) = A2(:,end);
    end    
    Xs1 = Xs1(1,:);
    X1 = X1(1,:);
    Xs2 = Xs2(1,:);
    X2 = X2(1,:);
end

function lnb_pdf = log_nb_pdf(y, r, mu)
    %lnb_pdf = log(gamma(y+r)) - log(gamma(y)) - log(factorial(y)) + r.*(log(r) -log(r+mu)) + y.*(log(mu) -log(r+mu));
    lnb_pdf =  r.*(-log(r+mu)) + y.*(log(mu) -log(r+mu));
end


function  indx  = resampleSystematic( w )

    N = length(w);
    Q = cumsum(w);
    indx = zeros(1,N);

    T = linspace(0,1-1/N,N) + rand(1)/N;
    T(N+1) = 1;
    
    i=1;
    j=1;
    
    while (i<=N)
        if (T(i)<Q(j))
            indx(i)=j;
            i=i+1;
        else
            j=j+1;        
        end
    end
end


%%%%%% H function, grad of discrete likelihood
function     h= H(Y, X, Xs, theta, delta_t, times)    
    
    X = X';
    Xs = Xs';

    a = theta(1)/theta(3) - theta(2)/theta(3) * exp(theta(3)*Xs);
    grad_a = zeros(length(Xs),4);
    grad_a(:,1) = 1/theta(3);
    grad_a(:,2) = - 1/theta(3) * exp(theta(3)*Xs);
    grad_a(:,3) = -theta(1)/(theta(3))^2 + theta(2)/(theta(3))^2 * exp(theta(3)*Xs) - theta(2)/theta(3) * exp(theta(3)*Xs).*Xs;
    grad_a(:,4) = 0;

    grad_log_g = zeros(length(X),4);
    grad_log_g(:,1) = 0;
    grad_log_g(:,2) = 0;
    grad_log_g(:,3) = (-theta(4)-Y(:,1)).*X.*(exp(theta(3)*X))./(theta(4)+exp(theta(3)*X)) + Y(:,1).*X + ...
        (-theta(4)-Y(:,2)).*X.*(exp(theta(3)*X))./(theta(4)+exp(theta(3)*X)) + Y(:,2).*X;
    grad_log_g(:,4) = psi(Y(:,1)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) + exp(theta(3)*X)) - ...
                        theta(4)./(theta(4)+exp(theta(3)*X)) - Y(:,1)./(theta(4)+exp(theta(3)*X)) + ...
                        psi(Y(:,2)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) + exp(theta(3)*X)) - ...
                        theta(4)./(theta(4)+exp(theta(3)*X)) - Y(:,2)./(theta(4)+exp(theta(3)*X));


    grad_log_initial = zeros(1,4);
    grad_log_initial(:,1) = 0;
    grad_log_initial(:,2) = 0;
    grad_log_initial(:,4) = 0;
    grad_log_initial(:,3) = 1/theta(3) - X(1).*(theta(3)*X(1)-5)/100;


    h = -sum(grad_a(1:end-1,:).*a(1:end-1,:),1)*delta_t + ...
         sum(grad_a(1:end-1,:).*(Xs(2:end)-Xs(1:end-1)),1) + ...
         sum(grad_log_g,1) + ...
         grad_log_initial;

end


function h = H_p_2(Y, X, Xs, theta_p, delta_t, times)    
    X = X';
    Xs = Xs';
    theta = [exp(theta_p(1)), exp(theta_p(2)), exp(theta_p(3)), exp(theta_p(4))];
    b = ((theta(3)^2)/2+theta(1)-theta(2)*Xs(:)).*Xs(:);
    grad_b = zeros(length(Xs),4);
    grad_b(:,1) = Xs;
    grad_b(:,2) = -Xs.*Xs;
    grad_b(:,3) = 0;
    grad_b(:,4) = 0;
    grad_log_g = zeros(length(X),4);
    grad_log_g(:,1) = 0;
    grad_log_g(:,2) = 0;
    grad_log_g(:,3) = 0;
    grad_log_g(:,4) =   psi(Y(:,1)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) + X) - ...
                        theta(4)./(theta(4)+X) - Y(:,1)./(theta(4)+X) + ...
                        psi(Y(:,2)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) +X) - ...
                        theta(4)./(theta(4)+X) - Y(:,2)./(theta(4)+X);
    grad_log_initial = zeros(1,4);
    grad_log_initial(:,1) = 2/theta(3)^2*(-psi(2*theta(1)/theta(3)^2)+log(X(1)*2*theta(2)/theta(3)^2));
    grad_log_initial(:,2) = 2/(theta(2)*theta(3)^2)*(theta(1)-theta(2)*X(1));
    grad_log_initial(:,4) = 0;
    grad_log_initial(:,3) = 0;

    % nultiply by Jacobian
    grad_b(:,1) = grad_b(:,1) * theta(1);
    grad_b(:,2) = grad_b(:,2) * theta(2);
    grad_b(:,3) = grad_b(:,3) * theta(3);
    grad_b(:,4) = grad_b(:,4) * theta(4);
    grad_log_g(:,1) = grad_log_g(:,1) * theta(1);
    grad_log_g(:,2) = grad_log_g(:,2) * theta(2);
    grad_log_g(:,3) = grad_log_g(:,3) * theta(3);
    grad_log_g(:,4) = grad_log_g(:,4) * theta(4);
    grad_log_initial(:,1) = grad_log_initial(:,1) * theta(1);
    grad_log_initial(:,2) = grad_log_initial(:,2) * theta(2);
    grad_log_initial(:,3) = grad_log_initial(:,3) * theta(3);
    grad_log_initial(:,4) = grad_log_initial(:,4) * theta(4);

    %ch_me=sum(-b(1:end-1,:).^2,1)*delta_t/2+ ...
    %     sum(b(1:end-1,:).*(Xs(2:end)-Xs(1:end-1))./(Xs(1:end-1)*theta(3)),1);

    ch_me=sum(b(1:end-1,:).*(b(1:end-1,:).*(delta_t/2)-(Xs(2:end)-Xs(1:end-1))./(Xs(1:end-1).*theta(3))) ,1);

    h = -sum(grad_b(1:end-1,:).*b(1:end-1,:),1)*delta_t + ...
         sum(grad_b(1:end-1,:).*(Xs(2:end)-Xs(1:end-1))./(Xs(1:end-1)*theta(3)),1) + ...
         sum(grad_log_g,1) + ...
         grad_log_initial;
    h(3)=-ch_me;

end



function h = H_p(Y, X, Xs, theta_p, delta_t, times)    
    X = X';
    Xs = Xs';
    theta = [theta_p(1), exp(theta_p(2)), exp(theta_p(3)), exp(theta_p(4))];
    a = theta(1)/theta(3) - theta(2)/theta(3) * exp(theta(3)*Xs);
    grad_a = zeros(length(Xs),4);
    grad_a(:,1) = 1/theta(3);
    grad_a(:,2) = - 1/theta(3) * exp(theta(3)*Xs);
    grad_a(:,3) = -theta(1)/(theta(3))^2 + theta(2)/(theta(3))^2 * exp(theta(3)*Xs) - theta(2)/theta(3) * exp(theta(3)*Xs).*Xs;
    grad_a(:,4) = 0;
    grad_log_g = zeros(length(X),4);
    grad_log_g(:,1) = 0;
    grad_log_g(:,2) = 0;
    grad_log_g(:,3) = (-theta(4)-Y(:,1)).*X.*(exp(theta(3)*X))./(theta(4)+exp(theta(3)*X)) + Y(:,1).*X + ...
        (-theta(4)-Y(:,2)).*X.*(exp(theta(3)*X))./(theta(4)+exp(theta(3)*X)) + Y(:,2).*X;
    grad_log_g(:,4) = psi(Y(:,1)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) + exp(theta(3)*X)) - ...
                        theta(4)./(theta(4)+exp(theta(3)*X)) - Y(:,1)./(theta(4)+exp(theta(3)*X)) + ...
                        psi(Y(:,2)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) + exp(theta(3)*X)) - ...
                        theta(4)./(theta(4)+exp(theta(3)*X)) - Y(:,2)./(theta(4)+exp(theta(3)*X));
    grad_log_initial = zeros(1,4);
    grad_log_initial(1) = 0;
    grad_log_initial(2) = 0;
    grad_log_initial(4) = 0;
    grad_log_initial(3) = 1/theta(3) - X(1).*(theta(3)*X(1)-5)/100;


    % nultiply by Jacobian
    grad_a(:,2) = grad_a(:,2) * theta(2);
    grad_a(:,3) = grad_a(:,3) * theta(3);
    grad_a(:,4) = grad_a(:,4) * theta(4);
    grad_log_g(:,2) = grad_log_g(:,2) * theta(2);
    grad_log_g(:,3) = grad_log_g(:,3) * theta(3);
    grad_log_g(:,4) = grad_log_g(:,4) * theta(4);
    grad_log_initial(:,2) = grad_log_initial(:,2) * theta(2);
    grad_log_initial(:,3) = grad_log_initial(:,3) * theta(3);
    grad_log_initial(:,4) = grad_log_initial(:,4) * theta(4);


    h = -sum(grad_a(1:end-1,:).*a(1:end-1,:),1)*delta_t + ...
         sum(grad_a(1:end-1,:).*(Xs(2:end)-Xs(1:end-1)),1) + ...
         sum(grad_log_g,1) + ...
         grad_log_initial;

end





function a = applyExp(A)
    a = A;
    a(:,2:4) = exp(A(:,2:4));
end

function a = applyLog(A)
    a = A;
    a(:,2:4) = log(A(:,2:4));
end

