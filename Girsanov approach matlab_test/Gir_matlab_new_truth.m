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
Y_data = kangaroo_Data(:,1:2);
% MLUSMA parameters
L_max = 9;

L_min = 8;

p_max = 0;      % useless now
p_min = 0;      % p_min is always 1.
N_0 =1000;
mcmc_link=500;
gammas=[2,3,0,10]*5e-4;
alpha=0.5;
particleCount = 500;
iterCount = 500; % replicates count
theta0 = [2.397, 4.429e-03, 0.840, 17.631];
theta0_p=log(theta0(:));
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
thetas=zeros(iterCount,N_0+1,4);
disp(size(thetas(:,1,:)));
thetas(:,1,1) = theta0(1);
thetas(:,1,2) = theta0(2);
thetas(:,1,3) = theta0(3);
thetas(:,1,4) = theta0(4);
hs=zeros(iterCount,N_0,4);
hs1=zeros(iterCount,N_0,4);
hs2=zeros(iterCount,N_0,4);
ch_me=zeros(iterCount);
thetas_p_level=zeros(iterCount,4);
start_0=tic;
X_in=zeros(particleCount,1)+X0;

end_0=toc(start_0);
tic;
X_sm=zeros(iterCount,N_0,length(DataTimes));
X1_sm=zeros(iterCount,N_0,length(DataTimes));
X2_sm=zeros(iterCount,N_0,length(DataTimes));


for i=1:iterCount
    %tic
    [L, L_density] = sample_l(L_max, L_min);
    L=L_max;
    delta_t = 2^(-L);
    %[p, p_density] = sample_p(L, L_max, p_max);
    %[p, p_density] = sample_p_given_l(L, L_max, p_max);
    [p, p_density] = sample_p_given_l(L-2, L_max-2, p_max);
    N = N_0;
    j=1;
    used_p_try_iter(i) = p;
    used_L_try_iter(i) = L;
    used_p_prob_try_iter(i) = p_density(p-p_min+1);
    used_L_prob_try_iter(i) = L_density(L-L_min+1);
    theta_level = 0;
    approx_DataTimes = round((DataTimes-DataTimes(1))/delta_t/2)*delta_t*2;
    start_t=tic;
    if mod(i,1) == 0
        disp(['i = ', num2str(i), ', p = ', num2str(p), ', L = ', num2str(L), ', N = ', num2str(N)]);
    end
    if L == L_min
        iter_time_start = tic;
        theta_p = zeros(N+1,4);
        theta_p(1,:) = theta0_p;
        X0 = gamrnd(2*theta0(1)/theta0(3)^2,theta0(3)^2/(2*theta0(2)),1);
        Xs = generate_discrete_Kang(delta_t, approx_DataTimes(end) - approx_DataTimes(1), X0, theta0);
        for n=1:N
            gamma = get_gamma_2(n,gammas,alpha);
            theta = [exp(theta_p(n,1)), exp(theta_p(n,2)), exp(theta_p(n,3)), exp(theta_p(n,4))];
            [Xs,X] = Conditional_Particle_Filter_2(delta_t, approx_DataTimes, particleCount, Y_data, theta, Xs);
            X_sm(i,n,:)=X;
            
            h = H_p_2(Y_data, X, Xs, theta_p(n,:), delta_t, approx_DataTimes);
            hs(i,n,:)=h;
            if mod(n,mcmc_link)==0

                theta_p(n+1,:) = theta_p(n,:) + gamma .* mean(hs(i,n-mcmc_link+1:n,:));
            else
                theta_p(n+1,:) = theta_p(n,:); %+ gamma .* h;
            end
            thetas(i,n+1,:)=exp(theta_p(n+1,:));
            %if any(abs(gamma.*h) > 0.1)
            %    theta_p(n+1,:) = theta_p(n,:);
            %end
            %h_values(i,tryIndex,n) = h;
        end
        theta_p_level =  theta_p(end,:);
        
        %if p > p_min
            %N = Calculate_N_p(p-1, L_max_local);
            %theta_level =  theta_level - theta(N);
            theta_p_level =  theta_p_level - theta_p(round(N/2),:);
            %theta_level =  theta_level - mean(theta(round(N/2)+1:(N+1)));
            %theta_level =  theta_level - mean(theta(1:(N+1)));
        %end  
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
        X0 = gamrnd(2*theta0(1)/theta0(3)^2,theta0(3)^2/(2*theta0(2)),1);
        [Xs1,Xs2] = generate_discrete_coupled_Kang_2(delta_t, approx_DataTimes(end) - approx_DataTimes(1), X0, X0, theta0, theta0);
        for n=1:N
            theta1 = [exp(theta_p_1(n,1)), exp(theta_p_1(n,2)), exp(theta_p_1(n,3)), exp(theta_p_1(n,4))];
            theta2 = [exp(theta_p_2(n,1)), exp(theta_p_2(n,2)), exp(theta_p_2(n,3)), exp(theta_p_2(n,4))];
            [Xs1,X1,Xs2,X2] = Coupled_Conditional_Particle_Filter_4(delta_t, approx_DataTimes, particleCount, ...
                Y_data, theta1, theta2, Xs1, Xs2);
            
            h1 = H_p_2(Y_data, X1, Xs1, theta_p_1(n,:), delta_t, approx_DataTimes);
            hs1(i,n,:)=h1;
            X2_sm(i,n,:)=X1;
            %theta1(n+1) = theta1(n) - gamma * (h/(sqrt(h^2 + 1)));
            %theta1(n+1) = theta1(n) - gamma * max(min(1,h),-1);
            if mod(n,mcmc_link)==0
                gamma = get_gamma_2(j,gammas,alpha);
                j=j+1;
                disp(n);
           
                disp(exp(theta_p_1(n,:)));
                theta_p_1(n+1,:) = theta_p_1(n,:) + gamma .*(  squeeze(mean(hs1(i,n-mcmc_link+1:n,:),2))).';
                
            else
                theta_p_1(n+1,:) = theta_p_1(n,:) ;%+ gamma .* h1;
            end
            
            h2 = H_p_2(Y_data, X2, Xs2, theta_p_2(n,:), 2*delta_t);
            hs2(i,n,:)=h2;
            X2_sm(i,n,:)=X2;
            if mod(n,mcmc_link)==0
                theta_p_2(n+1,:) = theta_p_2(n,:) + gamma .*(  squeeze(mean(hs2(i,n-mcmc_link+1:n,:),2))).';
            else
            
                theta_p_2(n+1,:) = theta_p_2(n,:) ;%+ gamma .* h2;
            end
            %6thetas(i,n,:)=exp(theta_p_2(n+1));
            %X_sm(i,n,:)=X1;
            %hs(i,n,:)=h1;
            thetas(i,n+1,:)=exp(theta_p_1(n+1,:));
            %if any(abs(gamma.*h1) > 0.1) || any(abs(gamma.*h2) > 0.1)
            %    theta_p_1(n+1,:) = theta_p_1(n,:);
            %    theta_p_2(n+1,:) = theta_p_2(n,:);
            %end
        end
        %Cost_MSA_try_iter(i,tryIndex,level) = 1.5*Cost_MSA_try_iter(i,tryIndex,level);
        theta_p_level = theta_p_1(end,:) - theta_p_2(end,:);

        %theta_level = mean(theta1(round(N/2)+1:(N+1))) - mean(theta2(round(N/2)+1:(N+1)));
        %theta_level = mean(theta1(1:(N+1))) - mean(theta2(1:(N+1)));
        %if p > p_min
            %N = Calculate_N_p(p-1, L_max_local);
            %theta_level = theta_level - (theta1(N) - theta2(N));
            theta_p_level = theta_p_level - (theta_p_1(round(N/2),:) - theta_p_2(round(N/2),:));
            thetas_p_level(i,:)=theta_p_level;
        %end
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


X1_flat = reshape(X1_sm, [], size(X1_sm, 3));  % or reshape(A, [], 1) for full vector
X2_flat = reshape(X2_sm, [], size(X2_sm, 3)); 
h1_flat= reshape(hs1, [], size(hs1, 3));
h2_flat= reshape(hs2, [], size(hs2, 3));
thetas_flat= reshape(thetas, [], size(thetas, 3));
writematrix(h1_flat, 'Observationsdata_matlab/Grads1_sm_31.txt');
writematrix(X1_flat, 'Observationsdata_matlab/X1_sm_31.txt');
writematrix(h2_flat, 'Observationsdata_matlab/Grads2_sm_31.txt');
writematrix(X2_flat, 'Observationsdata_matlab/X2_sm_31.txt');
writematrix(thetas_flat, 'Observationsdata_matlab/thetas_sm_31.txt');
writematrix(thetas_p_level,'Observationsdata_matlab/thetas_p_level_31.txt');
%X_mean=mean(X_sm,[1,2]);
%X_est=mean(X_sm,2);
%X_var=var(X_est,1);


%figure
%plot(squeeze(DataTimes), squeeze(X_mean(1,1,:)))
%hold on
%plot(squeeze(DataTimes), squeeze(Y_data(:,1)))
%hold on 
%plot(squeeze(DataTimes), squeeze(Y_data(:,2)))
%hold off
%legend ("sm","data 0", "data 1")
%legend show 
%plot(squeeze(DataTimes),squeeze(X_var))

%par_1=1;
%par_2=2;
%figure
%h_mean=mean(hs,[1,2]);
%disp(h_mean);
%disp(var(hs,0,[1,2]));
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



%theta1=[2,2/500,1];
%theta2=[2.2,2.2/500,1.4];
%particleCount=50;
%[X0_1,X0_2]=rej_max_coup_gamma_ind_dist(particleCount,theta1,theta2);
%disp(size(X0_2));

%p1 = @(x,a,b) gampdf(x, a, b);
%p2 = @(y,a,b) gampdf(y, a, b);
%
%particleCount = 5e4;
%
%alpha2=2*theta2(1)/theta2(3)^2;
%beta2=theta2(3)^2/(2*theta2(2));
%
%alpha1=2*theta1(1)/theta1(3)^2;
%beta1=theta1(3)^2/(2*theta1(2));
%
%plot_joint_with_marginals(particleCount, theta1, theta2, ...
%    p1, alpha1, beta1, p2, alpha2, beta2);








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


function gamma=get_gamma_2(n,gammas,alpha)
    gamma=gammas/n^(alpha+0.5);
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

function [Xs1, Xs2] = generate_discrete_coupled_Kang_2(delta_t, T, X0_1, X0_2, theta1, theta2)
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
        Xs1(:,i) = Xs1(:,i-1) + (theta1(3)^2/2+theta1(1) - theta1(2)*Xs1(:,i-1)).*Xs1(:,i-1)*delta_t + theta1(3)*Xs1(:,(i-1)).*dWf;
        
        if mod(i,2)==1
            Xs2(:,(i+1)/2) = Xs2(:,(i-1)/2) + (theta2(3)^2/2+theta2(1) - theta2(2)*Xs2(:,(i-1)/2)).*Xs2(:,(i-1)/2)*2*delta_t + theta2(3)*Xs2(:,(i-1)/2).*dWc;
            
            dWc = zeros(particleCount, 1);
        end
        
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




function [X0_1,X0_2]=rej_max_coup_gamma_ind_dist(particleCount,theta1,theta2)

    alpha_2=2*theta2(1)/theta2(3)^2;
    theta_2=theta2(3)^2/(2*theta2(2));

    alpha_1=2*theta1(1)/theta1(3)^2;
    theta_1=theta1(3)^2/(2*theta1(2));

    X0_2=gamrnd(alpha_2,theta_2,particleCount,1);
    pX0_2=gampdf(X0_2,alpha_2,theta_2);

    X0_1=X0_2;
    qX0_2=gampdf(X0_2,alpha_1,theta_1);
    for i=1:particleCount
        w=rand*pX0_2(i);
        if w>qX0_2(i)
            X0_1(i)=gamrnd(alpha_1,theta_1);
            qX0_1=gampdf(X0_1(i),alpha_1,theta_1);
            w_s=rand*qX0_1;
            pX0_1=gampdf(X0_1(i),alpha_2,theta_2);
            while w_s<=pX0_1
                X0_1(i)=gamrnd(alpha_1,theta_1);
                qX0_1=gampdf(X0_1(i),alpha_1,theta_1);
                w_s=rand*qX0_1;
                pX0_1=gampdf(X0_1(i),alpha_2,theta_2);
                
            end
        end
    end


end


function plot_joint_with_marginals( ...
        particleCount, theta1, theta2, ...
        p1, alpha1, beta1, ...
        p2, alpha2, beta2)

% Visualize coupled samples with joint heatmap + marginals vs analytic PDFs
% p1, p2 are function handles:
%   p1 = @(x1, a1, b1) ...    % returns pdf of X1 at x1
%   p2 = @(x2, a2, b2) ...    % returns pdf of X2 at x2

    % ---- sample -------------------------------------------------------
    [X0_1, X0_2] = rej_max_coup_gamma_ind_dist(particleCount, theta1, theta2);
    X = X0_1(:); Y = X0_2(:);
    good = isfinite(X) & isfinite(Y); X = X(good); Y = Y(good);

    % ---- layout -------------------------------------------------------
    tl = tiledlayout(4,4, "Padding","compact", "TileSpacing","compact");
    axTop   = nexttile(tl, [1 3]);
    axJoint = nexttile(tl, [3 3]);
    axRight = nexttile(tl, [3 1]);

    xlims = padlims(min(X), max(X));
    ylims = padlims(min(Y), max(Y));

    % ---- joint density (2D) ---------------------------------------------
    nb = max(20, round(sqrt(numel(X))/3));   % reasonable bin count
    axes(axJoint);
    h2 = histogram2(X, Y, [nb nb], ...
    "Normalization","pdf", "DisplayStyle","tile", "EdgeColor","none");
    colormap(axJoint, parula); colorbar('eastoutside');
xlabel(axJoint, 'x_1'); ylabel(axJoint, 'x_2');
xlim(axJoint, padlims(min(X),max(X))); ylim(axJoint, padlims(min(Y),max(Y)));
title(axJoint, 'Joint distribution');

% share axes so frames align
linkaxes([axTop,   axJoint], 'x');
linkaxes([axRight, axJoint], 'y');

% Pull the exact binning used by the 2-D plot:
xedges = h2.XBinEdges;  yedges = h2.YBinEdges;

% ---- top marginal (X1) — use SAME x-edges ---------------------------
axes(axTop); cla(axTop); hold(axTop,'on');
histogram(axTop, X, "Normalization","pdf", ...
    "BinEdges", xedges, ...
    "FaceColor",[0.3 0.6 0.9], "EdgeColor","none", "FaceAlpha",0.45);
xg = linspace(xedges(1), xedges(end), 500);
plot(axTop, xg, p1(xg, alpha1, beta1), 'k-', 'LineWidth', 1.8);
set(axTop,'XTickLabel',[]); ylabel(axTop,'density'); box(axTop,'on');
title(axTop,'Marginal of x_1');
legend(axTop, 'empirical', 'p_1(x_1|\alpha_1,\beta_1)', 'Location','northeast');

% ---- right marginal (X2) — use SAME y-edges --------------------------
axes(axRight); cla(axRight); hold(axRight,'on');
histogram(axRight, Y, "Normalization","pdf", "Orientation","horizontal", ...
    "BinEdges", yedges, ...
    "FaceColor",[0.9 0.5 0.3], "EdgeColor","none", "FaceAlpha",0.45);
yg = linspace(yedges(1), yedges(end), 500);
plot(axRight, p2(yg, alpha2, beta2), yg, 'k-', 'LineWidth', 1.8);
xlabel(axRight,'density'); set(axRight,'YTickLabel',[]); box(axRight,'on');
title(axRight,'Marginal of x_2');
legend(axRight, 'empirical', 'p_2(x_2|\alpha_2,\beta_2)', 'Location','southeast');
end

function lims = padlims(a,b)
    if a==b, d = max(1, abs(a))*0.05; else, d = 0.05*(b-a); end
    lims = [a-d, b+d];
end



function [Xs1, X1, Xs2, X2] = Coupled_Conditional_Particle_Filter_4(delta_t, times, particleCount, Y, theta1, theta2, ref_path1, ref_path2)
    
    N1 = (times(end)-times(1))/delta_t;
    N2 = (times(end)-times(1))/delta_t/2;
    Xs1 = zeros(particleCount,N1+1);
    X1 = zeros(particleCount,length(times));
    Xs2 = zeros(particleCount,N2+1);
    X2 = zeros(particleCount,length(times));

    ref_path_1_at_unit_times = ref_path1((times - times(1))/delta_t+1);
    ref_path_2_at_unit_times = ref_path2((times - times(1))/delta_t/2+1);
    % Change in the following two lines

    alpha_2=2*theta2(1)/theta2(3)^2;
    theta_2=theta2(3)^2/(2*theta2(2));

    alpha_1=2*theta1(1)/theta1(3)^2;
    theta_1=theta1(3)^2/(2*theta1(2));

    X0_2=gamrnd(alpha_2,theta_2,particleCount,1);
    pX0_2=gampdf(X0_2,alpha_2,theta_2);

    X0_1=X0_2;
    qX0_2=gampdf(X0_2,alpha_1,theta_1);
    for i=1:particleCount
        w=rand*pX0_2(i);
        if w>qX0_2(i)
            X0_1(i)=gamrnd(alpha_1,theta_1);
            qX0_1=gampdf(X0_1(i),alpha_1,theta_1);
            w_s=rand*qX0_1;
            pX0_1=gampdf(X0_1(i),alpha_2,theta_2);
            while w_s<=pX0_1
                X0_1(i)=gamrnd(alpha_1,theta_1);
                qX0_1=gampdf(X0_1(i),alpha_1,theta_1);
                w_s=rand*qX0_1;
                pX0_1=gampdf(X0_1(i),alpha_2,theta_2);
                
            end
        end


    end


    %X0_1 = 10/theta1(3)*randn(particleCount,1) + 5/theta1(3);
    %X0_2 = 10/theta2(3)*randn(particleCount,1) + 5/theta2(3);

    X1(:,1) = X0_1;
    Xs1(:,1) = X0_1;
    X2(:,1) = X0_2;
    Xs2(:,1) = X0_2;
    log_W1 = zeros(particleCount,1);
    log_W2 = zeros(particleCount,1);
    for i=1:length(times)
        % resampling first for the previous step
        r1 = theta1(4);
        
        p1 = r1./(r1+X1(:,i));
        r2 = theta2(4);
        p2 = r2./(r2+X2(:,i));
        %log_w1 = log(nbinpdf(Y(i,1), r1, p1)) + log(nbinpdf(Y(i,2), r1, p1));
        log_w1 = log_nb_pdf(Y(i,1), theta1(4), (X1(:,i))) + log_nb_pdf(Y(i,2), theta1(4), X1(:,i));
        log_W1 = log_W1 + log_w1;
        W1 = exp(log_W1 - max(log_W1));
        W1 = W1/sum(W1);
        %log_w2 = log(nbinpdf(Y(i,1), r2, p2)) + log(nbinpdf(Y(i,2), r2, p2));
        log_w2 = log_nb_pdf(Y(i,1), theta2(4), X2(:,i)) + log_nb_pdf(Y(i,2), theta2(4), X2(:,i));
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

        
        [A1,A2] = generate_discrete_coupled_Kang_2(delta_t, times(i+1)-times(i), X1(:,i), X2(:,i), theta1, theta2);

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
    b = ((theta(3)^2)/2+theta(1)-theta(2)*Xs)/theta(3);
    grad_b = zeros(length(Xs),4);
    grad_b(:,1) = 1/theta(3);
    grad_b(:,2) = -Xs/theta(3);
    grad_b(:,3) = 0;
    grad_b(:,4) = 0;
    grad_log_g = zeros(length(X),4);
    grad_log_g(:,1) = 0;
    grad_log_g(:,2) = 0;
    grad_log_g(:,3) = 0;
    grad_log_g(:,4) = psi(Y(:,1)+theta(4)) - psi(theta(4)) + log(theta(4)) + 1 - log(theta(4) + X) - ...
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

    
    h = -sum(grad_b(1:end-1,:).*b(1:end-1,:),1)*delta_t + ...
         sum(grad_b(1:end-1,:).*(Xs(2:end)-Xs(1:end-1))./(Xs(1:end-1)*theta(3)),1) + ...
         sum(grad_log_g,1) + ...
         grad_log_initial;
    

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

