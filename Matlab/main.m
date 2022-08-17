clear all; clc;
% Parameters setting for simulation
n_seq = [200 100 50 10 1];                       % Number of measurements
m = 1024;                                        % 
d = 256;                                         % 
spar_seq = round(m * [0.01:0.01:0.18]);          % Sparsity
times_exper = 100;                               % Times of simulation
sigma = 0.001;                                   % Variance of Gaussian noise
% Parameters setting for algorithm
maxIter = 200;                                   % Maximum iteration
innMaxIter = 30; innEps = 1E-6;                  % Used in Newton method
if ~exist('./Outputs','dir')
    mkdir('Outputs')
end

for n = n_seq                                    % For each n
    X1 = zeros(m, n);                            % Initial X
    for s = spar_seq                             % For each sparsity
        % Error = zeros(times_exper, 4);
        REtemp = zeros(times_exper, 10);
        CPUTimetemp = zeros(times_exper, 10);
        for t = 1:times_exper                    % For each experiment
            A = randn(d, m);                     % Matrix A
            Xtrue = zeros(m, n);                 % Matrix X
            indtrue = randperm(m, s);            % 
            Xtrue(indtrue, :) = randn(s, n);     % 
            B = A * Xtrue + sigma * randn(d, n); % Matrix B
            % Standalization
            NoA = norm(A, 2); A = A/NoA; B = B/NoA;
            % Perform algorithms
            % when p = 2
            [X_L20, t_L20] = L20(A, B, X1, s, maxIter);
            [X_L21, t_L21] = L21(A, B, X1, s, maxIter);
            [X_L2half, t_L2half] = L2half(A, B, X1, s, maxIter);
            [X_L2TwoThir, t_L2TwoThir] = L2TwoThir(A, B, X1, s, maxIter);
            [X_L202, t_L202] = L2newton(A, B, X1, s, 0.2, maxIter, innMaxIter, innEps);
            [X_L208, t_L208] = L2newton(A, B, X1, s, 0.8, maxIter, innMaxIter, innEps);
            % when p = 1
            [X_L10, t_L10] = L10(A, B, X1, s, maxIter);
            [X_L11, t_L11] = L11(A, B, X1, s, maxIter);
            [X_L1half, t_L1half] = L1half(A, B, X1, s, maxIter);
            [X_L1TwoThir, t_L1TwoThir] = L1TwoThir(A, B, X1, s, maxIter);
            % Record predicted error
            Error = [norm(X_L20 - Xtrue, 'fro'), norm(X_L21 - Xtrue, 'fro'),...
                norm(X_L2half - Xtrue, 'fro'), norm(X_L2TwoThir - Xtrue, 'fro'),...
                norm(X_L202 - Xtrue, 'fro'), norm(X_L208 - Xtrue, 'fro'),...
                norm(X_L10 - Xtrue, 'fro'), norm(X_L11 - Xtrue, 'fro'),...
                norm(X_L1half - Xtrue, 'fro'), norm(X_L1TwoThir - Xtrue, 'fro')];
            RE = Error/norm(Xtrue, 'fro'); REtemp(t,:) = RE;
            CPUTimetemp(t,:) = [t_L20, t_L21, t_L2half, t_L2TwoThir, t_L202, t_L208,...
                t_L10, t_L11, t_L1half, t_L1TwoThir];
        end
        filename = strcat('Outputs/RE_n_', int2str(n), '_s_', int2str(s), '.txt');
        dlmwrite(filename, REtemp, 'precision', '%.6f');
        filename = strcat('Outputs/CPUTime_n_', int2str(n), '_s_', int2str(s), '.txt');
        dlmwrite(filename, CPUTimetemp, 'precision', '%.6f');
    end
end
