function [X, t] = L2newton(A, B, X, s, q, maxIter, innMaxIter, innEps)
    st = cputime; % Record cputime
    % Initialization
    v = 0.5; % stepsize
    Bu1 = 2 * v * A' * B;
    Bu2 = 2 * v * A' * A;
    [nr0, nc0] = size(Bu1);
    I = eye(nc0);
    
    for k = 1:maxIter
        % Gradient descent
        Bu = X + Bu1 - Bu2 * X;
        
        % L2-1/2 threhsolding operator
        normBu = zeros(nr0, 1);
        for i = 1:nr0
            normBu(i) = norm(Bu(i,:), 2);
        end
        Bu0 = sort(normBu); criterion = Bu0(nr0-s);
        lambda = criterion^(2-q) / v;
        
        % Consider what if s-th largest group is not the only one
        if criterion == Bu0(nr0-s-1)
            ind = find(normBu >= criterion);
        else
            ind = find(normBu > criterion);
        end
        
        % Update matrix
        Xnew = zeros(nr0, nc0);
        for j = 1:length(ind)
            ind_temp = ind(j);
            rowDa = Bu(ind_temp, :);
            rowDaTemp = rowDa;
            
            % Newton method
            for t = 1:innMaxIter
                normTemp = norm(rowDaTemp, 1);
                H = lambda * q * rowDaTemp + (rowDaTemp - rowDa) * normTemp^(2 - q) ./ v;
                DH = lambda * q * I + normTemp^(-q) * (normTemp^2 * I + (2 - q) * (rowDaTemp - rowDa) * rowDaTemp') ./ v;
                
                rowDaTemp = rowDaTemp - (DH\(H'))';
                HTemp = lambda * q * rowDaTemp + (rowDaTemp - rowDa) * norm(rowDaTemp, 2)^(2 - q) / v;
                if norm(HTemp, 1) < innEps
                    break;
                end
            end
            Xnew(ind_temp, :) = rowDaTemp;
        end
        
        % Update and report
        X = Xnew;
        disp(['Complete the ', num2str(k), '-th iteration.']);
    end
    t = cputime - st;
end