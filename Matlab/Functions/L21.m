function [X, t] = L21(A, B, X, s, maxIter)
    st = cputime; % Record cputime
    % Initialization
    v = 0.5; % stepsize
    Bu1 = 2 * v * A' * B;
    Bu2 = 2 * v * A' * A;
    [nr0, nc0] = size(Bu1);
    
    for k = 1:maxIter
        % Gradient descent
        Bu = X + Bu1 - Bu2 * X;
        
        % L21 threhsolding operator
        normBu = zeros(nr0);
        for i = 1:nr0
            normBu(i) = norm(Bu(i,:), 2);
        end
        Bu0 = sort(normBu); criterion = Bu0(nr0-s); lambda = criterion / v;
        
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
            normRow = norm(rowDa, 2);
            Xnew(ind_temp, :) = (1 - v * lambda / normRow) * rowDa;
        end
        
        % Update and report
        X = Xnew;
        disp(['Complete the ', num2str(k), '-th iteration.']);
    end
    t = cputime - st;
end