function [X, t] = L2half(A, B, X, s, maxIter)
    st = cputime; % Record cputime
    % Initialization
    v = 0.5; % stepsize
    Va1 = (2/3)^(1.5) / v;
    Bu1 = 2 * v * A' * B;
    Bu2 = 2 * v * A' * A;
    [nr0, nc0] = size(Bu1);
    
    for k = 1:maxIter
        % Gradient descent
        Bu = X + Bu1 - Bu2 * X;
        
        % L2-1/2 threhsolding operator
        normBu = zeros(nr0, 1);
        for i = 1:nr0
            normBu(i) = norm(Bu(i,:), 2);
        end
        Bu0 = sort(normBu); criterion = Bu0(nr0-s);
        lambda = Va1 * criterion^(1.5); q = lambda * v / 4;
        
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
            phi = acos(q * (3/normRow)^(1.5));
            eta = 16 * normRow^(3/2) * cos((pi - phi)/3)^3;
            Xnew(ind_temp, :) = (eta/(3 * sqrt(3) * lambda * v + eta)) * rowDa;
        end
        
        % Update and report
        X = Xnew;
        disp(['Complete the ', num2str(k), '-th iteration.']);
    end
    t = cputime - st;
end