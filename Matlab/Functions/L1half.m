function [X, t] = L1half(A, B, X, s, maxIter)
    st = cputime; % Record cputime
    % Initialization
    v = 0.5; % stepsize
    Bu1 = 2 * v * A' * B;
    Bu2 = 2 * v * A' * A;
    [nr0, nc0] = size(Bu1);
    
    for k = 1:maxIter
        % Gradient descent
        Bu = X + Bu1 - Bu2 * X;
        
        % L1-1/2 threhsolding operator
        normBu = zeros(nr0, 1);
        for i = 1:nr0
            normBu(i) = norm(Bu(i,:), 1);
        end
        Bu0 = sort(normBu); criterion = Bu0(nr0-s);
        lambda = 4 / (v * nc0) * (criterion/3)^(1.5); q = lambda * v / 4;
        
        % Consider what if s-th largest group is not the only one
        if criterion == Bu0(nr0-s-1)
            ind = find(q * nc0 * (3./normBu).^(1.5) <= 1 - 1e-8);
        else
            ind = find(q * nc0 * (3./normBu).^(1.5) < 1 - 1e-8);
        end
        
        % Update matrix
        Xnew = zeros(nr0, nc0);
        for j = 1:length(ind)
            ind_temp = ind(j);
            rowDa = Bu(ind_temp, :);
            normRow = norm(rowDa, 1);
            eta = q * (3/normRow)^(0.5);
            phi = acos(q * nc0 * (3/normRow)^(1.5));
            Xnew(ind_temp, :) = rowDa - eta / cos((pi - phi) / 3) * sign(rowDa);
        end
            
        % Update and report
        X = Xnew;
        disp(['Complete the ', num2str(k), '-th iteration.']);
    end
    t = cputime - st;
end