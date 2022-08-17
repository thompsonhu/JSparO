function [X, t] = L2TwoThir(A, B, X, s, maxIter)
    st = cputime; % Record cputime
    % Initialization
    v = 0.5; % stepsize
    Va1 = (0.5)^(4/3) * 1.5 / v;
    Bu1 = 2 * v * A' * B;
    Bu2 = 2 * v * A' * A;
    [nr0, nc0] = size(Bu1);
    
    for k = 1:maxIter
        % Gradient descent
        Bu = X + Bu1 - Bu2 * X;
        
        % L2-2/3 threhsolding operator
        normBu = zeros(nr0);
        for i = 1:nr0
            normBu(i) = norm(Bu(i,:), 2);
        end
        Bu0 = sort(normBu); criterion = Bu0(nr0-s);
        lambda = Va1 * criterion^(4/3); q = 2 * lambda * v;
        
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
            phi = acosh(27 * normRow^2 / (16 * q^(1.5)));
            aa = 2 * q^0.25 * (cosh(phi/3))^(0.5) / (sqrt(3));
            eta = 3 * (aa^(1.5) + sqrt(2 * normRow - aa^3))^4;
            Xnew(ind_temp, :) = (eta/(32 * lambda * v * aa^2 + eta)) * rowDa;
        end
        
        % Update and report
        X = Xnew;
        disp(['Complete the ', num2str(k), '-th iteration.']);
    end
    t = cputime - st;
end