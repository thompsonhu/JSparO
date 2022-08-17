function [X, t] = L20(A, B, X, s, maxIter)
    st = cputime; % Record cputime
    % Initialization
    v = 0.5; % stepsize
    Bu1 = 2 * v * A' * B;
    Bu2 = 2 * v * A' * A;
    nr0 = size(Bu1, 1);
    
    for k = 1:maxIter
        % Gradient descent
        Bu = X + Bu1 - Bu2 * X;
        
        % L20 threhsolding operator
        normBu = zeros(nr0);
        for i = 1:nr0
            normBu(i) = norm(Bu(i,:), 2);
        end
        Bu0 = sort(normBu); criterion = Bu0(nr0-s);
        
        % Consider what if s-th largest group is not the only one
        if criterion == Bu0(nr0-s-1)
            ind = find(normBu >= criterion);
        else
            ind = find(normBu > criterion);
        end
        
        % Update matrix
        Xnew = Bu;
        Xnew(setdiff(1:nr0, ind), :) = 0;
        
        % Update and report
        X = Xnew;
        disp(['Complete the ', num2str(k), '-th iteration.']);
    end
    t = cputime - st;
end