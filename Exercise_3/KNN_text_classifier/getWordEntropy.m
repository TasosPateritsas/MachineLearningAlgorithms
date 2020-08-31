function [e]=getWordEntropy(f)
    %Calculate the vector of word entropies e 
    %from the Term Frequency matrix f
    %Add your code here
    
    % Calculate normalized frequency p and entropy e

    ND = size(f,1);
    NT = size(f,2);
   for j= 1:NT 
        s = 0;
        for i = 1:ND 
            p(i, j) = f(i, j) / sum(f(:, j));
            s = s + p(i, j) * log(p(i, j) + 1);
        end
        e(j) = 1+s/log(size(f, 1)+1);
    end

end