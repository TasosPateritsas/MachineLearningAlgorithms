function index = findBin(value, binEnds)
    numBins = length(binEnds) - 1;
    % return the index of the bin where value belongs {1, ..., numBins}. 
    index = numBins;
    for i = 1:numBins
      if (value >= binEnds(i)) && (value <=binEnds(i+1))
        if i == 1 
          index = [1 0 0]';
        end
        if i == 2 
          index = [0 1 0]';
        end
        if i == 3
          index = [0 0 1]';
        end
        
      end
    end 
end