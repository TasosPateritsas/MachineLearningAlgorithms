function [dists] = Euclidean_Distance(testX, trainX)
%     Compute the Euclidean  distance between the 
%     current test sample and all training samples
%
% 	  testX: a single feature vector (test)
% 	  trainX: a matrix containing the set of training samples
%     dists: vector of the distances from the training samples

%  ADD your code here

    % Euclidean distance
    A = repmat(testX, size(trainX, 1), 1) - trainX;
    dists = sqrt(sum(A.^2, 2));
    
end