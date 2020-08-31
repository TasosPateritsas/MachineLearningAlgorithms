function dists = CosineSimilarity_Distance(testX, trainX)
%     Compute the Cosine Similarity distance between the 
%     current test sample and all training samples
%
% 	  testX: a single feature vector (test)
% 	  trainX: a matrix containing the set of training samples
%     dists: vector of the distances from the training samples

% ADD your code here

    dists = zeros(size(trainX, 1), 1);
    for i = 1:size(trainX, 1)

        dists(i) = dot(testX, trainX(i, :)) / ((norm(testX) * norm(trainX(i, :)))+0.0001);
    end

end
