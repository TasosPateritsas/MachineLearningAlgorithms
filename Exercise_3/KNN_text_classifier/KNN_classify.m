function [test_z] = KNN_classify(k, train_f, train_y, test_f, dtype)
   % K-NN classification algorithm
   % k: Number of nearest neighbours
   % train_f: The matrix of training feature vectors
   % train_y: The labels of the training data
   % test_f: The matrix of the test feature vectors
   % dtype: Integer which defines the distance metric
   %    dtype=1: Call the function Euclidean_Distance
   %    dtype=2: Call the function CosineSimilarity_Distance
 
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Initialization
    test_z = zeros(size(test_f, 1), 1);

    for i = 1:size(test_f, 1)

        if dtype == 1 
            dist = Euclidean_Distance(test_f(i, :), train_f);
            [map, idx] = sort(dist);
        else
            dist = CosineSimilarity_Distance(test_f(i, :), train_f);
            [map, idx] = sort(dist, 'descend');
        end

        neighbours = train_y(idx(1:k));
        edges = unique(neighbours);
        counts = histc(neighbours, edges);
        [~, max_c] = max(counts);
        test_z(i) = edges(max_c);
    
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end