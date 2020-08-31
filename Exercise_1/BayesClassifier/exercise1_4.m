data_file = './data/mnist.mat';

data = load(data_file);

images = zeros(size(data.trainX, 1), 28, 28);
labels = zeros(size(data.trainY), 1);

for i = 1:size(data.trainX, 1)
    img = data.trainX(i, :);
    images(i, :, :) = reshape(img, 28, 28)';
    labels(i) = data.trainY(i);
end

digit_C1_indices = find(labels == 1); % digit 1
digit_C2_indices = find(labels == 2); % digit 2

digit_C1_images = images(digit_C1_indices, :, :);
digit_C2_images = images(digit_C2_indices, :, :);


aRatios_c1 = zeros(size(img,1)); % 
aRatios_c2 = zeros(size(img,1)); % 

for i = 1:size(digit_C1_images)
  aRatios_c1(i) = computeAspectRatio(digit_C1_images(i,:,:));
end

for i = 1:size(digit_C2_images)
  aRatios_c2(i) = computeAspectRatio(digit_C2_images(i,:,:));
end

% Compute the aspect ratios of all images and store the value of the i-th image in aRatios(i)

minAspectRatio = min([min(aRatios_c1) min(aRatios_c2)])
maxAspectRatio = max([max(aRatios_c1) max(aRatios_c2)])

numBins = 3;

binEnds = linspace(minAspectRatio, maxAspectRatio, numBins+1);

C1_bins = zeros(numBins, 1);
C2_bins = zeros(numBins, 1);
all_bins = zeros(numBins, 1);

% Use the findBin function to get the counts for the histogram
for i = 1:size(digit_C1_images)
  C1_bins = C1_bins+findBin(aRatios_c1(i),binEnds);
end

for i = 1:size(digit_C2_images)
  C2_bins = C2_bins+findBin(aRatios_c2(i),binEnds);
end

all_bins = C1_bins +C2_bins 

figure(1);
bar(C1_bins,C1_bins,'FaceColor','g');

figure(2);
bar(C2_bins,C2_bins,'FaceColor','r');

% Prior Probabilities
PC1 = size(digit_C1_images)/(size(digit_C1_images )+ size(digit_C2_images))
PC2 = size(digit_C2_images)/(size(digit_C1_images )+ size(digit_C2_images))

% Likelihoods
PC1_L = (C1_bins(1) / size(digit_C1_images,1));
PC1_M = (C1_bins(2) / size(digit_C1_images,1));
PC1_H = (C1_bins(3) / size(digit_C1_images,1));
PgivenC1 = [PC1_L PC1_M PC1_H]'
PC2_L = (C2_bins(1) / size(digit_C2_images,1));
PC2_M = (C2_bins(2) / size(digit_C2_images,1));
PC2_H = (C2_bins(3) / size(digit_C2_images,1));
PgivenC2 = [PC2_L PC2_M PC2_H]'


% Evidence 
P_L=PC1_L*PC1 + PC2_L*PC2
P_M=PC1_M*PC1 + PC2_M*PC2
P_H=PC1_H*PC1 + PC2_H*PC2
% Posterior Probabilities
PC1givenL = (PC1_L*PC1)/P_L
PC2givenL = (PC2_L*PC2)/P_L