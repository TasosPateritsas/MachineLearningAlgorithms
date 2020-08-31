function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
  [NumSamples NumFeatures] = size(Samples);
	A=zeros(NumFeatures,NewDim);
    
	
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes)  %The number of classes


    %For each class i
	%Find the necessary statistics
%Calculate the Class Prior Probability
  for i=1:NumClasses %For all classes
      P(i)=sum(Labels==(i-1))/NumLabels;
  end

  %Calculate the Class Mean 
  for i=1:NumClasses 
        mu(:,i)=mean(Samples(Labels==(i-1),:));
  end
  %Calculate the Within Class Scatter Matrix
  Sw=zeros(NumFeatures, NumFeatures);
  for i=1:NumClasses 
     Sw=Sw+P(i)*cov(Samples(Labels==(i-1),:));
  end
  %Calculate the Global Mean
  m0 = mean(mu);

  %Calculate the Between Class Scatter Matrix
  Sb=zeros(NumFeatures, NumFeatures);
  for i=1:NumClasses %For all classes
    means = (mu(:,i)-m0)*(mu(:,i)-m0)';
    Sb=Sb+P(i)*means;
  end
    
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    [V,D]=eig(EigMat);
    %Perform Eigendecomposition
    eigenval=diag(D); 
    [eigenval,ind]=sort(eigenval,1,'descend'); 
    eigenvec=V(:,ind);
  

    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A=eigenvec(:,1:NewDim);  % Return the LDA projection vectors
