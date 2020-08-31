clear all; close all
%-------------------------------------------------------------------------
%    K-NN Classification of Web-Pages
%-------------------------------------------------------------------------

%%%%% Initializations
NClasses = 4; %Number of Classes
nS = zeros(NClasses,1); %Number of Samples per class
 
%Define the names of the .csv files that contain the data
course_filename = 'Data/course_TDM.csv';
faculty_filename = 'Data/faculty_TDM.csv';
project_filename = 'Data/project_TDM.csv';
student_filename = 'Data/student_TDM.csv';

% Selecting method and amount
R = input('Enter the number of words to be used (300 recommended): ');
dtype = input('Enter (1) for norm2 and (2) for Cosine Similarity: ');
fprintf('Number of words selected: %d, ',R);
if dtype == 1
	disp('Distance Metric: norm2');
elseif dtype == 2
	disp('Distance Metric: Cosine Similarity');
else
	dtype = 2;
	disp('Not known distance selected. Considering Cosine Similarity');
end

K=[1 3 5 10]; %Different values for K nearest Neighbours


%-------------------------------------------------------------------------
%   READ and Preprocess the DATA
%-------------------------------------------------------------------------

%Get the vocabulary used in the term-documents from one of the files
Vocabulary=GetVocabulary(course_filename);  %Select one of the files as input

%Get the term frequencies from input files
f_course = (dlmread(course_filename,';',1,1)) ; 
f_student = (dlmread(student_filename,';',1,1)) ;
f_faculty = (dlmread(faculty_filename,';',1,1)) ;
f_project = (dlmread(project_filename,';',1,1)) ;

%Merge the 4 term frequency matrices
f=[f_course ; f_faculty ; f_project ; f_student];
    
%Create their corresponding labels (classes) (1,2,3 and 4)
lab_course=ones(size(f_course,1),1); 
lab_faculty=2*ones(size(f_faculty,1),1);
lab_project=3*ones(size(f_project,1),1);
lab_student=4*ones(size(f_student,1),1);
labels=[lab_course; lab_faculty; lab_project; lab_student];
            
%Entropy Transformation of the frequency matrix
%Also returns the entropy transformed features
e = getWordEntropy(f); %%%Complete your code in the function
tfidfFea = tfidf(f, 1);  %TF-IDF features

%Transform the feature vectors
f=tfidfFea;  %TF-IDF features

%Create the vocabulary of the top R entropy terms
%%%Complete your code in the function
[newVocabulary, newTermEntropy, new_f] = GetNewVocabulary(R, Vocabulary, e, f);

new_f = full(new_f);
%-------------------------------------------------------------------------
%        kNN Classification
%-------------------------------------------------------------------------
    for k=K
        x1=sprintf('\nNumber of K nearest neighbors: %d',k);
    	disp(x1);
        
		%Initializations for Classification
        NFolds=5; % Cross Validation Folds
		    NumErrors=zeros(NFolds,1);
        FoldAcc=zeros(NFolds,1);
		
        %Prepare the Cross Validation process
		    %foldInd: the indices per fold
		    %foldInd: the indices per fold
        foldInd=ones(1,length(labels)); %Fold indices
        for g = 1:NClasses %for each class
            gInd=find(labels==g); %Indices for class
            nS(g)=length(find(labels==g)); %Number of Class samples
            randInd = gInd(randperm(nS(g)));
            nSFC=floor(nS(g)/NFolds); %Num Samples per Fold per Class
            for cv=2:NFolds
                foldInd(randInd(1+(cv-1)*nSFC:cv*nSFC))=ones(1,nSFC)*cv;
            end
            %Assing to a random fold the remainder
            nRem=mod(nS(g),NFolds);
            randF=randi([1,NFolds],1,1);
            if nRem~=0
                foldInd(randInd(end-nRem+1:end))=ones(1,nRem)*randF;
            end
        end

		%Start the Classification and Cross Validation
        for cv = 1:NFolds
 
            test_f =  new_f(find(foldInd == cv), :);           % test features for fold cv
            test_y =  labels(find(foldInd == cv), :);      % test labels for fold cv
            train_f = new_f(find(foldInd ~= cv), :);          % train features for fold cv
            train_y =  labels(find(foldInd ~= cv), :);    % train labels for fold cv
            test_z = KNN_classify(k, train_f, train_y, test_f, dtype); %Return the predicted labels

            %Check Accuracy for fold
            
            FoldAcc(cv) = length(find((test_z == test_y)==1)) / size(test_z, 1);   %Accuracy of fold cv 
            
            fprintf('\nFold: %d, Accuracy: %f ',cv, FoldAcc(cv));
        end
        TotalAccuracy =  mean(FoldAcc);  %Total Accuracy 
        fprintf('\nK=%d -- Total Accuracy: %f',k, TotalAccuracy);
    end        


