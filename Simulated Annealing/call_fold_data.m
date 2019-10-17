function D1=call_fold_data() % ver 03/19
  %  
  % Calculation of PCA for train and validation data
  % for a given number (nf) of cross-validation folds 
  % and a given PCA cutoff (thresh) [see MATLAB PCA]
  % Output D1 contains the results
  % 
  nf = 3; % nof cross-validation folds 
  thresh = 1e-5; % a given PCA cutoff 
  load im.mat data target noc;
  [X,n] = call_norm_brightness(data); 
  T = target;
  D = repmat(struct('PC1',[],'T1',[],...
    'PC2',[],'T2',[]),nf,1);
  for i = 1:nf
    fprintf('.fold=%i \n',i)
    I2 = i:nf:n; % validation image indexes
    I1 = 1:n;
    I1(I2) = []; % train image indexes
    X1 = X(:,I1);
    T1 = T(I1);
    X2 = X(:,I2);
    T2 = T(I2);
	% PCA for training
    [X1p,Coef] = processpca(X1, thresh);
    [X1p,PS] = mapstd(X1p);
	% PCA for validation
    X2p = processpca('apply',X2,Coef);
    X2p = mapstd('apply',X2p,PS); 
    D(i) = struct('PC1',X1p,'T1',T1,'PC2',X2p,'T2',T2);
  end
  D1 = struct('D',D,'noc',noc);
return

function [X,n]=call_norm_brightness(data) % ver 03/19
  n = size(data,2); % nof images
  X = zeros(size(data)); 
  for i = 1:n
    A = data(:,i);
    mm = minmax(A');
% pixel brightness ranges in [0,255]:
    A = 255*(A - mm(1))/(mm(2) - mm(1)); 
    X(:,i) = A;
  end
return