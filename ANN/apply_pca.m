function [PS1,PS2,C,M]=apply_pca(X1,X2) % 04/02
  %
  % Notation: 
  %     nofpc is the number of principal components extracted from the
  %     images
  % Inputs: 
  %     X1,X2 are the training and validation images prepared 
  %     by the function split_image_set().
  % Function:
  %     PCA is applied to the set X1 to find the components P1 and matrix of
  %     coefficients C.
  % Ouputs: 
  %     PS1 is the (nofpc by n1) matrix of components for training set.
  %     PS2 is the (nofpc by n2) matrix of components for training set.
  % Settings:
  %     nofpc=100 by default can be reasonably changed to improve 
  %     the recognition accuracy
  %
  nofpc=100;
  % train set:
  [P1,C]=processpca(X1);   
  [PM1,M]=mapstd(P1);      
  % test set:
  P2=processpca('apply',X2,C);   
  PM2=mapstd('apply',P2,M);
  % get components:
  PS1=PM1(1:nofpc,:);
  PS2=PM2(1:nofpc,:);
return

