function [X1,T1,X2,T2]=split_image_set(X,T) % 04/02
  %
  % Inputs: 
  %     X, T are the input data and labels calculated by the function read_image_set().
  % Function:
  %     The data X,T are split into training and validation subsets X1,T1 and X2,T2.
  %     Images in the subsets are randomly assigned accordingly 
  %     to the given val_ratio.
  % Ouputs: 
  %     X1 is the (L by n1) matrix of training set.
  %     T1 is the n1-element target vector of images in the validation set. 
  %     X2 is the (L by n2) matrix of validation set.
  %     T2 is the n2-element target vector of images in the validation set.   
  % Settings:
  %     val_ratio=0.3. This ratio can be reasonably changed to achive 
  %     the maximal accuracy of unseeen images     .
  %
  val_ratio=0.3; 
  nofim=size(X,2);  % nof images
  I1=true(1,nofim); 
  nofvalimages=round(nofim*val_ratio);
  A2=randsample(nofim,nofvalimages);  % validation images
  I1(A2)=false;    % flags for train images
  I2=~I1;           % flags for validation images
  X1=X(:,I1);
  T1=T(I1);
  X2=X(:,I2);
  T2=T(I2);
return

