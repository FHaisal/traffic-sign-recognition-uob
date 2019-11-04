
function Net=train_net(X1,X2,T1,T2) % 04/02
  %
  % Inputs: 
  %     X1,X2 are the training and validation images prepared  by the function
  %     split_image_set().
  % Function: 
  %     Train an artificial neural network (Net) with the given number (nofhn) 
  %     of heeden neurons and training method (trainscq).
  %     Validate the accuracy of the trained network (Net) on the validation data X2
  % Ouputs: 
  %     Net is the trained feedforward ANN
  % Settings: 
  %     nofhn=20 by default can be reasonably changed to improve 
  %     the recognition accuracy
  %
  nofhn=20; 
  Net=feedforwardnet(nofhn,'trainscg');
  Net.trainParam.epochs = 1000;
  TV1=full(ind2vec(T1));
  Net=train(Net,X1,TV1); % ?? overfitting on validation ?
  YV2=Net(X2);
  Y2=vec2ind(YV2);
  vacc=mean(T2==Y2);
  fprintf('. FR accuracy on validation set is %5.3f\n',vacc)
return


%train_net.m
%Open with
%Displaying train_net.m.