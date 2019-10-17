function [NE1]=call_pw_annealing_main(D1,fi) % ver 03/2019
%
% AIM: Annealing Search Algorithm to optimise a PWANN
% INPUTS:
% - D1 are the image data created by function
% - fi=1:5 is the folder index
%
global nobc noc Cb nf
 D=D1.D; % image data
 noc=D1.noc; % nof classes (persons)
 Cb = combnk(1:noc,2); % binary classifiers indexes
 nf = size(D,2); % nof folds
 nobc=noc*(noc-1)/2; % nof binary (pairwise) classifiers
 D1=D(fi);
 [Y2,E2,NE1]=call_bin_ANN_train(D1);
 Y=call_pw_ANN(Y2);
 pf=mean(Y'==D1.T2);
 call_print(pf,E2,NE1,fi);
return


function call_print(pf,E2,NE,fi) % 12/03
 global Cb noc
 fprintf('.fold=%i, Pf=%5.3f\n',fi,pf)
 figure(1)
 plot(E2)
 title(sprintf('Validation errors on fold %i \n',fi))
 grid
 xlabel('Pairwise ANNs')
 ylabel('Error')
 fprintf('.ANNs with largest validation error:\n')
 [Me,Ie]=sort(E2,'descend');
 for i=1:noc
  ci=Ie(i);
  fprintf(' %2i: %3i (%2i/%2i) %5.3f,acpr=%3.1f,nopc=%3.0f  \n',...
  i,ci,Cb(ci,:),Me(i),NE(ci).accr,NE(ci).nopc)
 end
return


function [Y2,E2,NE1]=call_bin_ANN_train(D1) % 12/03
 % Y2 are PWANN outcomes on validation set
 % E2 are validation errors
 % NE1 are PWANN
 % NES are settings
 global nobc noc nohn esize Av minpc nofcp
 nohn=1; % nof hidden neurons
 esize=3; % ensemble size
 nofcp=80; % prior nof pc
 minpc=20; % min nof pc
 av=2; % annealing variance  NES=struct('nohn',nohn,'esize',esize,'nofcp',nofcp,'minpc',minpc,'av',av);
 Av1=av*(1:5); % Annealing variance of pc
 Av=[-Av1, Av1];
 X1=D1.PC1;
 T1=D1.T1;
 X2=D1.PC2;
 T2=D1.T2;
 n2 = size(X2,2);
 Y2=zeros(n2,nobc); % outcomes of ANNs on validation set
 E2=zeros(nobc,1); % train error of binary ANNs
 Ac=zeros(nobc,1); % acceptance rates
 NE1=repmat(struct('NE',{},'accr',0,'nopc',0),nobc,1);
 warning ('off','NNET:Obsolete');
 ic=0;
 for i1=1:noc-1
  I1=T1==i1; % mask of i1-th person images
  n1 = sum(I1);
  for i2=i1+1:noc
   I2=T1==i2; % mask of i2-th person images
   n2=sum(I2);
   T=[ones(1,n1) -1*ones(1,n2)]; % targets for binary ANN
   X=[X1(:,I1) X1(:,I2)];
   [NE,accr,nopc]=call_tr_ANN_ens(X,T);
   ic=ic+1;
   Ac(ic)=accr;
   Y2(:,ic)=call_ts_ANN_ens(NE,X2);
   % val_error:
   J1=T2==i1;
   J2=T2==i2;
   v1=sum(J1);
   v2=sum(J2);
   Tv=[ones(1,v1), -1*ones(1,v2)];
   Yv=call_ts_ANN_ens(NE,[X2(:,J1) X2(:,J2)]);
   E2(ic)=mean(sign(Yv)~=Tv');
   NE1(ic)=struct('NE',NE,'accr',accr,'nopc',nopc);
  end
 end
return


function [NE,acr,nopc]=call_tr_ANN_ens(X,T) % 12/03
 % train ANN ensemble
 global nohn esize Av minpc nofcp
 lenAv=length(Av);
 Ac=zeros(esize,2); % acceptance
 NE=repmat(struct('net',{},'nofin',0),esize,1);
 maxpc=size(X,1); % max nof pc
 Nc=zeros(esize,1);
 nofc=nofcp; % prior on nof pc
 lik=-Inf;
 for i=1:esize
  v1=nofc+Av(randi(lenAv)); % proposed nof pc
  v1=min(v1,maxpc);
  v1=max(v1,minpc);
  V=1:v1;
  net=newff(minmax(X(V,:)),[nohn 1],{'tansig' 'tansig'},'trainlm');
  net.trainParam.show = NaN;
  net.trainParam.epochs = 50;
  net.trainParam.showWindow = false;
  net=train(net,X(V,:),T);
  Y=sim(net,X(V,:));
  lik1=call_lik(Y,T);
  r=exp(lik1-lik);
  if rand < r % accept proposal
   ac=1;
   lik=lik1;
   nofc=v1;
  else
   ac=0;
   lik=lik1;
  end
  Nc(i)=nofc;
  NE(i)=struct('net',net,'nofin',nofc);
  Ac(i,:)=[ac,lik];
 end
 I=Ac(:,1)==0; % indexes of rejected ann
 NE(I)=[];
 acr=mean(Ac(:,1));
 nopc=mean(Nc(~I));
return


function lik=call_lik(Y0,T0) % 17/03
 % Likelihood (-Inf, 0) of ANN, Netlab book page 125 Eq 4.16,  t={0,1}
 % Y0= (1,-1) are ANN outcomes, T0={1,-1} labels of classes 1 and 2
 I=T0==-1; % indexes of images of second class
 Y=(Y0+1)/2;
 lik0=sum(log(1-Y(I)));
 lik1=sum(log(Y(~I)));
 lik=lik0+lik1;
return


function Ye=call_ts_ANN_ens(NE,X) % 12/03
 % test ANN ensemble on X for a given # components
 % Ye is ensemble output
 ntest=size(X,2);
 sizeNE=size(NE,2);
 Y=zeros(ntest,sizeNE);
 for i=1:sizeNE
  net=NE(i).net;
  V=1:NE(i).nofin;
  Y(:,i)=sim(net,X(V,:));
 end
 Ye=single(sum(Y,2));
return


function Yi=call_pw_ANN(Yb) % 12/03
 % output layer for PWANNs
 % Yi are predicted classes
 global Cb noc
 n2 = size(Yb,1); % nof validation images
 Y = zeros(n2,noc); % outcomes
 for c = 1:noc
  I1 = Cb(:,1) == c; % +1 outputs
  I2 = Cb(:,2) == c; % -1 outputs
  B1 = sum(Yb(:,I1),2);
  B2 = sum(Yb(:,I2),2);
  Y(:,c) = B1 - B2;
 end
[~,Yi]=max(Y,[],2);
return
