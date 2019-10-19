[X,T]=call_im(50, 3);

[X1,T1,X2,T2]=split_image_set(X,T);
[PS1,PS2,C,M]=apply_pca(X1,X2);
Net=train_net(X1,X2,T1,T2);