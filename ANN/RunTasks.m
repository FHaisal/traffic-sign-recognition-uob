[X,T]=call_im(3000, 43);

for i=0:4
    [X1,T1,X2,T2]=split_image_set(X,T);
    [PS1,PS2,C,M]=apply_pca(X1,X2);
    Net=train_net(X1,X2,T1,T2);
end