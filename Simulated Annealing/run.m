%call_im(3000, 43);
D1 = call_fold_data();
    
for i=1:5
    [NE1] = call_pw_annealing_main(D1, i); 
end