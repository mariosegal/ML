z=zeros(size(x1,1));
for i = 1:size(x1,1)
   for j=1:size(x2,1)
         aux=[tx(i) ty(j)];
   		 z(i,j) = multivariateGaussian(aux, mu, sigma2);
    endfor;
endfor;
