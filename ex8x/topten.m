function mylist = topten(list,X,num)

%this calculates the distance not squared this time, but I added 
%abs as it made sense to me not to negate nefgatives and positives;
aux=sum(abs(X-ones(size(X,1),1)*X(num,:)),2);
%since that row may be tied at zero make an impossible lage number;
aux(num) = -9999999999999999999999;
[aux idx]=sort(aux); 
%now take elments 2:11;
mylist=list(idx(2:11));

end;





