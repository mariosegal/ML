function [idx list] = findMovies(list1,str1)

IndexC = strfind(list1, str1);
idx = find(not(cellfun('isempty', IndexC)));
list = list1(idx);

end;
