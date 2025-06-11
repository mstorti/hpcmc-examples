###key proc.m

k = 0;
kinc = 20;
while 1
  file = sprintf("./STEPS/tempo%d.h5",k);
  if !exist(file); break; endif
  load("-f",file);
  n = length(u);
  m = round(sqrt(n));
  assert(n==m^2);
  u = reshape(u,m,m);
  x = ((1:m)'-1)/m;
  plot(x,u(:,1:3:end));
  axis([0,1,0,1]);
  title(sprintf("k %d",k));
  pause(0.1);
  k += kinc;
endwhile
