###key proc.m

load -f temp.h5
u=reshape(u,101,101)';
plot(u(:,1:5:end));
