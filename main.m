%%%%% Semiparametric inference for a two-stage outcome-dependent sampling design with interval-censored failure time data
%%%%% This paper has been accepted by Lifetime Data Analysis


%%%%%%%%%%%%%%% generate data

N = 3000;
sizes = [300,25,25];
nv = sum(sizes);
nnv = N-nv;
n0 = sizes(1); n1 = sizes(2); n2 = sizes(3);

be0 = [log(2),0.5]';
p = length(be0);
u = 2.5; l = 0;
a1 = 0.57; a2 = 1.23;

inat = 0.2;
nt = ceil(u/inat*2);
pv = 0.2;
m = 3;

ru = random('Uniform',0,1,N,1);
x = random('Normal',0,1,N,1);
z = random('Normal',0,1,N,1);
T = -log(1-ru).*exp(-[x,z]*be0)/0.1;

U = zeros(N,1); V = U; del1 = U; del2 = U;
for i = 1:N
  obst1 = cumsum(random('Uniform',0,2*inat,1,nt));
  obst1 = obst1(obst1<u); ant1 = length(obst1); mi = zeros(1,ant1);
  while sum(mi) == 0
    mi = random('Binomial',1,pv,1,ant1);
  end
  obst = nonzeros(obst1.*mi); ant = length(obst);
  if T(i) <= obst(1)
    U(i) = obst(1); V(i) = obst(min(2,ant))*(ant>1)+u*(ant==1); del1(i) = 1;
  elseif ant > 1
   for j = 2:ant
     if (T(i) > obst(j-1)) && (T(i) <= obst(j))
       U(i) = obst(j-1); V(i) = obst(j); del2(i) = 1;
     end
   end
  end
  if T(i) > obst(ant)
     U(i) = obst(max(1,ant-1))*(ant>1); V(i) = obst(ant);
  end
end

dataf = [U,V,del1,del2,x,z];
index1 = ((del1==1)&(U<=a1))|((del2==1)&(V<=a1)); N1 = sum(index1);
index2 = ((del2==1)&(U>=a2)); N2 = sum(index2);
index3 = (((del1==1)|(del2==1))&(1-index1))&(1-index2); N3 = sum(index3);
index4 = ((del1==0)&(del2==0)); N4 = sum(index4);

ind0 = zeros(N,1); ind0(datasample(1:N,n0,'Replace',false)) = 1;
data0 = dataf(ind0==1,:);
ind01 = ind0&index1; data01 = dataf(ind01,:); n01 = sum(ind01);
ind02 = ind0&index2; data02 = dataf(ind02,:); n02 = sum(ind02);
ind03 = ind0&index3; data03 = dataf(ind03,:); n03 = sum(ind03);
ind04 = ind0&index4; data04 = dataf(ind04,:); n04 = sum(ind04);

ind1 = index1&(1-ind0); temp1 = zeros(sum(ind1),1);
temp1(datasample(1:sum(ind1),n1,'Replace',false)) = 1;
ind1(ind1==1) = temp1;
data1 = dataf(ind1,:);

ind2 = index2&(1-ind0); temp2 = zeros(sum(ind2),1);
temp2(datasample(1:sum(ind2),n2,'Replace',false)) = 1;
ind2(ind2==1) = temp2;
data2 = dataf(ind2,:);

datav = [data01;data1;data02;data2;data03;data04];

indnv = ((1-ind0)&(1-ind1))&(1-ind2);
datanv = dataf(indnv,:);


%%%%%%%%%%%%%%% proposed method

vbu = zeros(nv,(m+1)); vbv = vbu;
for i = 0:m
   vbu(:,(i+1)) = bern(i,m,l,u,datav(:,1));
   vbv(:,(i+1)) = bern(i,m,l,u,datav(:,2));
end

nvbu = zeros(nnv,(m+1)); nvbv = nvbu;
for i = 0:m
   nvbu(:,(i+1)) = bern(i,m,l,u,datanv(:,1));
   nvbv(:,(i+1)) = bern(i,m,l,u,datanv(:,2));
end

w1 = [N1/(N*(n1+n01)),N2/(N*(n2+n02)),N3/(N*n03),N4/(N*n04)];
sz = [(n1+n01),(n2+n02),n03,n04];
initial = [repmat(0.5,p,1);repmat(-0.5,(m+1),1)];

opt1 = optimset('LargeScale','off','MaxFunEvals',10000,'MaxIter',10000,'Display','off');
[ml1,fval1,exitflag1] = fminunc(@(y) likvn(y,m,p,vbu,vbv,nvbu,nvbv,datav,datanv,w1,sz),initial,opt1);
% ml1(1:p) is the estimate of beta

w2 = [((N1/N)^2)/(n1+n01),((N2/N)^2)/(n2+n02),((N3/N)^2)/n03,((N4/N)^2)/n04];
hessian1 = likhess(ml1,m,p,vbu,vbv,nvbu,nvbv,datav,datanv,w1,sz);
Sigmid1 = Sig(ml1,m,p,nvbu,nvbv,datav,datanv,w1,w2,sz,N);
Sigma1 = pinv(hessian1)+pinv(hessian1)*Sigmid1*pinv(hessian1);
seb1 = sqrt(diag(Sigma1(1:p,1:p)));  % variance estimate for beta


