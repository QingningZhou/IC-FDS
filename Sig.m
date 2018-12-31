%%%%% middle part of variance sandwich

function output = Sig(para,m,p,nvbu,nvbv,datav,datanv,w1,w2,sz,N)
    beta = para(1:p); et = exp(para((p+1):(p+m+1))); eta = cumsum(et);
    
    covv = datav(:,5:(p+4)); znv = datanv(:,6:(p+4));
    del1nv = datanv(:,3); del2nv = datanv(:,4);
    nvLu = nvbu*eta; nvLv = nvbv*eta;
    csz = cumsum(sz);
    res = zeros((p+m+1),(p+m+1));
    for l = 1:4
      Sigl1 = zeros(sz(l),(p+m+1));
      for j = 1:size(datanv,1)
        temp1 = 0;  temp2 = zeros((p+m+1),1);
        for k = 1:4
          xnv = covv((1+(k>1)*csz(max(k-1,1))):csz(k),1);
          covnv = [xnv,repmat(znv(j,:),sz(k),1)];
          nvSu = exp(-nvLu(j)*exp(covnv*beta));
          nvSv = exp(-nvLv(j)*exp(covnv*beta));
          temp3 = ((1-nvSu).^del1nv(j)).*((nvSu-nvSv).^del2nv(j)).*(nvSv.^(1-del1nv(j)-del2nv(j)));
          if k == l
            temp5 = temp3;
          end
          temp1 = temp1+w1(k)*sum(temp3);
          
          temp4 = zeros(size(covnv,1),(p+m+1));
          for i = 1:p
            nvSbu = nvSu.*(-nvLu(j)*exp(covnv*beta)).*covnv(:,i);
            nvSbv = nvSv.*(-nvLv(j)*exp(covnv*beta)).*covnv(:,i);
            temp4(:,i) = del1nv(j).*(-nvSbu)./(1-nvSu)+del2nv(j).*(nvSbu-nvSbv)./(nvSu-nvSv)+(1-del1nv(j)-del2nv(j)).*nvSbv./nvSv;
          end
          for i = 1:(m+1)
            nvSeu = nvSu.*(-exp(covnv*beta))*sum(nvbu(j,i:(m+1)))*et(i);
            nvSev = nvSv.*(-exp(covnv*beta))*sum(nvbv(j,i:(m+1)))*et(i);
            temp4(:,i+p) = del1nv(j).*(-nvSeu)./(1-nvSu)+del2nv(j).*(nvSeu-nvSev)./(nvSu-nvSv)+(1-del1nv(j)-del2nv(j)).*nvSev./nvSv;
          end
          if k == l
            temp6 = temp4;
          end
          temp2 = temp2+w1(k)*(temp4'*temp3);
        end
        Sigl1 = Sigl1+temp6/temp1-temp5*temp2'/(temp1^2);
      end
      Sigl = zeros((p+m+1),(p+m+1)); mSigl1 = mean(Sigl1);
      for i = 1:sz(l)
        Sigl = Sigl+(Sigl1(i,:)-mSigl1)'*(Sigl1(i,:)-mSigl1);
      end
      res = res+w2(l)*Sigl/(sz(l)-1)/(N^2);
    end
    
    output = res;
end