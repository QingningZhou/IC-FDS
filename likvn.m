%%%%% pseudo-likelihood function

function output = likvn(para,m,p,vbu,vbv,nvbu,nvbv,datav,datanv,w,sz)
    beta = para(1:p); eta = cumsum(exp(para((p+1):(p+m+1))));
    
    %%% validataion sample
    
    del1v = datav(:,3); del2v = datav(:,4); covv = datav(:,5:(p+4));
    vLu = vbu*eta; vSu = exp(-vLu.*exp(covv*beta));
    vLv = vbv*eta; vSv = exp(-vLv.*exp(covv*beta));
    
    vlogl = sum(del1v.*log(1-vSu)+del2v.*log(vSu-vSv)+(1-del1v-del2v).*log(vSv));
    
    %%% nonvalidation sample
    
    del1nv = datanv(:,3); del2nv = datanv(:,4); znv = datanv(:,6:(p+4));
    nvLu = nvbu*eta; nvLv = nvbv*eta;
    csz = cumsum(sz); temp = zeros(size(datanv,1),1);
    for j = 1:size(datanv,1)
      for k = 1:4
        xnv = covv((1+(k>1)*csz(max(k-1,1))):csz(k),1);
        covnv = [xnv,repmat(znv(j,:),sz(k),1)];
        nvSu = exp(-nvLu(j)*exp(covnv*beta));
        nvSv = exp(-nvLv(j)*exp(covnv*beta));
        temp(j) = temp(j)+w(k)*sum(((1-nvSu).^del1nv(j)).*((nvSu-nvSv).^del2nv(j)).*(nvSv.^(1-del1nv(j)-del2nv(j))));
      end
    end
    
    output = -vlogl-sum(log(temp));
end