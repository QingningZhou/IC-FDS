%%%%% hessian matrix

function output = likhess(para,m,p,vbu,vbv,nvbu,nvbv,datav,datanv,w,sz)
    beta = para(1:p); et = exp(para((p+1):(p+m+1))); eta = cumsum(et);
    
    %%% validataion sample
    
    del1v = datav(:,3); del2v = datav(:,4); covv = datav(:,5:(p+4));
    vLu = vbu*eta; vSu = exp(-vLu.*exp(covv*beta));
    vLv = vbv*eta; vSv = exp(-vLv.*exp(covv*beta));
    
    vcbu = fliplr(cumsum(fliplr(vbu),2));
    vcbv = fliplr(cumsum(fliplr(vbv),2));
    
    vhess = zeros((p+m+1),(p+m+1));
    
    for i = 1:size(datav,1)
      vSbu = vSu(i)*(-vLu(i)*exp(covv(i,:)*beta))*covv(i,:);
      vSbv = vSv(i)*(-vLv(i)*exp(covv(i,:)*beta))*covv(i,:);
      vSeu = vSu(i)*(-exp(covv(i,:)*beta))*(vcbu(i,:).*et');
      vSev = vSv(i)*(-exp(covv(i,:)*beta))*(vcbv(i,:).*et');
      
      vStu = [vSbu,vSeu]; vStv = [vSbv,vSev];  
      
      vSbbu = (-vLu(i)*exp(covv(i,:)*beta))*covv(i,:)'*(vSbu+vSu(i)*covv(i,:));
      vSbbv = (-vLv(i)*exp(covv(i,:)*beta))*covv(i,:)'*(vSbv+vSv(i)*covv(i,:));
      vSeeu = (-exp(covv(i,:)*beta))*(vcbu(i,:)'.*et)*vSeu;
      vSeev = (-exp(covv(i,:)*beta))*(vcbv(i,:)'.*et)*vSev;
      vSbeu = (-exp(covv(i,:)*beta))*covv(i,:)'*(vSeu*vLu(i)+vSu(i)*(vcbu(i,:).*et'));
      vSbev = (-exp(covv(i,:)*beta))*covv(i,:)'*(vSev*vLv(i)+vSv(i)*(vcbv(i,:).*et'));
      
      vSttu = [vSbbu,vSbeu;vSbeu',vSeeu]; vSttv = [vSbbv,vSbev;vSbev',vSeev]; 
      
      temp1 = -vSttu/(1-vSu(i))-vStu'*vStu/((1-vSu(i))^2);
      temp2 = (vSttu-vSttv)/(vSu(i)-vSv(i))-(vStu-vStv)'*(vStu-vStv)/((vSu(i)-vSv(i))^2);
      temp3 = vSttv/vSv(i)-vStv'*vStv/(vSv(i)^2);
      
      vhess = vhess+(del1v(i)*temp1+del2v(i)*temp2+(1-del1v(i)-del2v(i))*temp3);
    end
    
    %%% nonvalidation sample
    
    del1nv = datanv(:,3); del2nv = datanv(:,4); znv = datanv(:,6:(p+4));
    nvLu = nvbu*eta; nvLv = nvbv*eta; 
    
    nvcbu = fliplr(cumsum(fliplr(nvbu),2));
    nvcbv = fliplr(cumsum(fliplr(nvbv),2));
    
    csz = cumsum(sz); nvhess = zeros((p+m+1),(p+m+1));     
    for j = 1:size(datanv,1)
      temp1 = 0;  temp2 = zeros((p+m+1),1);  temp3 = zeros((p+m+1),(p+m+1));
      for k = 1:4
        xnv = covv((1+(k>1)*csz(max(k-1,1))):csz(k),1);
        covnv = [xnv,repmat(znv(j,:),sz(k),1)];
        nvSu = exp(-nvLu(j)*exp(covnv*beta));
        nvSv = exp(-nvLv(j)*exp(covnv*beta));
        temp4 = ((1-nvSu).^del1nv(j)).*((nvSu-nvSv).^del2nv(j)).*(nvSv.^(1-del1nv(j)-del2nv(j)));
        temp1 = temp1+w(k)*sum(temp4);
        
        temp5 = zeros((p+m+1),1);  temp6 = zeros((p+m+1),(p+m+1));
        for i = 1:size(covnv,1)
          nvSbu = nvSu(i)*(-nvLu(j)*exp(covnv(i,:)*beta))*covnv(i,:);
          nvSbv = nvSv(i)*(-nvLv(j)*exp(covnv(i,:)*beta))*covnv(i,:);
          nvSeu = nvSu(i)*(-exp(covnv(i,:)*beta))*(nvcbu(j,:).*et');
          nvSev = nvSv(i)*(-exp(covnv(i,:)*beta))*(nvcbv(j,:).*et');
      
          nvStu = [nvSbu,nvSeu]; nvStv = [nvSbv,nvSev];  
      
          nvSbbu = (-nvLu(j)*exp(covnv(i,:)*beta))*covnv(i,:)'*(nvSbu+nvSu(i)*covnv(i,:));
          nvSbbv = (-nvLv(j)*exp(covnv(i,:)*beta))*covnv(i,:)'*(nvSbv+nvSv(i)*covnv(i,:));
          nvSeeu = (-exp(covnv(i,:)*beta))*(nvcbu(j,:)'.*et)*nvSeu;
          nvSeev = (-exp(covnv(i,:)*beta))*(nvcbv(j,:)'.*et)*nvSev;
          nvSbeu = (-exp(covnv(i,:)*beta))*covnv(i,:)'*(nvSeu*nvLu(j)+nvSu(i)*(nvcbu(j,:).*et'));
          nvSbev = (-exp(covnv(i,:)*beta))*covnv(i,:)'*(nvSev*nvLv(j)+nvSv(i)*(nvcbv(j,:).*et'));
          
          nvSttu = [nvSbbu,nvSbeu;nvSbeu',nvSeeu]; nvSttv = [nvSbbv,nvSbev;nvSbev',nvSeev]; 
          
          dlogl = del1nv(j).*(-nvStu)./(1-nvSu(i))+del2nv(j).*(nvStu-nvStv)./(nvSu(i)-nvSv(i))+(1-del1nv(j)-del2nv(j)).*nvStv./nvSv(i);
                    
          temp7 = -nvSttu/(1-nvSu(i))-nvStu'*nvStu/((1-nvSu(i))^2);
          temp8 = (nvSttu-nvSttv)/(nvSu(i)-nvSv(i))-(nvStu-nvStv)'*(nvStu-nvStv)/((nvSu(i)-nvSv(i))^2);
          temp9 = nvSttv/nvSv(i)-nvStv'*nvStv/(nvSv(i)^2);
          ddlogl = del1nv(j)*temp7+del2nv(j)*temp8+(1-del1nv(j)-del2nv(j))*temp9;
          
          temp5 = temp5+dlogl'*temp4(i);
          temp6 = temp6+(ddlogl+dlogl'*dlogl)*temp4(i);
        end
        temp2 = temp2+w(k)*temp5;
        temp3 = temp3+w(k)*temp6;
      end
      nvhess = nvhess+(temp3*temp1-temp2*temp2')/(temp1^2);
    end
    
    output = -vhess-nvhess;
end