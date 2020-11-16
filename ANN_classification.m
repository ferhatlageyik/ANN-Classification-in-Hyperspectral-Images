close all;
clear all;

HSI = load('salinas.mat');            %hiperspektral görüntü
HSI = HSI.salinas;
HSI_GT = load('salinas_gt.mat');      %Groundtruth
HSI_GT = HSI_GT.salinas_gt;

ratio1 = 0.04;      % etiketi bilinen piksellerin yüzde kaçý kadar hücre oluþturulsun
ratio2= 0.3;        % yol aðýrlýklarýnýn güncellenme iþlemindeki iterasyon sayýsýna denk gelir
OK = 0.003;         % öðrenme katsayýsý     

[spat1,spat2,spec]=size(HSI);
[row, col, val] = find(HSI_GT);   
                    
Num_of_Classes = max(HSI_GT(:));  %kaç farklý sýnýf var


t=0;
 % her etiketten rasgele noktalar seçme
for n=1:1:Num_of_Classes
    [row, col, val] = find(HSI_GT == n);
    [Num_of_sample , one] = size(val);
    a=round(Num_of_sample*ratio1);
    if(a<8)   %her sýnýftan en az 8 hücre oluþturulsun.
        a=8;
    end
    b=randperm(a,a);% rasgele seçim
    for i=1:1:a
        noron_points(i+t,1)=row(b(i));  % seçilen piksellerin satýr sutün ve etiket bilgisini ayrý bir matrsite tut
        noron_points(i+t,2)=col(b(i));
        noron_points(i+t,3)=HSI_GT(row(b(i)),col(b(i)));
    end
    t=t+a;
end
n=size(noron_points,1); % n= hücre sayýsý

norons=zeros(n,1,spec); % örnek sayýsý kadar yeni hücre oluþtur.
norons_labels=zeros(1,n); % hücre etieketleri için bir dizi daha oluþtur.

for a=1:1:n       %hücrelere ilk yol aðýrlýklarýný atama
   norons(a,1,1:end)=HSI(noron_points(a,1),noron_points(a,2),1:end);
   norons_labels(1,a)=noron_points(a,3);
end

t=0;
% eðitim için her etiketten rasgele noktalar seçme
% hücre seçimi iþleminin aynýsý sadece oran farklý

for m=1:1:Num_of_Classes
    [row, col, val] = find(HSI_GT == m);
    [Num_of_sample , one] = size(val);
    k=round(Num_of_sample*ratio2);
    p=randperm(k,k);
    for r=1:1:k
        teach_points(r+t,1)=row(p(r));
        teach_points(r+t,2)=col(p(r));
        teach_points(r+t,3)=HSI_GT(row(p(r)),col(p(r)));
    end
    t=t+k;
end

counter=size(teach_points,1);

for i=1:1:counter %% eðitime girecek her örnek için

        for j=1:1:n %% bütün hücrelerle karþýlaþtýr
         distance=0;
            for band=1:1:spec
           distance = distance + (HSI(teach_points(i,1),teach_points(i,2),band)-norons(j,1,band))^2; %öklid mesafesi
            end
         dist(j,1)=sqrt(distance);
        end
        
        
    [v , index]=sort(dist(:,1)); % en küçük oklid mesafesine göre sýralama yapýlýr
    E = OK*dist(index(1),1);     % en düþük öklid mesafesini hata sayýsý olarak tanýmlayýp direkt öðrenme katsayýsýyla çarpýp bir deðiþkene atadým
    
    
    %LVQ-X yöntemine göre yol aðýrlýklarýnýn güncellenme iþlemi
    while(E~=0)  % en düþük mesafenin 0 gelmesi olasýlýðýna karþý E deðerinin 0 olmasý durumunda döngüye girmemesi için.
        
         if(((teach_points(i,3))== (norons_labels(1,index(1)))) && ((teach_points(i,3))== (norons_labels(1,index(2))))) %% en düþük mesafeye sahip ilk iki hücrenin de etiketi doðru mu ?
     
              for k=1:1:spec
                norons(index(1),1,k) = norons(index(1),1,k)+ E; %% 1. hücrenin yol aðýrlýðýný kuvvetlendir
                norons(index(2),1,k) = norons(index(2),1,k)- E; %% 2. hücrenin yol aðýrlýðýný zayýflat.
              end
              
        elseif(((teach_points(i,3)) ~= (norons_labels(1,index(1)))) && ((teach_points(i,3)) ~= (norons_labels(1,index(2))))) %% en düþük mesafeye sahip ilk iki hücrenin de etiketi yanlýþ mý ?
     
        for m=3:1:n %% en düþük mesafeye sahip yerel en iyiyi bul
          
            if((teach_points(i,3))==(norons_labels(1,index(m))))
                  
                for o=1:1:spec
                       norons(index(m),1,o) = norons(index(m),1,o)+ E; %% yerel en iyinin yol aðýrlýðýný kuvvetlendir.
                       norons(index(1),1,o) = norons(index(1),1,o)- E; %% genel en iyinin yol aðýrlýðýný zayýflat.
                   end
                break
                
            end
            
        end
        
        
        elseif(((teach_points(i,3)) ~= (norons_labels(1,index(1)))) && ((teach_points(i,3)) == (norons_labels(1,index(2))))) %% 1. hücrenin etiketi yanlýþ 2. hücrenin etiketi doðru ise 
         
            for s=1:1:spec
                norons(index(1),1,s) = norons(index(1),1,s) -E; %% 1. hücrenin yol aðýrlýðýný zayýflat.
                norons(index(2),1,s) = norons(index(2),1,s) +E; %% 2. hücrenin yol aðýrlýðýný kuvvlendir.
            end
         
        else    
    end
    E=0;  %% 0 olmayan herhangi bir E deðeri için döngünün bir kez dönmesi gerekiyor.
    end
end
    distance=0;
    
    
    tagged=zeros(spat1,spat2); %yeni etiket deðerleri için boþ bir matris oluþtuduk.
for x=1:1:spat1
    for y=1:1:spat2
        for z=1:1:n  %% bütün hücrelere sor
          for band=1:1:spec
              distance = distance + (HSI(x,y,band)-norons(z,1,band))^2;
          end
          dist(z,1)=sqrt(distance);
          distance=0;
        end
        [v , index] = sort(dist(:,1));
        tagged(x,y)= norons_labels(1,index(1));
        
    end
end

 figure;
 subplot(1,2,1); imagesc(tagged); title('ANN classified image');
 subplot(1,2,2); imagesc(HSI_GT);title('Groundtruth');



 true=0;
 false=0;
 for x=1:spat1 
      for y=1:spat2
          if(HSI_GT(x,y)~=0)
              if(tagged(x,y)==HSI_GT(x,y))
                  true = true+1;
              else
                  false = false+1;
              end
          end
      end
 end
 success_rate= true*100 / (true + false);
 print=['overall accuravy of ANN classification = %',num2str(success_rate)];
 disp(print)

 average_accuracy=zeros(Num_of_Classes,1);

 for n=1:1:Num_of_Classes
 [row, col, val] = find(HSI_GT == n);
 [num_of_label , one] = size(val);
 
 count=0;
    for i=1:1:num_of_label
        if(tagged(row(i),col(i))==HSI_GT(row(i),col(i)))
            count=count+1;
        end
    end
     average_accuracy(n,1)=count*100/num_of_label;
 end
 
 
 
 
 
 
 
 
 
 