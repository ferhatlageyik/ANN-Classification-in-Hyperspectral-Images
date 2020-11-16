close all;
clear all;

HSI = load('salinas.mat');            %hiperspektral g�r�nt�
HSI = HSI.salinas;
HSI_GT = load('salinas_gt.mat');      %Groundtruth
HSI_GT = HSI_GT.salinas_gt;

ratio1 = 0.04;      % etiketi bilinen piksellerin y�zde ka�� kadar h�cre olu�turulsun
ratio2= 0.3;        % yol a��rl�klar�n�n g�ncellenme i�lemindeki iterasyon say�s�na denk gelir
OK = 0.003;         % ��renme katsay�s�     

[spat1,spat2,spec]=size(HSI);
[row, col, val] = find(HSI_GT);   
                    
Num_of_Classes = max(HSI_GT(:));  %ka� farkl� s�n�f var


t=0;
 % her etiketten rasgele noktalar se�me
for n=1:1:Num_of_Classes
    [row, col, val] = find(HSI_GT == n);
    [Num_of_sample , one] = size(val);
    a=round(Num_of_sample*ratio1);
    if(a<8)   %her s�n�ftan en az 8 h�cre olu�turulsun.
        a=8;
    end
    b=randperm(a,a);% rasgele se�im
    for i=1:1:a
        noron_points(i+t,1)=row(b(i));  % se�ilen piksellerin sat�r sut�n ve etiket bilgisini ayr� bir matrsite tut
        noron_points(i+t,2)=col(b(i));
        noron_points(i+t,3)=HSI_GT(row(b(i)),col(b(i)));
    end
    t=t+a;
end
n=size(noron_points,1); % n= h�cre say�s�

norons=zeros(n,1,spec); % �rnek say�s� kadar yeni h�cre olu�tur.
norons_labels=zeros(1,n); % h�cre etieketleri i�in bir dizi daha olu�tur.

for a=1:1:n       %h�crelere ilk yol a��rl�klar�n� atama
   norons(a,1,1:end)=HSI(noron_points(a,1),noron_points(a,2),1:end);
   norons_labels(1,a)=noron_points(a,3);
end

t=0;
% e�itim i�in her etiketten rasgele noktalar se�me
% h�cre se�imi i�leminin ayn�s� sadece oran farkl�

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

for i=1:1:counter %% e�itime girecek her �rnek i�in

        for j=1:1:n %% b�t�n h�crelerle kar��la�t�r
         distance=0;
            for band=1:1:spec
           distance = distance + (HSI(teach_points(i,1),teach_points(i,2),band)-norons(j,1,band))^2; %�klid mesafesi
            end
         dist(j,1)=sqrt(distance);
        end
        
        
    [v , index]=sort(dist(:,1)); % en k���k oklid mesafesine g�re s�ralama yap�l�r
    E = OK*dist(index(1),1);     % en d���k �klid mesafesini hata say�s� olarak tan�mlay�p direkt ��renme katsay�s�yla �arp�p bir de�i�kene atad�m
    
    
    %LVQ-X y�ntemine g�re yol a��rl�klar�n�n g�ncellenme i�lemi
    while(E~=0)  % en d���k mesafenin 0 gelmesi olas�l���na kar�� E de�erinin 0 olmas� durumunda d�ng�ye girmemesi i�in.
        
         if(((teach_points(i,3))== (norons_labels(1,index(1)))) && ((teach_points(i,3))== (norons_labels(1,index(2))))) %% en d���k mesafeye sahip ilk iki h�crenin de etiketi do�ru mu ?
     
              for k=1:1:spec
                norons(index(1),1,k) = norons(index(1),1,k)+ E; %% 1. h�crenin yol a��rl���n� kuvvetlendir
                norons(index(2),1,k) = norons(index(2),1,k)- E; %% 2. h�crenin yol a��rl���n� zay�flat.
              end
              
        elseif(((teach_points(i,3)) ~= (norons_labels(1,index(1)))) && ((teach_points(i,3)) ~= (norons_labels(1,index(2))))) %% en d���k mesafeye sahip ilk iki h�crenin de etiketi yanl�� m� ?
     
        for m=3:1:n %% en d���k mesafeye sahip yerel en iyiyi bul
          
            if((teach_points(i,3))==(norons_labels(1,index(m))))
                  
                for o=1:1:spec
                       norons(index(m),1,o) = norons(index(m),1,o)+ E; %% yerel en iyinin yol a��rl���n� kuvvetlendir.
                       norons(index(1),1,o) = norons(index(1),1,o)- E; %% genel en iyinin yol a��rl���n� zay�flat.
                   end
                break
                
            end
            
        end
        
        
        elseif(((teach_points(i,3)) ~= (norons_labels(1,index(1)))) && ((teach_points(i,3)) == (norons_labels(1,index(2))))) %% 1. h�crenin etiketi yanl�� 2. h�crenin etiketi do�ru ise 
         
            for s=1:1:spec
                norons(index(1),1,s) = norons(index(1),1,s) -E; %% 1. h�crenin yol a��rl���n� zay�flat.
                norons(index(2),1,s) = norons(index(2),1,s) +E; %% 2. h�crenin yol a��rl���n� kuvvlendir.
            end
         
        else    
    end
    E=0;  %% 0 olmayan herhangi bir E de�eri i�in d�ng�n�n bir kez d�nmesi gerekiyor.
    end
end
    distance=0;
    
    
    tagged=zeros(spat1,spat2); %yeni etiket de�erleri i�in bo� bir matris olu�tuduk.
for x=1:1:spat1
    for y=1:1:spat2
        for z=1:1:n  %% b�t�n h�crelere sor
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
 
 
 
 
 
 
 
 
 
 