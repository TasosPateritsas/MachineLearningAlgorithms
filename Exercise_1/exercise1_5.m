clear all ;
close all;
% this is only if you run this script on pc
%pkg load statistics 

S1 = [1.2 -0.4; -0.4 1.2];
S2 = [1.2 0.4; 0.4 1.2];
mu1=[3 3];
mu2=[6 6];


x1 = [-4 : 0.01 : 12] ;
x2 = [-4 : 0.01 : 12] ;
[X1,X2] = meshgrid(x1,x2);
%------------------------------------------ S1!=S2--------------------------------------------------------
figure(1)
hold on
Y1 = mvnpdf([X1(:) X2(:)] ,mu1,S1);
Y1_reshape = reshape(Y1,length(x2),length(x1));
contour(x1,x2,Y1_reshape,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999])
grid on

hold on
Y2 = mvnpdf([X1(:) X2(:)] ,mu2,S2);
Y2_reshape = reshape(Y2,length(x2),length(x1));
contour(x1,x2,Y2_reshape,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999])
grid on

Pw1 = 0.1 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,'b','LineWidth',2)
grid on

Pw1 = 0.25 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,'g','LineWidth',2)
grid on

Pw1 = 0.5 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,'r','LineWidth',2)
grid on

Pw1 = 0.75 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,[0.9 0 0.9],'LineWidth',2)
grid on

Pw1 = 0.9 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,[0.5 0.3 0.7],'LineWidth',2)
grid on

legend('isoupseis1','isoupseis2','Pw1=0.1','Pw1=0.25','Pw1=0.5','Pw1=0.75','Pw1=0.9');

hold off

%------------------------------------------ S1=S2--------------------------------------------------------


figure(2)
hold on
Y1 = mvnpdf([X1(:) X2(:)] ,mu1,S2);
Y1_reshape = reshape(Y1,length(x2),length(x1));
contour(x1,x2,Y1_reshape,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999])
grid on

hold on
Y2 = mvnpdf([X1(:) X2(:)] ,mu2,S2);
Y2_reshape = reshape(Y2,length(x2),length(x1));
contour(x1,x2,Y2_reshape,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999])
grid on

Pw1 = 0.1 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,'b','LineWidth',2)
grid on

Pw1 = 0.25 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,'g','LineWidth',2)
grid on

Pw1 = 0.5 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,'r','LineWidth',2)
grid on

Pw1 = 0.75 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,[0.9 0 0.9],'LineWidth',2)
grid on

Pw1 = 0.9 ;
Pw2 = 1- Pw1 ;
eqn1 = Pw1.*Y1 - Pw2.*Y2;
eqn1_reshape = reshape(eqn1,length(x2),length(x1));
eqn1_reshape (eqn1_reshape<0)= 0;
eqn1_reshape (eqn1_reshape>0) =1;
hold on 
contour(x1,x2,eqn1_reshape,[0.5 0.3 0.7],'LineWidth',2)
grid on

legend('isoupseis1','isoupseis2','Pw1=0.1','Pw1=0.25','Pw1=0.5','Pw1=0.75','Pw1=0.9');

hold off