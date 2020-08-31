close all;
clear;
clc;

#pkg load statistics
H=[3.1803, 3.0208, 6.3968, 5.0169, 4.0323,6.2086, 5.3474, 3.5154, 7.0094, 5.4343,7.2184, 6.7070, 3.9738, 3.0621, 4.8906, 4.5041, 3.7346, 5.7467, 7.2327, 4.1803,3.7299, 4.6305, 5.9945, 3.7187, 3.1980]';

% A erwthma
x = -0.5:0.1:8;
m0 = 0;                         
s = 1.25^2;             
s0 = 10*s;          
n = size(H,1);


figure;
hold on;
for i=1:n
    plot(x,normpdf(x,(n*s0/(n*s0+s))*mean(H(1:i,1)),sqrt((s0*s)/(i*s0+s))),'linewidth',2)
end
hold off;

% B erwthma
x = -3:0.1:10;


figure;
hold on;

plot(x,normpdf(x,(n*s0/(n*s0+s))*mean(H(1:n,1)),sqrt(s+(s0*s)/(n*s0+s))),'linewidth',2)

s0 = s;
plot(x,normpdf(x,(n*s0/(n*s0+s))*mean(H(1:n,1)),sqrt(s+(s0*s)/(n*s0+s))),'linewidth',2)

s0 = 0.1*s;
plot(x,normpdf(x,(n*s0/(n*s0+s))*mean(H(1:n,1)),sqrt(s+(s0*s)/(n*s0+s))),'linewidth',2)

s0 = 0.01*s;
plot(x,normpdf(x,(n*s0/(n*s0+s))*mean(H(1:n,1)),sqrt(s+(s0*s)/(n*s0+s))),'linewidth',2)

hold off