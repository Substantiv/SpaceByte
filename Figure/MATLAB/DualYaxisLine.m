clc;clear;close all;

% x 
x = 0 : 0.01 : 20;

% y1
y1 = 200*exp(-0.05*x).*sin(x);

% y2
y2 = 0.8*exp(-0.5*x).*sin(10*x);

yyaxis left,  % Activates the left y-axis
plot(x,y1,'LineWidth',1.5,'LineStyle','--');
ylabel('Curve in Left Side (thousands)')

yyaxis right, % Activates the right y-axis
plot(x,y2,'LineWidth',2.0,'LineStyle','-');
ylabel('Curve in Right Side (%)')
xlabel('X-axes variable')

ax = gca;
ax.YAxis(1).Exponent = 3;

legtxt = {'$y_1=200exp(-0.05x)sin(x)$','$y_2=0.8exp (-0.5x)sin(10x)$'};

% Title
title('Line chart with two y-axes','FontSize',14);
legend(legtxt,'Interpreter','Latex','FontSize',12)
grid on