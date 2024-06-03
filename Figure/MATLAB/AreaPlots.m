% https://mp.weixin.qq.com/s/cs97Qt_YahuAq-yEzlwOTg
clc;clear;close all;
%面积图
color=[0.925490196078431  0.705882352941177  0.690196078431373;
0.862745098039216  0.450980392156863  0.466666666666667;
0.823529411764706  0.635294117647059  0.619607843137255;
0.701960784313725  0.360784313725490  0.368627450980392;
0.611764705882353  0.698039215686275  0.788235294117647;
0.250980392156863  0.501960784313726  0.639215686274510;
0.368627450980392  0.462745098039216  0.552941176470588;
0.411764705882353  0.525490196078431  0.615686274509804];
x=[1:1:9];
y1=[0,2,2.5,2.7,3,2.4,1.9,1.6,0];
y2=[0,1.2,1.4,1.5,2,1.6,1.3,0.7,0];
y3=[0,1.2,1.4,1.3,1.5,1.3,1.0,0.6,0];
area(x,y1,'FaceAlpha',.7,'FaceColor',color(1,:),'EdgeColor',color(2,:),'LineWidth',2)
hold on
area(x+1,y1,'FaceAlpha',.7,'FaceColor',color(3,:),'EdgeColor',color(4,:),'LineWidth',2)
hold on
area(x+5,y2,'FaceAlpha',.6,'FaceColor',color(5,:),'EdgeColor',color(6,:),'LineWidth',2)
hold on
area(x+7,y3,'FaceAlpha',.6,'FaceColor',color(7,:),'EdgeColor',color(8,:),'LineWidth',2)
hold on
ax = gca;
ax.YLim=[0,4];
set(gca,"FontName","Times New Roman","FontSize",12,"LineWidth",1.5)
box off