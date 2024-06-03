clear all; close all; clc;
figure(1)

% 先创建一个符合尺寸的图片
set(gcf,'unit','centimeters','position',[10 5 17.4 10]); % 10cm*17.4cm
set(gcf,'ToolBar','none','ReSize','off');   % 移除工具栏
set(gcf,'color','w'); % 背景设为白色

% 准备几个数据画图
t=0:0.1:1;
y1=t;
y2=t.^2;
y3=t+0.1;
y4=t.^4+0.1;
y5=t.^3+0.1;
y6=t.^4-0.1*t;


subplot(2,2,1) 
 
p1 = plot(t,y1,'b--','LineWidth',1.5);
hold on
p2 = plot(t,y2,'color',[0 0.81 0.82],'LineWidth',1.5);
set(gca,'Position',[0.1 0.6 0.43 0.27]);%第(1)个图的位置
g = get(p1,'Parent');%对应p1所在的坐标轴
set(g,'Linewidth',1.5,'FontSize',10,'FontName','Arial','FontWeight','bold');
ylabel('Error [m]','FontSize',10,'FontName','Arial','FontWeight','bold');
ylim([0,2])
xlabel({'t [s]','(a)'},'FontSize',10,'FontName','Arial','FontWeight','bold');
 
%增加局部放大
axes('position',[0.15,0.75,0.1,0.1]);%局部放大图位置
p1 = plot(t,y1,'b--','LineWidth',1.5);
hold on
p2 = plot(t,y2,'color',[0 0.81 0.82],'LineWidth',1.5);
axis([0.85 1 0.8 0.9]);%坐标范围设置
set(gca,'YTick',0.8:0.05:0.9,'FontName','Arial','FontSize',10,'FontWeight','bold');%y坐标轴设置
set(gca,'XTick',0.85:0.05:1,'FontName','Arial','FontSize',10,'FontWeight','bold');%x坐标轴设置
set(gca,'LineWidth',1.5);
 
subplot(2,2,2) 
p3 = plot(t,y3,'k--','LineWidth',1.5);
set(gca,'Position',[0.54 0.6 0.43 0.27]);%第(2)个图的位置
g = get(p3,'Parent');%对应p1所在的坐标轴
set(g,'Linewidth',1.5,'FontSize',10,'FontName','Arial','FontWeight','bold');
ylim([0,2])
% 共用坐标轴的时候可以隐藏其中一些坐标轴
set(g,'YTick',[]);
%ylabel('Error [m]','FontSize',10,'FontName','Arial','FontWeight','bold');
xlabel({'t [s]','(b)'},'FontSize',10,'FontName','Arial','FontWeight','bold');
 
subplot(2,2,3) 
p4 = plot(t,y4,'r--','LineWidth',1.5);
hold on
p5 = plot(t,y5,'color',[0 0.3 0.82],'LineWidth',1.5);
set(gca,'Position',[0.1 0.16 0.43 0.27]);%第(3)个图的位置
g = get(p4,'Parent');%对应p1所在的坐标轴
set(g,'Linewidth',1.5,'FontSize',10,'FontName','Arial','FontWeight','bold');
ylim([0,2])
ylabel('Error [m]','FontSize',10,'FontName','Arial','FontWeight','bold');
xlabel({'t [s]','(c)'},'FontSize',10,'FontName','Arial','FontWeight','bold');
 
subplot(2,2,4) 
p6 = plot(t,y6,'g','LineWidth',1.5);
set(gca,'Position',[0.54 0.16 0.43 0.27]);%第(4)个图的位置
g = get(p6,'Parent');%对应p1所在的坐标轴
set(g,'Linewidth',1.5,'FontSize',10,'FontName','Arial','FontWeight','bold');
ylim([0,2])
set(g,'YTick',[]);
%ylabel('Error [m]','FontSize',10,'FontName','Arial');
xlabel({'t [s]','(d)'},'FontSize',10,'FontName','Arial','FontWeight','bold');
 
h1=legend([p1 p2 p3 p4 p5 p6],'\fontname{Arial}y_1','\fontname{Arial}y_2','\fontname{Arial}y_3',...
    '\fontname{Arial}y_4','\fontname{Arial}y5','\fontname{Arial}y6','Orientation','horizontal');
set(h1,'Linewidth',1.5,'FontSize',10,'FontWeight','bold');
set(h1,'position',[0.4,0.9,0.2,0.1]);%legend位置
set(h1,'Box','off');
 
%text(0.05,0.6,'(a)')