
clc;clear;close all;
%定义原始结构数据
D1 = [5 3 9 1 2];
D2 = [5 8 7 2 9];
D3 = [8 2 1 4 6];
P = [D1; D2; D3];

%创建蜘蛛网绘图函数类
s = spider_plot_class(P);

%设置蜘蛛网绘图属性
s.AxesLabels = {'S1', 'S2', 'S3', 'S4', 'S5'};
s.AxesInterval = 4;
s.AxesPrecision = 0;
s.AxesDisplay = 'one';
s.AxesLimits = [1, 2, 1, 1, 1; 10, 8, 9, 5, 10];
s.FillOption = 'on';
s.FillTransparency =  0.2;
s.Color = [1, 0, 0; 0, 1, 0; 0, 0, 1];
s.LineStyle = '--';
s.LineWidth = 3;
s.LineTransparency = 1;
s.Marker =  'd';
s.MarkerSize = 10;
s.MarkerTransparency = 1;
s.AxesFont = 'Times New Roman';
s.LabelFont = 'Times New Roman';
s.AxesFontSize = 12;
s.LabelFontSize = 10;
s.Direction = 'clockwise';
s.AxesDirection = {'reverse', 'normal', 'normal', 'normal', 'normal'};
s.AxesLabelsOffset = 0.2;
s.AxesDataOffset = 0.1;
s.AxesScaling = 'linear';
s.AxesColor = [0.6, 0.6, 0.6];
s.AxesLabelsEdge = 'none';
s.AxesOffset = 1;
s.LegendLabels = {'D1', 'D2', 'D3'};
s.LegendHandle.Location = 'northeastoutside';
s.PlotVisible = 'on';
s.AxesTickLabels = 'data';
s.AxesInterpreter = 'tex';
s.BackgroundColor = 'w';
s.MinorGrid = 'off';
s.MinorGridInterval = 2;
s.AxesZero = 'off';
s.AxesZeroColor = 'k';
s.AxesZeroWidth = 2;
s.AxesRadial = 'on';
s.AxesWeb = 'on';
s.AxesShaded = 'off';
s.AxesShadedLimits = [];
s.AxesShadedColor = 'g';
s.AxesShadedTransparency = 0.2;