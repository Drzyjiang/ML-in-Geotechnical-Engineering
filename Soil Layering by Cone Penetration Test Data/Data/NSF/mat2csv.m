%% 2025.5.10: This script converts NSF .mat data to .csv

% Load data
load NorthAmerica_Dataset_V1.mat

% This is the dataId specified in the reference document
dataId = 981;

%% get data
clear data;

% multipliers for converting meter to ft, MPa to tsf
m2ft = 1/0.3048;
MPa2tsf = 9.324;

% use zero as porepressure
shape = size(NorthAmerica{1,dataId}.depth_m');
rowSize = shape(1,1);
porepressure = zeros(rowSize, 1);

data = [NorthAmerica{1,dataId}.depth_m' * m2ft, NorthAmerica{1,dataId}.qc_mpa'*MPa2tsf, ...
    NorthAmerica{1,dataId}.fs_mpa' * MPa2tsf, porepressure];


%% Write data
fileName = strcat(string(dataId), ".csv");
header = {"Depth (ft)",	"Cone resistance (tsf)", "Sleeve friction (tsf)", "Pore pressure u2 (psi)"};

%writematrix(header, fileName, "WriteMode", 'append')
%T = array2table(data, 'VariableNames', header);
combined = [header; num2cell(data)];
writecell(combined, fileName);
%writetable(T, fileName);
%csvwrite(fileName, header, 0,0);
%csvwrite(fileName, data, 1,0);

%% Simply plot data
subplot(1,2,1)
plot(NorthAmerica{1,dataId}.qc_mpa'*MPa2tsf, NorthAmerica{1,dataId}.depth_m' * m2ft); 
axis ij;
grid on;

subplot(1,2,2)
plot(NorthAmerica{1,dataId}.fs_mpa' * MPa2tsf, NorthAmerica{1,dataId}.depth_m' * m2ft); 
axis ij;
grid on;