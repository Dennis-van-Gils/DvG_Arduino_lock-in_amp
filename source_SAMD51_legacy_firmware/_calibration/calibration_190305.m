V_setp     = [0     0.1   0.2   0.5   1.0   1.5   2.0   2.5   3.0   3.2   3.3];
V_read_osc = [0.003 0.104 0.203 0.505 1.00  1.50  2.00  2.51  3.01  3.21  3.30];
V_read_ard = [0.060 0.110 0.203 0.505 1.007 1.510 2.013 2.516 3.018 3.219 3.299];

V_diff_osc = V_read_osc - V_setp;
V_diff_ard = V_read_ard - V_setp;

h1 = figure(1); clf
plot(V_setp, V_read_osc, '-ok', 'LineWidth', 2)
hold on
plot(V_setp, V_read_ard, '-or', 'LineWidth', 2)

h2 = figure(2); clf
plot(V_setp, V_diff_osc, '-ok', 'LineWidth', 2, 'DisplayName', 'read oscilloscope')
hold on
plot(V_setp, V_diff_ard, '-or', 'LineWidth', 2, 'DisplayName', 'read Arduino')

str_title = sprintf('%s - Feather M4 Express', datestr(date, 'dd-mm-yyyy'));
title(str_title)
xlabel('Arduino output voltage (V)')
ylabel('deviation (V)')

% Fit over the selection range: 0.2 <= V <= 3.2
sel_V_setp = V_setp(3:end-1);
sel_V_diff_ard = V_diff_ard(3:end-1);
pfit = polyfit(sel_V_setp, sel_V_diff_ard, 1);

fit_V = 0:.01:3.3;
str_fit = sprintf('fit: dev = %.4f x + %.4f (V)', pfit(1), pfit(2));
plot(fit_V, polyval(pfit, fit_V), '-b', 'LineWidth', 2, 'DisplayName', str_fit)
legend('Location', 'NE')

saveFig(2, str_title)