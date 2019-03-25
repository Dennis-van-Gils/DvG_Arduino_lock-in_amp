set(0, 'DefaultLineLineWidth', 2)
set(0, 'DefaultAxesColor', [0 0 0])
set(0, 'DefaultAxesXColor', [1 0 0])
set(0, 'DefaultAxesYColor', [1 0 0])
set(0, 'DefaultAxesFontSize', 14)
set(0, 'DefaultAxesXGrid', 'on')
set(0, 'DefaultAxesYGrid', 'on')

% ---------------------
%  Settings
% ---------------------

f_ref = 100;      % Reference frequency [Hz]
A_ref = 1;        % Reference amplitude [V]

f_sig = 100;      % Signal main frequency [Hz]
A_sig = 1;        % Signal amplitude [V]
phi_deg = 10;     % Phase offset [deg]
phi = phi_deg/180*pi; % Phase offset [rad]

g = 1;            % Pre-amplification factor

% ---------------------
%  Perform
% ---------------------

Fs = 5e3;            % sample rate [Hz]
t = 0:1/Fs:2;     % time [s]

V_I  = g * A_sig * cos(2*pi*f_sig*t + phi);
V_RX =     A_ref * cos(2*pi*f_ref*t);
V_RY =     A_ref * sin(2*pi*f_ref*t);

V_MX = V_I .* V_RX;
V_MY = V_I .* V_RY;

lpFilt = designfilt('lowpassfir', ...
                    'FilterOrder', 2000, ...
                    'CutoffFrequency', 195, ...
                    'DesignMethod', 'window', ...
                    'Window', 'blackman', ...
                    'SampleRate', Fs);
V_out_X = filtfilt(lpFilt, V_MX);
V_out_Y = filtfilt(lpFilt, V_MY);

V_out_amp = sqrt(V_out_X.^2 + V_out_Y.^2);
V_out_phi = atand(V_out_Y ./ V_out_X);
                  
% ---------------------
%  Plotting
% ---------------------

h1 = figure(1); clf

h1a = subplot(4, 1, 1);
plot(t, V_RX, '-m', 'DisplayName', 'V\_RX')
hold on
plot(t, V_RY, '-y', 'DisplayName', 'V\_RY')
plot(t, V_I, '-c', 'DisplayName', 'V\_I')
title('Reference and signal')
h_leg = legend('Location', 'NEO');
set(h_leg, 'TextColor', [1, 1, 1])

h1b = subplot(4, 1, 2);
plot(t, V_MX, '-m', 'DisplayName', 'V\_MX')
hold on
plot(t, V_MY, '-y', 'DisplayName', 'V\_MY')
title('Mixer')
h_leg = legend('Location', 'NEO');
set(h_leg, 'TextColor', [1, 1, 1])

h1c = subplot(4, 1, 3);
plot(t, V_out_X, '-m', 'DisplayName', 'V\_MX_LP')
title('V\_MX\_LP')
h1d = subplot(4, 1, 4);
plot(t, V_out_Y, '-y', 'DisplayName', 'V\_MY_LP')
title('V\_MY\_LP')


h2 = figure(2); clf

h2a = subplot(2, 1, 1);
plot(t, V_out_amp, '-m')
title('LIA output: amplitude')
h2b = subplot(2, 1, 2);
plot(t, V_out_phi, '-m')
title('LIA output: phase')

linkaxes([h1a, h1b, h1c, h1d, h2a, h2b], 'x')
xlim([0.5 0.520])