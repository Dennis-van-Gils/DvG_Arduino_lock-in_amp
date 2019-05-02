% Low-pass filter design at analog inputs to prevent aliasing
% Simple RC low-pass filter

% Given
if 1
  R = 12;       % [Ohm]
  C = 4.7e-6;   % [F]
else
  R = 560;      % [Ohm]
  C = 100e-9;   % [F]
end

% Results in corner frequency:
f_c = 1/(2*pi*R*C);   % [Hz]

% Calculate voltage output at
V_in = 1;                 % [VAC]
% and over frequency range
f_max = 1e4;              % [Hz]
f = 0:1:f_max;            % [Hz]
% --> Capacitance reactance
X_C = 1./(2*pi*f*C);      % [Ohm]
% --> Impedance
Z = sqrt(R^2 + X_C.^2);   % [Ohm]
% --> Voltage output
V_out = V_in * X_C ./ Z;  % [VAC]
% --> Phase response
phi = -atand(2*pi*f*R*C); % [deg]

fprintf('R = %.1f Ohm\n', R)
fprintf('C = %.1e F\n', C)
fprintf('f_c = %.1f Hz\n', f_c)

figure(1); clf

subplot(3, 1, 1)
semilogx(f, 20*log10(V_out/V_in), '.-')
xlim([1 f_max])
hold on
plot([f_c f_c], ylim, '-k')
plot(xlim, 10*[-log10(2) -log10(2)], '-k')
xlabel('f (Hz)')
ylabel('amplitude attenuation (dB)')
title(sprintf(['low-pass RC circuit\n' ...
               'R = %.1f \\Omega, C = %.0f nF, f_c = %.1f Hz'], ...
               R, C*1e9, f_c))

subplot(3, 1, 2)
semilogx(f, phi, '.-')
xlim([1 f_max])
hold on
plot([f_c f_c], ylim, '-k')
plot(xlim, [-45 -45], '-k')
xlabel('f (Hz)')
ylabel('phase response (deg)')

subplot(3, 1, 3)
semilogx(f, Z, '.-')
xlim([1 f_max])
hold on
plot([f_c f_c], ylim, '-k')
xlabel('f (Hz)')
ylabel('impedance (Ohm)')