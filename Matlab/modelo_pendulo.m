clear all; clc;

s = tf("s");

load("lambda.mat")

% Parâmetros

g = 9.81;

Mh = 0.149; %kg
Mr = 0.144; %kg
L = 0.14298; %m
d = 0.0987; %m
i_stall = 1.8; %A
tau_stall = 0.3136; %Nm
w2_noload = 380; %RPM
i_noload = 0.1; %A
r = 100e-3;
r_in = r-8.9e-3;

u_max = 12; %V

Jh = 1/3 * Mh * L^2;
Jr = 1/2 * Mr * (r^2+r_in^2);
b1 = 0;
b2 = 2*lambda*(Jh+Jr);

Kt = 0;
Kv = 0;

%Rm = u_nom / i_stall
Rm = u_max / i_stall;

% Matrizes REE
l_31 = -(Mr*g*L + Mh*g*d)/(Mr*L^2 + Jh);
l_33 = -b1/(Mr*L^2 + Jh);
l_34 = (b2 + Kt*Kv/Rm)/(Mr*L^2 + Jh);
l_41 = (Mr*g*L + Mh*g*d)/(Mr*L^2 + Jh);
l_43 = b1/(Mr*L^2 + Jh);
l_44 = -((Mr*L^2 + Jh + Jr)*(b2 + Kt*Kv/Rm))/(Jr*(Mr*L^2 + Jh));

A = [0     0    1    0;
     0     0    0    1;
     l_31  0    l_33 l_34;
     l_41  0    l_43 l_44];

l_3 = -(12*Kt)/(Rm*(Mr*L^2 + Jh));
l_4 = (12*Kt*(Mr*L^2 + Jh + Jr))/(Rm*Jr*(Mr*L^2 + Jh));

B = [0
     0;
     l_3;
     l_4];

C = [1 0 0 0];

D = 0;

% Simulação

sys = ss(A, B, C, D); % Create state-space system
t = 0:0.01:5; % Time vector for simulation
u = u_max * ones(size(t)); % Step input
y = lsim(sys, u, t); % Simulate the system response

