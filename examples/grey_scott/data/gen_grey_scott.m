%% 1. Rolls
% The Gray-Scott equations are a pair of coupled reaction-diffusion

% The equations are
% $$ u_t = \varepsilon_1\Delta u + b(1-u) - uv^2, \quad
% v_t = \varepsilon_2\Delta v - dv + uv^2, $$
% where $\Delta$ is the Laplacian and $\varepsilon_1,
% \varepsilon_2,b,d$ are parameters.
% To begin with we choose these values.

% nondimensionalization factor
k = 1000;

ep1 = 0.0002 * k; ep2 = 0.0001 * k;
b1 = 0.04 * k; c1 = k;
b2 = 0.1 * k; c2 = k;


nn = 200;
steps = 100;
dt = 1 / k;

dom = [-1 1 -1 1]; x = chebfun('x',dom(1:2)); t = linspace(0,2000/k, steps+1);
S = spinop2(dom, t);
S.lin = @(u,v) [ep1*lap(u); ep2*lap(v)];
S.nonlin = @(u,v)  [b1*(1-u)- c1 * u.*v.^2; 
                   -b2*v + c2 * u.*v.^2];
S.init = chebfun2v(@(x,y) 1-exp(-10*((x+.05).^2+(y+.02).^2)), ...
                   @(x,y) exp(-10*((x-.05).^2+(y-.02).^2)),dom);
               
tic, u = spin2(S,nn,dt, 'plot', 'off');

plot(u{1, steps}), view(0,90), axis equal, axis off
time_in_seconds = toc

N = 200;
[X,Y] = meshgrid(linspace(-1,1, N), linspace(-1,1, N));

usol = zeros(steps+1, N, N);
for i = 1:steps+1
    usol(i,:,:) = u{1, i}(X,Y);
end

vsol = zeros(steps+1, N, N);
for i = 1:steps+1
    vsol(i,:,:) = u{2, i}(X,Y);
end

x = linspace(-1,1, N);
y = linspace(-1,1, N);

save('grey_scott.mat', 'b1', 'b2', 'c1', 'c2',  'ep1', 'ep2', 'usol', 'vsol', 't', 'x', 'y')







