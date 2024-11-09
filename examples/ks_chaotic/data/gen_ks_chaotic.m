%% Kuramoto-Sivashinsky equation and chaos
nn = 511;
steps = 250;

dom = [0 2*pi]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) - 100 / 16^2 *  diff(u,2) - 100 / 16^4 * diff(u,4);
S.nonlin = @(u) - 100 / 16 * 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = cos(x).*(1+sin(x));
% S.init = -sin(pi*x/50);
u = spin(S,nn,1e-5, 'plot', 'off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(0,2*pi,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('ks_chaotic.mat','t','x','usol')